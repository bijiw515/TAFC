"""
Batch version of tafc_generate.py - loads model once and processes multiple prompts.
"""
import argparse
from datetime import datetime
import logging
import os
import sys
import warnings
import json
from pathlib import Path

warnings.filterwarnings('ignore')

import torch, random
import torch.distributed as dist
from PIL import Image

import wan
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS, SUPPORTED_SIZES
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import cache_video, cache_image, str2bool

import gc
from contextlib import contextmanager
import torchvision.transforms.functional as TF
import torch.cuda.amp as amp
import numpy as np
import math
from wan.modules.model import sinusoidal_embedding_1d
from wan.utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                              get_sampling_sigmas, retrieve_timesteps)
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from tqdm import tqdm
from util_seacache import rel_l1, apply_sea_with_scheduler
from util_tafc import (
    TAFCBranch,
    TAFCController,
    get_physical_timestep,
    print_tafc_stats,
    print_tafc_summary,
)

# Import the necessary functions from tafc_generate.py
import tafc_generate
from tafc_generate import (
    t2v_generate,
    i2v_generate,
    tafc_forward,
    _parse_args,
    _validate_args,
    _init_logging,
    _make_controller,
    _apply_tafc_hooks,
    EXAMPLE_PROMPT,
)


def generate_batch(args, prompt_list_file, output_dir):
    """
    Load model once and generate videos for all prompts in the list.

    Args:
        args: Arguments from argparse
        prompt_list_file: Path to file containing one prompt per line
        output_dir: Directory to save generated videos
    """
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(f"offload_model is not specified, set to {args.offload_model}.")

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)
    else:
        assert not (args.t5_fsdp or args.dit_fsdp), \
            f"t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (args.ulysses_size > 1 or args.ring_size > 1), \
            f"context parallel are not supported in non-distributed environments."

    # Load prompt list
    with open(prompt_list_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]

    logging.info(f"Loaded {len(prompts)} prompts from {prompt_list_file}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load stats file if it exists
    stats_file = os.path.join(output_dir, "gen_stats.json")
    if os.path.exists(stats_file):
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        logging.info(f"Loaded existing stats: {len(stats)} videos already generated")
    else:
        stats = {}

    cfg = WAN_CONFIGS[args.task]
    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")
    logging.info("Using TAFC (Target-Anchored Flow Caching) with target drift estimation")

    if args.tafc_no_pid:
        logging.info("TAFC control: open-loop (PID disabled)")
    else:
        tgt = "auto" if args.tafc_e_target <= 0 else f"{args.tafc_e_target:g}"
        logging.info(
            f"TAFC control: closed-loop | e_target={tgt} | reject={args.tafc_reject:g} | "
            f"pid=({args.tafc_kp:g},{args.tafc_ki:g},{args.tafc_kd:g}) | "
            f"gain=[{args.tafc_gain_min:g},{args.tafc_gain_max:g}] | "
            f"brake_beta={args.tafc_brake_beta:g} | calib_steps={args.tafc_calib_steps}")

    # Initialize model ONCE
    logging.info("=" * 80)
    logging.info("LOADING MODEL (once for all prompts)")
    logging.info("=" * 80)

    if "t2v" in args.task or "t2i" in args.task:
        wan_t2v = wan.WanT2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
            t5_cpu=args.t5_cpu,
        )
        _apply_tafc_hooks(wan_t2v, t2v_generate, args)
        pipeline = wan_t2v
    else:
        # i2v task not supported in batch mode for now
        raise NotImplementedError("i2v task not supported in batch mode yet")

    logging.info("Model loaded successfully")
    logging.info("=" * 80)

    # Process each prompt
    for idx, prompt in enumerate(prompts):
        # Use the same naming format as the original script: {prompt}-0.mp4
        safe_prompt = prompt.replace("/", "_").strip()
        video_filename = f"{safe_prompt}-0.mp4"
        video_path = os.path.join(output_dir, video_filename)

        # Skip if already generated (check file existence)
        if os.path.exists(video_path):
            logging.info(f"[{idx+1}/{len(prompts)}] SKIP (already exists): {prompt[:60]}...")
            continue

        logging.info(f"[{idx+1}/{len(prompts)}] Generating: {prompt}")

        try:
            # Set seed for this generation
            if args.base_seed >= 0:
                seed = args.base_seed + idx
            else:
                seed = random.randint(0, sys.maxsize)

            # Generate video
            video = pipeline.generate(
                prompt,
                size=SIZE_CONFIGS[args.size],
                frame_num=args.frame_num,
                shift=args.sample_shift,
                sample_solver=args.sample_solver,
                sampling_steps=args.sample_steps,
                guide_scale=args.sample_guide_scale,
                seed=seed,
                offload_model=args.offload_model)

            # Save video
            if rank == 0:
                if "t2i" in args.task:
                    cache_image(
                        tensor=video.squeeze(1)[None],
                        save_file=video_path,
                        nrow=1,
                        normalize=True,
                        value_range=(-1, 1))
                else:
                    cache_video(
                        tensor=video[None],
                        save_file=video_path,
                        fps=cfg.sample_fps,
                        nrow=1,
                        normalize=True,
                        value_range=(-1, 1))

                # Update stats
                stats[safe_prompt] = {
                    "prompt": prompt,
                    "video_file": video_filename,
                    "seed": seed,
                    "timestamp": datetime.now().isoformat(),
                    "index": idx,
                }

                # Save stats after each video
                with open(stats_file, 'w') as f:
                    json.dump(stats, f, indent=2, ensure_ascii=False)

                logging.info(f"Saved to {video_path}")

            # Clean up
            del video
            torch.cuda.empty_cache()

        except Exception as e:
            logging.error(f"Failed to generate video {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    logging.info("=" * 80)
    logging.info(f"Batch generation complete: {len(stats)}/{len(prompts)} videos")
    logging.info("=" * 80)


if __name__ == "__main__":
    # CRITICAL: Disable cuDNN early to prevent CUDNN_STATUS_NOT_INITIALIZED
    if torch.cuda.is_available():
        torch.cuda.init()
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.cuda.synchronize()
        print("[CUDA] Initialized with cuDNN disabled (using native CUDA backend)")

    parser = argparse.ArgumentParser(description="Batch video generation with TAFC")
    parser.add_argument("--prompt_list", type=str, required=True,
                        help="File containing one prompt per line")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save generated videos")

    # Import all arguments from tafc_generate
    original_parser = argparse.ArgumentParser()
    tafc_generate._parse_args.__wrapped__ = lambda: original_parser

    # Add all original arguments (copy from tafc_generate.py _parse_args)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--config_name", type=str, default="t2v-14B")
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--task", type=str, default="t2v-14B")
    parser.add_argument("--size", type=str, default="1280x720")
    parser.add_argument("--frame_num", type=int, default=81)
    parser.add_argument("--offload_model", type=str2bool, default=None)
    parser.add_argument("--ulysses_size", type=int, default=1)
    parser.add_argument("--ring_size", type=int, default=1)
    parser.add_argument("--t5_fsdp", action="store_true", default=False)
    parser.add_argument("--t5_cpu", action="store_true", default=False)
    parser.add_argument("--dit_fsdp", action="store_true", default=False)
    parser.add_argument("--base_seed", type=int, default=-1)
    parser.add_argument("--sample_solver", type=str, default='unipc')
    parser.add_argument("--sample_steps", type=int, default=None)
    parser.add_argument("--sample_shift", type=float, default=None)
    parser.add_argument("--sample_guide_scale", type=float, default=5.0)
    parser.add_argument("--tafc_thresh", type=float, default=0.2)
    parser.add_argument("--tafc_max_cache", type=float, default=6.0)
    parser.add_argument("--use_ret_steps", action="store_true", default=False)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--tafc_no_pid", action="store_true", default=False)
    parser.add_argument("--tafc_e_target", type=float, default=0.0)
    parser.add_argument("--tafc_auto_tol", type=float, default=2.5)
    parser.add_argument("--tafc_reject", type=float, default=3.0)
    parser.add_argument("--tafc_kp", type=float, default=0.4)
    parser.add_argument("--tafc_ki", type=float, default=0.05)
    parser.add_argument("--tafc_kd", type=float, default=0.2)
    parser.add_argument("--tafc_gain_min", type=float, default=0.3)
    parser.add_argument("--tafc_gain_max", type=float, default=3.0)
    parser.add_argument("--tafc_brake_beta", type=float, default=2.0)
    parser.add_argument("--tafc_brake_floor", type=float, default=0.3)
    parser.add_argument("--tafc_calib_steps", type=int, default=3)

    args = parser.parse_args()

    generate_batch(args, args.prompt_list, args.output_dir)
