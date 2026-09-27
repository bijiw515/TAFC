#!/usr/bin/env python3
"""
Batch version of tafc_generate.py for HunyuanVideo - loads model once and processes multiple prompts.

This enables efficient VBench evaluation by avoiding repeated model loading overhead.
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List

import torch
from loguru import logger

# HunyuanVideo imports
from hyvideo.config import parse_args
from hyvideo.inference import HunyuanVideoSampler
from hyvideo.utils.file_utils import save_videos_grid

# TAFC imports
from util_tafc import (
    TAFCBranch,
    TAFCController,
    get_physical_timestep,
    print_tafc_stats,
    print_tafc_summary,
    should_force_compute,
)


# ============================================================================
# Import TAFC forward pass and configuration from tafc_generate.py
# ============================================================================

from tafc_generate import (
    tafc_forward,
    configure_tafc,
    add_tafc_args,
)


def load_prompts(prompt_list_file: str) -> List[str]:
    """Load prompts from a text file (one prompt per line)."""
    with open(prompt_list_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


def safe_prompt_name(prompt: str) -> str:
    """Convert prompt to filesystem-safe name."""
    return prompt.replace("/", "_").strip()


def generate_batch(args, sampler: HunyuanVideoSampler, prompts: List[str], output_dir: Path, stats_fn):
    """
    Generate videos for all prompts using a single loaded model.

    Args:
        args: Parsed arguments
        sampler: Pre-loaded HunyuanVideoSampler
        prompts: List of text prompts
        output_dir: Directory to save generated videos
        stats_fn: Function to get TAFC statistics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load or initialize stats file
    stats_file = output_dir / "gen_stats.json"
    if stats_file.exists():
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        logger.info(f"Loaded existing stats: {len([k for k in stats if k != '_config'])} videos")
    else:
        stats = {}

    # Save config info
    if '_config' not in stats:
        stats['_config'] = {
            'tafc_thresh': float(args.tafc_thresh),
            'tafc_max_cache': float(args.tafc_max_cache),
            'use_ret_steps': bool(args.use_ret_steps),
            'infer_steps': int(args.infer_steps),
            'video_size': args.video_size,
            'video_length': int(args.video_length),
            'seed': int(args.seed),
        }

    logger.info(f"Processing {len(prompts)} prompts")
    logger.info(f"Output directory: {output_dir}")

    # Verify CUDA is ready (cuDNN should remain disabled)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
        logger.info(f"cuDNN status: {'DISABLED' if not torch.backends.cudnn.enabled else 'enabled'}")

    for idx, prompt in enumerate(prompts):
        safe_name = safe_prompt_name(prompt)
        video_filename = f"{safe_name}-0.mp4"
        video_path = output_dir / video_filename

        # Skip if already generated
        if video_path.exists():
            logger.info(f"[{idx+1}/{len(prompts)}] SKIP (exists): {prompt[:60]}...")
            continue

        logger.info(f"[{idx+1}/{len(prompts)}] Generating: {prompt}")

        try:
            # Ensure model is in eval mode and on correct device
            sampler.pipeline.transformer.eval()

            # Reset TAFC state explicitly before each generation
            # Reset INSTANCE attributes (matching FLUX implementation)
            tr = sampler.pipeline.transformer
            tr.cnt = 0
            if hasattr(tr, 'tafc_branch'):
                tr.tafc_branch.reset()
                logger.debug(f"[{idx+1}/{len(prompts)}] TAFC state reset")

            # Ensure CUDA is ready
            if idx > 0 and torch.cuda.is_available():
                torch.cuda.synchronize()

            # Use base seed + index for reproducibility
            current_seed = args.seed + idx

            # Generate video
            t0 = time.time()
            outputs = sampler.predict(
                prompt=prompt,
                height=args.video_size[0],
                width=args.video_size[1],
                video_length=args.video_length,
                seed=current_seed,
                negative_prompt=args.neg_prompt,
                infer_steps=args.infer_steps,
                guidance_scale=args.cfg_scale,
                num_videos_per_prompt=1,
                flow_shift=args.flow_shift,
                batch_size=args.batch_size,
                embedded_guidance_scale=args.embedded_cfg_scale
            )
            elapsed = time.time() - t0
            samples = outputs['samples']

            # Get TAFC stats
            tafc_stats = stats_fn()

            logger.info(
                f"[{idx+1}/{len(prompts)}] Done in {elapsed:.1f}s | "
                f"cache_rate={tafc_stats['cache_rate_pct']:.1f}% | "
                f"cached={tafc_stats['cached_steps']}/{tafc_stats['total_steps']}"
            )

            # Save video
            if 'LOCAL_RANK' not in os.environ or int(os.environ['LOCAL_RANK']) == 0:
                sample = samples[0].unsqueeze(0)
                save_videos_grid(sample, str(video_path), fps=24)
                logger.info(f"Saved to: {video_path}")

                # Update stats
                stats[safe_name] = {
                    'prompt': prompt,
                    'video_file': video_filename,
                    'seed': current_seed,
                    'seconds': round(elapsed, 1),
                    'cache_rate_pct': tafc_stats['cache_rate_pct'],
                    'cached_steps': tafc_stats['cached_steps'],
                    'computed_steps': tafc_stats['computed_steps'],
                    'total_steps': tafc_stats['total_steps'],
                    'timestamp': datetime.now().isoformat(),
                    'index': idx,
                }

                # Add diagnostics if available
                for k in ['veto_max_consec', 'veto_e_reject', 'veto_force_compute', 'gain', 'brake']:
                    if k in tafc_stats:
                        stats[safe_name][k] = tafc_stats[k]

                # Save stats after each video
                with open(stats_file, 'w') as f:
                    json.dump(stats, f, indent=2, ensure_ascii=False)

            # Cleanup - aggressive CUDA memory management
            del outputs, samples
            torch.cuda.synchronize()  # Wait for all ops to complete
            torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"[{idx+1}/{len(prompts)}] ERROR: {e}")
            import traceback
            traceback.print_exc()

            # Record error in stats
            stats[safe_name] = {
                'prompt': prompt,
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'index': idx,
            }
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)

            # After an error, aggressively clean up CUDA state
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            continue

    # Print summary
    done = [v for k, v in stats.items()
            if k != '_config' and 'error' not in v]
    if done:
        avg_time = sum(v['seconds'] for v in done) / len(done)
        avg_cache = sum(v['cache_rate_pct'] for v in done) / len(done)
        logger.info("=" * 80)
        logger.info(f"Batch complete: {len(done)}/{len(prompts)} videos")
        logger.info(f"Average time: {avg_time:.1f}s/video")
        logger.info(f"Average cache rate: {avg_cache:.1f}%")
        logger.info("=" * 80)


def main():
    # CRITICAL: Disable cuDNN early to prevent CUDNN_STATUS_NOT_INITIALIZED
    if torch.cuda.is_available():
        torch.cuda.init()
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.cuda.synchronize()
        logger.info("CUDA initialized with cuDNN disabled (using native CUDA backend)")

    # Parse TAFC arguments first
    tafc_parser = argparse.ArgumentParser(add_help=False)
    add_tafc_args(tafc_parser)

    # Add batch-specific arguments
    tafc_parser.add_argument("--prompt_list", type=str, required=True,
                            help="File containing one prompt per line")
    tafc_parser.add_argument("--output_dir", type=str, required=True,
                            help="Directory to save generated videos")
    tafc_parser.add_argument("--flow-reverse", action="store_true", default=False,
                            help="Enable flow reverse")

    # Show help if requested
    if "-h" in sys.argv[1:] or "--help" in sys.argv[1:]:
        print("=" * 80)
        print("HunyuanVideo Batch Generation with TAFC")
        print("=" * 80)
        print(tafc_parser.format_help())

    tafc_args, remaining = tafc_parser.parse_known_args()

    # Parse HunyuanVideo arguments
    argv_backup = sys.argv
    sys.argv = [argv_backup[0]] + remaining
    try:
        args = parse_args(namespace=tafc_args)
    finally:
        sys.argv = argv_backup

    # Validate paths
    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"Model path not found: {models_root_path}")

    # Load prompts
    prompts = load_prompts(args.prompt_list)
    logger.info(f"Loaded {len(prompts)} prompts from {args.prompt_list}")

    # Load model ONCE
    logger.info("=" * 80)
    logger.info("LOADING MODEL (once for all prompts)")
    logger.info("=" * 80)

    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(
        models_root_path, args=args)

    # Get updated args from sampler
    args = hunyuan_video_sampler.args

    # Override video dimensions to ensure consistent output
    args.video_size = (480, 832)  # height=480, width=832
    args.video_length = 65  # 65 frames
    logger.info(f"[Video Config] Fixed dimensions: {args.video_size[0]}x{args.video_size[1]}, {args.video_length} frames")

    # Configure TAFC
    stats_fn = configure_tafc(hunyuan_video_sampler, args)

    tr = hunyuan_video_sampler.pipeline.transformer
    mode = "open-loop" if args.tafc_no_pid else "CLOSED-LOOP (PID)"
    logger.info(
        f"[TAFC] {mode} | thresh={args.tafc_thresh} | max_cache={args.tafc_max_cache} "
        f"| steps={args.infer_steps} | ret_steps={tr.ret_steps} "
        f"| cutoff_steps={tr.cutoff_steps}")

    if not args.tafc_no_pid:
        tgt = "auto" if args.tafc_e_target <= 0 else f"{args.tafc_e_target:g}"
        logger.info(
            f"[TAFC] e_target={tgt} (auto_tol={args.tafc_auto_tol}) "
            f"| kp={args.tafc_kp} ki={args.tafc_ki} kd={args.tafc_kd} "
            f"| gain=[{args.tafc_gain_min}, {args.tafc_gain_max}] "
            f"| brake_beta={args.tafc_brake_beta} floor={args.tafc_brake_floor}")

    logger.info("Model loaded successfully")
    logger.info("=" * 80)

    # Generate all videos
    generate_batch(args, hunyuan_video_sampler, prompts, args.output_dir, stats_fn)


if __name__ == "__main__":
    main()
