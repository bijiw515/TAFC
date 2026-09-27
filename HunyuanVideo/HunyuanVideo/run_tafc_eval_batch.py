#!/usr/bin/env python3
"""
Batch driver for TAFC evaluation on HunyuanVideo - loads model once and processes all prompts.

This script drives tafc_generate_batch.py to generate videos for VBench evaluation.
Output format matches VBench's expected structure: {prompt}-0.mp4
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent.resolve()


def main():
    parser = argparse.ArgumentParser(
        description="Run TAFC evaluation on VBench prompt list for HunyuanVideo (batch mode)"
    )
    parser.add_argument("--prompt_file", type=str,
                        default="/data/dev2/lgbi/TeaCache/eval/teacache/vbench/VBench_full_info.json",
                        help="VBench prompt file (JSON format)")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory (default: eval_samples/<config>)")
    parser.add_argument("--model_base", type=str,
                        default="/data/dev2/lgbi/SeaCache/HunyuanVideo/HunyuanVideo/ckpts/hunyuan-video-t2v-720p",
                        help="Model base directory containing HunyuanVideo checkpoints")
    parser.add_argument("--video_size", type=str, default="480,832",
                        help="Video size as height,width (e.g., 480,832)")
    parser.add_argument("--video_length", type=int, default=65,
                        help="Video length in frames (65 frames)")
    parser.add_argument("--infer_steps", type=int, default=50,
                        help="Number of inference steps")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base seed for generation")
    parser.add_argument("--cfg_scale", type=float, default=1.0,
                        help="Classifier-free guidance scale")
    parser.add_argument("--embedded_cfg_scale", type=float, default=6.0,
                        help="Embedded guidance scale")
    parser.add_argument("--flow_shift", type=float, default=7.0,
                        help="Flow shift parameter")
    parser.add_argument("--flow-reverse", action="store_true", default=False,
                        help="Enable flow reverse")
    parser.add_argument("--use_ret_steps", action="store_true", default=True,
                        help="Use retention steps (force first 5 steps)")
    parser.add_argument("--config", type=str, required=True,
                        help="Config name (e.g., tafc_0.3)")
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=None,
        help="GPU ID to use. If not set, inherits CUDA_VISIBLE_DEVICES from parent."
    )
    parser.add_argument("--tafc_thresh", type=str, default=None,
                        help="TAFC threshold (extracted from config name if not specified)")
    parser.add_argument("--tafc_max_cache", type=float, default=6.0,
                        help="Max consecutive cache budget")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="TAFC logging interval")
    parser.add_argument("--neg_prompt", type=str, default="",
                        help="Negative prompt")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size")
    args = parser.parse_args()

    # Extract threshold from config name if not explicitly set
    if args.tafc_thresh is None and "tafc_" in args.config:
        args.tafc_thresh = args.config.split("_")[1]
    elif args.tafc_thresh is None:
        args.tafc_thresh = "0.3"

    # Load prompts from VBench JSON
    with open(args.prompt_file) as f:
        prompt_data = json.load(f)
        prompts = [x["prompt_en"] for x in prompt_data]

    print(f"[driver] Loaded {len(prompts)} prompts", flush=True)

    # Prepare environment for subprocess
    env = os.environ.copy()

    # Add PyTorch library path for flash-attn
    torch_lib_path = os.path.join(os.path.dirname(sys.executable), '..', 'lib', 'python3.10', 'site-packages', 'torch', 'lib')
    if os.path.exists(torch_lib_path):
        if 'LD_LIBRARY_PATH' in env:
            env['LD_LIBRARY_PATH'] = f"{torch_lib_path}:{env['LD_LIBRARY_PATH']}"
        else:
            env['LD_LIBRARY_PATH'] = torch_lib_path

    if args.gpu_id is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
        env['LOCAL_RANK'] = '0'
        env['RANK'] = '0'
        env['WORLD_SIZE'] = '1'
        print(f"[driver] Using GPU {args.gpu_id} (mapped to cuda:0 in subprocess)", flush=True)
    else:
        if 'LOCAL_RANK' not in env:
            env['LOCAL_RANK'] = '0'
        if 'RANK' not in env:
            env['RANK'] = '0'
        if 'WORLD_SIZE' not in env:
            env['WORLD_SIZE'] = '1'
        print(f"[driver] Using CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', 'default')}", flush=True)

    # Output directory
    if args.out_dir is None:
        output_dir = HERE / "eval_samples" / args.config
    else:
        output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write prompts to a temporary text file (one prompt per line)
    temp_prompt_file = HERE / f"temp_prompts_{args.config}.txt"
    with open(temp_prompt_file, 'w', encoding='utf-8') as f:
        for p in prompts:
            f.write(p + '\n')

    # Parse video size
    video_height, video_width = map(int, args.video_size.split(','))

    # Build command for batch processing
    if args.config == "baseline":
        # Use baseline batch generation script (without TAFC)
        cmd = [
            sys.executable, 'baseline_generate_batch.py',
            '--prompt_list', str(temp_prompt_file),
            '--output_dir', str(output_dir),
            '--model-base', args.model_base,
            '--video-size', str(video_height), str(video_width),
            '--video-length', str(args.video_length),
            '--infer-steps', str(args.infer_steps),
            '--seed', str(args.seed),
            '--cfg-scale', str(args.cfg_scale),
            '--embedded-cfg-scale', str(args.embedded_cfg_scale),
            '--flow-shift', str(args.flow_shift),
            '--neg-prompt', args.neg_prompt,
            '--batch-size', str(args.batch_size),
        ]
        if getattr(args, 'flow_reverse', False):
            cmd.append('--flow-reverse')
    else:
        # Use TAFC generation script
        cmd = [
            sys.executable, 'tafc_generate_batch.py',
            '--prompt_list', str(temp_prompt_file),
            '--output_dir', str(output_dir),
            '--model-base', args.model_base,
            '--video-size', str(video_height), str(video_width),
            '--video-length', str(args.video_length),
            '--infer-steps', str(args.infer_steps),
            '--seed', str(args.seed),
            '--cfg-scale', str(args.cfg_scale),
            '--embedded-cfg-scale', str(args.embedded_cfg_scale),
            '--flow-shift', str(args.flow_shift),
            '--neg-prompt', args.neg_prompt,
            '--batch-size', str(args.batch_size),
            '--tafc_thresh', str(args.tafc_thresh),
            '--tafc_max_cache', str(args.tafc_max_cache),
            '--log_interval', str(args.log_interval),
        ]
        if getattr(args, 'flow_reverse', False):
            cmd.append('--flow-reverse')
        if args.use_ret_steps:
            cmd.append('--use_ret_steps')

    print(f"[driver] ({args.config}) Starting batch generation (model loads once)", flush=True)
    print(f"[driver] Output: {output_dir}", flush=True)
    print(f"[driver] TAFC config: thresh={args.tafc_thresh}, max_cache={args.tafc_max_cache}", flush=True)

    # Run batch subprocess
    proc = subprocess.run(cmd, cwd=HERE, env=env)

    # Clean up temp file
    if temp_prompt_file.exists():
        temp_prompt_file.unlink()

    if proc.returncode != 0:
        print(f"[driver] ERROR: batch generation failed with rc={proc.returncode}", file=sys.stderr)
        sys.exit(proc.returncode)

    print(f"[driver] ({args.config}) Batch generation complete", flush=True)

    # Print summary from gen_stats.json
    stats_file = output_dir / "gen_stats.json"
    if stats_file.exists():
        with open(stats_file) as f:
            stats = json.load(f)

        done = [v for k, v in stats.items()
                if k != "_config" and "error" not in v]

        if done:
            avg_time = sum(v["seconds"] for v in done) / len(done)
            avg_cache = sum(v["cache_rate_pct"] for v in done) / len(done)
            print(f"[driver] === Summary: {len(done)}/{len(prompts)} videos ===", flush=True)
            print(f"[driver] Average time: {avg_time:.1f}s/video", flush=True)
            print(f"[driver] Average cache rate: {avg_cache:.1f}%", flush=True)


if __name__ == "__main__":
    main()
