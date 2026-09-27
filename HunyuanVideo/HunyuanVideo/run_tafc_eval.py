#!/usr/bin/env python
"""
Single-prompt evaluation driver for TAFC on HunyuanVideo with VBench.

Generates videos for VBench prompts under different configs:
  - baseline  (no cache, baseline generation)
  - tafc_0.1  (tafc_generate.py, thresh 0.1)
  - tafc_0.2  (tafc_generate.py, thresh 0.2)
  - tafc_0.3  (tafc_generate.py, thresh 0.3)

Output layout (files named "{prompt}-0.mp4" to match VBench standard):
  eval_samples/baseline/{prompt}-0.mp4
  eval_samples/tafc_0.1/{prompt}-0.mp4
  ...

Per-config timing and TAFC cache-rate are parsed from stdout and written to
eval_samples/<config>/gen_stats.json.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent.resolve()

# TAFC prints a summary line: "[TAFC Summary] Total cached: 60/100 (60.0%)"
CACHE_RE = re.compile(r"\[TAFC Summary\] Total cached:\s*(\d+)/(\d+)\s*\(([\d.]+)%\)")


def build_cmd(config, prompt, save_path, args):
    """Return the subprocess command list for a given config."""
    common = [
        '--model-base', args.model_base,
        '--video-size', str(args.video_height), str(args.video_width),
        '--video-length', str(args.video_length),
        '--infer-steps', str(args.infer_steps),
        '--seed', str(args.seed),
        '--cfg-scale', str(args.cfg_scale),
        '--embedded-cfg-scale', str(args.embedded_cfg_scale),
        '--flow-shift', str(args.flow_shift),
        '--neg-prompt', args.neg_prompt,
        '--batch-size', str(args.batch_size),
        '--prompt', prompt,
        '--save-path', save_path,
    ]

    if args.flow_reverse:
        common.append('--flow-reverse')

    if config == "baseline":
        # Use the baseline generation script (without TAFC)
        # For now, use tafc_generate.py with very high threshold to effectively disable caching
        cmd = [sys.executable, str(HERE / "tafc_generate.py")] + common
        cmd += ['--tafc_thresh', '999.0']  # Effectively disable caching
    else:
        # Extract threshold from config name (e.g., tafc_0.1 -> 0.1)
        thresh = config.split("_")[1]
        cmd = [sys.executable, str(HERE / "tafc_generate.py")] + common
        cmd += ['--tafc_thresh', thresh]
        cmd += ['--tafc_max_cache', str(args.tafc_max_cache)]
        cmd += ['--log_interval', str(args.log_interval)]
        if args.use_ret_steps:
            cmd += ['--use_ret_steps']

    return cmd


def safe_name(prompt):
    """Filesystem-safe base name; keep it reversible-ish and unique per prompt."""
    return prompt.replace("/", "_").strip()


def main():
    parser = argparse.ArgumentParser(
        description="Run TAFC evaluation on VBench prompts for HunyuanVideo"
    )
    parser.add_argument("--prompt_file", type=str,
                        default="/data/dev2/lgbi/TeaCache/eval/teacache/vbench/VBench_subset_info.json",
                        help="VBench prompt file")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory (default: eval_samples)")
    parser.add_argument("--model_base", type=str,
                        default="/data/dev2/lgbi/SeaCache/HunyuanVideo/HunyuanVideo/ckpts",
                        help="Model checkpoint directory")
    parser.add_argument("--video_size", type=str, default="480,832",
                        help="Video size as height,width (480,832 matches Wan2.1's 832*480)")
    parser.add_argument("--video_length", type=int, default=65,
                        help="Video length in frames")
    parser.add_argument("--infer_steps", type=int, default=50,
                        help="Number of inference steps")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base seed for generation")
    parser.add_argument("--cfg_scale", type=float, default=1.0,
                        help="CFG scale")
    parser.add_argument("--embedded_cfg_scale", type=float, default=6.0,
                        help="Embedded CFG scale")
    parser.add_argument("--flow_shift", type=float, default=7.0,
                        help="Flow shift")
    parser.add_argument("--flow_reverse", action="store_true", default=False,
                        help="Enable flow reverse")
    parser.add_argument("--tafc_max_cache", type=float, default=6.0,
                        help="Max consecutive cache budget at the LAST denoising step")
    parser.add_argument("--use_ret_steps", action="store_true", default=True,
                        help="Use retention steps (force first 5 steps)")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="Logging interval")
    parser.add_argument("--neg_prompt", type=str, default="",
                        help="Negative prompt")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size")
    parser.add_argument("--configs", nargs="+",
                        default=["baseline", "tafc_0.1", "tafc_0.2", "tafc_0.3"],
                        help="Configs to evaluate")
    parser.add_argument("--skip_existing", action="store_true", default=True,
                        help="Skip already generated videos")
    parser.add_argument("--gpu_id", type=int, default=None,
                        help="GPU ID to use")
    args = parser.parse_args()

    # Parse video size
    args.video_height, args.video_width = map(int, args.video_size.split(','))

    # Load prompts
    with open(args.prompt_file) as f:
        prompts = [x["prompt_en"] for x in json.load(f)]

    print(f"[driver] {len(prompts)} prompts, configs={args.configs}, "
          f"tafc_max_cache={args.tafc_max_cache}", flush=True)

    # Prepare environment
    env = os.environ.copy()
    if args.gpu_id is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
        env['LOCAL_RANK'] = '0'
        env['RANK'] = '0'
        env['WORLD_SIZE'] = '1'
        print(f"[driver] Using GPU {args.gpu_id} (mapped to cuda:0 in subprocesses)", flush=True)
    else:
        if 'LOCAL_RANK' not in env:
            env['LOCAL_RANK'] = '0'
        if 'RANK' not in env:
            env['RANK'] = '0'
        if 'WORLD_SIZE' not in env:
            env['WORLD_SIZE'] = '1'

    # Output directory
    if args.out_dir is None:
        out_dir = HERE / "eval_samples"
    else:
        out_dir = Path(args.out_dir)

    # Process each config
    for config in args.configs:
        cfg_dir = out_dir / config
        cfg_dir.mkdir(parents=True, exist_ok=True)

        stats = {}
        if config != "baseline":
            stats["_config"] = {
                "tafc_max_cache": args.tafc_max_cache,
                "tafc_thresh": float(config.split("_")[1]),
                "infer_steps": args.infer_steps,
                "use_ret_steps": bool(args.use_ret_steps)
            }

        # Process each prompt
        for i, prompt in enumerate(prompts):
            base = safe_name(prompt)
            save_path = cfg_dir / f"{base}-0.mp4"

            if args.skip_existing and save_path.exists():
                print(f"[driver] ({config}) [{i+1}/{len(prompts)}] skip existing: {base}", flush=True)
                continue

            # Create a temporary directory for this generation
            temp_dir = cfg_dir / f"temp_{base}"
            temp_dir.mkdir(exist_ok=True)

            cmd = build_cmd(config, prompt, str(temp_dir), args)

            print(f"[driver] ({config}) [{i+1}/{len(prompts)}] generating: {prompt[:60]}", flush=True)
            t0 = time.time()
            proc = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True, env=env)
            dt = time.time() - t0

            if proc.returncode != 0:
                print(f"[driver] ERROR ({config}) prompt={prompt!r} rc={proc.returncode}", flush=True)
                print(proc.stdout[-2000:], flush=True)
                print(proc.stderr[-2000:], flush=True)
                stats[base] = {"seconds": dt, "error": True}
                continue

            # Parse cache rate from output
            m = CACHE_RE.search(proc.stdout)
            cache_rate = float(m.group(3)) if m else None

            # Find and move the generated video
            generated_files = list(temp_dir.glob("*.mp4"))
            if generated_files:
                # Move the first found video to the target location
                generated_files[0].rename(save_path)
                print(f"[driver] ({config}) moved {generated_files[0].name} to {save_path.name}", flush=True)
            else:
                print(f"[driver] WARNING ({config}) no video found in {temp_dir}", flush=True)

            # Clean up temp directory
            try:
                temp_dir.rmdir()
            except:
                pass  # Directory not empty, leave it

            stats[base] = {
                "seconds": round(dt, 1),
                "cache_rate_pct": cache_rate,
            }

            print(f"[driver] ({config}) done in {dt:.1f}s cache={cache_rate}%", flush=True)

            # Save stats after each video
            with open(cfg_dir / "gen_stats.json", "w") as f:
                json.dump(stats, f, indent=2)

        # Summary
        done = [v for k, v in stats.items()
                if k != "_config" and not v.get("error")]
        if done:
            avg_t = sum(v["seconds"] for v in done) / len(done)
            crs = [v["cache_rate_pct"] for v in done if v.get("cache_rate_pct") is not None]
            avg_cr = sum(crs) / len(crs) if crs else None
            print(f"[driver] === {config}: avg {avg_t:.1f}s/video, avg cache {avg_cr}% ===", flush=True)


if __name__ == "__main__":
    main()
