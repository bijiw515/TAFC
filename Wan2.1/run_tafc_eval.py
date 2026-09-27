#!/usr/bin/env python
"""
Batch generation driver for TAFC evaluation with TeaCache eval scripts.

Generates videos for a subset of VBench prompts under 4 configs:
  - baseline  (no cache, generate.py)     -> used as GT for PSNR/LPIPS/SSIM
  - tafc_0.1  (tafc_generate.py, thresh 0.1)
  - tafc_0.2  (tafc_generate.py, thresh 0.2)
  - tafc_0.3  (tafc_generate.py, thresh 0.3)

Output layout (files named "{prompt}-0.mp4" to match VBench vbench_standard):
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

HERE = os.path.dirname(os.path.abspath(__file__))

# TAFC prints a summary line: "[TAFC Summary] Total cached: 60/100 (60.0%)"
CACHE_RE = re.compile(r"\[TAFC Summary\] Total cached:\s*(\d+)/(\d+)\s*\(([\d.]+)%\)")
BRANCH_RE = re.compile(r"\[TAFC Branch\] (cond|uncond):\s*(\d+)/(\d+)\s*\(([\d.]+)%\)")


def build_cmd(config, prompt, save_file, args):
    """Return the subprocess command list for a given config."""
    common = [
        sys.executable,
        "--task", args.task,
        "--ckpt_dir", args.ckpt_dir,
        "--size", args.size,
        "--sample_steps", str(args.sample_steps),
        "--frame_num", str(args.frame_num),
        "--base_seed", str(args.base_seed),
        "--prompt", prompt,
        "--save_file", save_file,
        "--offload_model", str(args.offload_model),
    ]
    if config == "baseline":
        cmd = [sys.executable, os.path.join(HERE, "generate.py")] + common[1:]
    else:
        thresh = config.split("_")[1]
        cmd = [sys.executable, os.path.join(HERE, "tafc_generate.py")] + common[1:]
        cmd += ["--tafc_thresh", thresh]
        cmd += ["--tafc_max_cache", str(args.tafc_max_cache)]
        if args.use_ret_steps:
            cmd += ["--use_ret_steps"]
    return cmd


def safe_name(prompt):
    """Filesystem-safe base name; keep it reversible-ish and unique per prompt."""
    return prompt.replace("/", "_").strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_file", type=str,
                        default="/data/dev2/lgbi/TeaCache/eval/teacache/vbench/VBench_subset_info.json")
    parser.add_argument("--out_dir", type=str, default=os.path.join(HERE, "eval_samples"))
    parser.add_argument("--task", type=str, default="t2v-1.3B")
    parser.add_argument("--ckpt_dir", type=str, default="/data/dev2/lgbi/SeaCache/Wan2.1/Wan2.1-T2V-1.3B")
    parser.add_argument("--size", type=str, default="832*480")
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--frame_num", type=int, default=65,
                        help="Number of frames to generate")
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--tafc_max_cache", type=float, default=4.0,
                        help="Max consecutive cache budget at the LAST denoising step, "
                             "passed through to tafc_generate.py. Budget ramps linearly "
                             "from 1 to this value across denoising. Ignored by baseline.")
    parser.add_argument("--use_ret_steps", action="store_true", default=True)
    parser.add_argument("--offload_model", type=str, default="False",
                        help="Offload model to CPU each step. False is much faster on large-VRAM GPUs.")
    parser.add_argument("--configs", nargs="+",
                        default=["baseline", "tafc_0.1", "tafc_0.2", "tafc_0.3"])
    parser.add_argument("--skip_existing", action="store_true", default=True)
    parser.add_argument("--gpu_id", type=int, default=None,
                        help="GPU ID to use. If not set, inherits CUDA_VISIBLE_DEVICES from parent.")
    args = parser.parse_args()

    with open(args.prompt_file) as f:
        prompts = [x["prompt_en"] for x in json.load(f)]
    print(f"[driver] {len(prompts)} prompts, configs={args.configs}, "
          f"tafc_max_cache={args.tafc_max_cache}", flush=True)

    # Prepare environment for subprocesses to avoid OOM
    env = os.environ.copy()
    if args.gpu_id is not None:
        # Set CUDA_VISIBLE_DEVICES to the specified GPU
        env['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
        env['LOCAL_RANK'] = '0'  # Critical: when using single GPU, LOCAL_RANK must be 0
        env['RANK'] = '0'
        env['WORLD_SIZE'] = '1'
        print(f"[driver] Using GPU {args.gpu_id} (mapped to cuda:0 in subprocesses)", flush=True)
    else:
        # Ensure LOCAL_RANK is set to 0 for single-GPU mode
        if 'LOCAL_RANK' not in env:
            env['LOCAL_RANK'] = '0'
        if 'RANK' not in env:
            env['RANK'] = '0'
        if 'WORLD_SIZE' not in env:
            env['WORLD_SIZE'] = '1'

    for config in args.configs:
        cfg_dir = os.path.join(args.out_dir, config)
        os.makedirs(cfg_dir, exist_ok=True)
        stats = {}
        if config != "baseline":
            stats["_config"] = {"tafc_max_cache": args.tafc_max_cache,
                                "tafc_thresh": float(config.split("_")[1]),
                                "sample_steps": args.sample_steps,
                                "use_ret_steps": bool(args.use_ret_steps)}
        for i, prompt in enumerate(prompts):
            base = safe_name(prompt)
            save_file = os.path.join(cfg_dir, f"{base}-0.mp4")
            if args.skip_existing and os.path.exists(save_file):
                print(f"[driver] ({config}) [{i+1}/{len(prompts)}] skip existing: {base}", flush=True)
                continue
            cmd = build_cmd(config, prompt, save_file, args)
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
            m = CACHE_RE.search(proc.stdout)
            cache_rate = float(m.group(3)) if m else None

            # Extract per-branch cache rates
            branch_stats = {}
            for bm in BRANCH_RE.finditer(proc.stdout):
                branch_name = bm.group(1)
                branch_cached = int(bm.group(2))
                branch_total = int(bm.group(3))
                branch_pct = float(bm.group(4))
                branch_stats[f"{branch_name}_cached"] = branch_cached
                branch_stats[f"{branch_name}_pct"] = branch_pct

            stats[base] = {
                "seconds": round(dt, 1),
                "cache_rate_pct": cache_rate,
                **branch_stats
            }

            if branch_stats:
                print(f"[driver] ({config}) done in {dt:.1f}s cache={cache_rate}% "
                      f"(cond={branch_stats.get('cond_pct', '?')}% uncond={branch_stats.get('uncond_pct', '?')}%)",
                      flush=True)
            else:
                print(f"[driver] ({config}) done in {dt:.1f}s cache={cache_rate}%", flush=True)
            with open(os.path.join(cfg_dir, "gen_stats.json"), "w") as f:
                json.dump(stats, f, indent=2)

        # summary
        done = [v for k, v in stats.items()
                if k != "_config" and not v.get("error")]
        if done:
            avg_t = sum(v["seconds"] for v in done) / len(done)
            crs = [v["cache_rate_pct"] for v in done if v.get("cache_rate_pct") is not None]
            avg_cr = sum(crs) / len(crs) if crs else None
            print(f"[driver] === {config}: avg {avg_t:.1f}s/video, avg cache {avg_cr}% ===", flush=True)


if __name__ == "__main__":
    main()
