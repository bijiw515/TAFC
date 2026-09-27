#!/usr/bin/env python3
"""
Batch generation driver for the FLUX DrawBench benchmark.

FLUX counterpart of Wan2.1/run_seacache_eval.py. The key difference: FLUX images
take seconds rather than minutes, so a fresh subprocess per prompt (as in the
Wan2.1 driver) would spend most of its wall clock loading a 43 GB pipeline. Here
each config is one subprocess that loops over the whole benchmark internally,
and the per-image timing / cache-rate bookkeeping lives in gen_stats.json
written by the generation script itself.

Configs are named ``baseline``, ``tafc_<thresh>[_ret]``, ``seacache_<thresh>``:
  baseline          -> flux_generate.py      (GT for PSNR/SSIM/LPIPS)
  tafc_0.2          -> tafc_generate.py --tafc_thresh 0.2
  tafc_0.2_ret      -> tafc_generate.py --tafc_thresh 0.2 --use_ret_steps
  seacache_0.3      -> seacache_generate.py --seacache_thresh 0.3

Output layout:
  eval_samples/<config>/0000.png ... 0199.png
  eval_samples/<config>/gen_stats.json

Examples
--------
  # full benchmark, all configs, one after another on this GPU
  python run_drawbench_eval.py

  # quick smoke test
  python run_drawbench_eval.py --num_prompts 4 --configs baseline tafc_0.2

  # one config per GPU, in parallel
  CUDA_VISIBLE_DEVICES=0 python run_drawbench_eval.py --configs baseline &
  CUDA_VISIBLE_DEVICES=1 python run_drawbench_eval.py --configs tafc_0.2 &
"""

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PROMPTS = os.path.join(HERE, "benchmarks", "drawbench.csv")


def parse_config(config):
    """``tafc_0.2_ret`` -> ('tafc', 0.2, True). Raises on unknown names."""
    if config == "baseline":
        return "baseline", None, False
    parts = config.split("_")
    method = parts[0]
    if method not in ("tafc", "seacache") or len(parts) < 2:
        raise ValueError(
            f"unknown config {config!r}; expected 'baseline', 'tafc_<thresh>[_ret]' "
            f"or 'seacache_<thresh>'"
        )
    try:
        thresh = float(parts[1])
    except ValueError:
        raise ValueError(f"config {config!r}: {parts[1]!r} is not a threshold")
    use_ret = "ret" in parts[2:]
    return method, thresh, use_ret


def build_cmd(config, cfg_dir, args):
    """Command line for one config. Generation args are identical across
    configs apart from the caching knobs -- that is what keeps the images
    comparable."""
    method, thresh, use_ret = parse_config(config)

    common = [
        "--prompt_file", args.prompt_file,
        "--output_dir", cfg_dir,
        "--stats_file", os.path.join(cfg_dir, "gen_stats.json"),
        "--name_mode", "index",
        "--width", str(args.width),
        "--height", str(args.height),
        "--num_inference_steps", str(args.num_inference_steps),
        "--guidance", str(args.guidance),
        "--seed", str(args.base_seed),
        "--num_images_per_prompt", str(args.num_images_per_prompt),
        "--model_name", args.model_name,
        "--dtype", args.dtype,
        "--warmup_steps", str(args.warmup_steps),
        "--shard", str(args.shard),
        "--num_shards", str(args.num_shards),
    ]
    if args.num_prompts:
        common += ["--num_prompts", str(args.num_prompts)]
    if args.model_id:
        common += ["--model_id", args.model_id]
    if args.offload:
        common += ["--offload"]
    if args.skip_existing:
        common += ["--skip_existing"]

    if method == "baseline":
        return [sys.executable, os.path.join(HERE, "flux_generate.py")] + common
    if method == "tafc":
        cmd = [sys.executable, os.path.join(HERE, "tafc_generate.py")] + common
        cmd += ["--tafc_thresh", str(thresh),
                "--tafc_max_cache", str(args.tafc_max_cache),
                "--log_interval", str(args.log_interval)]
        if use_ret:
            cmd += ["--use_ret_steps"]
        return cmd
    cmd = [sys.executable, os.path.join(HERE, "seacache_generate.py")] + common
    cmd += ["--seacache_thresh", str(thresh)]
    return cmd


def read_summary(cfg_dir):
    path = os.path.join(cfg_dir, "gen_stats.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f).get("summary")
    except (json.JSONDecodeError, OSError):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_file", type=str, default=DEFAULT_PROMPTS,
                        help="DrawBench CSV (default: benchmarks/drawbench.csv, 200 prompts).")
    parser.add_argument("--out_dir", type=str, default=os.path.join(HERE, "eval_samples"))
    parser.add_argument("--configs", nargs="+",
                        default=["baseline", "seacache_0.3", "tafc_0.2", "tafc_0.2_ret"],
                        help="baseline | tafc_<thresh>[_ret] | seacache_<thresh>")
    parser.add_argument("--num_prompts", type=int, default=None,
                        help="Use only the first N DrawBench prompts (smoke tests).")
    parser.add_argument("--num_images_per_prompt", type=int, default=1)

    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--model_name", type=str, default="flux-dev",
                        choices=["flux-dev", "flux-schnell"])
    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--offload", action="store_true", default=False)
    parser.add_argument("--warmup_steps", type=int, default=0,
                        help="Throwaway generation before timing starts (0 disables).")

    parser.add_argument("--tafc_max_cache", type=float, default=6.0)
    parser.add_argument("--log_interval", type=int, default=0,
                        help="TAFC per-step logging interval (0 = quiet, the default for batch runs).")

    parser.add_argument("--shard", type=int, default=0,
                        help="Round-robin shard for splitting one config across GPUs.")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--skip_existing", action="store_true", default=True,
                        help="Resume: keep images that already exist.")
    parser.add_argument("--no_skip_existing", dest="skip_existing", action="store_false")
    parser.add_argument("--dry_run", action="store_true", help="Print commands and exit.")
    args = parser.parse_args()

    if not os.path.exists(args.prompt_file):
        raise SystemExit(f"prompt file not found: {args.prompt_file}")
    for config in args.configs:
        parse_config(config)  # fail fast on typos before spending GPU hours

    print(f"[driver] prompts={args.prompt_file} configs={args.configs} "
          f"steps={args.num_inference_steps} size={args.width}x{args.height} "
          f"seed={args.base_seed}", flush=True)

    results = {}
    for config in args.configs:
        cfg_dir = os.path.join(args.out_dir, config)
        os.makedirs(cfg_dir, exist_ok=True)
        cmd = build_cmd(config, cfg_dir, args)

        if args.dry_run:
            print(f"[driver] ({config}) {' '.join(cmd)}", flush=True)
            continue

        log_path = os.path.join(cfg_dir, "generate.log")
        print(f"[driver] ({config}) starting -> {cfg_dir} (log: {log_path})", flush=True)
        t0 = time.time()
        with open(log_path, "a") as log:
            log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)} =====\n")
            log.flush()
            # Stream to the log file; the generation script already prints one
            # line per image, so tail the log to follow progress.
            proc = subprocess.run(cmd, cwd=HERE, stdout=log, stderr=subprocess.STDOUT, text=True)
        dt = time.time() - t0

        if proc.returncode != 0:
            print(f"[driver] ERROR ({config}) rc={proc.returncode} — tail of {log_path}:", flush=True)
            with open(log_path) as f:
                print("".join(f.readlines()[-25:]), flush=True)
            results[config] = {"error": True, "wall_seconds": round(dt, 1)}
            continue

        summary = read_summary(cfg_dir) or {}
        summary["wall_seconds"] = round(dt, 1)
        results[config] = summary
        print(f"[driver] ({config}) done in {dt/60:.1f} min | "
              f"avg {summary.get('avg_seconds')}s/image | "
              f"cache {summary.get('avg_cache_rate_pct')}%", flush=True)

    if args.dry_run or not results:
        return

    with open(os.path.join(args.out_dir, "driver_summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    base = results.get("baseline", {}).get("avg_seconds")
    print("\n" + "=" * 78)
    print(f"{'config':<22}{'images':>7}{'s/image':>10}{'speedup':>10}{'cache %':>10}")
    print("-" * 78)
    for config, r in results.items():
        if r.get("error"):
            print(f"{config:<22}{'ERROR':>7}")
            continue
        avg = r.get("avg_seconds")
        speedup = f"{base / avg:.2f}x" if (base and avg) else "-"
        cache = r.get("avg_cache_rate_pct")
        print(f"{config:<22}{r.get('num_images', 0):>7}{avg:>10.2f}{speedup:>10}"
              f"{(f'{cache:.1f}' if cache is not None else '-'):>10}")
    print("=" * 78)
    print(f"summary: {os.path.join(args.out_dir, 'driver_summary.json')}")
    print("next:    python eval_drawbench.py --gt_dir "
          f"{os.path.join(args.out_dir, 'baseline')} --gen_dir <config_dir>", flush=True)


if __name__ == "__main__":
    main()
