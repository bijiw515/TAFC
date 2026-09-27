#!/usr/bin/env python3
"""
Batch generation driver for the HPSv3 image-generation benchmark.

Sibling of run_drawbench_eval.py. Same three generation backends, same config
naming, same output layout -- only the prompt source and the default prompt
count differ, because HPSv3 is a much bigger benchmark than DrawBench:

  DrawBench   200 prompts, short prompts, categories from the CSV
  HPSv3     12,000 prompts (12 categories x 1,000), long VLM captions,
            always 1024x1024 (aspect ratio 1.0 in HPDv3)

The full benchmark is ~70 GPU-hours per config at 21 s/image, so the default
here is --num_prompts 1200 (100 per category). Because
benchmarks/hpsv3_benchmark.json interleaves the categories (index 0..11 is one
prompt per category, then it repeats), truncating to any multiple of 12 stays
exactly category-balanced. Pass --num_prompts 0 for the full 12,000.

Configs are named ``baseline``, ``tafc_<thresh>[_ret]``, ``seacache_<thresh>``:
  baseline          -> flux_generate.py      (GT for PSNR/SSIM/LPIPS)
  tafc_0.2          -> tafc_generate.py --tafc_thresh 0.2
  tafc_0.2_ret      -> tafc_generate.py --tafc_thresh 0.2 --use_ret_steps
  seacache_0.3      -> seacache_generate.py --seacache_thresh 0.3

Output layout (kept separate from the DrawBench run):
  eval_samples_hpsv3/<config>/0000.png ... 1199.png
  eval_samples_hpsv3/<config>/gen_stats.json

Examples
--------
  # default subset (1,200 prompts), all configs, sequentially on this GPU
  python run_hpsv3_eval.py

  # quick smoke test: one prompt per category
  python run_hpsv3_eval.py --num_prompts 12 --configs baseline tafc_0.2

  # the paper's full 12,000-prompt benchmark
  python run_hpsv3_eval.py --num_prompts 0

  # one config per GPU, in parallel
  CUDA_VISIBLE_DEVICES=0 python run_hpsv3_eval.py --configs baseline &
  CUDA_VISIBLE_DEVICES=1 python run_hpsv3_eval.py --configs tafc_0.2 &

  # split ONE config across 4 GPUs (shards are round-robin, so each shard gets
  # whole categories -- only the merged directory covers all 12)
  for s in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$s python run_hpsv3_eval.py --configs baseline \
        --shard $s --num_shards 4 &
  done
"""

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PROMPTS = os.path.join(HERE, "benchmarks", "hpsv3_benchmark.json")
# 12 categories x 100 prompts. HPSv3's own benchmark is 1,000 per category;
# 100 keeps a full run to ~7 GPU-hours per config while staying balanced.
DEFAULT_NUM_PROMPTS = 12000


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

    # Shards share one image directory but must NOT share one stats file, or
    # whichever worker finishes last overwrites the others and the summary ends
    # up covering only its own slice. summarize_hpsv3.py merges the parts.
    stats_name = ("gen_stats.json" if args.num_shards == 1
                  else f"gen_stats_shard{args.shard}of{args.num_shards}.json")

    common = [
        "--prompt_file", args.prompt_file,
        "--output_dir", cfg_dir,
        "--stats_file", os.path.join(cfg_dir, stats_name),
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
    # --num_prompts 0 means "the whole benchmark": just omit the flag.
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


def read_summary(cfg_dir, args):
    stats_name = ("gen_stats.json" if args.num_shards == 1
                  else f"gen_stats_shard{args.shard}of{args.num_shards}.json")
    path = os.path.join(cfg_dir, stats_name)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f).get("summary")
    except (json.JSONDecodeError, OSError):
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate the HPSv3 benchmark for every FLUX caching config."
    )
    parser.add_argument("--prompt_file", type=str, default=DEFAULT_PROMPTS,
                        help="HPSv3 prompt JSON (default: benchmarks/hpsv3_benchmark.json, "
                             "12,000 prompts; build it with benchmarks/build_hpsv3_benchmark.py).")
    parser.add_argument("--out_dir", type=str,
                        default=os.path.join(HERE, "eval_samples_hpsv3"))
    parser.add_argument("--configs", nargs="+",
                        default=["baseline", "seacache_0.3", "tafc_0.2", "tafc_0.2_ret"],
                        help="baseline | tafc_<thresh>[_ret] | seacache_<thresh>")
    parser.add_argument("--num_prompts", type=int, default=DEFAULT_NUM_PROMPTS,
                        help=f"Use the first N prompts (default {DEFAULT_NUM_PROMPTS} = 100 per "
                             f"category). 0 = the full 12,000-prompt benchmark. Keep it a "
                             f"multiple of 12 to stay category-balanced.")
    parser.add_argument("--num_images_per_prompt", type=int, default=1)

    # HPDv3 benchmark captions all describe aspect-ratio-1.0 images -> square.
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
        raise SystemExit(
            f"prompt file not found: {args.prompt_file}\n"
            f"build it first: python benchmarks/build_hpsv3_benchmark.py"
        )
    for config in args.configs:
        parse_config(config)  # fail fast on typos before spending GPU hours
    if args.num_prompts and args.num_prompts % 12:
        print(f"[driver] WARNING --num_prompts {args.num_prompts} is not a multiple of 12; "
              f"the last categories will have one prompt fewer.", flush=True)

    n_desc = args.num_prompts or "all"
    print(f"[driver] prompts={args.prompt_file} (n={n_desc}) configs={args.configs} "
          f"steps={args.num_inference_steps} size={args.width}x{args.height} "
          f"seed={args.base_seed} shard={args.shard}/{args.num_shards}", flush=True)

    results = {}
    for config in args.configs:
        cfg_dir = os.path.join(args.out_dir, config)
        os.makedirs(cfg_dir, exist_ok=True)
        cmd = build_cmd(config, cfg_dir, args)

        if args.dry_run:
            print(f"[driver] ({config}) {' '.join(cmd)}", flush=True)
            continue

        log_path = os.path.join(cfg_dir, "generate.log" if args.num_shards == 1
                                else f"generate_shard{args.shard}of{args.num_shards}.log")
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

        summary = read_summary(cfg_dir, args) or {}
        summary["wall_seconds"] = round(dt, 1)
        results[config] = summary
        print(f"[driver] ({config}) done in {dt/60:.1f} min | "
              f"avg {summary.get('avg_seconds')}s/image | "
              f"cache {summary.get('avg_cache_rate_pct')}%", flush=True)

    if args.dry_run or not results:
        return

    # Sharded runs each write their own summary so parallel workers do not
    # clobber one another.
    name = ("driver_summary.json" if args.num_shards == 1
            else f"driver_summary_shard{args.shard}of{args.num_shards}.json")
    with open(os.path.join(args.out_dir, name), "w") as f:
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
    print(f"summary: {os.path.join(args.out_dir, name)}")
    print("next:    conda activate hpsv3 && python eval_hpsv3.py --gen_dir "
          f"{os.path.join(args.out_dir, args.configs[-1])} "
          f"--prompt_file {args.prompt_file}", flush=True)


if __name__ == "__main__":
    main()
