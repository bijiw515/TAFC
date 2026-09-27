#!/usr/bin/env python3
"""HPSv3 human-preference scoring for FLUX caching methods.

Counterpart of eval_drawbench.py for the HPSv3 image-generation benchmark
(HPSv3: Towards Wide-Spectrum Human Preference Score, ICCV 2025). HPSv3 is a
Qwen2-VL-7B reward model: it takes (prompt, image) and returns mu/sigma, where
mu is the preference score reported in the paper. Higher is better; the paper's
1024px numbers sit around 8-11 for modern models (FLUX.1-dev overall 10.43).

Why this is a separate script from eval_drawbench.py
----------------------------------------------------
* It needs its own conda env. HPSv3 pins transformers==4.45.2 (Qwen2-VL), while
  the generation/pyiqa env here runs transformers 5.x. See README_HPSV3.md.
* It loads a 16 GB reward model, so it batches over images and is worth running
  once per config rather than alongside the pyiqa metrics.

Why not use HPSv3's own evaluate/benchmark.py
---------------------------------------------
That script discovers prompts from a sidecar ``<image>.txt`` next to every
image, which the FLUX harness does not write (it names files by benchmark index
so full-reference metrics line up across configs). This script reads the
prompts straight from the benchmark JSON instead, and reports the same
per-category means the paper's table does. Scoring itself calls the official
``hpsv3.inference.HPSv3RewardInferencer``, unmodified.

Aggregation matches the paper and HPSv3's benchmark.py: mean over images within
a category, then an unweighted mean over the 12 categories for OVERALL. With a
category-balanced prompt file the two are equal anyway.

Usage
-----
  conda activate hpsv3

  # score one config
  python eval_hpsv3.py --gen_dir eval_samples_hpsv3/baseline \
      --prompt_file benchmarks/hpsv3_benchmark.json

  # several configs in one model load (the model costs ~2 min to load)
  python eval_hpsv3.py --gen_dir eval_samples_hpsv3/baseline \
      eval_samples_hpsv3/tafc_0.2 --prompt_file benchmarks/hpsv3_benchmark.json

  # split one config across 4 GPUs
  for s in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$s python eval_hpsv3.py --gen_dir eval_samples_hpsv3/baseline \
        --prompt_file benchmarks/hpsv3_benchmark.json --shard $s --num_shards 4 &
  done
  wait
  python eval_hpsv3.py --merge_shards eval_samples_hpsv3/baseline
"""

import argparse
import csv
import glob
import json
import os
import time

import numpy as np

from bench_common import load_prompts

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")
# The 12 HPDv3 categories, in the order the paper's table lists them.
CATEGORY_ORDER = [
    "Characters", "Arts", "Design", "Architecture", "Animals",
    "Natural Scenery", "Transportation", "Products", "Others", "Plants",
    "Food", "Science",
]


def prompt_index(prompt_file):
    """Map ``0007`` (file stem = benchmark index) -> {prompt, category}."""
    return {f"{it['idx']:04d}": it for it in load_prompts(prompt_file)}


def list_images(gen_dir):
    return sorted(f for f in os.listdir(gen_dir) if f.lower().endswith(IMAGE_EXTS))


def category_stats(rows, score_key="hpsv3"):
    """Per-category mean/std/min/max plus an OVERALL row.

    OVERALL is the unweighted mean of the per-category means, which is what the
    paper's Overall column and HPSv3's benchmark.py both report.
    """
    cats = [c for c in CATEGORY_ORDER if any(r["category"] == c for r in rows)]
    # keep any unexpected category rather than silently dropping it
    cats += sorted({r["category"] for r in rows if r["category"] and r["category"] not in cats})
    if not cats:
        cats = [""]

    stats = {}
    for cat in cats:
        vals = [r[score_key] for r in rows if r["category"] == cat]
        if not vals:
            continue
        stats[cat or "(uncategorised)"] = {
            "count": len(vals),
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }

    means = [s["mean"] for s in stats.values()]
    all_vals = [r[score_key] for r in rows]
    stats["OVERALL"] = {
        "count": len(rows),
        "mean": float(np.mean(means)),                 # macro over categories
        "mean_over_images": float(np.mean(all_vals)),  # micro, for reference
        "std": float(np.std(all_vals)),
        "min": float(np.min(all_vals)),
        "max": float(np.max(all_vals)),
    }
    return stats


def print_stats(stats, title):
    print(f"\n=== {title} ===")
    print(f"{'Category':<20}{'Count':>7}{'Mean':>10}{'Std':>9}{'Min':>9}{'Max':>9}")
    print("-" * 64)
    for cat, s in stats.items():
        if cat == "OVERALL":
            continue
        print(f"{cat:<20}{s['count']:>7}{s['mean']:>10.4f}{s['std']:>9.4f}"
              f"{s['min']:>9.4f}{s['max']:>9.4f}")
    o = stats["OVERALL"]
    print("-" * 64)
    print(f"{'OVERALL':<20}{o['count']:>7}{o['mean']:>10.4f}{o['std']:>9.4f}"
          f"{o['min']:>9.4f}{o['max']:>9.4f}")
    print(f"{'(mean over images)':<20}{'':>7}{o['mean_over_images']:>10.4f}")


def write_outputs(out_dir, tag, rows, stats, meta, suffix=""):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{tag}_hpsv3_per_image{suffix}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "category", "hpsv3", "sigma", "prompt"])
        writer.writeheader()
        writer.writerows(rows)

    json_path = os.path.join(out_dir, f"{tag}_hpsv3{suffix}.json")
    with open(json_path, "w") as f:
        json.dump({**meta, "statistics": stats}, f, indent=2)
    return csv_path, json_path


def merge_shards(gen_dir, prompt_file=None):
    """Combine ``*_hpsv3_per_image_shard*.csv`` into one full-run result."""
    tag = os.path.basename(os.path.normpath(gen_dir))
    parts = sorted(glob.glob(os.path.join(gen_dir, f"{tag}_hpsv3_per_image_shard*.csv")))
    if not parts:
        raise SystemExit(f"no shard CSVs found in {gen_dir}")

    rows, seen = [], set()
    for path in parts:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["image"] in seen:
                    continue
                seen.add(row["image"])
                row["hpsv3"] = float(row["hpsv3"])
                row["sigma"] = float(row["sigma"]) if row["sigma"] not in ("", None) else None
                rows.append(row)
    rows.sort(key=lambda r: r["image"])

    stats = category_stats(rows)
    meta = {
        "gen_dir": os.path.abspath(gen_dir),
        "benchmark": "HPSv3",
        "prompt_file": os.path.abspath(prompt_file) if prompt_file else None,
        "num_images": len(rows),
        "merged_from": [os.path.basename(p) for p in parts],
        "aggregation": "mean within category, then unweighted mean over categories",
    }
    csv_path, json_path = write_outputs(gen_dir, tag, rows, stats, meta)
    print_stats(stats, f"HPSv3 — {tag} (merged {len(parts)} shards)")
    print(f"\nper-image CSV: {csv_path}\nsummary JSON:  {json_path}")


def score_dir(inferencer, gen_dir, prompts, args):
    """Score every image in one config directory."""
    names = list_images(gen_dir)
    if not names:
        print(f"⚠ No images found in {gen_dir}, skipping...", flush=True)
        return None
    if args.num_shards > 1:
        names = [n for i, n in enumerate(names) if i % args.num_shards == args.shard]

    unknown = [n for n in names if os.path.splitext(n)[0].split("_")[0] not in prompts]
    if unknown:
        raise SystemExit(
            f"{len(unknown)} images have no prompt in {args.prompt_file}, first: {unknown[0]}. "
            f"HPSv3 scoring needs --name_mode index filenames and the same prompt file "
            f"used for generation."
        )

    tag = args.tag or os.path.basename(os.path.normpath(gen_dir))
    rows = []
    started = time.time()
    skipped = 0

    for start in range(0, len(names), args.batch_size):
        batch = names[start:start + args.batch_size]
        paths = [os.path.join(gen_dir, n) for n in batch]
        items = [prompts[os.path.splitext(n)[0].split("_")[0]] for n in batch]
        texts = [it["prompt"] for it in items]

        # Filter out corrupt or unreadable images
        valid_indices = []
        valid_paths = []
        valid_texts = []
        valid_items = []
        valid_names = []

        for i, (path, text, item, name) in enumerate(zip(paths, texts, items, batch)):
            try:
                # Try to open the image to verify it's valid
                from PIL import Image
                with Image.open(path) as img:
                    img.verify()
                valid_indices.append(i)
                valid_paths.append(path)
                valid_texts.append(text)
                valid_items.append(item)
                valid_names.append(name)
            except Exception as e:
                print(f"⚠ Skipping corrupt image {name}: {e}", flush=True)
                skipped += 1

        if not valid_paths:
            print(f"⚠ All images in batch starting at {start} are corrupt, skipping", flush=True)
            continue

        try:
            rewards = inferencer.reward(prompts=valid_texts, image_paths=valid_paths)

            for n, it, reward in zip(valid_names, valid_items, rewards):
                # The reward head pools to [B, output_dim] with output_dim=2, i.e.
                # [mu, sigma] per image. mu is the score the paper reports.
                flat = reward.flatten()
                mu = float(flat[0].item())
                sigma = float(flat[1].item()) if flat.numel() > 1 else None
                rows.append({"image": n, "category": it.get("category", ""),
                             "hpsv3": mu, "sigma": sigma, "prompt": it["prompt"]})
        except Exception as e:
            print(f"⚠ Error processing batch starting at {start}: {e}", flush=True)
            skipped += len(valid_names)
            continue

        done = min(start + args.batch_size, len(names))
        rate = (done - skipped) / max(time.time() - started, 1e-6)
        last_score = rows[-1]['hpsv3'] if rows else 0.0
        print(f"[{tag}] {done}/{len(names)} images ({skipped} skipped) | {rate:.2f} img/s | "
              f"last mu={last_score:.4f}", flush=True)

    if skipped > 0:
        print(f"⚠ Total skipped images: {skipped}/{len(names)}", flush=True)

    if not rows:
        raise SystemExit(f"No valid images could be scored in {gen_dir}")

    stats = category_stats(rows)
    suffix = "" if args.num_shards == 1 else f"_shard{args.shard}of{args.num_shards}"
    meta = {
        "gen_dir": os.path.abspath(gen_dir),
        "benchmark": "HPSv3",
        "prompt_file": os.path.abspath(args.prompt_file),
        "checkpoint_path": args.checkpoint_path,
        "config_path": args.config_path,
        "num_images": len(rows),
        "num_skipped": skipped,
        "shard": f"{args.shard}/{args.num_shards}",
        "aggregation": "mean within category, then unweighted mean over categories",
        "elapsed_sec": round(time.time() - started, 1),
    }
    out_dir = args.out_dir or gen_dir
    csv_path, json_path = write_outputs(out_dir, tag, rows, stats, meta, suffix)
    print_stats(stats, f"HPSv3 — {tag}" + (f" shard {args.shard}/{args.num_shards}"
                                           if args.num_shards > 1 else ""))
    print(f"\nper-image CSV: {csv_path}\nsummary JSON:  {json_path}")
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Score FLUX images with the HPSv3 reward model (run in the hpsv3 env)."
    )
    parser.add_argument("--gen_dir", nargs="+",
                        help="One or more config directories of images to score.")
    parser.add_argument("--prompt_file", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "benchmarks", "hpsv3_benchmark.json"),
        help="The prompt JSON used for generation.")
    parser.add_argument("--merge_shards", default=None,
                        help="Merge an already-scored directory's shard CSVs and exit "
                             "(no model load, no GPU).")

    parser.add_argument("--config_path", default=None,
                        help="HPSv3 model yaml (default: the packaged HPSv3_7B.yaml).")
    parser.add_argument("--checkpoint_path", default=None,
                        help="HPSv3.safetensors (default: downloaded from the HF hub).")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Images per forward pass. 8 fits comfortably on a 48 GB card; "
                             "HPSv3 resizes every image to 256*28*28 pixels internally.")

    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--out_dir", default=None, help="default: each --gen_dir")
    parser.add_argument("--tag", default=None,
                        help="Output file prefix (default: the gen_dir basename).")
    args = parser.parse_args()

    if args.merge_shards:
        merge_shards(args.merge_shards, args.prompt_file)
        return
    if not args.gen_dir:
        parser.error("--gen_dir is required (or use --merge_shards)")
    if not os.path.exists(args.prompt_file):
        raise SystemExit(f"prompt file not found: {args.prompt_file}")
    for d in args.gen_dir:
        if not os.path.isdir(d):
            raise SystemExit(f"not a directory: {d}")
    if len(args.gen_dir) > 1 and args.tag:
        parser.error("--tag cannot be combined with multiple --gen_dir")

    prompts = prompt_index(args.prompt_file)

    # Imported here so that --merge_shards and argument errors do not need the
    # 16 GB model or even a working GPU.
    from hpsv3 import HPSv3RewardInferencer

    print(f"[hpsv3] loading reward model on {args.device} "
          f"(checkpoint: {args.checkpoint_path or 'HF hub MizzenAI/HPSv3'})", flush=True)
    t0 = time.time()
    inferencer = HPSv3RewardInferencer(
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        device=args.device,
    )
    print(f"[hpsv3] model ready in {time.time() - t0:.0f}s", flush=True)

    overall = {}
    for gen_dir in args.gen_dir:
        stats = score_dir(inferencer, gen_dir, prompts, args)
        if stats:  # Only add to overall if directory had images
            overall[os.path.basename(os.path.normpath(gen_dir))] = stats["OVERALL"]["mean"]

    if len(overall) > 1:
        print("\n=== HPSv3 overall, all configs ===")
        for cfg, mean in overall.items():
            print(f"  {cfg:<24}{mean:>9.4f}")


if __name__ == "__main__":
    main()
