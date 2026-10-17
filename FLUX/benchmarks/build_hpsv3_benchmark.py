#!/usr/bin/env python3
"""Build the HPSv3 image-generation benchmark prompt file.

The HPSv3 paper (ICCV 2025, sec. 5.1.2) samples 1,000 prompts from each of the
12 HPDv3 categories -> 12,000 prompts, and reports one HPSv3 score per category
plus an overall number. The prompt lists live in the HPDv3 dataset repo as
``benchmark/benchmark_<Category>.json``; each entry is a VLM caption of a
high-quality real image, always at aspect ratio 1.0 (so 1024x1024 square).

This script downloads those 12 files and merges them into one prompt file that
``bench_common.load_prompts`` can read.

Category order is *interleaved*, not concatenated: index 0 is Animals[0],
index 1 is Architecture[0], ... index 11 is Transportation[0], index 12 is
Animals[1], and so on. That way both ``--num_prompts N`` (smoke tests) and
``--num_shards`` (round-robin splitting across GPUs) stay balanced across
categories without any extra bookkeeping.

  python benchmarks/build_hpsv3_benchmark.py
  python benchmarks/build_hpsv3_benchmark.py --per_category 20 \
      --out benchmarks/hpsv3_benchmark_smoke.json
"""

import argparse
import json
import os
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))

# The 12 HPDv3 categories, in the order the paper's table lists them.
CATEGORIES = [
    "Animals", "Architecture", "Arts", "Characters", "Design", "Food",
    "Natural Scenery", "Others", "Plants", "Products", "Science",
    "Transportation",
]
BASE_URL = "https://huggingface.co/datasets/MizzenAI/HPDv3/resolve/main/benchmark"


def fetch_category(category, cache_dir):
    """Download (and cache) one category's prompt list."""
    os.makedirs(cache_dir, exist_ok=True)
    local = os.path.join(cache_dir, f"benchmark_{category}.json")
    if not os.path.exists(local):
        url = f"{BASE_URL}/benchmark_{urllib.parse.quote(category)}.json"
        print(f"  downloading {category} ...", flush=True)
        tmp = local + ".tmp"
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, local)
    with open(local, encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(HERE, "hpsv3_benchmark.json"))
    parser.add_argument("--cache_dir", default=os.path.join(HERE, "hpdv3_raw"),
                        help="Where the raw per-category JSONs are kept.")
    parser.add_argument("--per_category", type=int, default=None,
                        help="Keep only the first N prompts of each category "
                             "(default: all 1000, i.e. the paper's benchmark).")
    args = parser.parse_args()

    per_cat = {}
    for category in CATEGORIES:
        rows = fetch_category(category, args.cache_dir)
        prompts = [r["caption"].strip() for r in rows if (r.get("caption") or "").strip()]
        if args.per_category:
            prompts = prompts[:args.per_category]
        per_cat[category] = prompts
        print(f"  {category:<18} {len(prompts)} prompts", flush=True)

    # Interleave: one prompt per category per round.
    items = []
    for i in range(max(len(v) for v in per_cat.values())):
        for category in CATEGORIES:
            prompts = per_cat[category]
            if i < len(prompts):
                items.append({"prompt": prompts[i], "category": category,
                              "category_index": i})

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=1)

    print(f"\n{len(items)} prompts over {len(CATEGORIES)} categories -> {args.out}")
    print("index 0..11 is one prompt per category, then it repeats — so "
          "--num_prompts / --num_shards stay category-balanced.")


if __name__ == "__main__":
    main()
