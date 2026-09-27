#!/usr/bin/env python3
"""Collect DrawBench results across configs into one table.

Reads, for every config directory under --out_dir:
  gen_stats.json                    (written by the generation scripts)
  <config>_drawbench.json           (written by eval_drawbench.py)

and prints speed / cache-rate / quality side by side, with speedup relative to
the baseline config.

  python summarize_drawbench.py --out_dir eval_samples

--rank additionally prints the CycleReward average-rank table per speed-matched
group of competing methods (see eval_cyclereward.py); it reads the per-image
CSVs written there, so score the configs first.

  python summarize_drawbench.py --out_dir eval_samples --rank
"""

import argparse
import json
import os


def load_json(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def collect(out_dir, baseline):
    configs = sorted(
        d for d in os.listdir(out_dir)
        if os.path.isdir(os.path.join(out_dir, d))
    )
    # baseline first, the rest alphabetically
    configs.sort(key=lambda c: (c != baseline, c))

    rows = []
    for config in configs:
        cfg_dir = os.path.join(out_dir, config)
        gen = load_json(os.path.join(cfg_dir, "gen_stats.json")) or {}
        ev = load_json(os.path.join(cfg_dir, f"{config}_drawbench.json")) or {}
        # eval_cyclereward.py runs separately (it loads its own BLIP reward
        # model), so its means land in a sibling file; fold them into the same
        # metric row when present.
        crwd = load_json(os.path.join(cfg_dir, f"{config}_cyclereward.json")) or {}
        summary = gen.get("summary") or {}
        if not summary and not ev and not crwd:
            continue
        rows.append({
            "config": config,
            "method": (gen.get("config") or {}).get("method", "baseline"),
            "num_images": (summary.get("num_images") or ev.get("num_images")
                           or crwd.get("num_images")),
            "avg_seconds": summary.get("avg_seconds"),
            "avg_cache_rate_pct": summary.get("avg_cache_rate_pct"),
            "metrics": {**(ev.get("mean") or {}), **(crwd.get("mean") or {})},
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "eval_samples"))
    parser.add_argument("--baseline", default="baseline")
    parser.add_argument("--json_out", default=None, help="also write the table as JSON")
    parser.add_argument("--rank", action="store_true",
                        help="Also print the CycleReward average-rank table for each "
                             "speed-matched group (needs eval_cyclereward.py's CSVs).")
    parser.add_argument("--per_category", action="store_true",
                        help="With --rank: break the ranks down by DrawBench category.")
    args = parser.parse_args()

    rows = collect(args.out_dir, args.baseline)
    if not rows:
        raise SystemExit(f"no results found under {args.out_dir}")

    base = next((r for r in rows if r["config"] == args.baseline), None)
    base_t = base["avg_seconds"] if base else None

    metric_names = []
    for r in rows:
        for m in r["metrics"]:
            if m not in metric_names:
                metric_names.append(m)

    # cyclereward_* names are much longer than psnr/ssim, so size the metric
    # columns to whatever is actually present
    w = max([11] + [len(m) + 2 for m in metric_names])
    header = f"{'config':<20}{'n':>5}{'s/img':>9}{'speedup':>9}{'cache%':>8}"
    header += "".join(f"{m:>{w}}" for m in metric_names)
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in rows:
        t = r["avg_seconds"]
        speed = f"{base_t / t:.2f}x" if (base_t and t) else "-"
        cache = r["avg_cache_rate_pct"]
        line = (f"{r['config']:<20}{(r['num_images'] or 0):>5}"
                f"{(f'{t:.2f}' if t else '-'):>9}{speed:>9}"
                f"{(f'{cache:.1f}' if cache is not None else '-'):>8}")
        for m in metric_names:
            val = r["metrics"].get(m)
            line += f"{(f'{val:.4f}' if val is not None else '-'):>{w}}"
        print(line)
    print("=" * len(header))

    if args.rank:
        # same rule and same implementation as summarize_hpsv3.py --rank
        from summarize_hpsv3 import print_rank_groups
        print_rank_groups(args.out_dir, args.per_category)

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"written: {args.json_out}")


if __name__ == "__main__":
    main()
