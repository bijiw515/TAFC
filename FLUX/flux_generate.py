#!/usr/bin/env python3
"""
FLUX baseline generation (no caching).

This is the reference/GT arm of the DrawBench benchmark: same prompt order,
seeds, resolution, step count and file names as tafc_generate.py /
seacache_generate.py, but with the stock diffusers forward pass. Full-reference
metrics (PSNR / SSIM / LPIPS) are computed against images produced here.

Single prompt:
  python flux_generate.py --prompt "a red car" --output_dir outputs/base

Whole benchmark:
  python flux_generate.py --prompt_file benchmarks/drawbench.csv \
      --output_dir eval_samples/baseline --name_mode index \
      --stats_file eval_samples/baseline/gen_stats.json
"""

import argparse
import os
import time
from datetime import datetime

import torch
from diffusers import DiffusionPipeline

from bench_common import image_name, load_prompts, sample_seed, write_stats


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def add_common_args(parser):
    """CLI shared by all three FLUX generation entry points."""
    parser.add_argument("--prompt_file", type=str, default=None,
                        help="Benchmark file: .csv (DrawBench), .json, or .txt (one prompt per line).")
    parser.add_argument("--prompt", type=str, default="a photo of an astronaut riding a horse",
                        help="Single prompt. Used only when --prompt_file is absent.")
    parser.add_argument("--num_prompts", type=int, default=None,
                        help="Use only the first N prompts of the benchmark.")
    parser.add_argument("--shard", type=int, default=0, help="Round-robin shard index for this worker.")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of workers.")

    parser.add_argument("--output_dir", type=str, required=True, help="Directory for generated images.")
    parser.add_argument("--name_mode", type=str, default="index", choices=["index", "legacy"],
                        help="'index' -> 0007.png (comparable across configs); 'legacy' -> prompt in filename.")
    parser.add_argument("--stats_file", type=str, default=None,
                        help="Where to write per-image timing/cache stats (default: <output_dir>/gen_stats.json).")
    parser.add_argument("--skip_existing", action="store_true", default=False,
                        help="Do not regenerate images that already exist.")

    parser.add_argument("--width", type=int, default=1024, help="Image width (multiple of 16).")
    parser.add_argument("--height", type=int, default=1024, help="Image height (multiple of 16).")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Sampling steps.")
    parser.add_argument("--guidance", type=float, default=3.5, help="Guidance scale.")
    parser.add_argument("--seed", type=int, default=42, help="Base seed; per-image seed is derived from prompt index.")
    parser.add_argument("--num_images_per_prompt", type=int, default=1, help="Images per prompt.")

    parser.add_argument("--model_name", type=str, default="flux-dev", choices=["flux-dev", "flux-schnell"])
    parser.add_argument("--model_id", type=str, default=None, help="Explicit HF model id; overrides --model_name.")
    parser.add_argument("--offload", action="store_true", help="Use enable_model_cpu_offload().")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--warmup_steps", type=int, default=4,
                        help="Steps for one throwaway generation before timing (0 disables). "
                             "Keeps the first measured image from absorbing kernel autotuning.")
    return parser


def resolve_model(args):
    if args.model_id is not None:
        return args.model_id, ("flux-schnell" if "schnell" in args.model_id else "flux-dev")
    model_id = ("black-forest-labs/FLUX.1-dev" if args.model_name == "flux-dev"
                else "black-forest-labs/FLUX.1-schnell")
    return model_id, args.model_name


def build_pipeline(args, model_id):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if (args.dtype == "bf16" and device == "cuda") else torch.float16
    print(f"[{now_str()}] Loading {model_id} ({torch_dtype})", flush=True)
    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch_dtype)
    if args.offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    return pipe, device, torch_dtype


def collect_prompts(args, parser):
    """Return ``[{idx, prompt, category}]`` plus a description of the source."""
    if args.prompt_file:
        items = load_prompts(args.prompt_file, limit=args.num_prompts,
                             shard=args.shard, num_shards=args.num_shards)
        return items, args.prompt_file
    if args.prompt:
        return [{"idx": 0, "prompt": args.prompt, "category": ""}], "<inline-prompt>"
    parser.error("You must provide either --prompt_file or --prompt.")


def pipe_kwargs(args, model_name, num_steps):
    return dict(
        num_inference_steps=num_steps,
        height=(args.height // 16) * 16,
        width=(args.width // 16) * 16,
        guidance_scale=0.0 if model_name == "flux-schnell" else float(args.guidance),
        max_sequence_length=256 if model_name == "flux-schnell" else 512,
        num_images_per_prompt=1,
    )


def run_warmup(pipe, args, model_name, device, reset_fn=None):
    """One short throwaway generation so timings are not polluted by autotuning."""
    if args.warmup_steps <= 0:
        return
    if reset_fn is not None:
        reset_fn(args.warmup_steps)
    kwargs = pipe_kwargs(args, model_name, args.warmup_steps)
    print(f"[{now_str()}] Warmup ({args.warmup_steps} steps)", flush=True)
    pipe(prompt="warmup", generator=torch.Generator(device=device).manual_seed(0), **kwargs)
    if device == "cuda":
        torch.cuda.synchronize()


def generate_loop(pipe, args, model_name, model_id, device, num_steps, items,
                  prompt_source, tag, config_extra=None, reset_fn=None, per_image_stats=None):
    """Shared generation loop: seeds, file names, timing and incremental stats.

    ``reset_fn(num_steps)`` clears cache state before each image; ``per_image_stats()``
    returns cache counters for the image that was just produced.
    """
    os.makedirs(args.output_dir, exist_ok=True)
    stats_path = args.stats_file or os.path.join(args.output_dir, "gen_stats.json")
    kwargs = pipe_kwargs(args, model_name, num_steps)

    payload = {
        "tag": tag,
        "model_id": model_id,
        "model_name": model_name,
        "steps": num_steps,
        "guidance": kwargs["guidance_scale"],
        "size": f"{kwargs['width']}x{kwargs['height']}",
        "base_seed": int(args.seed),
        "num_images_per_prompt": int(args.num_images_per_prompt),
        "prompt_source": prompt_source,
        "shard": f"{args.shard}/{args.num_shards}",
        "warmup_steps": int(args.warmup_steps),
        "config": config_extra or {},
        "images": {},
    }

    run_warmup(pipe, args, model_name, device, reset_fn=reset_fn)

    n_per = int(args.num_images_per_prompt)
    total = len(items) * n_per
    done = 0
    for item in items:
        for i in range(n_per):
            done += 1
            idx, prompt = item["idx"], item["prompt"]
            fn = image_name(idx, i, name_mode=args.name_mode, prompt=prompt, tag=tag)
            out_path = os.path.join(args.output_dir, fn)
            key = os.path.splitext(fn)[0]

            if args.skip_existing and os.path.exists(out_path):
                print(f"[{done}/{total}] skip existing {fn}", flush=True)
                continue

            seed_this = sample_seed(args.seed, idx, i, n_per)
            if reset_fn is not None:
                reset_fn(num_steps)
            generator = torch.Generator(device=device).manual_seed(seed_this)

            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            out = pipe(prompt=prompt, generator=generator, **kwargs)
            if device == "cuda":
                torch.cuda.synchronize()
            dt = time.time() - t0

            out.images[0].save(out_path)

            row = {
                "prompt": prompt,
                "category": item.get("category", ""),
                "seed": seed_this,
                "seconds": round(dt, 3),
                "file": fn,
            }
            if per_image_stats is not None:
                row.update(per_image_stats())
            payload["images"][key] = row

            extra = ""
            if row.get("cache_rate_pct") is not None:
                extra = f" cache={row['cache_rate_pct']:.1f}%"
            print(f"[{done}/{total}] {dt:6.2f}s{extra} -> {fn} | {prompt[:60]}", flush=True)

            write_stats(stats_path, payload)

    finalize_stats(payload, stats_path)
    return payload


def finalize_stats(payload, stats_path):
    rows = list(payload["images"].values())
    if rows:
        secs = [r["seconds"] for r in rows]
        rates = [r["cache_rate_pct"] for r in rows if r.get("cache_rate_pct") is not None]
        payload["summary"] = {
            "num_images": len(rows),
            "avg_seconds": round(sum(secs) / len(secs), 3),
            "total_seconds": round(sum(secs), 1),
            "avg_cache_rate_pct": round(sum(rates) / len(rates), 2) if rates else None,
        }
        write_stats(stats_path, payload)
        s = payload["summary"]
        print("\n" + "=" * 70)
        print(f"[Summary] {payload['tag']} | {s['num_images']} images | "
              f"avg {s['avg_seconds']:.2f}s/image | total {s['total_seconds']:.0f}s")
        if s["avg_cache_rate_pct"] is not None:
            print(f"          avg cache rate {s['avg_cache_rate_pct']:.1f}%")
        print(f"          images: {payload_dir(stats_path)}")
        print(f"          stats:  {stats_path}")
        print("=" * 70, flush=True)
    else:
        print("[Summary] nothing generated (all skipped?)", flush=True)


def payload_dir(stats_path):
    return os.path.dirname(os.path.abspath(stats_path))


def main():
    parser = argparse.ArgumentParser(
        description="FLUX baseline (no caching) — DrawBench benchmark arm / GT for fidelity metrics."
    )
    add_common_args(parser)
    args = parser.parse_args()

    items, prompt_source = collect_prompts(args, parser)
    model_id, model_name = resolve_model(args)
    num_steps = args.num_inference_steps if args.num_inference_steps else (
        4 if model_name == "flux-schnell" else 50)

    pipe, device, torch_dtype = build_pipeline(args, model_id)

    print(f"[{now_str()}] baseline | steps={num_steps} | guidance={args.guidance} | "
          f"prompts={len(items)} | seed={args.seed} | dtype={torch_dtype}", flush=True)

    generate_loop(pipe, args, model_name, model_id, device, num_steps, items,
                  prompt_source, tag="baseline")


if __name__ == "__main__":
    main()
