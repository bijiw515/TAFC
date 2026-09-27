"""Full-reference video fidelity evaluation with pyiqa (PSNR / SSIM / LPIPS ...).

Compares each generated video against the same-named video in the GT dir
(usually the no-cache baseline), frame by frame.

Aggregation: metric is averaged over frames within a video, then averaged over
videos (macro average, one vote per video). This differs from TeaCache's
eval.py, which averages per-frame values inside fixed-size video batches and
then averages the batch means -- with a ragged last batch that silently
up-weights the tail videos.

Usage:
  python eval_pyiqa.py --gt_dir eval_samples/baseline --gen_dir eval_samples/tafc_0.3
"""

import argparse
import csv
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import imageio.v3 as iio
import numpy as np
import pyiqa
import torch

# metric name -> (pyiqa metric id, extra create_metric kwargs)
METRIC_SPECS = {
    "psnr": ("psnr", {}),  # RGB, no Y-channel conversion
    "psnr_y": ("psnr", {"test_y_channel": True}),
    "ssim": ("ssim", {}),  # pyiqa default: Y channel (YIQ), no downsample
    "ssim_rgb": ("ssim", {"test_y_channel": False, "channels": 3}),
    "ms_ssim": ("ms_ssim", {}),
    "lpips": ("lpips", {}),  # AlexNet, v0.1 weights
    "lpips_vgg": ("lpips-vgg", {}),
    "dists": ("dists", {}),
}


def read_video(path):
    """Return float32 tensor [T, 3, H, W] in [0, 1] on CPU."""
    arr = iio.imread(path, plugin="FFMPEG")  # [T, H, W, C] uint8
    if arr.ndim == 3:  # single frame
        arr = arr[None]
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    tensor = torch.from_numpy(np.ascontiguousarray(arr)).permute(0, 3, 1, 2)
    return tensor.float().div_(255.0)


def align(gt, gen):
    """Trim to the common frame count; require identical spatial size."""
    if gt.shape[-2:] != gen.shape[-2:]:
        raise ValueError(f"resolution mismatch: gt {tuple(gt.shape[-2:])} vs gen {tuple(gen.shape[-2:])}")
    t = min(gt.shape[0], gen.shape[0])
    return gt[:t], gen[:t], gt.shape[0], gen.shape[0]


@torch.no_grad()
def score_pair(metrics, gt, gen, device, chunk):
    """Per-frame scores for every metric on one video pair."""
    out = {name: [] for name in metrics}
    for start in range(0, gt.shape[0], chunk):
        a = gt[start : start + chunk].to(device, non_blocking=True)
        b = gen[start : start + chunk].to(device, non_blocking=True)
        for name, metric in metrics.items():
            val = metric(b, a)  # (test, reference) order per pyiqa convention
            out[name].extend(torch.as_tensor(val).flatten().float().cpu().tolist())
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", required=True, help="reference videos (e.g. no-cache baseline)")
    parser.add_argument("--gen_dir", required=True, help="videos under test")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["psnr", "ssim", "lpips"],
        choices=sorted(METRIC_SPECS),
        help="which metrics to compute",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk", type=int, default=16, help="frames per forward pass")
    parser.add_argument("--out_dir", default=None, help="where to write results (default: gen_dir)")
    parser.add_argument("--tag", default=None, help="output file prefix (default: gen_dir basename)")
    args = parser.parse_args()

    names = sorted(f for f in os.listdir(args.gen_dir) if f.lower().endswith(".mp4"))
    if not names:
        raise SystemExit(f"no .mp4 files in {args.gen_dir}")
    missing = [n for n in names if not os.path.exists(os.path.join(args.gt_dir, n))]
    if missing:
        raise SystemExit(f"{len(missing)} videos have no GT counterpart, first: {missing[0]}")

    metrics = {}
    for name in args.metrics:
        metric_id, kwargs = METRIC_SPECS[name]
        metrics[name] = pyiqa.create_metric(metric_id, device=args.device, **kwargs)

    out_dir = args.out_dir or args.gen_dir
    tag = args.tag or os.path.basename(os.path.normpath(args.gen_dir))
    os.makedirs(out_dir, exist_ok=True)

    per_video = []
    skipped = []
    started = time.time()
    # Decoding is the bottleneck and releases the GIL, so prefetch the next pair.
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = pool.submit(
            lambda n: (read_video(os.path.join(args.gt_dir, n)), read_video(os.path.join(args.gen_dir, n))),
            names[0],
        )
        for idx, name in enumerate(names):
            try:
                gt, gen = pending.result()
                if idx + 1 < len(names):
                    nxt = names[idx + 1]
                    pending = pool.submit(
                        lambda n: (read_video(os.path.join(args.gt_dir, n)), read_video(os.path.join(args.gen_dir, n))),
                        nxt,
                    )
                gt, gen, t_gt, t_gen = align(gt, gen)
                frames = score_pair(metrics, gt, gen, args.device, args.chunk)
                row = {"video": name, "frames": gt.shape[0]}
                if t_gt != t_gen:
                    row["frames_gt"], row["frames_gen"] = t_gt, t_gen
                for metric_name, values in frames.items():
                    row[metric_name] = float(np.mean(values))
                    row[f"{metric_name}_min"] = float(np.min(values))
                per_video.append(row)
                summary = ", ".join(f"{m}={row[m]:.4f}" for m in args.metrics)
                print(f"[{idx + 1}/{len(names)}] {summary}  {name}", flush=True)
            except Exception as e:
                skipped.append({"video": name, "error": str(e)})
                print(f"[{idx + 1}/{len(names)}] SKIPPED (error: {e})  {name}", flush=True)
                # Prefetch next video even after error
                if idx + 1 < len(names):
                    nxt = names[idx + 1]
                    pending = pool.submit(
                        lambda n: (read_video(os.path.join(args.gt_dir, n)), read_video(os.path.join(args.gen_dir, n))),
                        nxt,
                    )

    if not per_video:
        raise SystemExit("All videos failed to process. Check errors above.")

    means = {m: float(np.mean([r[m] for r in per_video])) for m in args.metrics}
    stds = {m: float(np.std([r[m] for r in per_video])) for m in args.metrics}

    csv_path = os.path.join(out_dir, f"{tag}_pyiqa_per_video.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_video[0].keys()))
        writer.writeheader()
        writer.writerows(per_video)

    if skipped:
        skipped_path = os.path.join(out_dir, f"{tag}_pyiqa_skipped.csv")
        with open(skipped_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["video", "error"])
            writer.writeheader()
            writer.writerows(skipped)

    result = {
        "gt_dir": os.path.abspath(args.gt_dir),
        "gen_dir": os.path.abspath(args.gen_dir),
        "num_videos": len(per_video),
        "num_skipped": len(skipped),
        "metrics": args.metrics,
        "mean": means,
        "std_across_videos": stds,
        "aggregation": "mean over frames per video, then mean over videos",
        "library": f"pyiqa {pyiqa.__version__}",
        "elapsed_sec": round(time.time() - started, 1),
    }
    json_path = os.path.join(out_dir, f"{tag}_pyiqa.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    print("\n=== pyiqa full-reference results ===")
    print(f"{len(per_video)} videos processed | {len(skipped)} skipped | GT: {args.gt_dir} | test: {args.gen_dir}")
    for m in args.metrics:
        print(f"  {m:10s} {means[m]:.4f}  (std across videos {stds[m]:.4f})")
    print(f"per-video CSV: {csv_path}")
    print(f"summary JSON:  {json_path}")
    if skipped:
        skipped_path = os.path.join(out_dir, f"{tag}_pyiqa_skipped.csv")
        print(f"skipped CSV:   {skipped_path}")
        print(f"\nWarning: {len(skipped)} video(s) were skipped due to errors. See {skipped_path} for details.")


if __name__ == "__main__":
    main()
