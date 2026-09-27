#!/usr/bin/env python3
"""DrawBench evaluation for FLUX caching methods.

Image counterpart of Wan2.1/eval_pyiqa.py. Two families of metrics:

* Full-reference (needs --gt_dir): PSNR / SSIM / LPIPS against the no-cache
  baseline. This is the standard way caching papers report quality loss -- it
  measures how far the accelerated sampler drifted from the exact same sampler
  without caching, at the same seed.
* No-reference: CLIP score (prompt alignment), ImageReward (human preference --
  the metric DrawBench is usually reported with), and pyiqa no-reference
  quality metrics. The first two need the DrawBench prompt file; none of them
  need a baseline, so they are the only meaningful numbers for the baseline arm
  itself.

Aggregation is a macro average: per image, then over images. A per-image CSV is
written alongside the JSON summary so per-category breakdowns stay possible.

Usage
-----
  # quality loss vs baseline
  python eval_drawbench.py --gt_dir eval_samples/baseline --gen_dir eval_samples/tafc_0.2

  # + prompt alignment and human preference
  python eval_drawbench.py --gt_dir eval_samples/baseline --gen_dir eval_samples/tafc_0.2 \
      --prompt_file benchmarks/drawbench.csv \
      --metrics psnr ssim lpips clip_score image_reward topiq_nr

  # baseline on its own (no GT)
  python eval_drawbench.py --gen_dir eval_samples/baseline \
      --prompt_file benchmarks/drawbench.csv --metrics clip_score image_reward topiq_nr
"""

import argparse
import csv
import json
import os
import time

import numpy as np
import torch
from PIL import Image

from bench_common import load_prompts

# name -> (pyiqa id, kwargs, needs_reference)
PYIQA_SPECS = {
    "psnr": ("psnr", {}, True),               # RGB, no Y-channel conversion
    "psnr_y": ("psnr", {"test_y_channel": True}, True),
    "ssim": ("ssim", {}, True),               # pyiqa default: Y channel, no downsample
    "ssim_rgb": ("ssim", {"test_y_channel": False, "channels": 3}, True),
    "ms_ssim": ("ms_ssim", {}, True),
    "lpips": ("lpips", {}, True),             # AlexNet, v0.1 weights
    "lpips_vgg": ("lpips-vgg", {}, True),
    "dists": ("dists", {}, True),
    "topiq_fr": ("topiq_fr", {}, True),
    # no-reference image quality
    "topiq_nr": ("topiq_nr", {}, False),
    "musiq": ("musiq", {}, False),
    "niqe": ("niqe", {}, False),
    "maniqa": ("maniqa", {}, False),
}
# metrics implemented here rather than through pyiqa
EXTRA_METRICS = {"clip_score", "image_reward"}
ALL_METRICS = sorted(set(PYIQA_SPECS) | EXTRA_METRICS)
# metrics that score an image against its prompt -> need --prompt_file
NEEDS_PROMPT = {"clip_score", "image_reward"}
# higher is better for these; lower is better for the rest
HIGHER_IS_BETTER = {"psnr", "psnr_y", "ssim", "ssim_rgb", "ms_ssim", "topiq_fr",
                    "topiq_nr", "musiq", "maniqa", "clip_score", "image_reward"}

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")


def read_image(path):
    """float32 tensor [1, 3, H, W] in [0, 1]."""
    with Image.open(path) as im:
        arr = np.array(im.convert("RGB"), dtype=np.uint8)  # np.array copies -> writable
    t = torch.from_numpy(arr).permute(2, 0, 1)[None]
    return t.float().div_(255.0)


def prompt_index(prompt_file):
    """Map ``0007`` (file stem, benchmark index) -> prompt text / category."""
    if not prompt_file:
        return {}
    items = load_prompts(prompt_file)
    return {f"{it['idx']:04d}": it for it in items}


class ClipScore:
    """CLIP image-text cosine similarity, the usual DrawBench alignment metric.

    Reported as the raw cosine similarity (typically 0.20-0.40 for
    ViT-L/14). Note this is a relative measure between configs on the same
    prompts, not an absolute quality number.
    """

    def __init__(self, model_id, device):
        from transformers import CLIPModel, CLIPProcessor
        self.device = device
        self.model = CLIPModel.from_pretrained(model_id).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_id)

    @torch.no_grad()
    def __call__(self, pil_image, prompt):
        inputs = self.processor(
            text=[prompt], images=[pil_image], return_tensors="pt",
            padding="max_length", truncation=True, max_length=77,
        ).to(self.device)
        out = self.model(**inputs)
        img = torch.nn.functional.normalize(out.image_embeds, dim=-1)
        txt = torch.nn.functional.normalize(out.text_embeds, dim=-1)
        return float((img * txt).sum())


def _patch_image_reward():
    """Make the ImageReward package importable on modern transformers/timm.

    ImageReward 1.5 vendors BLIP code written against transformers 4.15 and
    pins timm==0.6.13. Everything it actually needs still exists, just moved,
    so we forward the moved symbols instead of downgrading the env (which
    would break pyiqa and diffusers). Each patch is applied only if the
    original attribute is missing, so a correctly pinned env is untouched.
    """
    import sys
    import types

    # clip: only imported by ImageReward's CLIPScore/AestheticScore, which we
    # do not use. openai-clip needs pkg_resources (gone in setuptools >= 81).
    try:
        import clip  # noqa: F401
    except ModuleNotFoundError as exc:
        if exc.name != "pkg_resources":
            raise
        import packaging
        shim = types.ModuleType("pkg_resources")
        shim.packaging = packaging
        sys.modules["pkg_resources"] = shim

    # transformers >= 4.31 moved these out of modeling_utils into pytorch_utils.
    import transformers.modeling_utils as modeling_utils
    from transformers import pytorch_utils
    for name in ("apply_chunking_to_forward", "prune_linear_layer",
                 "find_pruneable_heads_and_indices"):
        if not hasattr(modeling_utils, name):
            # find_pruneable_heads_and_indices is imported but only used by
            # BertModel._prune_heads, which scoring never calls.
            setattr(modeling_utils, name, getattr(pytorch_utils, name, None))

    import ImageReward.models.BLIP.med as med
    # transformers 5 PreTrainedModel: init_weights() consults these, and
    # get_head_mask moved off the model class. BLIP's BertModel is built from
    # a local config and loaded from a checkpoint, so there is nothing to tie.
    if not hasattr(med.BertModel, "all_tied_weights_keys"):
        med.BertModel.all_tied_weights_keys = {}
    if not hasattr(med.BertModel, "get_head_mask"):
        med.BertModel.get_head_mask = (
            lambda self, head_mask, num_hidden_layers, is_attention_chunked=False:
            [None] * num_hidden_layers
        )

    # transformers 5 dropped tokenizer.additional_special_tokens_ids.
    import ImageReward.models.BLIP.blip_pretrain as blip_pretrain
    from transformers import BertTokenizer

    def init_tokenizer():
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        tokenizer.add_special_tokens({"additional_special_tokens": ["[ENC]"]})
        tokenizer.enc_token_id = tokenizer.convert_tokens_to_ids("[ENC]")
        return tokenizer

    blip_pretrain.init_tokenizer = init_tokenizer


class ImageRewardScore:
    """ImageReward (BLIP + MLP head trained on 137k human comparisons).

    The standard human-preference metric for DrawBench: it judges prompt
    fidelity and aesthetics jointly, which CLIP score alone does not. Scores
    are z-normalised, so they straddle zero (roughly -2.5 to +1.5 for FLUX at
    1024px) and only mean something relative to another config on the same
    prompts.
    """

    def __init__(self, device, model_path="ImageReward-v1.0", download_root=None,
                 med_config=None):
        _patch_image_reward()
        import ImageReward
        self.model = ImageReward.load(model_path, device=device,
                                      download_root=download_root,
                                      med_config=med_config)

    @torch.no_grad()
    def __call__(self, pil_image, prompt):
        return float(self.model.score(prompt, pil_image))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_dir", required=True, help="images under test")
    parser.add_argument("--gt_dir", default=None,
                        help="reference images (the no-cache baseline). Required for full-reference metrics.")
    parser.add_argument("--prompt_file", default=None,
                        help="DrawBench CSV; required for --metrics clip_score / image_reward, "
                             "also adds categories to the CSV.")
    parser.add_argument("--metrics", nargs="+", default=["psnr", "ssim", "lpips"],
                        choices=ALL_METRICS, metavar="METRIC",
                        help=f"any of: {', '.join(ALL_METRICS)}")
    parser.add_argument("--clip_model", default="openai/clip-vit-large-patch14")
    parser.add_argument("--image_reward_model", default="ImageReward-v1.0",
                        help="ImageReward name or a path to ImageReward.pt")
    parser.add_argument("--image_reward_root", default=None,
                        help="checkpoint cache dir (default: ~/.cache/ImageReward)")
    parser.add_argument("--image_reward_med_config", default=None,
                        help="BLIP med_config.json (default: downloaded next to the checkpoint)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_dir", default=None, help="default: --gen_dir")
    parser.add_argument("--tag", default=None, help="output file prefix (default: gen_dir basename)")
    args = parser.parse_args()

    names = sorted(f for f in os.listdir(args.gen_dir) if f.lower().endswith(IMAGE_EXTS))
    if not names:
        raise SystemExit(f"no images in {args.gen_dir}")

    fr_metrics = [m for m in args.metrics if m in PYIQA_SPECS and PYIQA_SPECS[m][2]]
    nr_metrics = [m for m in args.metrics if m in PYIQA_SPECS and not PYIQA_SPECS[m][2]]
    want_clip = "clip_score" in args.metrics
    want_reward = "image_reward" in args.metrics
    prompt_metrics = [m for m in args.metrics if m in NEEDS_PROMPT]

    if fr_metrics and not args.gt_dir:
        raise SystemExit(f"--gt_dir is required for full-reference metrics: {fr_metrics}")
    if prompt_metrics and not args.prompt_file:
        raise SystemExit(f"--prompt_file is required for: {', '.join(prompt_metrics)}")

    if args.gt_dir:
        missing = [n for n in names if not os.path.exists(os.path.join(args.gt_dir, n))]
        if missing:
            raise SystemExit(
                f"{len(missing)}/{len(names)} images have no GT counterpart in {args.gt_dir}, "
                f"first: {missing[0]}. Did both configs finish with --name_mode index?"
            )

    prompts = prompt_index(args.prompt_file)
    if prompt_metrics:
        unknown = [n for n in names if os.path.splitext(n)[0].split("_")[0] not in prompts]
        if unknown:
            raise SystemExit(
                f"{len(unknown)} images have no prompt in {args.prompt_file}, first: {unknown[0]}. "
                f"{', '.join(prompt_metrics)} need --name_mode index filenames."
            )

    metrics = {}
    if fr_metrics or nr_metrics:
        import pyiqa
        for name in fr_metrics + nr_metrics:
            metric_id, kwargs, _ = PYIQA_SPECS[name]
            metrics[name] = pyiqa.create_metric(metric_id, device=args.device, **kwargs)
    clip = ClipScore(args.clip_model, args.device) if want_clip else None
    reward = ImageRewardScore(args.device, args.image_reward_model,
                              args.image_reward_root,
                              args.image_reward_med_config) if want_reward else None

    out_dir = args.out_dir or args.gen_dir
    tag = args.tag or os.path.basename(os.path.normpath(args.gen_dir))
    os.makedirs(out_dir, exist_ok=True)

    per_image = []
    started = time.time()
    with torch.no_grad():
        for i, name in enumerate(names):
            gen_path = os.path.join(args.gen_dir, name)
            gen = read_image(gen_path).to(args.device)
            key = os.path.splitext(name)[0].split("_")[0]
            item = prompts.get(key, {})

            row = {"image": name, "prompt": item.get("prompt", ""),
                   "category": item.get("category", "")}

            if fr_metrics:
                gt = read_image(os.path.join(args.gt_dir, name)).to(args.device)
                if gt.shape[-2:] != gen.shape[-2:]:
                    raise SystemExit(
                        f"{name}: resolution mismatch gt {tuple(gt.shape[-2:])} "
                        f"vs gen {tuple(gen.shape[-2:])}"
                    )
                for m in fr_metrics:
                    row[m] = float(torch.as_tensor(metrics[m](gen, gt)).flatten()[0])
                del gt
            for m in nr_metrics:
                row[m] = float(torch.as_tensor(metrics[m](gen)).flatten()[0])
            if clip is not None or reward is not None:
                # both take the PIL image at native resolution and do their own
                # resize/normalise, so read it once and share it
                with Image.open(gen_path) as im:
                    pil = im.convert("RGB")
                    if clip is not None:
                        row["clip_score"] = clip(pil, item["prompt"])
                    if reward is not None:
                        row["image_reward"] = reward(pil, item["prompt"])

            per_image.append(row)
            shown = ", ".join(f"{m}={row[m]:.4f}" for m in args.metrics)
            print(f"[{i + 1}/{len(names)}] {shown}  {name}", flush=True)

    means = {m: float(np.mean([r[m] for r in per_image])) for m in args.metrics}
    stds = {m: float(np.std([r[m] for r in per_image])) for m in args.metrics}

    by_category = {}
    cats = sorted({r["category"] for r in per_image if r["category"]})
    for cat in cats:
        rows = [r for r in per_image if r["category"] == cat]
        by_category[cat] = {"count": len(rows),
                            **{m: float(np.mean([r[m] for r in rows])) for m in args.metrics}}

    csv_path = os.path.join(out_dir, f"{tag}_drawbench_per_image.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_image[0].keys()))
        writer.writeheader()
        writer.writerows(per_image)

    result = {
        "gen_dir": os.path.abspath(args.gen_dir),
        "gt_dir": os.path.abspath(args.gt_dir) if args.gt_dir else None,
        "prompt_file": os.path.abspath(args.prompt_file) if args.prompt_file else None,
        "benchmark": "DrawBench",
        "num_images": len(per_image),
        "metrics": args.metrics,
        "models": {k: v for k, v in (("clip_score", args.clip_model if want_clip else None),
                                     ("image_reward", args.image_reward_model if want_reward else None))
                   if v},
        "mean": means,
        "std_across_images": stds,
        "by_category": by_category,
        "aggregation": "mean over images (macro average)",
        "elapsed_sec": round(time.time() - started, 1),
    }
    json_path = os.path.join(out_dir, f"{tag}_drawbench.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    print("\n=== DrawBench results ===")
    print(f"{len(per_image)} images | test: {args.gen_dir}" +
          (f" | GT: {args.gt_dir}" if args.gt_dir else ""))
    for m in args.metrics:
        arrow = "up" if m in HIGHER_IS_BETTER else "down"
        print(f"  {m:12s} {means[m]:9.4f}  (std {stds[m]:.4f}, {arrow} is better)")
    if by_category:
        print("\nby category:")
        for cat, vals in by_category.items():
            body = "  ".join(f"{m}={vals[m]:.4f}" for m in args.metrics)
            print(f"  {cat:<22} n={vals['count']:<4} {body}")
    print(f"\nper-image CSV: {csv_path}")
    print(f"summary JSON:  {json_path}")


if __name__ == "__main__":
    main()
