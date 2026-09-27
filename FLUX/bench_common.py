"""Shared helpers for the FLUX benchmark harness (DrawBench).

Used by the three generation entry points (flux_generate.py / tafc_generate.py /
seacache_generate.py), the batch driver (run_drawbench_eval.py) and the metric
script (eval_drawbench.py) so that every config sees exactly the same prompt
order, seeds and file names -- which is what makes full-reference metrics
against the no-cache baseline meaningful.

Conventions
-----------
* A prompt's ``idx`` is its position in the *full* benchmark file and never
  changes with --num_prompts / sharding.
* Benchmark images are named ``{idx:04d}.png`` so the same prompt lands on the
  same file name in every config directory.
* Per-image seed is ``base_seed + idx * num_images_per_prompt + i``.
"""

import csv
import json
import os
import re


# ----------------------------------------------------------------------------
# Prompt loading
# ----------------------------------------------------------------------------

def _from_csv(path):
    """DrawBench layout: columns ``Prompts`` and ``Category``."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        prompt_key = next(
            (k for k in ("Prompts", "prompt", "Prompt", "prompt_en", "text") if k in fields),
            fields[0] if fields else None,
        )
        if prompt_key is None:
            raise ValueError(f"{path}: no columns found")
        cat_key = next((k for k in ("Category", "category", "dimension") if k in fields), None)
        rows = []
        for row in reader:
            prompt = (row.get(prompt_key) or "").strip()
            if not prompt:
                continue
            rows.append((prompt, (row.get(cat_key) or "").strip() if cat_key else ""))
    return rows


def _from_json(path):
    """Accepts a list of strings, or VBench-style list of dicts."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("prompts", data.get("data", []))
    rows = []
    for item in data:
        if isinstance(item, str):
            prompt, cat = item.strip(), ""
        else:
            prompt = next(
                (item[k] for k in ("prompt_en", "prompt", "Prompts", "text") if item.get(k)),
                "",
            ).strip()
            cat = str(item.get("category") or item.get("dimension") or "")
        if prompt:
            rows.append((prompt, cat))
    return rows


def _from_txt(path):
    with open(path, encoding="utf-8") as f:
        return [(line.strip(), "") for line in f if line.strip()]


def load_prompts(path, limit=None, shard=0, num_shards=1):
    """Return ``[{idx, prompt, category}]`` for the requested slice.

    ``limit`` truncates the benchmark; ``shard``/``num_shards`` then take a
    round-robin slice so parallel workers get an even mix of prompt lengths.
    ``idx`` always refers to the position in the full file.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        rows = _from_csv(path)
    elif ext == ".json":
        rows = _from_json(path)
    else:
        rows = _from_txt(path)
    if not rows:
        raise ValueError(f"no prompts found in {path}")

    items = [{"idx": i, "prompt": p, "category": c} for i, (p, c) in enumerate(rows)]
    if limit is not None and limit > 0:
        items = items[:limit]
    if num_shards > 1:
        items = [it for it in items if it["idx"] % num_shards == shard]
    return items


# ----------------------------------------------------------------------------
# File naming / seeds
# ----------------------------------------------------------------------------

def safe_filename(name):
    name = re.sub(r"\s+", "_", name.strip())
    name = re.sub(r"[^0-9A-Za-z._-]", "", name)
    return name or "img"


def image_name(idx, sample=0, name_mode="index", prompt="", tag="IMG"):
    """Output file name for one generated image.

    ``index`` (benchmark mode) gives ``0007.png`` / ``0007_1.png``, identical
    across configs. ``legacy`` keeps the older prompt-in-filename scheme.
    """
    if name_mode == "index":
        suffix = "" if sample == 0 else f"_{sample}"
        return f"{idx:04d}{suffix}.png"
    return f"{tag}_{idx:05d}-{safe_filename(prompt)[:80]}.png"


def sample_seed(base_seed, idx, sample=0, num_images_per_prompt=1):
    """Seed tied to the prompt index, so results are reproducible per prompt
    regardless of --num_prompts, sharding or skipped/resumed images."""
    return int(base_seed) + int(idx) * int(num_images_per_prompt) + int(sample)


# ----------------------------------------------------------------------------
# Stats
# ----------------------------------------------------------------------------

def write_stats(path, payload):
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)
