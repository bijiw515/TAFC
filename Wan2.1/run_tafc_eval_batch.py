#!/usr/bin/env python3
"""
Batch driver for TAFC evaluation - loads model once and processes all prompts.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent.resolve()


def main():
    parser = argparse.ArgumentParser(
        description="Run TAFC evaluation on VBench prompt list (batch mode - model loads once)"
    )
    parser.add_argument("--prompt_file", type=str,
                        default="/data/dev2/lgbi/TeaCache/eval/teacache/vbench/VBench_full_info.json",
                        help="VBench prompt file")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory (default: eval_samples/<config>)")
    parser.add_argument("--task", type=str, default="t2v-1.3B",
                        help="Task name (t2v-1.3B or t2v-14B)")
    parser.add_argument("--ckpt_dir", type=str,
                        default="/data/dev2/lgbi/SeaCache/Wan2.1/Wan2.1-T2V-1.3B",
                        help="Model checkpoint directory")
    parser.add_argument("--size", type=str, default="832*480",
                        help="Video size (832*480 or 1280x720)")
    parser.add_argument("--sample_steps", type=int, default=50,
                        help="Sampling steps")
    parser.add_argument("--base_seed", type=int, default=42,
                        help="Base seed for generation")
    parser.add_argument("--use_ret_steps", action="store_true", default=True,
                        help="Use retention steps")
    parser.add_argument("--offload_model", type=str, default="False",
                        help="Offload model to CPU")
    parser.add_argument("--config", type=str, required=True,
                        help="Config name (e.g., tafc_0.3)")
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=None,
        help="GPU ID to use. If not set, inherits CUDA_VISIBLE_DEVICES from parent."
    )
    parser.add_argument("--tafc_thresh", type=str, default="0.2",
                        help="TAFC threshold (extracted from config name if not specified)")
    parser.add_argument("--tafc_max_cache", type=float, default=6.0,
                        help="Max consecutive cache budget")
    args = parser.parse_args()

    # Extract threshold from config name if not explicitly set
    if "tafc_" in args.config:
        args.tafc_thresh = args.config.split("_")[1]

    # Load prompts
    with open(args.prompt_file) as f:
        prompt_data = json.load(f)
        prompts = [x["prompt_en"] for x in prompt_data]

    print(f"[driver] Loaded {len(prompts)} prompts", flush=True)

    # Prepare environment for subprocess
    env = os.environ.copy()
    if args.gpu_id is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
        env['LOCAL_RANK'] = '0'
        env['RANK'] = '0'
        env['WORLD_SIZE'] = '1'
        print(f"[driver] Using GPU {args.gpu_id} (mapped to cuda:0 in subprocess)", flush=True)
    else:
        if 'LOCAL_RANK' not in env:
            env['LOCAL_RANK'] = '0'
        if 'RANK' not in env:
            env['RANK'] = '0'
        if 'WORLD_SIZE' not in env:
            env['WORLD_SIZE'] = '1'
        print(f"[driver] Using CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', 'default')}", flush=True)

    # Output directory
    if args.out_dir is None:
        output_dir = HERE / "eval_samples" / args.config
    else:
        output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write prompts to a temporary text file (one prompt per line)
    temp_prompt_file = HERE / f"temp_prompts_{args.config}.txt"
    with open(temp_prompt_file, 'w', encoding='utf-8') as f:
        for p in prompts:
            f.write(p + '\n')

    # Build command for batch processing
    cmd = [
        sys.executable, 'tafc_generate_batch.py',
        '--prompt_list', str(temp_prompt_file),
        '--output_dir', str(output_dir),
        '--ckpt_dir', args.ckpt_dir,
        '--task', args.task,
        '--size', args.size,
        '--frame_num', str(81),
        '--sample_steps', str(args.sample_steps),
        '--sample_shift', str(5.0),
        '--sample_guide_scale', str(5.0),
        '--sample_solver', 'unipc',
        '--offload_model', args.offload_model,
        '--base_seed', str(args.base_seed),
        '--tafc_thresh', str(args.tafc_thresh),
        '--tafc_max_cache', str(args.tafc_max_cache),
        '--log_interval', str(20),
    ]

    if args.use_ret_steps:
        cmd.append('--use_ret_steps')

    print(f"[driver] ({args.config}) Starting batch generation (model loads once)", flush=True)
    print(f"[driver] Output: {output_dir}", flush=True)

    # Run batch subprocess
    proc = subprocess.run(cmd, cwd=HERE, env=env)

    # Clean up temp file
    if temp_prompt_file.exists():
        temp_prompt_file.unlink()

    if proc.returncode != 0:
        print(f"[driver] ERROR: batch generation failed with rc={proc.returncode}", file=sys.stderr)
        sys.exit(proc.returncode)

    print(f"[driver] ({args.config}) Batch generation complete", flush=True)


if __name__ == "__main__":
    main()
