# TAFC for Wan2.1
This is the official implementation of TAFC for Wan2.1.

## Installation
1. Clone the original [Wan2.1](https://github.com/Wan-Video/Wan2.1) repository and install all required dependencies (Python, CUDA, checkpoints, etc.) as described there.
2. Place the provided TAFC scripts (`tafc_generate.py`, `util_tafc.py`, and `util_seacache.py` from the SeaCache repo, which `tafc_generate.py` imports) into the Wan2.1 repository root.
3. For benchmark evaluation, also place `run_tafc_eval.py`, `run_tafc_eval_batch.py`, `tafc_generate_batch.py` and `eval_pyiqa.py` in the same directory.

## Usage
The main code is `tafc_generate.py`. Make sure to run the script inside the Wan2.1 directory.

### Text-to-Video (1.3B model)

    python tafc_generate.py \
        --task t2v-1.3B \
        --size 832*480 \
        --ckpt_dir ./Wan2.1-T2V-1.3B \
        --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
        --base_seed 42 \
        --offload_model True \
        --sample_steps 50 \
        --tafc_thresh 0.15

Arguments:

- `--task`: model task to run (e.g., `t2v-1.3B`)
- `--size` : output resolution (w*h) (`480p: 832*480, 720p: 1280*720`)
- `--ckpt_dir` : path to the checkpoint directory
- `--prompt` : text prompt for video generation
- `--base_seed` : base random seed
- `--sample_steps` : number of sampling steps for the solver
- `--tafc_thresh` : base curvature threshold, controls the cache ratio (lower = stricter)
- `--tafc_max_cache` : consecutive-cache budget at the last step (default `6.0`, ramps from 1)
- `--use_ret_steps` : force-compute the first 5 steps
- `--log_interval` : print diagnostics every N steps (`0` disables)

> **Note:** You can sweep over `tafc_thresh` to find the balanced quality-latency trade-offs.
> Wan2.1 uses classifier-free guidance, so TAFC runs one independent cache loop per CFG branch (cond / uncond).

### Closed-loop (PID) arguments

| flag | default | meaning |
|---|---|---|
| `--tafc_no_pid` | off | disable the closed loop (A/B baseline) |
| `--tafc_e_target` | 0.0 | absolute error target; 0 = auto-calibrate |
| `--tafc_auto_tol` | 2.5 | tolerated multiple of the one-step baseline |
| `--tafc_reject` | 3.0 | hard veto above this multiple of the budget |
| `--tafc_kp` / `--tafc_ki` / `--tafc_kd` | 0.4 / 0.05 / 0.2 | PID gains |
| `--tafc_gain_min` / `--tafc_gain_max` | 0.3 / 3.0 | clamps on the budget multiplier |
| `--tafc_brake_beta` | 2.0 | feedforward brake strength (0 disables) |
| `--tafc_brake_floor` | 0.3 | hardest the brake may squeeze the budget |
| `--tafc_calib_steps` | 3 | forced single-step computes before caching is allowed |

---
## Benchmark Evaluation
We follow the [TeaCache](https://github.com/ali-vilab/TeaCache) / [SeaCache](https://github.com/jiwoogit/SeaCache) protocol: generate the VBench prompts with the no-cache baseline and each TAFC setting, compute fidelity (PSNR / SSIM / LPIPS) against the baseline, and quality with VBench.

> **Note:** VBench and Wan2.1 require incompatible `transformers` versions, so use two separate conda environments (e.g. `tafc` for generation and fidelity, `vbench` for VBench scoring).

### 1. Generate samples
`run_tafc_eval.py` runs the baseline (`generate.py`, no cache) and TAFC configs. Videos are named `{prompt}-0.mp4` to match VBench's expected format.

    # full VBench prompt set, 4 configs, sequentially
    python run_tafc_eval.py \
        --prompt_file ./vbench/VBench_full_info.json \
        --ckpt_dir ./Wan2.1-T2V-1.3B \
        --task t2v-1.3B --size 832*480 \
        --out_dir ./eval_samples \
        --configs baseline tafc_0.03 tafc_0.15

    # one config per GPU, in parallel
    CUDA_VISIBLE_DEVICES=0 python run_tafc_eval.py --configs baseline &
    CUDA_VISIBLE_DEVICES=1 python run_tafc_eval.py --configs tafc_0.03 &
    CUDA_VISIBLE_DEVICES=2 python run_tafc_eval.py --configs tafc_0.15 &

Outputs:

    eval_samples/<config>/{prompt}-0.mp4
    eval_samples/<config>/gen_stats.json    # per-video latency and cache rate

Config names are `baseline` or `tafc_<thresh>`. Existing videos are skipped (`--skip_existing`), so interrupted runs can be resumed.

To avoid reloading the model for every prompt, use the batch driver (one process per config):

    python run_tafc_eval_batch.py \
        --config tafc_0.15 \
        --prompt_file ./vbench/VBench_full_info.json \
        --ckpt_dir ./Wan2.1-T2V-1.3B

### 2. Fidelity: PSNR / SSIM / LPIPS
Each generated video is compared frame-by-frame with the same-named baseline video (macro-averaged over videos).

    python eval_pyiqa.py \
        --gt_dir eval_samples/baseline \
        --gen_dir eval_samples/tafc_0.15 \
        --metrics psnr ssim lpips

Results are written next to `--gen_dir` (override with `--out_dir`, `--tag`). Other supported metrics: `psnr_y`, `ssim_rgb`, `ms_ssim`, `lpips_vgg`, `dists`.

Alternatively, the TeaCache script can be used:

    python common_metrics/eval.py \
        --gt_video_dir eval_samples/baseline \
        --generated_video_dir eval_samples/tafc_0.15

### 3. Quality: VBench
Run from the `eval/teacache` directory of the TeaCache repo, inside the `vbench` environment:

    conda activate vbench
    python vbench/run_vbench.py \
        --video_path /path/to/eval_samples/tafc_0.15 \
        --save_path ./vbench_scores/tafc_0.15
    python vbench/cal_vbench.py --score_dir ./vbench_scores/tafc_0.15

> **Note:** `run_vbench.py` reads the prompt info from `./vbench/VBench_full_info.json`. When evaluating a subset, point it at the subset info file instead.

### 4. Latency and cache rate
Latency and cache rate are read from `gen_stats.json`. TFLOPs and speedup are computed relative to the `baseline` config.

---
## Reproducing the Paper Results
Wan2.1-T2V-1.3B, 832×480, 65 frames, 50 steps, `sample_shift` 5.0, guidance 5.0, seed 42, full VBench prompt set. Operating points used in the paper:

| Speedup | `tafc_thresh` | `tafc_max_cache` |
|:--:|:--:|:--:|
| ~2× | 0.03 | 2.0 |
| ~3× | 0.15 | 5.0 |

    # no-cache reference
    python run_tafc_eval.py --configs baseline --frame_num 65 \
        --prompt_file ./vbench/VBench_full_info.json --ckpt_dir ./Wan2.1-T2V-1.3B

    # ~2x
    python run_tafc_eval_batch.py --config tafc_0.03 --tafc_max_cache 2.0 \
        --prompt_file ./vbench/VBench_full_info.json --ckpt_dir ./Wan2.1-T2V-1.3B
    # ~3x
    python run_tafc_eval_batch.py --config tafc_0.15 --tafc_max_cache 5.0 \
        --prompt_file ./vbench/VBench_full_info.json --ckpt_dir ./Wan2.1-T2V-1.3B

Then compute fidelity and VBench as described in Benchmark Evaluation.

> **Note:** The paper's Wan2.1 samples were generated with a DiffSynth-Studio port of the same TAFC code (no `--use_ret_steps`, tiled VAE decoding). The scripts here use the official Wan2.1 pipeline, where `--use_ret_steps` defaults to on in the evaluation drivers, so numbers can differ slightly from the paper.

---
## Inference Results
Latency may vary depending on your GPU.

> **Note:** `tafc_thresh = δ`

### Qualitative samples (T2V 1.3B, 480p)

Left to right: original (50 steps), SeaCache (δ = 0.2), TAFC (δ = 0.03). If the videos do not play inline, use the mp4 links.

*a backpack and an umbrella*

<table>
<tr><th align='center'>Original</th><th align='center'>SeaCache (δ = 0.2)</th><th align='center'>TAFC (δ = 0.03)</th></tr>
<tr>
<td><video src="../assets/samples/wan2.1/01_a_backpack_and_an_umbrella/baseline.mp4" controls muted loop playsinline width="280"></video><br><a href="../assets/samples/wan2.1/01_a_backpack_and_an_umbrella/baseline.mp4">mp4</a></td>
<td><video src="../assets/samples/wan2.1/01_a_backpack_and_an_umbrella/seacache.mp4" controls muted loop playsinline width="280"></video><br><a href="../assets/samples/wan2.1/01_a_backpack_and_an_umbrella/seacache.mp4">mp4</a></td>
<td><video src="../assets/samples/wan2.1/01_a_backpack_and_an_umbrella/tafc.mp4" controls muted loop playsinline width="280"></video><br><a href="../assets/samples/wan2.1/01_a_backpack_and_an_umbrella/tafc.mp4">mp4</a></td>
</tr>
</table>

*bar*

<table>
<tr><th align='center'>Original</th><th align='center'>SeaCache (δ = 0.2)</th><th align='center'>TAFC (δ = 0.03)</th></tr>
<tr>
<td><video src="../assets/samples/wan2.1/02_bar/baseline.mp4" controls muted loop playsinline width="280"></video><br><a href="../assets/samples/wan2.1/02_bar/baseline.mp4">mp4</a></td>
<td><video src="../assets/samples/wan2.1/02_bar/seacache.mp4" controls muted loop playsinline width="280"></video><br><a href="../assets/samples/wan2.1/02_bar/seacache.mp4">mp4</a></td>
<td><video src="../assets/samples/wan2.1/02_bar/tafc.mp4" controls muted loop playsinline width="280"></video><br><a href="../assets/samples/wan2.1/02_bar/tafc.mp4">mp4</a></td>
</tr>
</table>


| Method (T2V 1.3B, 480p) | Latency ↓ | TFLOPs ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | VBench ↑ |
|:--|--:|--:|--:|--:|--:|--:|
| Original (50 steps) | 316.7s | 31949 | - | - | - | 81.49% |
| TAFC (δ = 0.03) | 154.2s | 15556 | 25.56 | 0.862 | 0.092 | 81.53% |
| TAFC (δ = 0.15) | 103.8s | 10471 | 21.17 | 0.779 | 0.176 | 81.18% |
