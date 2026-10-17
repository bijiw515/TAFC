# TAFC for HunyuanVideo
This is the official implementation of TAFC for HunyuanVideo.

## Installation
1. Clone the original [HunyuanVideo](https://github.com/Tencent/HunyuanVideo) repository and install all required dependencies (Python, CUDA, checkpoints, etc.) as described there.
2. Place the provided TAFC scripts (`tafc_generate.py`, `util_tafc.py`) into the HunyuanVideo repository root.
3. For benchmark evaluation, also place `tafc_generate_batch.py`, `run_tafc_eval.py` and `run_tafc_eval_batch.py` in the same directory. Fidelity metrics use `eval_pyiqa.py` from the [Wan2.1](../Wan2.1/README.md) folder.

## Usage
The main code is `tafc_generate.py`. Make sure to run the script inside the HunyuanVideo directory.

### Text-to-Video Inference

```bash
cd HunyuanVideo

python3 tafc_generate.py \
    --video-size 720 1280 \
    --video-length 33 \
    --infer-steps 50 \
    --prompt "A cat walks on the grass, realistic style." \
    --flow-reverse \
    --use-cpu-offload \
    --save-path ./tafc_results \
    --tafc_thresh 0.15
```

Arguments:
- `--video-size`: output resolution `(height width)`
- `--video-length`: number of frames to generate
- `--infer-steps`: number of sampling steps
- `--prompt`: text prompt for video generation
- `--save-path`: directory to store generated videos
- `--tafc_thresh`: base curvature threshold, controls the cache ratio (lower = stricter)
- `--tafc_max_cache`: consecutive-cache budget at the last step (default `6.0`, ramps from 1)
- `--use_ret_steps`: force-compute the first 5 steps
- `--log_interval`: print diagnostics every N steps (`0` disables)

> **Note:** You can sweep over `tafc_thresh` to find the balanced quality-latency trade-offs.
> HunyuanVideo embeds guidance, so the transformer is called once per step and a single cached residual stream suffices.

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

> **Note:** VBench and HunyuanVideo/Wan2.1 require incompatible `transformers` versions, so use two separate conda environments (e.g. `tafc` for generation and fidelity, `vbench` for VBench scoring).

### 1. Generate samples
`run_tafc_eval.py` runs the baseline (no cache) and TAFC configs. Videos are named `{prompt}-0.mp4` to match VBench's expected format.

    # full VBench prompt set, configs run sequentially
    python run_tafc_eval.py \
        --prompt_file ./vbench/VBench_full_info.json \
        --model_base ./ckpts \
        --video_size 480,832 --video_length 65 \
        --infer_steps 50 --flow_reverse \
        --out_dir ./eval_samples \
        --configs baseline tafc_0.02 tafc_0.15

    # one config per GPU, in parallel
    CUDA_VISIBLE_DEVICES=0 python run_tafc_eval.py --configs baseline &
    CUDA_VISIBLE_DEVICES=1 python run_tafc_eval.py --configs tafc_0.02 &
    CUDA_VISIBLE_DEVICES=2 python run_tafc_eval.py --configs tafc_0.15 &

Outputs:

    eval_samples/<config>/{prompt}-0.mp4
    eval_samples/<config>/gen_stats.json    # per-video latency and cache rate

Config names are `baseline` or `tafc_<thresh>`. Existing videos are skipped (`--skip_existing`), so interrupted runs can be resumed.

To avoid reloading the model for every prompt, use the batch driver (one process per config):

    python run_tafc_eval_batch.py \
        --config tafc_0.15 \
        --prompt_file ./vbench/VBench_full_info.json \
        --model_base ./ckpts \
        --video_size 480,832 --video_length 65 \
        --flow-reverse

### 2. Fidelity: PSNR / SSIM / LPIPS
Each generated video is compared frame-by-frame with the same-named baseline video (macro-averaged over videos).

    python eval_pyiqa.py \
        --gt_dir eval_samples/baseline \
        --gen_dir eval_samples/tafc_0.15 \
        --metrics psnr ssim lpips

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
HunyuanVideo, 480×832, 65 frames, 50 steps, seed 42, `embedded_cfg_scale` 6.0, `flow_shift` 7.0, `--use_ret_steps` on, full VBench prompt set. Operating points used in the paper:

| Speedup | `tafc_thresh` | `tafc_max_cache` |
|:--:|:--:|:--:|
| ~2× | 0.02 | 1.0 |
| ~3× | 0.15 | 3.0 |

    # no-cache reference
    python run_tafc_eval.py --configs baseline --flow_reverse \
        --prompt_file ./vbench/VBench_full_info.json --model_base ./ckpts

    # ~2x
    python run_tafc_eval.py --configs tafc_0.02 --tafc_max_cache 1.0 --flow_reverse \
        --prompt_file ./vbench/VBench_full_info.json --model_base ./ckpts
    # ~3x
    python run_tafc_eval.py --configs tafc_0.15 --tafc_max_cache 3.0 --flow_reverse \
        --prompt_file ./vbench/VBench_full_info.json --model_base ./ckpts

Then compute fidelity and VBench as described in Benchmark Evaluation.

> **Note:** Latency depends on the GPU, so absolute times will differ from the paper.

---
## Inference Results
Latency may vary depending on your GPU.

> **Note:** `tafc_thresh = δ`

### Qualitative samples

Left to right: original (50 steps), SeaCache (δ = 0.19), TAFC (δ = 0.02). If the videos do not play inline, use the mp4 links.

*a donut on the top of broccoli, front view*

<table>
<tr><th align='center'>Original</th><th align='center'>SeaCache (δ = 0.19)</th><th align='center'>TAFC (δ = 0.02)</th></tr>
<tr>
<td><video src="../assets/samples/hunyuanvideo/01_a_donut_on_the_top_of_broccoli_front_view/baseline.mp4" controls muted loop playsinline width="280"></video><br><a href="../assets/samples/hunyuanvideo/01_a_donut_on_the_top_of_broccoli_front_view/baseline.mp4">mp4</a></td>
<td><video src="../assets/samples/hunyuanvideo/01_a_donut_on_the_top_of_broccoli_front_view/seacache.mp4" controls muted loop playsinline width="280"></video><br><a href="../assets/samples/hunyuanvideo/01_a_donut_on_the_top_of_broccoli_front_view/seacache.mp4">mp4</a></td>
<td><video src="../assets/samples/hunyuanvideo/01_a_donut_on_the_top_of_broccoli_front_view/tafc.mp4" controls muted loop playsinline width="280"></video><br><a href="../assets/samples/hunyuanvideo/01_a_donut_on_the_top_of_broccoli_front_view/tafc.mp4">mp4</a></td>
</tr>
</table>

*A tranquil tableau of a cup*

<table>
<tr><th align='center'>Original</th><th align='center'>SeaCache (δ = 0.19)</th><th align='center'>TAFC (δ = 0.02)</th></tr>
<tr>
<td><video src="../assets/samples/hunyuanvideo/02_A_tranquil_tableau_of_a_cup/baseline.mp4" controls muted loop playsinline width="280"></video><br><a href="../assets/samples/hunyuanvideo/02_A_tranquil_tableau_of_a_cup/baseline.mp4">mp4</a></td>
<td><video src="../assets/samples/hunyuanvideo/02_A_tranquil_tableau_of_a_cup/seacache.mp4" controls muted loop playsinline width="280"></video><br><a href="../assets/samples/hunyuanvideo/02_A_tranquil_tableau_of_a_cup/seacache.mp4">mp4</a></td>
<td><video src="../assets/samples/hunyuanvideo/02_A_tranquil_tableau_of_a_cup/tafc.mp4" controls muted loop playsinline width="280"></video><br><a href="../assets/samples/hunyuanvideo/02_A_tranquil_tableau_of_a_cup/tafc.mp4">mp4</a></td>
</tr>
</table>


| Method (HunyuanVideo) | Latency ↓ | TFLOPs ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | VBench ↑ |
|:--|--:|--:|--:|--:|--:|--:|
| Original (50 steps) | 160.4s | 44457 | - | - | - | 81.66% |
| TAFC (δ = 0.02) | 78.8s | 20791 | 29.24 | 0.895 | 0.103 | 81.70% |
| TAFC (δ = 0.15) | 52.2s | 13981 | 24.69 | 0.836 | 0.167 | 81.38% |
