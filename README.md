# Accelerating Diffusion Models with Physical Velocity Evolution

<h5 align="center">

[![arXiv](https://img.shields.io/badge/arXiv-Coming%20Soon-b31b1b.svg?logo=arXiv)](https://arxiv.org/)
[![Project Page](https://img.shields.io/badge/Project-Website-blue.svg)](https://github.com/bijiw515/TAFC)

</h5>

## TAFC

**Trajectory-Aligned Flow Caching (TAFC)** is a training-free acceleration
framework for diffusion and flow-based generative models.

Existing caching methods typically rely on feature-level differences or other
representation-space signals to decide when computation can be reused.
However, these signals do not directly characterize how cache approximation
errors affect the generative trajectory over **physical sampling time**.

TAFC instead aligns cache reuse with the physical-time evolution of the
generative process through:

- **Physical-Time Velocity Extrapolation** — predicts local residual evolution
  over the scheduler's actual physical sampling intervals.
- **Curvature-Aware Reuse Gating** — uses magnitude and directional variation
  to determine whether extrapolation remains reliable.
- **Self-Calibrating Error Feedback** — measures forecast error at refresh
  steps and automatically adjusts subsequent reuse budgets.

TAFC requires **no additional training or fine-tuning**.


## Method Overview

<p align="center">
  <img width="2306" height="1140" alt="480d71705ddacc4b706113925bd20ae0" src="https://github.com/user-attachments/assets/258d1fe0-69ec-4b38-bd13-b68074225cc8" />

</p>

<p align="center">
  <b>Figure 1:</b> Overview of TAFC. A priori extrapolation risk and residual
  curvature determine whether to forecast or refresh. At refresh steps,
  observed forecast errors drive closed-loop feedback to recalibrate future
  reuse budgets.
</p>

TAFC maintains two model-evaluated residual anchors and uses their evolution
over physical sampling time to forecast subsequent cached steps. A refresh is
triggered when the extrapolation risk or curvature exceeds its adaptive budget.
The newly evaluated residual then provides an a posteriori error signal for
closed-loop calibration.

The current formulation uses first-order residual forecasting rather than the
zero-order residual reuse adopted by conventional caching methods.


## Results

### FLUX.1-dev

| Method | Latency ↓ | PSNR ↑ | LPIPS ↓ | HPSv3 ↑ |
|---|---:|---:|---:|---:|
| Baseline (50 steps) | 20.90s | - | - | 10.6060 |
| SeaCache (δ=0.3) | 9.82s | 26.29 | 0.106 | 10.5126 |
| **TAFC (δ=0.04)** | **9.37s** | **29.87** | **0.051** | **10.6623** |
| SeaCache (δ=0.6) | 6.43s | 21.33 | 0.226 | 10.3573 |
| **TAFC (δ=0.15)** | **5.94s** | **21.35** | **0.196** | **10.8605** |
| SeaCache (δ=0.8) | 5.07s | 18.49 | 0.266 | 10.1945 |
| **TAFC (δ=0.30)** | **4.61s** | **19.41** | **0.240** | **10.9129** |

### Video Generation

| Model | Setting | Latency ↓ | PSNR ↑ | VBench ↑ |
|---|---|---:|---:|---:|
| HunyuanVideo | Original | 160.4s | - | 81.66% |
| HunyuanVideo | **TAFC (δ=0.02)** | **78.8s** | **29.24** | **81.70%** |
| HunyuanVideo | **TAFC (δ=0.15)** | **52.2s** | **24.69** | **81.38%** |
| Wan2.1-1.3B | Original | 316.7s | - | 81.49% |
| Wan2.1-1.3B | **TAFC (δ=0.03)** | **154.2s** | **25.56** | **81.53%** |
| Wan2.1-1.3B | **TAFC (δ=0.15)** | **103.8s** | **21.17** | **81.18%** |

TAFC achieves favorable efficiency–fidelity trade-offs across image and video
generation, including approximately 2×–4× acceleration on FLUX.1-dev.:chatgpt-content-reference{index="2"} :chatgpt-content-reference{index="3"}


## Supported Models

### Text-to-Image
- [FLUX](./FLUX/README_TAFC.md)
  - FLUX.1-dev
  - FLUX.1-schnell

### Text-to-Video
- [Wan2.1](./Wan2.1/README_TAFC.md)
  - Wan2.1-T2V-1.3B
  - Wan2.1-T2V-14B
- [HunyuanVideo](./HunyuanVideo/README_TAFC.md)

### Image-to-Video
- [Wan2.1](./Wan2.1/README_TAFC.md)
  - Wan2.1-I2V-14B


## Quick Start

### FLUX

```bash
cd FLUX
pip install torch diffusers transformers accelerate

python tafc_generate.py \
    --prompt "a photo of an astronaut riding a horse" \
    --output_dir ./outputs \
    --num_inference_steps 50 \
    --tafc_thresh 0.2
```

### Wan2.1

```bash
cd Wan2.1
pip install -r requirements.txt

python tafc_generate.py \
    --prompt "A cat walks on the grass" \
    --save_path ./results \
    --tafc_thresh 0.3
```

### HunyuanVideo

```bash
cd HunyuanVideo/HunyuanVideo

python tafc_generate.py \
    --video-size 720 1280 \
    --video-length 33 \
    --infer-steps 50 \
    --prompt "A cat walks on the grass, realistic style." \
    --tafc_thresh 0.3
```

See the model-specific READMEs for installation details, parameters, and
evaluation scripts.


## Evaluation

We evaluate TAFC on:

- **FLUX.1-dev**: DrawBench, HPSv3, CLIP Score, ImageReward, PSNR, SSIM, LPIPS
- **HunyuanVideo**: VBench, PSNR, SSIM, LPIPS
- **Wan2.1**: VBench, PSNR, SSIM, LPIPS

All reported experiments use a 50-step unaccelerated trajectory as the
reference.:chatgpt-content-reference{index="4"}


## News

- **[2026-09]** Released TAFC for FLUX, HunyuanVideo, and Wan2.1.
- **[Coming Soon]** arXiv preprint.


## Citation

If you find this work useful, please cite:

```bibtex
@article{tafc2026,
  title   = {Accelerating Diffusion Models with Physical Velocity Evolution},
  author  = {Anonymous Authors},
  journal = {arXiv preprint},
  year    = {2026}
}
```


## Acknowledgement

This repository builds upon
[SeaCache](https://github.com/jiwoogit/SeaCache),
[FLUX](https://github.com/black-forest-labs/flux),
[Wan2.1](https://github.com/Wan-Video/Wan2.1),
[HunyuanVideo](https://github.com/Tencent/HunyuanVideo), and
[Diffusers](https://github.com/huggingface/diffusers).


## License

Apache 2.0. See [LICENSE](LICENSE) for details.
