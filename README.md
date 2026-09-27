# Accelerating Diffusion Models with Physical Velocity Evolution

<h5 align="center">

[![arXiv](https://img.shields.io/badge/arXiv-Coming%20Soon-b31b1b.svg?logo=arXiv)](https://arxiv.org/)
[![Project Page](https://img.shields.io/badge/Project-Website-blue.svg)](https://github.com/bijiw515/TAFC)

</h5>


## Trajectory-Aligned Flow Caching

**Trajectory-Aligned Flow Caching (TAFC)** is a training-free acceleration
framework for diffusion and flow-based generative models.

Existing caching methods typically determine reuse from feature-level
differences or other representation-space surrogate signals. However, these
signals do not directly characterize how approximation errors perturb the
generative trajectory over **physical sampling time**.

TAFC instead grounds cache reuse in the **physical-time evolution of the
generative trajectory**. It models the reusable Transformer residual over
physical sampling time, predicts its local evolution from model-evaluated
anchors, evaluates whether the prediction remains reliable, and uses errors
observed at refresh steps to recalibrate subsequent reuse decisions.

TAFC consists of three complementary components:

- **Physical-Time Velocity Extrapolation** — predicts local residual evolution
  using the scheduler's actual physical sampling intervals rather than integer
  inference-step indices.
- **Curvature-Aware Reuse Gating** — evaluates local magnitude and directional
  variation to determine whether the observed dynamics still support reliable
  extrapolation.
- **Self-Calibrating Error Feedback** — measures a posteriori approximation
  errors at refresh steps and adapts subsequent reuse budgets through
  closed-loop feedback.

Together, TAFC turns caching into a **trajectory-aligned, predictive, and
self-correcting process**.


## Motivation

<p align="center">
  <img src="assets/tafc_mechanism_comparison.png" width="850" />
</p>

<p align="center">
  <b>Figure 1: Motivation and conceptual overview.</b>
  Existing feature-based caching relies on representation-level surrogate
  signals that may not reflect the physical trajectory perturbation introduced
  by reuse. TAFC instead connects cache decisions to physical-time dynamics
  and uses approximation errors observed at refresh steps to calibrate
  subsequent reuse.
</p>


## Method Overview

<p align="center">
  <img src="assets/tafc_method.png" width="1000" />
</p>

<p align="center">
  <b>Figure 2: Practical TAFC under the trajectory-aligned formulation.</b>
  Two evaluated residuals define a first-order forecast, while extrapolation
  risk and residual curvature are compared with adaptive budgets to determine
  forecast or refresh. At refresh steps, the observed approximation error
  drives PID feedback, and the resulting gain recalibrates both budgets for
  subsequent decisions.
</p>


## Highlights

- **Training-free** — no additional training or fine-tuning.
- **Physical-time aware** — explicitly handles non-uniform sampling intervals.
- **First-order residual forecasting** — replaces conventional zero-order
  residual reuse with a local temporal model.
- **Reliability-aware reuse** — jointly considers extrapolation risk and
  residual curvature.
- **Closed-loop calibration** — refresh-time errors continuously recalibrate
  reuse aggressiveness.
- **Image and video generation** — evaluated on FLUX.1-dev, HunyuanVideo, and
  Wan2.1-T2V-1.3B.


## Performance

### FLUX.1-dev

Evaluation at 1024×1024 resolution using a 50-step unaccelerated reference.

| Method | Latency ↓ | TFLOPs ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | HPSv3 ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Baseline (50 steps) | 20.90s | 1668 | - | - | - | 10.6060 |
| TeaCache (δ=0.3) | 10.34s | 862 | 20.76 | 0.810 | 0.211 | 10.5814 |
| TaylorSeer (S=3) | 9.96s | 748 | 22.78 | 0.828 | 0.163 | 10.2312 |
| SeaCache (δ=0.3) | 9.82s | 782 | 26.29 | 0.893 | 0.106 | 10.5126 |
| **TAFC (δ=0.04)** | **9.37s** | **703** | **29.87** | **0.941** | **0.051** | **10.6623** |
| SeaCache (δ=0.6) | 6.43s | 518 | 21.33 | 0.798 | 0.226 | 10.3573 |
| **TAFC (δ=0.15)** | **5.94s** | **451** | **21.35** | **0.820** | **0.196** | **10.8605** |
| SeaCache (δ=0.8) | 5.07s | 391 | 18.49 | 0.788 | 0.266 | 10.1945 |
| **TAFC (δ=0.30)** | **4.61s** | **327** | **19.41** | **0.792** | **0.240** | **10.9129** |

TAFC provides favorable efficiency–fidelity trade-offs across approximately
**2×, 3×, and 4× acceleration regimes**.


### HunyuanVideo

| Method | Latency ↓ | TFLOPs ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | VBench ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Original (50 steps) | 160.4s | 44457 | - | - | - | 81.66% |
| TeaCache (δ=0.12) | 82.9s | 22694 | 23.27 | 0.808 | 0.201 | 81.18% |
| TaylorSeer (S=2) | 81.7s | 22163 | 24.08 | 0.817 | 0.155 | 80.43% |
| SeaCache (δ=0.19) | 79.6s | 21374 | 27.60 | 0.863 | 0.136 | 81.64% |
| **TAFC (δ=0.02)** | **78.8s** | **20791** | **29.24** | **0.895** | **0.103** | **81.70%** |
| SeaCache (δ=0.35) | 53.1s | 14623 | 20.07 | 0.750 | 0.291 | 80.77% |
| **TAFC (δ=0.15)** | **52.2s** | **13981** | **24.69** | **0.836** | **0.167** | **81.38%** |


### Wan2.1-T2V-1.3B

| Method | Latency ↓ | TFLOPs ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | VBench ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Original (50 steps) | 316.7s | 31949 | - | - | - | 81.49% |
| TeaCache (δ=0.09) | 156.2s | 15758 | 20.72 | 0.724 | 0.258 | 81.05% |
| TaylorSeer (S=2) | 155.4s | 15677 | 16.21 | 0.547 | 0.332 | 80.45% |
| SeaCache (δ=0.20) | 154.9s | 15626 | 22.67 | 0.808 | 0.217 | 81.42% |
| **TAFC (δ=0.03)** | **154.2s** | **15556** | **25.56** | **0.862** | **0.092** | **81.53%** |
| SeaCache (δ=0.35) | 104.4s | 10532 | 17.52 | 0.728 | 0.242 | 80.61% |
| **TAFC (δ=0.15)** | **103.8s** | **10471** | **21.17** | **0.779** | **0.176** | **81.18%** |


## How TAFC Works

### 1. Residual Caching Formulation

Let `h(t)` denote the hidden representation at physical sampling time `t`, and
let `Fθ` denote the mapping implemented by the cached Transformer layers.

We define the reusable Transformer residual as

```text
r(t) = Fθ(h(t), t) - h(t)
```

so that a full evaluation gives

```text
Fθ(h(t), t) = h(t) + r(t)
```

Conventional caching simply reuses the most recently evaluated residual:

```text
r_hat(t) = r(t_b)
```

where `t_b` denotes the latest refresh time.

This implicitly assumes that the residual remains locally constant over
physical sampling time.


### 2. Physical-Time First-Order Forecasting

TAFC instead models how the reusable residual evolves over physical sampling
time.

Let `(t_a, r(t_a))` and `(t_b, r(t_b))` denote the two most recent
model-evaluated residual anchors.

The local temporal variation is estimated as

```text
d_b =
[r(t_b) - r(t_a)] /
|t_b - t_a|
```

and the residual at a subsequent sampling time `t` is forecast as

```text
r_hat(t) =
r(t_b) +
d_b |t - t_b|
```

Therefore, conventional caching uses

```text
Zero-order reuse:
r_hat(t) = r(t_b)
```

whereas TAFC uses

```text
First-order forecast:
r_hat(t) = r(t_b) + local_trend × physical_time
```

The two anchors are updated only when a fresh model evaluation is performed.
Cached predictions never become new anchors.


### 3. Reliability-Aware Refresh

First-order extrapolation is only locally reliable. TAFC therefore determines
whether the current forecast remains trustworthy before allowing another
cached step.

#### A Priori Extrapolation Risk

TAFC first measures the relative extrapolated correction

```text
E_hat_k =
||r_hat(t_k) - r(t_b)|| /
(||r(t_b)|| + eps)
```

which measures how far the forecast has moved away from the latest
model-evaluated anchor.

#### Curvature Proxy

TAFC additionally measures local magnitude and directional variation between
the two evaluated anchors:

```text
kappa_k =
| ||r(t_b)|| - ||r(t_a)|| | /
(||r(t_a)|| + eps)

+ 2 * (
    1 -
    <r(t_b), r(t_a)> /
    max(||r(t_b)|| ||r(t_a)||, eps)
)
```

The first term captures changes in residual magnitude, while the second captures
changes in direction.

The curvature proxy is used as a **local reliability signal** rather than as
the quantity directly optimized by TAFC.


### 4. Adaptive Reuse Budgets

The extrapolation-risk and curvature signals are compared with adaptive budgets:

```text
B_e(t_k) =
e_base · G · beta_k
```

```text
B_kappa(t_k) =
kappa_base · tau(t_k) · G · beta_k
```

where:

- `e_base` is the base extrapolation-risk budget,
- `kappa_base` is the base curvature budget,
- `tau(t_k)` provides time-dependent modulation,
- `beta_k` provides the prescribed scheduling modulation,
- `G` is the feedback-controlled gain.

The practical refresh rule can be summarized as:

```python
if calibration_region:
    refresh()

elif cutoff_region:
    refresh()

elif extrapolation_risk >= risk_budget:
    refresh()

elif curvature >= curvature_budget:
    refresh()

elif consecutive_cached >= max_cache:
    refresh()

else:
    forecast()
```

When both reliability criteria pass, the cached Transformer output is

```text
h_out(t_k) =
h(t_k) + r_hat(t_k)
```

Otherwise, TAFC performs a fresh model evaluation:

```text
h_out(t_k) =
Fθ(h(t_k), t_k)
```


### 5. Refresh-Time Error Feedback

The actual forecast error is unavailable during cached steps because the true
residual is intentionally not evaluated.

Once a refresh occurs, TAFC can measure the a posteriori approximation error:

```text
E_post =
||r(t_k) - r_hat(t_k)|| /
(||r(t_k)|| + eps)
```

Warm-up refreshes establish a target error scale:

```text
E_target =
max(median(E_calib), eps)
```

After calibration, each refresh generates a controller error:

```text
e_q =
log(
    (E_target + eps) /
    (E_post + eps)
)
```

The PID controller then updates the feedback-controlled gain:

```text
S_q =
S_(q-1) + e_q
```

```text
DeltaG_q =
K_P e_q
+ K_I S_q
+ K_D (e_q - e_(q-1))
```

followed by

```text
log G <- log G + DeltaG_q
```

If the observed approximation error is above the calibrated target, the
controller tightens subsequent reuse. If the error is below the target, it
allows more permissive reuse.

The resulting closed loop is

```text
Physical-time forecast
        ↓
Reliability gating
        ↓
Forecast / Refresh
        ↓
Refresh-time error measurement
        ↓
PID gain update
        ↓
Adaptive budget update
        ↓
Next reuse decision
```


## Why Physical Sampling Time Matters

Diffusion and flow-based generative models evolve a latent state over
continuous sampling time.

However, many practical schedulers are **non-uniform**, meaning that adjacent
inference iterations do not correspond to equal physical-time intervals.

Therefore,

```text
representation change ≠ physical trajectory perturbation
```

At a fixed state and sampling time, an approximation error affects the
subsequent latent update through the corresponding physical interval:

```text
trajectory perturbation
∝
approximation error × physical Δt
```

The same approximation error can therefore have different consequences at
different locations along the sampling trajectory.

TAFC explicitly models reuse over the scheduler's physical sampling time
instead of treating inference iterations as uniformly spaced.


## Why First-Order Forecasting?

Conventional caching effectively assumes

```text
r(t) ≈ constant
```

between refreshes.

TAFC instead assumes that over a sufficiently local physical-time interval,

```text
r(t) ≈ local linear evolution
```

Under the local smoothness assumptions in our analysis, zero-order residual
reuse incurs an

```text
O(Δ)
```

local approximation error, whereas the first-order TAFC forecast achieves an

```text
O(Δ²)
```

local error bound.

This provides a theoretical motivation for using two evaluated anchors rather
than repeatedly reusing the latest residual.


## Comparison with Existing Caching Methods

| Method | Reuse Signal | Approximation Strategy | Physical-Time Aware | Error Feedback |
|---|---|---|---|---|
| **TeaCache** | Feature difference | Zero-order reuse | ✗ | ✗ |
| **SeaCache** | Spectrally processed feature difference | Zero-order reuse | ✗ | ✗ |
| **TaylorSeer** | Temporal feature trend | Feature forecasting | ✗ | ✗ |
| **TAFC** | **Extrapolation risk + residual curvature** | **First-order residual forecasting** | **✓** | **✓** |

TAFC separates two fundamental questions:

1. **How should the reusable computation be predicted?**
2. **When is that prediction still reliable enough to reuse?**

The first is addressed by **physical-time first-order forecasting**, while the
second is handled by **reliability-aware gating and self-calibrating feedback**.


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
  - HunyuanVideo

### Image-to-Video

- [Wan2.1](./Wan2.1/README_TAFC.md)
  - Wan2.1-I2V-14B


## Quick Start

### FLUX

```bash
cd FLUX

# Install dependencies
pip install torch diffusers transformers accelerate

# Generate with TAFC
python tafc_generate.py \
    --prompt "a photo of an astronaut riding a horse" \
    --output_dir ./outputs \
    --num_inference_steps 50 \
    --tafc_thresh 0.2
```


### Wan2.1

```bash
cd Wan2.1

# Install dependencies
pip install -r requirements.txt

# Generate with TAFC
python tafc_generate.py \
    --prompt "A cat walks on the grass" \
    --save_path ./results \
    --video_length 81 \
    --tafc_thresh 0.3
```


### HunyuanVideo

```bash
cd HunyuanVideo/HunyuanVideo

# See HunyuanVideo/README_TAFC.md for installation details

python tafc_generate.py \
    --video-size 720 1280 \
    --video-length 33 \
    --infer-steps 50 \
    --prompt "A cat walks on the grass, realistic style." \
    --tafc_thresh 0.3
```


## Recommended Operating Points

The exact operating point is model- and task-dependent.

| Model | TAFC Setting | Approx. Regime |
|---|---:|---:|
| FLUX.1-dev | δ=0.04 | ~2× |
| FLUX.1-dev | δ=0.15 | ~3× |
| FLUX.1-dev | δ=0.30 | ~4× |
| HunyuanVideo | δ=0.02 | ~2× |
| HunyuanVideo | δ=0.15 | ~3× |
| Wan2.1-T2V-1.3B | δ=0.03 | ~2× |
| Wan2.1-T2V-1.3B | δ=0.15 | ~3× |

Acceleration regimes are approximate. Actual wall-clock performance depends on
the model, hardware, scheduler, resolution, and implementation.


## Core Parameters

- `--tafc_thresh`  
  Controls the base reuse tolerance. Lower values result in more refreshes and
  higher reconstruction fidelity, while higher values allow more aggressive
  reuse.

- `--tafc_max_cache`  
  Maximum number of consecutive cached steps before a forced refresh.

- `--use_ret_steps`  
  Forces full evaluations during the initial warm-up / calibration region.

- `--tafc_no_pid`  
  Disables refresh-time closed-loop feedback for ablation experiments.

Example:

```bash
python tafc_generate.py \
    --tafc_thresh 0.2 \
    --tafc_no_pid
```


## Evaluation

### FLUX.1-dev

- DrawBench
- HPSv3
- CLIP Score
- ImageReward
- PSNR / SSIM / LPIPS
- Latency / TFLOPs

### HunyuanVideo

- VBench
- PSNR / SSIM / LPIPS
- Latency / TFLOPs

### Wan2.1

- VBench
- PSNR / SSIM / LPIPS
- Latency / TFLOPs


## Qualitative Results

<p align="center">
  <img src="assets/qualitative_flux.png" width="900" />
</p>

<p align="center">
  <b>FLUX.1-dev:</b>
  qualitative reconstruction comparison under different refresh ratios.
</p>

<p align="center">
  <img src="assets/qualitative_video.png" width="900" />
</p>

<p align="center">
  <b>HunyuanVideo and Wan2.1:</b>
  qualitative reconstruction comparison under aggressive caching.
</p>


## Ablation Study

<p align="center">
  <img src="assets/ablation.png" width="750" />
</p>

First-order forecasting, physical-time modeling, adaptive reliability budgets,
and closed-loop feedback provide complementary improvements. Removing any
individual component degrades the reconstruction fidelity–acceleration
trade-off, particularly under aggressive caching.


## Refresh Pattern Analysis

<p align="center">
  <img src="assets/cache_pattern.png" width="850" />
</p>

TAFC produces sample-dependent refresh patterns instead of relying on a fixed
periodic schedule. Model evaluations are dynamically allocated according to
the local trajectory dynamics and feedback-controlled reliability criteria.


## Latest News

- **[2026-09]** TAFC evaluated on FLUX.1-dev, HunyuanVideo, and
  Wan2.1-T2V-1.3B.
- **[2026-09]** Released inference and evaluation code.
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

Citation information will be updated after the public release.


## Acknowledgement

This repository builds upon excellent prior work:

- [SeaCache](https://github.com/jiwoogit/SeaCache)
- [FLUX](https://github.com/black-forest-labs/flux)
- [Wan2.1](https://github.com/Wan-Video/Wan2.1)
- [HunyuanVideo](https://github.com/Tencent/HunyuanVideo)
- [Diffusers](https://github.com/huggingface/diffusers)

We thank the authors and contributors for their open research and code.


## License

This project is licensed under the Apache 2.0 License.

See [LICENSE](LICENSE) for details.


## Contact

For questions, issues, or discussions, please open an issue on GitHub.
