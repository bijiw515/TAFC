# TAFC: Target-Anchored Flow Caching for Accelerating Diffusion Models

<h5 align="center">

[![arXiv](https://img.shields.io/badge/arXiv-Coming%20Soon-b31b1b.svg?logo=arXiv)](https://arxiv.org/)
[![Home Page](https://img.shields.io/badge/Project-Website-blue.svg)](https://github.com/bijiw515/TAFC)

</h5>

## Method Overview

TAFC (Target-Anchored Flow Caching) is a training-free acceleration technique for Flow Matching diffusion models that achieves 2-3× speedup through trajectory curvature measurement and first-order extrapolation with closed-loop error control.

#### Key Innovations:

- **Trajectory Curvature Measurement**: Combines magnitude change (tangential acceleration) and direction change (normal acceleration) to detect when the flow trajectory is bending sharply
- **Physical Time Awareness**: Uses the scheduler's actual physical timesteps rather than integer step counts, crucial for non-uniform schedulers (e.g., shift=5.0 in FLUX)
- **First-Order Extrapolation**: Goes beyond simple cache reuse (zero-order hold) to extrapolate with velocity trends: `x_new = x_cached + velocity × Δt`
- **Adaptive Threshold**: Strict in early stages (0.7× base) for correct structure, relaxed in late stages (2.0× base) for faster detail refinement
- **Closed-Loop Control**: PID-based error control automatically adjusts caching aggressiveness based on measured local truncation error

<p align="center">
  <img src="assets/tafc_mechanism_comparison.png" width="800" />
</p>
<p align="center">
  <b>Figure 1:</b> TAFC vs SeaCache mechanism comparison. TAFC uses trajectory curvature and first-order extrapolation for more accurate drift estimation.
</p>

## Performance

Benchmark results on NVIDIA A100 (80GB) for 1024×1024 images, 50 steps:

| Method | Time | Speedup | Quality Loss |
|--------|------|---------|--------------|
| **Baseline** | 45.2s | 1.0× | - |
| **SeaCache (0.3)** | 23.1s | 2.0× | Slight |
| **TAFC (0.15)** | 22.8s | 2.0× | Minimal |
| **TAFC (0.2)** | 18.4s | 2.5× | Slight |
| **TAFC (0.3)** | 15.1s | 3.0× | Moderate |

**Key Observation**: TAFC 0.15 achieves similar speedup to SeaCache 0.3 but with better quality preservation due to first-order extrapolation reducing cumulative error.

## Latest News

- [2026-09-15] Released TAFC for FLUX, HunyuanVideo, and Wan2.1 models
- [2026-09-15] Published code, evaluation scripts, and documentation

## Supported Models

#### Text-to-Image
- [FLUX](./FLUX/README_TAFC.md) - FLUX.1-dev and FLUX.1-schnell

#### Text-to-Video
- [Wan2.1](./Wan2.1/README_TAFC.md) - Wan2.1-T2V-1.3B and T2V-14B
- [HunyuanVideo](./HunyuanVideo/README_TAFC.md) - HunyuanVideo text-to-video

#### Image-to-Video
- [Wan2.1](./Wan2.1/README_TAFC.md) - Wan2.1-I2V-14B

## Quick Start

### FLUX (Text-to-Image)

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

### Wan2.1 (Text-to-Video)

```bash
cd Wan2.1

# Install dependencies
pip install -r requirements.txt

# Generate video with TAFC
python tafc_generate.py \
    --prompt "A cat walks on the grass" \
    --save_path ./results \
    --video_length 81 \
    --tafc_thresh 0.3
```

### HunyuanVideo (Text-to-Video)

```bash
cd HunyuanVideo/HunyuanVideo

# Copy TAFC files to HunyuanVideo repo
# See HunyuanVideo/README_TAFC.md for details

python tafc_generate.py \
    --video-size 720 1280 \
    --video-length 33 \
    --infer-steps 50 \
    --prompt "A cat walks on the grass, realistic style." \
    --tafc_thresh 0.3
```

## How TAFC Works

### 1. Trajectory Curvature Detection

```python
# Pseudocode
v_prev = residual_at_step_t-1
v_curr = residual_at_step_t

# Magnitude change (tangential acceleration)
mag_change = |norm(v_curr) - norm(v_prev)| / norm(v_prev)

# Direction change (normal acceleration via cosine similarity)
cos_sim = dot(v_curr, v_prev) / (norm(v_curr) * norm(v_prev))
angle_penalty = 1 - cos_sim

# Combined curvature
curvature = mag_change + 2.0 * angle_penalty
```

### 2. Adaptive Decision

```python
# Normalized time [0, 1]
norm_t = current_step / total_steps

# Adaptive threshold: strict early, relaxed late
time_scale = 0.7 + 1.3 * norm_t  # 0.7 → 2.0
curvature_thresh = base_thresh * time_scale

# Decision logic
if curvature >= curvature_thresh:
    compute_fresh()  # Curvature too high, need fresh computation
elif consecutive_cached >= max_consec_cache:
    compute_fresh()  # Safety refresh after too many cached steps
else:
    use_cache_with_extrapolation()  # Safe to cache
```

### 3. First-Order Extrapolation

```python
# Compute residual velocity
velocity = (residual_new - residual_old) / physical_dt

# Extrapolate to next step
residual_extrapolated = residual_old + velocity * physical_dt_next

# Apply extrapolated residual
hidden_states = hidden_states + residual_extrapolated
```

### 4. Closed-Loop Error Control (Optional)

TAFC includes a PID-based closed-loop controller that automatically adjusts the caching threshold based on measured local truncation error:

- **Error Estimation**: Compares first-order extrapolation against zero-order hold (a-priori) and against the true residual (a-posteriori)
- **Auto-Calibration**: Learns the baseline error scale during the first few forced compute steps
- **PID Adjustment**: `threshold_{n+1} = threshold_n × (E_target / E_current)^p` with anti-windup
- **Feedforward Brake**: Tightens budget when curvature derivative is rising (entering turbulent regions)

Enable with default settings (recommended):
```bash
python tafc_generate.py --tafc_thresh 0.2  # PID enabled by default
```

Disable for A/B comparison:
```bash
python tafc_generate.py --tafc_thresh 0.2 --tafc_no_pid
```

## Evaluation

Each model directory includes evaluation scripts:

- **FLUX**: HPSv3, DrawBench, and CycleReward metrics
- **Wan2.1**: PyIQA quality metrics for video generation
- **HunyuanVideo**: VBench evaluation suite

See individual README files for detailed evaluation instructions.

## Comparison with Other Methods

| Method | Decision Metric | Extrapolation | Time-Aware | Closed-Loop |
|--------|----------------|---------------|------------|-------------|
| **DenoiseCache** | Relative L1 distance | Zero-order hold | ✗ | ✗ |
| **SeaCache** | SEA filter + L1 | Zero-order hold | ✗ | ✗ |
| **TAFC** | Trajectory curvature | First-order | ✓ | ✓ |

TAFC advantages:
- More accurate drift estimation (physical time)
- Better extrapolation (considers trends)
- Smarter decisions (curvature vs simple distance)
- Adaptive control (PID-based error feedback)

## Parameter Tuning Guide

### Recommended Presets by Use Case

| Use Case | tafc_thresh | tafc_max_cache | use_ret_steps | Expected Speedup |
|----------|-------------|----------------|---------------|------------------|
| **Production (publish)** | 0.10-0.15 | 4.0-5.0 | ✓ | 1.8-2.2× |
| **High Quality (art)** | 0.15-0.20 | 5.0-6.0 | ✓ | 2.0-2.5× |
| **Balanced (general)** | 0.20-0.25 | 6.0-7.0 | ✗ | 2.5-2.8× |
| **Fast (drafting)** | 0.25-0.35 | 7.0-9.0 | ✗ | 2.8-3.2× |
| **Rapid (preview)** | 0.35-0.50 | 9.0-12.0 | ✗ | 3.0-3.5× |

### Core Parameters

- `--tafc_thresh`: Base curvature threshold (lower = stricter, higher quality)
- `--tafc_max_cache`: Maximum consecutive cache steps at the end (early stages ramp from 1)
- `--use_ret_steps`: Force computation for the first 5 steps (recommended for high quality)
- `--tafc_no_pid`: Disable closed-loop control (for A/B testing)

See individual model READMEs for detailed parameter descriptions and tuning workflows.

## Citation

If you use TAFC in your research, please cite:

```bibtex
@article{tafc2026,
  title={TAFC: Target-Anchored Flow Caching for Accelerating Diffusion Models},
  author={Your Name},
  journal={arXiv preprint},
  year={2026}
}
```

## Acknowledgement

This repository builds upon excellent prior work:
- [SeaCache](https://github.com/jiwoogit/SeaCache) - Spectral-evolution-aware caching framework
- [FLUX](https://github.com/black-forest-labs/flux) - Flow matching text-to-image model
- [Wan2.1](https://github.com/Wan-Video/Wan2.1) - Large-scale video generation models
- [HunyuanVideo](https://github.com/Tencent/HunyuanVideo) - Text-to-video generation
- [Diffusers](https://github.com/huggingface/diffusers) - Hugging Face diffusion models library

We thank the authors and contributors for their open research and code.

## License

This project is licensed under the Apache 2.0 License. See [LICENSE](LICENSE) for details.

## Contact

For questions, issues, or discussions, please open an issue on GitHub or contact the authors.
