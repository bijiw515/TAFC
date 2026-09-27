# TAFC for HunyuanVideo

Target-Anchored Flow Caching with closed-loop error control. Same method as the
FLUX and Wan2.1 arms — `util_tafc.py` is byte-identical across all three, so
cross-model comparisons measure the model, not the implementation.

## Installation

1. Clone [HunyuanVideo](https://github.com/Tencent/HunyuanVideo) and install its
   dependencies and checkpoints as described there.
2. Copy `tafc_generate.py` and `util_tafc.py` into the HunyuanVideo repo root.

## Usage

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
    --tafc_thresh 0.3
```

## How it works

The cached quantity is the **network output residual** (`img - ori_img`), i.e.
the right-hand side of the flow-matching ODE, not a modulated input. A skipped
step replaces it with a first-order extrapolation over the physical time
actually coasted since the last compute — read from the scheduler, so a shifted
(non-uniform) schedule is handled correctly.

Four gates can veto a cache, and the summary reports which one fired:

| veto | meaning |
|---|---|
| `no_model` | no velocity yet, nothing to extrapolate with |
| `lte` | local truncation error over budget |
| `curvature` | trajectory bending too hard |
| `cap` | consecutive-cache safety net |

On top of the open-loop schedule sits a closed loop, borrowed from adaptive
step-size ODE solvers:

- **Curvature derivative** — a feedforward brake on *rising* curvature. Entering
  a turbulent region tightens the budget instead of relaxing it per the preset
  ramp.
- **Local truncation error** — first-order extrapolation scored against
  zero-order hold gives an a-priori error estimate `E_current` before the
  decision, and against the true residual an a-posteriori one after it.
- **PID law** — `τ_{n+1} = τ_n · (E_target / E_current)^p` in log space, with
  anti-windup and gain clamping. High cache rates in laminar flow, automatic
  braking in turbulence.

`E_target` auto-calibrates from the measured one-step error baseline during the
first few forced computes (`--tafc_calib_steps`), since the intrinsic error
scale depends on prompt and seed as much as on the schedule. The a-priori and
a-posteriori estimates live on different magnitude scales, so each keeps its own
baseline (`pid_baseline_pre` / `pid_baseline_post`). All closed-loop state is
per-trajectory: each video recalibrates.

HunyuanVideo calls the transformer once per denoising step (guidance is
embedded), so a single cached residual stream suffices. Wan2.1 needs one
independent loop per CFG pass; that is the only structural difference between
the arms.

## Arguments

Open loop:

| flag | default | meaning |
|---|---|---|
| `--tafc_thresh` | 0.3 | base curvature threshold; lower = stricter |
| `--tafc_max_cache` | 6.0 | consecutive-cache budget at the last step (ramps from 1) |
| `--use_ret_steps` | off | force-compute the first 5 steps |
| `--log_interval` | 10 | print diagnostics every N steps (0 disables) |

Closed loop:

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

To A/B the closed loop against the plain schedule, run the same prompt and seed
twice, adding `--tafc_no_pid` to one. Compare `cache_rate_pct` alongside a
quality metric — a higher cache rate is only a win if quality holds.
