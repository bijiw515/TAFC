"""
Utility functions for TAFC (Target-Anchored Flow Caching)

This module provides core computational functions for TAFC:
- Trajectory curvature computation (magnitude + direction changes)
- Curvature derivative (second-order signal) and its brake
- Local truncation error estimation (a-priori embedded pair + a-posteriori)
- Closed-loop PID threshold control
- TAFCBranch: one cached residual stream + its gate, so models that call the
  transformer twice per step (CFG cond/uncond, e.g. Wan2.1) can keep a fully
  independent closed loop per branch without duplicating the gate logic
- Target drift estimation using physics-based formulas
- Physical timestep extraction from schedulers
- Relative L1 distance computation
"""

import math
from typing import List, Optional, Tuple
import torch


def compute_trajectory_curvature(
    v_prev: torch.Tensor,
    v_curr: torch.Tensor,
    eps: float = 1e-8
) -> float:
    """
    Compute trajectory curvature combining magnitude and direction changes.

    Compares TWO consecutive residuals (velocities) to measure how much
    the flow is "turning" in the high-dimensional space.

    Physics intuition:
    - In Flow Matching, velocity v_t = model_output is the residual pointing toward x_0
    - Magnitude change = tangential acceleration (speed change)
    - Direction change = normal acceleration (turning)
    - Combined curvature predicts how much cached velocity will drift from true trajectory

    Args:
        v_prev: Previous velocity prediction (network output residual)
        v_curr: Current velocity prediction (network output residual)
        eps: Small constant for numerical stability

    Returns:
        Combined curvature metric (scalar). Higher values indicate sharper trajectory turns.
    """
    if v_prev is None or v_curr is None:
        return 0.0

    # Flatten for easier computation
    v_prev_flat = v_prev.reshape(v_prev.shape[0], -1)
    v_curr_flat = v_curr.reshape(v_curr.shape[0], -1)

    # 1. Magnitude change (tangential acceleration)
    norm_prev = torch.norm(v_prev_flat, dim=1, keepdim=True) + eps
    norm_curr = torch.norm(v_curr_flat, dim=1, keepdim=True) + eps
    mag_change = torch.abs(norm_curr - norm_prev) / norm_prev
    mag_change = float(mag_change.mean().detach().cpu())

    # 2. Direction change (normal acceleration via cosine similarity)
    cos_sim = (v_curr_flat * v_prev_flat).sum(dim=1, keepdim=True) / (norm_curr * norm_prev + eps)
    cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
    # angle_penalty: 0 when parallel, 2 when opposite
    angle_penalty = 1.0 - cos_sim
    angle_penalty = float(angle_penalty.mean().detach().cpu())

    # 3. Combine: weight direction changes more heavily
    # Direction changes are more critical than magnitude changes for quality
    alpha = 2.0
    curvature = mag_change + alpha * angle_penalty

    return curvature


def estimate_target_drift(
    curvature: float,
    physical_time_elapsed_squared: float
) -> float:
    """
    Estimate accumulated drift from target using physics-based formula:
    Drift ≈ 0.5 * curvature * (Δt_physical)²

    CRITICAL: Uses physical timestep differences from the ODE schedule,
    not integer skip counts. This accounts for non-uniform timestep spacing,
    especially important with shift=5.0 where early steps are tiny and
    late steps are large.

    Physics intuition:
    - When we cache a velocity and reuse it, we're assuming constant velocity
    - But the true trajectory has acceleration (curvature)
    - Local truncation error for constant velocity approximation is O(curvature * dt²)
    - This is analogous to position error in physics: Δx ≈ 0.5 * a * t²

    Args:
        curvature: Trajectory curvature (magnitude + direction change)
        physical_time_elapsed_squared: Square of PHYSICAL time elapsed (sum of Δt)

    Returns:
        Estimated drift magnitude
    """
    # Local truncation error for constant velocity approximation
    # is proportional to curvature * (physical_dt)²
    drift = 0.5 * curvature * physical_time_elapsed_squared
    return drift


def get_physical_timestep(
    scheduler,
    step_idx: int,
    num_steps: int
) -> float:
    """
    Extract the physical timestep value from a scheduler at a given step index.

    Different schedulers store timesteps differently:
    - FlowMatchEulerDiscreteScheduler: .timesteps attribute
    - UniPC/DPM schedulers: .timesteps attribute
    - Fallback: assume uniform spacing in [0, 1]

    Args:
        scheduler: Diffusers scheduler instance
        step_idx: Current step index (0-based)
        num_steps: Total number of inference steps

    Returns:
        Physical timestep value (typically in range [0, 1] for flow matching)
    """
    if hasattr(scheduler, 'timesteps') and step_idx < len(scheduler.timesteps):
        return float(scheduler.timesteps[step_idx])

    # Fallback: assume uniform spacing from 1.0 to 0.0
    return 1.0 - (step_idx / max(1, num_steps - 1))


def compute_physical_delta_t(
    scheduler,
    prev_step_idx: int,
    curr_step_idx: int,
    num_steps: int
) -> float:
    """
    Compute the physical time difference between two steps.

    This is critical for first-order extrapolation and drift estimation.
    Non-uniform schedulers (especially with shift > 1) have very different
    step sizes at different parts of the trajectory.

    Args:
        scheduler: Diffusers scheduler instance
        prev_step_idx: Previous step index
        curr_step_idx: Current step index
        num_steps: Total number of inference steps

    Returns:
        Physical time difference |t_curr - t_prev|
    """
    t_prev = get_physical_timestep(scheduler, prev_step_idx, num_steps)
    t_curr = get_physical_timestep(scheduler, curr_step_idx, num_steps)
    return abs(t_curr - t_prev)


def rel_l1(
    a: torch.Tensor,
    b: torch.Tensor,
    eps: float = 1e-16
) -> float:
    """
    Compute relative L1 distance between two tensors.

    This is used as an alternative distance metric to curvature,
    particularly useful for comparing features or activations.

    Formula: ||a - b||_1 / (||b||_1 + eps)

    Args:
        a: First tensor
        b: Second tensor (reference)
        eps: Small constant for numerical stability

    Returns:
        Relative L1 distance (scalar)
    """
    num = (a - b).abs().mean()
    den = b.abs().mean() + eps
    return float((num / den).detach().cpu())


def adaptive_threshold_schedule(
    normalized_time: float,
    base_threshold: float,
    early_scale: float = 0.7,
    late_scale: float = 2.0
) -> float:
    """
    Compute adaptive threshold that varies across the denoising trajectory.

    TAFC uses stricter thresholds early (when structure is being established)
    and more relaxed thresholds late (when refinements are being made).

    Args:
        normalized_time: Current position in trajectory [0, 1]
                        0 = start (early denoising)
                        1 = end (late denoising)
        base_threshold: Base threshold value set by user
        early_scale: Multiplier for early steps (default 0.7 = 30% stricter)
        late_scale: Multiplier for late steps (default 2.0 = 2x more relaxed)

    Returns:
        Scaled threshold value
    """
    time_scale = early_scale + (late_scale - early_scale) * normalized_time
    return base_threshold * time_scale


def adaptive_max_cache_schedule(
    normalized_time: float,
    max_cache_late: float
) -> float:
    """
    Compute adaptive maximum consecutive cache budget.

    Even if curvature is high, late stages should get some guaranteed caching
    to achieve the desired speedup. This provides a backup mechanism.

    Args:
        normalized_time: Current position in trajectory [0, 1]
        max_cache_late: Maximum consecutive cache steps at the end

    Returns:
        Maximum consecutive cache steps allowed at current position
    """
    # Linear ramp from 1.0 (start) to max_cache_late (end)
    return 1.0 + max_cache_late * normalized_time


def compute_residual_velocity(
    residual_new: torch.Tensor,
    residual_old: torch.Tensor,
    time_diff: float,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute velocity of residual change for first-order extrapolation.

    First-order extrapolation improves over zero-order hold (simple reuse)
    by accounting for the trend in how the residual is changing.

    Formula: velocity = (residual_new - residual_old) / Δt

    Args:
        residual_new: Current residual (network output - input)
        residual_old: Previous residual
        time_diff: Physical time difference between the two residuals
        eps: Small constant for numerical stability

    Returns:
        Residual velocity tensor (same shape as residuals)
    """
    return (residual_new - residual_old) / (time_diff + eps)


def extrapolate_residual_first_order(
    residual_cached: torch.Tensor,
    residual_velocity: torch.Tensor,
    time_step: float
) -> torch.Tensor:
    """
    Perform first-order extrapolation of a cached residual.

    Instead of just reusing the cached residual (zero-order hold),
    we extrapolate based on the observed velocity trend:

    residual_new ≈ residual_cached + velocity × Δt

    This is more accurate when the trajectory has consistent acceleration.

    Args:
        residual_cached: Previously computed residual
        residual_velocity: Rate of change of residual
        time_step: Physical time step to extrapolate forward

    Returns:
        Extrapolated residual tensor
    """
    return residual_cached + residual_velocity * time_step


def should_force_compute(
    step_idx: int,
    num_steps: int,
    ret_steps: int = 1,
    cutoff_steps: Optional[int] = None
) -> bool:
    """
    Determine if computation should be forced at this step regardless of caching decision.

    Certain steps are always computed:
    1. Retention steps (warmup): First few steps to build initial history
    2. Cutoff steps: Last few steps for final quality
    3. First and last steps: Always computed

    Args:
        step_idx: Current step index (0-based)
        num_steps: Total number of inference steps
        ret_steps: Number of retention (warmup) steps at the start
        cutoff_steps: Step index after which to force computation (None = compute last step only)

    Returns:
        True if should force computation, False if caching decision should be evaluated
    """
    # Always compute first step
    if step_idx < ret_steps:
        return True

    # Always compute last step
    if step_idx >= num_steps - 1:
        return True

    # Cutoff region (final quality assurance)
    if cutoff_steps is not None and step_idx >= cutoff_steps:
        return True

    return False


# ============================================================================
# Closed-loop control: curvature derivative + local truncation error + PID
# ============================================================================

def compute_curvature_rate(
    curvature: float,
    prev_curvature: Optional[float],
    steps_elapsed: int = 1,
    eps: float = 1e-6,
) -> float:
    """
    Relative growth rate of the trajectory curvature (a second-order signal).

    ``compute_trajectory_curvature`` tells us how sharply the flow is turning
    *right now*; its derivative tells us whether we are entering or leaving a
    turbulent region. A sudden curvature spike means the open-loop schedule --
    which blindly relaxes the threshold as ``normalized_time`` grows -- is about
    to cache straight through a violent part of the trajectory.

    We use the *relative* rate rather than d(curv)/dt because the curvature
    metric is dimensionless while the physical timesteps are O(20) in FLUX
    units; a relative rate keeps the signal O(1) and scheduler-agnostic.

    Args:
        curvature: Curvature measured at the current evaluation.
        prev_curvature: Curvature measured at the previous evaluation
                        (``None`` on the first evaluation -> rate 0).
        steps_elapsed: Compute steps between the two curvature samples,
                       so the rate is per-step rather than per-interval.
        eps: Numerical floor for the denominator.

    Returns:
        Relative curvature growth per step. Positive = sharpening turn
        (entering turbulence), negative = straightening out (laminar).
    """
    if prev_curvature is None:
        return 0.0
    denom = max(abs(prev_curvature), eps)
    return (curvature - prev_curvature) / denom / max(1, int(steps_elapsed))


def curvature_brake_factor(
    curvature_rate: float,
    beta: float = 2.0,
    floor: float = 0.3,
) -> float:
    """
    Convert a curvature growth rate into a multiplicative brake on the budget.

    Deliberately ASYMMETRIC: only a *rising* curvature brakes. A falling
    curvature is left to the PID loop, which relaxes the budget on the evidence
    of measured error rather than on the hope that the trajectory stays smooth.
    Rewarding falling curvature here as well would double-count the same
    observation and make the two mechanisms fight each other.

    Args:
        curvature_rate: Output of ``compute_curvature_rate``.
        beta: Brake strength. Higher = harder braking on a curvature spike.
        floor: Lower bound on the factor, so one spike cannot fully stall
               caching for the rest of the trajectory.

    Returns:
        Factor in ``[floor, 1.0]`` to multiply the error/curvature budget by.
    """
    rise = max(0.0, float(curvature_rate))
    factor = 1.0 / (1.0 + beta * rise)
    return max(floor, factor)


def estimate_extrapolation_error(
    residual_cached: torch.Tensor,
    residual_velocity: Optional[torch.Tensor],
    horizon: float,
    eps: float = 1e-8,
) -> Optional[float]:
    """
    A-PRIORI local truncation error estimate via an embedded pair.

    Classic adaptive ODE solvers (Dormand-Prince, RKF45) estimate the local
    error by differencing two solutions of different order at the same step.
    Here the embedded pair is free: the first-order extrapolation and the
    zero-order hold are both already available, and they differ by exactly the
    first-order correction term.

        E = || v * h || / || r ||

    This is the fraction of the residual that we are *asking the extrapolator
    to invent*. It grows with the coasting horizon ``h``, so it is a
    timestep-aware replacement for "how many steps have I cached in a row":
    at large physical steps the same budget buys fewer cached steps.

    Args:
        residual_cached: Last computed residual (the zero-order term).
        residual_velocity: d(residual)/dt from the last two computed steps.
        horizon: Physical time we are about to extrapolate across.
        eps: Numerical floor for the denominator.

    Returns:
        Dimensionless error estimate, or ``None`` if no velocity is available
        (in which case the caller has no first-order model and must compute).
    """
    if residual_velocity is None or residual_cached is None:
        return None
    corr = residual_velocity.abs().mean() * abs(float(horizon))
    base = residual_cached.abs().mean() + eps
    return float((corr / base).detach().cpu())


def measure_extrapolation_error(
    residual_true: torch.Tensor,
    residual_cached: torch.Tensor,
    residual_velocity: Optional[torch.Tensor],
    horizon: float,
    eps: float = 1e-8,
) -> Optional[float]:
    """
    A-POSTERIORI local truncation error: what the extrapolation actually cost.

    On every computed step we know the true residual, so we can score the
    prediction the first-order model *would* have made across the horizon we
    just coasted through:

        E = || r_true - (r_cached + v * h) || / || r_true ||

    This is the honest feedback signal for the controller. Unlike the a-priori
    estimate it measures the part the linear model got *wrong* (the curvature
    of the residual path), not merely how much work it was asked to do.

    Args:
        residual_true: Freshly computed residual at the current step.
        residual_cached: Residual at the last computed step before this one.
        residual_velocity: Velocity used for the extrapolation.
        horizon: Physical time between the two computed steps.
        eps: Numerical floor for the denominator.

    Returns:
        Dimensionless relative error, or ``None`` if velocity is unavailable.
    """
    if residual_velocity is None or residual_cached is None:
        return None
    predicted = residual_cached + residual_velocity * float(horizon)
    err = (residual_true - predicted).abs().mean()
    base = residual_true.abs().mean() + eps
    return float((err / base).detach().cpu())


class TAFCController:
    """
    Closed-loop PID controller for the TAFC caching budget.

    The open-loop TAFC schedule relaxes its threshold as a fixed function of
    ``normalized_time``, regardless of whether the extrapolation is actually
    holding up. This controller closes that loop: it watches the a-posteriori
    error of the extrapolations it authorised and steers a multiplicative gain
    so that the measured error tracks a target.

    The update is the adaptive-step-size law in log space, extended from the
    pure integral action of a classical ODE controller to full PID:

        e_n        = log(E_target / E_measured)
        log g_{n+1} = log g_n + kp*e_n + ki*sum(e) + kd*(e_n - e_{n-1})

    * kp (proportional) is the exponent ``p`` of the tau update law: it reacts
      to the current error ratio.
    * ki (integral) removes the steady-state bias that pure proportional
      control leaves behind, with clamped anti-windup.
    * kd (derivative) damps the oscillation that a delayed, noisy error signal
      would otherwise induce.

    The gain is applied to BOTH gates -- the curvature threshold and the local
    truncation error budget -- so a single control signal governs the caching
    aggressiveness coherently. The target itself is never scaled: it is the
    quality anchor the loop steers toward.

    Both budgets are calibrated against baselines measured on single-step
    forced computes during warmup, which makes every threshold in the loop
    dimensionless and self-scaling across models, resolutions and schedules.
    """

    def __init__(
        self,
        kp: float = 0.4,
        ki: float = 0.05,
        kd: float = 0.2,
        gain_min: float = 0.3,
        gain_max: float = 3.0,
        target: Optional[float] = None,
        auto_tol: float = 2.5,
        reject: float = 3.0,
        brake_beta: float = 2.0,
        brake_floor: float = 0.3,
        integral_clamp: float = 2.0,
        baseline_window: int = 8,
    ):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.gain_min = float(gain_min)
        self.gain_max = float(gain_max)
        self.fixed_target = None if target is None or target <= 0 else float(target)
        self.auto_tol = float(auto_tol)
        self.reject = float(reject)
        self.brake_beta = float(brake_beta)
        self.brake_floor = float(brake_floor)
        self.integral_clamp = float(integral_clamp)
        self.baseline_window = int(baseline_window)
        self.reset()

    def reset(self) -> None:
        """Clear all controller state (called once per trajectory)."""
        self._log_gain = 0.0
        self._integral = 0.0
        self._prev_err = None
        self.baseline_post: List[float] = []   # a-posteriori, 1-step horizon
        self.baseline_pre: List[float] = []    # a-priori, 1-step horizon
        self.updates = 0
        self.last_measured = None
        self.last_err = None

    # -- calibration ------------------------------------------------------

    def observe_baseline(self, err_post: Optional[float], err_pre: Optional[float]) -> None:
        """
        Record the intrinsic one-step error scale of this trajectory.

        Only called on forced single-step computes (warmup), so the baselines
        describe the model's inherent extrapolation error rather than the
        controller's own choices -- otherwise a permissive gain would inflate
        the very budget it is being judged against.
        """
        if err_post is not None and math.isfinite(err_post) and err_post > 0:
            self.baseline_post.append(float(err_post))
            del self.baseline_post[:-self.baseline_window]
        if err_pre is not None and math.isfinite(err_pre) and err_pre > 0:
            self.baseline_pre.append(float(err_pre))
            del self.baseline_pre[:-self.baseline_window]

    @staticmethod
    def _median(values: List[float]) -> Optional[float]:
        if not values:
            return None
        s = sorted(values)
        mid = len(s) // 2
        return s[mid] if len(s) % 2 else 0.5 * (s[mid - 1] + s[mid])

    @property
    def needs_calibration(self) -> bool:
        """True while the loop still lacks the baselines its gates need."""
        if not self.baseline_pre:
            return True
        return self.fixed_target is None and not self.baseline_post

    @property
    def target(self) -> Optional[float]:
        """Error the loop steers toward: user-fixed, or auto-calibrated."""
        if self.fixed_target is not None:
            return self.fixed_target
        base = self._median(self.baseline_post)
        return None if base is None else base * self.auto_tol

    @property
    def gain(self) -> float:
        """Current multiplicative gain on the caching budget."""
        return math.exp(self._log_gain)

    def brake(self, curvature_rate: float) -> float:
        """Instantaneous feedforward brake for a curvature spike."""
        return curvature_brake_factor(curvature_rate, self.brake_beta, self.brake_floor)

    # -- gates ------------------------------------------------------------

    def error_budget(self, curvature_rate: float = 0.0) -> Optional[float]:
        """
        Allowed a-priori extrapolation error, in units of the one-step baseline.

        ``reject`` steps of one-step-equivalent first-order correction is the
        nominal coasting budget; the PID gain stretches or shrinks it on
        measured evidence, and the brake cuts it when curvature spikes.
        """
        base = self._median(self.baseline_pre)
        if base is None:
            return None
        return base * self.reject * self.gain * self.brake(curvature_rate)

    def curvature_budget(self, scheduled_thresh: float, curvature_rate: float = 0.0) -> float:
        """Open-loop scheduled threshold, corrected by the closed loop."""
        return scheduled_thresh * self.gain * self.brake(curvature_rate)

    # -- update -----------------------------------------------------------

    def update(self, err_measured: Optional[float], eps: float = 1e-12) -> None:
        """
        Advance the PID loop on one a-posteriori error measurement.

        Args:
            err_measured: Relative error of the extrapolation we authorised.
                          ``None`` (no measurement available) is a no-op, so
                          the gain simply holds.
        """
        target = self.target
        if err_measured is None or target is None:
            return
        if not math.isfinite(err_measured):
            return

        self.last_measured = float(err_measured)
        err = math.log(target / max(float(err_measured), eps))
        self.last_err = err

        derivative = 0.0 if self._prev_err is None else (err - self._prev_err)
        integral = self._integral + err

        step = self.kp * err + self.ki * integral + self.kd * derivative
        log_gain = self._log_gain + step
        clamped = min(max(log_gain, math.log(self.gain_min)), math.log(self.gain_max))

        # Anti-windup: stop accumulating once the actuator is saturated.
        if clamped == log_gain:
            self._integral = min(max(integral, -self.integral_clamp), self.integral_clamp)

        self._log_gain = clamped
        self._prev_err = err
        self.updates += 1

    def diagnostics(self) -> dict:
        """Controller state for stats files and A/B analysis."""
        target = self.target
        return {
            "pid_gain": round(self.gain, 4),
            "pid_updates": int(self.updates),
            "pid_target": None if target is None else round(target, 6),
            "pid_baseline_post": None if not self.baseline_post else round(
                self._median(self.baseline_post), 6),
            "pid_baseline_pre": None if not self.baseline_pre else round(
                self._median(self.baseline_pre), 6),
            "pid_last_measured": None if self.last_measured is None else round(
                self.last_measured, 6),
        }


class TAFCBranch:
    """
    One independently-cached residual stream, with its own closed loop.

    FLUX evaluates the transformer once per denoising step, so a single set of
    residual history suffices. Wan2.1 and other CFG-in-two-passes models call the
    model TWICE per step -- once conditional, once unconditional -- and the two
    passes have genuinely different residual trajectories. Sharing one cache
    between them would compare a conditional residual against an unconditional
    one and measure a curvature that belongs to neither.

    So each branch gets its own instance: its own residual history, its own
    velocity, its own consecutive-cache counter, and its own PID controller.
    The gate logic itself is identical to the single-stream case, which is why it
    lives here instead of being written out twice in the forward pass.
    """

    def __init__(self, name: str, controller: Optional[TAFCController] = None):
        self.name = str(name)
        self.ctrl = controller
        self.reset()

    def reset(self) -> None:
        """Clear the branch's trajectory state (called once per video)."""
        self.previous_residual = None
        self.prev_prev_residual = None
        self.residual_velocity = None
        self.previous_t = None
        self.cache_step_count = 0
        self.cache_skip_count = 0
        self.last_compute_step = -1
        self.curv_ref_step = -1
        self.prev_curvature = None
        self.curvature_rate = 0.0
        self.forced_this_step = False
        self.vetoes = {}
        # Last-evaluated diagnostics, for logging.
        self.last_curvature = None
        self.last_thresh = None
        self.last_e_current = None
        self.last_e_budget = None
        self.last_max_consec = None
        if self.ctrl is not None:
            self.ctrl.reset()

    @property
    def has_history(self) -> bool:
        """True once there is a residual to extrapolate from."""
        return self.previous_residual is not None and self.previous_t is not None

    @property
    def needs_calibration(self) -> bool:
        return self.ctrl is not None and self.ctrl.needs_calibration

    def horizon(self, current_t: float) -> float:
        """
        Physical time we would extrapolate across if we cached now.

        ``previous_t`` only advances on COMPUTED steps, so during a cache run
        this keeps growing: it is the coasting distance, not one step.
        """
        if self.previous_t is None:
            return 0.0
        return abs(self.previous_t - float(current_t))

    # -- the gate ---------------------------------------------------------

    def should_compute(
        self,
        step_idx: int,
        normalized_time: float,
        base_thresh: float,
        max_cache_late: float,
        current_t: float,
        early_scale: float = 0.7,
        late_scale: float = 2.0,
    ) -> Tuple[bool, Optional[str]]:
        """
        Decide whether this branch must recompute, and why.

        Returns ``(should_calc, veto)`` where ``veto`` names the gate that
        forced the compute (``None`` when the step is cached).
        """
        curvature = compute_trajectory_curvature(
            self.prev_prev_residual, self.previous_residual,
        ) if self.prev_prev_residual is not None else 0.01

        # Second-order signal: curvature only changes when a fresh residual
        # arrives, so recompute the rate once per compute step and hold it
        # across a cache run.
        if self.last_compute_step != self.curv_ref_step:
            self.curvature_rate = compute_curvature_rate(
                curvature, self.prev_curvature,
                steps_elapsed=self.last_compute_step - self.curv_ref_step,
            )
            self.prev_curvature = curvature
            self.curv_ref_step = self.last_compute_step
        rate = self.curvature_rate

        # Open-loop schedule first, then the closed-loop correction on top.
        scheduled_thresh = adaptive_threshold_schedule(
            normalized_time, base_thresh, early_scale=early_scale, late_scale=late_scale,
        )
        max_consec_cache = adaptive_max_cache_schedule(normalized_time, max_cache_late)

        if self.ctrl is not None:
            thresh = self.ctrl.curvature_budget(scheduled_thresh, rate)
            e_budget = self.ctrl.error_budget(rate)
        else:
            thresh = scheduled_thresh
            e_budget = None

        e_current = estimate_extrapolation_error(
            self.previous_residual, self.residual_velocity, self.horizon(current_t),
        )

        self.last_curvature = curvature
        self.last_thresh = thresh
        self.last_e_current = e_current
        self.last_e_budget = e_budget
        self.last_max_consec = max_consec_cache

        # Any gate can veto the cache:
        #  1. no_model  - no velocity yet, nothing to extrapolate with
        #  2. lte       - truncation error over budget (timestep-aware)
        #  3. curvature - closed-loop-corrected quality gate
        #  4. cap       - hard consecutive-cache safety net
        veto = None
        if e_current is None:
            veto = "no_model"
        elif e_budget is not None and e_current > e_budget:
            veto = "lte"
        elif curvature >= thresh:
            veto = "curvature"
        elif self.cache_step_count >= max_consec_cache:
            veto = "cap"

        if veto is not None:
            self.cache_step_count = 0
            self.vetoes[veto] = self.vetoes.get(veto, 0) + 1
            return True, veto
        self.cache_step_count += 1
        return False, None

    # -- cache / compute --------------------------------------------------

    def extrapolate(self, current_t: float) -> torch.Tensor:
        """First-order extrapolation of the cached residual (zero-order fallback)."""
        self.cache_skip_count += 1
        if self.residual_velocity is None:
            return self.previous_residual
        return extrapolate_residual_first_order(
            self.previous_residual, self.residual_velocity, self.horizon(current_t),
        )

    def observe(self, new_residual: torch.Tensor, current_t: float, step_idx: int) -> None:
        """
        Close the loop on a freshly computed residual and roll the history.

        We now know the true residual, so we can score the first-order model over
        exactly the horizon it was asked to cover. This is the only honest error
        signal in the system: the a-priori estimate says how much work the
        extrapolator was given, this says how much of it it got wrong.
        """
        current_t = float(current_t)
        old_velocity = self.residual_velocity

        if self.ctrl is not None and self.has_history:
            h = abs(current_t - self.previous_t)
            gap = step_idx - self.last_compute_step
            err_post = measure_extrapolation_error(
                new_residual, self.previous_residual, old_velocity, h,
            )
            if self.forced_this_step and gap == 1:
                # Untainted one-step sample: the controller's own gain had no say
                # in this step, so it is a valid unit for both budgets.
                err_pre = estimate_extrapolation_error(
                    self.previous_residual, old_velocity, h,
                )
                self.ctrl.observe_baseline(err_post, err_pre)
            elif gap > 1:
                # We actually coasted `gap` steps -- steer on how that turned out.
                self.ctrl.update(err_post)

        if self.has_history:
            self.residual_velocity = compute_residual_velocity(
                new_residual, self.previous_residual, abs(current_t - self.previous_t),
            )
        else:
            self.residual_velocity = None

        self.last_compute_step = step_idx
        self.prev_prev_residual = (
            self.previous_residual.clone() if self.previous_residual is not None else None
        )
        self.previous_residual = new_residual
        self.previous_t = current_t

    def diagnostics(self, prefix: str = "") -> dict:
        """Per-branch counters and controller state, for stats files."""
        p = prefix or (self.name + "_" if self.name else "")
        row = {f"{p}cached_calls": int(self.cache_skip_count)}
        row.update({f"{p}veto_{k}": int(v) for k, v in self.vetoes.items()})
        if self.ctrl is not None:
            row.update({f"{p}{k}": v for k, v in self.ctrl.diagnostics().items()})
        return row


def print_tafc_stats(
    step_idx: int,
    num_steps: int,
    cache_skip_count: int,
    curvature: Optional[float] = None,
    curvature_thresh: Optional[float] = None,
    normalized_time: Optional[float] = None,
    max_consec_cache: Optional[float] = None,
    cache_step_count: int = 0,
    curvature_rate: Optional[float] = None,
    e_current: Optional[float] = None,
    e_budget: Optional[float] = None,
    gain: Optional[float] = None,
    veto: Optional[dict] = None,
    brake: Optional[float] = None,
) -> None:
    """
    Print formatted TAFC statistics for debugging and monitoring.

    Args:
        step_idx: Current step index
        num_steps: Total number of steps
        cache_skip_count: Total number of cached steps so far
        curvature: Current trajectory curvature (optional)
        curvature_thresh: Current curvature threshold (optional)
        normalized_time: Normalized time [0, 1] (optional)
        max_consec_cache: Maximum consecutive cache budget (optional)
        cache_step_count: Consecutive cache counter (optional)
        curvature_rate: Relative curvature growth per step (optional)
        e_current: A-priori extrapolation error estimate (optional)
        e_budget: Allowed extrapolation error at this step (optional)
        gain: Current PID gain on the budget (optional)
        brake: Current curvature brake factor (optional)
    """
    cache_rate = cache_skip_count / max(1, step_idx) * 100 if step_idx > 0 else 0.0

    base_msg = f"[TAFC] Step {step_idx}/{num_steps}, Cached: {cache_skip_count}/{step_idx} ({cache_rate:.1f}%)"

    if curvature is not None and curvature_thresh is not None:
        base_msg += (f", curv={curvature:.4f}, thresh={curvature_thresh:.4f}, "
                     f"norm_t={normalized_time:.2f}, consec={cache_step_count}, "
                     f"max_consec={max_consec_cache:.1f}")

    if curvature_rate is not None:
        base_msg += f", dcurv={curvature_rate:+.3f}"
    if e_current is not None:
        base_msg += f", E={e_current:.4f}"
    if e_budget is not None:
        base_msg += f"/{e_budget:.4f}"
    if gain is not None:
        base_msg += f", gain={gain:.3f}"
    if brake is not None and brake < 1.0:
        base_msg += f", brake={brake:.3f}"

    print(base_msg)


def print_tafc_summary(
    total_steps: int,
    cache_skip_count: int,
    speedup_estimate: Optional[float] = None,
    controller: Optional["TAFCController"] = None,
    vetoes: Optional[dict] = None,
) -> None:
    """
    Print final summary statistics after generation completes.

    Args:
        total_steps: Total number of steps executed
        cache_skip_count: Total number of cached steps
        speedup_estimate: Theoretical speedup estimate (optional)
        controller: Closed-loop controller, for its final state (optional)
        vetoes: Counts of why the gate forced a compute (optional)
    """
    cache_rate = cache_skip_count / total_steps * 100 if total_steps > 0 else 0.0
    compute_steps = total_steps - cache_skip_count

    print("=" * 70)
    print("[TAFC Summary]")
    print(f"  Total steps:                {total_steps}")
    print(f"  Computed steps:             {compute_steps}")
    print(f"  Cached steps:               {cache_skip_count}")
    print(f"  Cache rate:                 {cache_rate:.1f}%")

    if speedup_estimate is None and total_steps > 0:
        # Estimate speedup assuming cached steps are free (upper bound)
        speedup_estimate = total_steps / max(1, compute_steps)

    if speedup_estimate is not None:
        print(f"  Estimated speedup:          {speedup_estimate:.2f}x")

    if vetoes:
        parts = ", ".join(f"{k}={v}" for k, v in vetoes.items() if v)
        print(f"  Compute triggers:           {parts or 'none'}")

    if controller is not None:
        d = controller.diagnostics()
        print(f"  PID gain (final):           {d['pid_gain']:.3f} "
              f"({d['pid_updates']} updates)")
        if d["pid_target"] is not None:
            print(f"  Error target / last:        {d['pid_target']:.5f} / "
                  f"{d['pid_last_measured'] if d['pid_last_measured'] is not None else float('nan'):.5f}")
        if d["pid_baseline_pre"] is not None:
            print(f"  1-step baseline (pre/post): {d['pid_baseline_pre']:.5f} / "
                  f"{d['pid_baseline_post'] if d['pid_baseline_post'] is not None else float('nan'):.5f}")

    print("=" * 70)
