#!/usr/bin/env python3
"""
FLUX + TAFC (Target-Anchored Flow Caching) Generation Script

This script implements TAFC for FLUX models, combining:
- Trajectory curvature measurement (magnitude + direction changes)
- Physical timestep-aware drift estimation
- First-order extrapolation for cached steps
- Adaptive thresholding across the denoising trajectory
- CLOSED-LOOP error feedback (PID) over that schedule: the curvature
  derivative brakes on turbulence, and a local-truncation-error estimate
  steers the caching budget from measured extrapolation error instead of
  a fixed open-loop ramp
"""

from typing import Any, Dict, Optional, Union
import argparse

import numpy as np
import torch
from diffusers.models import FluxTransformer2DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_version,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)
from flux_generate import (
    add_common_args,
    build_pipeline,
    collect_prompts,
    generate_loop,
    now_str,
    resolve_model,
)
from util_tafc import (
    TAFCController,
    compute_trajectory_curvature,
    compute_curvature_rate,
    estimate_extrapolation_error,
    measure_extrapolation_error,
    get_physical_timestep,
    adaptive_threshold_schedule,
    adaptive_max_cache_schedule,
    compute_residual_velocity,
    extrapolate_residual_first_order,
    should_force_compute,
    print_tafc_stats,
    print_tafc_summary,
)

logger = logging.get_logger(__name__)


# ============================================================================
# TAFC Forward Pass for FLUX
# ============================================================================

def tafc_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    pooled_projections: torch.Tensor = None,
    timestep: torch.LongTensor = None,
    img_ids: torch.Tensor = None,
    txt_ids: torch.Tensor = None,
    guidance: torch.Tensor = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    controlnet_block_samples=None,
    controlnet_single_block_samples=None,
    return_dict: bool = True,
    controlnet_blocks_repeat: bool = False,
) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
    """
    Drop-in replacement for FluxTransformer2DModel.forward with TAFC caching.

    TAFC implementation with:
    1. TRUE velocity (network output residual)
    2. PHYSICAL timestep differences from ODE schedule
    3. Trajectory CURVATURE (magnitude + direction changes)
    4. FIRST-ORDER extrapolation
    """

    if joint_attention_kwargs is not None:
        joint_attention_kwargs = joint_attention_kwargs.copy()
        lora_scale = joint_attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        scale_lora_layers(self, lora_scale)
    else:
        if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
            logger.warning("Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective.")

    hidden_states = self.x_embedder(hidden_states)

    timestep = timestep.to(hidden_states.dtype) * 1000
    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000
    else:
        guidance = None

    temb = (
        self.time_text_embed(timestep, pooled_projections)
        if guidance is None
        else self.time_text_embed(timestep, guidance, pooled_projections)
    )
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)

    if txt_ids is not None and txt_ids.ndim == 3:
        logger.warning("`txt_ids` passed as 3D Tensor; dropping batch dim for rotary embedding cache.")
        txt_ids = txt_ids[0]
    if img_ids is not None and img_ids.ndim == 3:
        logger.warning("`img_ids` passed as 3D Tensor; dropping batch dim for rotary embedding cache.")
        img_ids = img_ids[0]

    if txt_ids is not None and img_ids is not None:
        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)
    else:
        image_rotary_emb = None

    # ============================================================================
    # TAFC Gating Logic
    # ============================================================================

    # Get current physical timestep and normalized time
    current_step_idx = self.cnt
    current_t = get_physical_timestep(self.scheduler, current_step_idx, self.num_steps)
    normalized_time = current_step_idx / max(1, self.num_steps - 1)

    should_calc = True
    physical_delta_t = 0.0
    ctrl = getattr(self, "tafc_ctrl", None)
    self.forced_this_step = False

    if getattr(self, "enable_tafc", False):
        # Physical horizon we would extrapolate across if we cached this step.
        # `previous_t` only advances on COMPUTED steps, so during a cache run
        # this keeps growing: it is the coasting distance, not one step.
        if self.previous_t is not None:
            physical_delta_t = abs(self.previous_t - current_t)

        needs_cal = ctrl is not None and ctrl.needs_calibration

        # Decision 1: Force computation for warmup or cutoff steps. While the
        # closed loop is still uncalibrated we force as well, so the controller
        # collects its one-step error baselines before authorising any
        # extrapolation. Costs ~2 extra computes out of 50.
        if should_force_compute(self.cnt, self.num_steps, self.ret_steps, self.cutoff_steps) or (
                needs_cal and self.cnt < getattr(self, "tafc_calib_steps", 3)):
            should_calc = True
            self.forced_this_step = True
            self.cumulative_physical_time = 0.0
        # Decision 2: First time entering caching mode - need to initialize
        elif self.previous_residual is None or self.previous_t is None:
            should_calc = True
            self.cumulative_physical_time = 0.0
        # Decision 3: Closed-loop evaluation of the cache decision
        else:
            # Track how many consecutive steps we've cached
            if not hasattr(self, 'cache_step_count'):
                self.cache_step_count = 0

            # Compute trajectory curvature from consecutive residuals
            if getattr(self, 'prev_prev_residual', None) is not None:
                curvature = compute_trajectory_curvature(
                    self.prev_prev_residual,
                    self.previous_residual
                )
            else:
                curvature = 0.01

            # Second-order signal: is the curvature itself spiking? Curvature
            # only changes when a fresh residual arrives, so the rate is
            # recomputed once per compute step and held across a cache run.
            if self.last_compute_step != self.curv_ref_step:
                self.curvature_rate = compute_curvature_rate(
                    curvature,
                    self.prev_curvature,
                    steps_elapsed=self.last_compute_step - self.curv_ref_step,
                )
                self.prev_curvature = curvature
                self.curv_ref_step = self.last_compute_step
            curvature_rate = self.curvature_rate

            # Open-loop schedule first, then the closed-loop correction.
            scheduled_thresh = adaptive_threshold_schedule(
                normalized_time,
                self.tafc_thresh,
                early_scale=0.7,
                late_scale=2.0
            )
            max_consec_cache = adaptive_max_cache_schedule(
                normalized_time,
                self.tafc_max_cache
            )

            if ctrl is not None:
                curvature_thresh = ctrl.curvature_budget(scheduled_thresh, curvature_rate)
                e_budget = ctrl.error_budget(curvature_rate)
            else:
                curvature_thresh = scheduled_thresh
                e_budget = None

            # A-priori local truncation error of the extrapolation we are about
            # to make, measured over the horizon we would actually coast.
            e_current = estimate_extrapolation_error(
                self.previous_residual,
                getattr(self, 'residual_velocity', None),
                physical_delta_t,
            )

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
            elif curvature >= curvature_thresh:
                veto = "curvature"
            elif self.cache_step_count >= max_consec_cache:
                veto = "cap"

            if veto is not None:
                should_calc = True
                self.cache_step_count = 0
                self.tafc_vetoes[veto] = self.tafc_vetoes.get(veto, 0) + 1
            else:
                should_calc = False
                self.cache_step_count += 1

            # Debug output
            log_every = getattr(self, "cache_log_interval", 10)
            if log_every and self.cnt > 0 and self.cnt % log_every == 0:
                cache_skips = getattr(self, 'cache_skip_count', 0)
                print_tafc_stats(
                    self.cnt, self.num_steps, cache_skips,
                    curvature, curvature_thresh, normalized_time,
                    max_consec_cache, self.cache_step_count,
                    curvature_rate=curvature_rate,
                    e_current=e_current,
                    e_budget=e_budget,
                    gain=None if ctrl is None else ctrl.gain,
                    veto=veto,
                )

    # ============================================================================
    # Main Block Compute / Skip
    # ============================================================================

    if getattr(self, "enable_tafc", False) and not should_calc and (self.previous_residual is not None):
        # Apply FIRST-ORDER extrapolation
        if hasattr(self, 'residual_velocity') and self.residual_velocity is not None:
            extrapolated_residual = extrapolate_residual_first_order(
                self.previous_residual,
                self.residual_velocity,
                physical_delta_t
            )
            hidden_states = hidden_states + extrapolated_residual
        else:
            # Fallback to zero-order hold if no velocity available
            hidden_states = hidden_states + self.previous_residual

        # Track cache usage
        if not hasattr(self, 'cache_skip_count'):
            self.cache_skip_count = 0
        self.cache_skip_count += 1

    else:
        # Compute fresh residual
        ori_hidden_states = hidden_states.clone()

        for index_block, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                def create_custom_forward(module):
                    def custom_forward(hidden_states, encoder_hidden_states, temb, image_rotary_emb):
                        return module(
                            hidden_states=hidden_states,
                            encoder_hidden_states=encoder_hidden_states,
                            temb=temb,
                            image_rotary_emb=image_rotary_emb,
                            joint_attention_kwargs=joint_attention_kwargs,
                        )
                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            # controlnet residual
            if controlnet_block_samples is not None:
                interval_control = int(np.ceil(len(self.transformer_blocks) / len(controlnet_block_samples)))
                if controlnet_blocks_repeat:
                    hidden_states = hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                else:
                    hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]

        for index_block, block in enumerate(self.single_transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                def create_custom_forward(module):
                    def custom_forward(hidden_states, encoder_hidden_states, temb, image_rotary_emb):
                        return module(
                            hidden_states=hidden_states,
                            encoder_hidden_states=encoder_hidden_states,
                            temb=temb,
                            image_rotary_emb=image_rotary_emb,
                            joint_attention_kwargs=joint_attention_kwargs,
                        )
                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            if controlnet_single_block_samples is not None:
                interval_control = int(np.ceil(len(self.single_transformer_blocks) / len(controlnet_single_block_samples)))
                hidden_states = hidden_states + controlnet_single_block_samples[index_block // interval_control]

        # Compute residual and velocity for TAFC
        if getattr(self, "enable_tafc", False):
            new_residual = hidden_states - ori_hidden_states
            old_velocity = getattr(self, 'residual_velocity', None)

            # ----------------------------------------------------------------
            # CLOSE THE LOOP. We now know the true residual, so we can score the
            # first-order model over exactly the horizon it was asked to cover.
            # This is the only honest error signal in the system: the a-priori
            # estimate says how much work the extrapolator was given, this says
            # how much of it the extrapolator got wrong.
            # ----------------------------------------------------------------
            if ctrl is not None and self.previous_residual is not None and self.previous_t is not None:
                horizon = abs(current_t - self.previous_t)
                gap = self.cnt - self.last_compute_step
                err_post = measure_extrapolation_error(
                    new_residual, self.previous_residual, old_velocity, horizon,
                )
                if self.forced_this_step and gap == 1:
                    # Untainted one-step sample: the controller's own gain had
                    # no say in this step, so it is a valid unit for both
                    # budgets. Measured on warmup/cutoff computes only.
                    err_pre = estimate_extrapolation_error(
                        self.previous_residual, old_velocity, horizon,
                    )
                    ctrl.observe_baseline(err_post, err_pre)
                elif gap > 1:
                    # We actually coasted `gap` steps on the extrapolation --
                    # steer the loop on how that turned out.
                    ctrl.update(err_post)

            # Compute residual velocity for first-order extrapolation
            if self.previous_residual is not None and self.previous_t is not None:
                self.residual_velocity = compute_residual_velocity(
                    new_residual,
                    self.previous_residual,
                    abs(current_t - self.previous_t)
                )
            else:
                # First time: no velocity yet
                self.residual_velocity = None

            self.last_compute_step = self.cnt

            # Update history: keep last two residuals for curvature computation
            if hasattr(self, 'previous_residual') and self.previous_residual is not None:
                self.prev_prev_residual = self.previous_residual.clone()
            else:
                self.prev_prev_residual = None

            self.previous_residual = new_residual
            self.previous_t = current_t

    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    if USE_PEFT_BACKEND:
        unscale_lora_layers(self, lora_scale)

    # Update step counter
    self.cnt += 1

    # Debug output every N steps
    log_every = getattr(self, "cache_log_interval", 10)
    if getattr(self, "enable_tafc", False) and log_every and self.cnt % log_every == 0:
        cache_skips = getattr(self, 'cache_skip_count', 0)
        print(f"[TAFC Progress] Step {self.cnt}/{self.num_steps}, Cached: {cache_skips}/{self.cnt} ({cache_skips/self.cnt*100:.1f}%)")

    # Reset at end of trajectory
    if self.cnt >= self.num_steps:
        if getattr(self, "enable_tafc", False):
            total_cached = getattr(self, 'cache_skip_count', 0)
            # Keep the totals of the trajectory we just finished: the counters
            # below are cleared for the next image, but the benchmark harness
            # reads them after pipe() returns.
            self.last_cached_steps = total_cached
            self.last_total_steps = self.num_steps
            vetoes = dict(getattr(self, "tafc_vetoes", {}) or {})
            diag = {f"veto_{k}": int(v) for k, v in vetoes.items()}
            if ctrl is not None:
                diag.update(ctrl.diagnostics())
            self.last_tafc_diag = diag
            if log_every:
                print_tafc_summary(self.num_steps, total_cached,
                                   vetoes=vetoes, controller=ctrl)
        # Reset for next generation
        self.cnt = 0
        if hasattr(self, 'cache_skip_count'):
            self.cache_skip_count = 0
        self.previous_residual = None
        self.prev_prev_residual = None
        self.residual_velocity = None
        self.previous_t = None
        self.cumulative_physical_time = 0.0
        if hasattr(self, 'cache_step_count'):
            self.cache_step_count = 0
        # Closed-loop state is per-trajectory: the baselines and the gain
        # describe THIS image's flow, not the next prompt's.
        self.last_compute_step = -1
        self.curv_ref_step = -1
        self.prev_curvature = None
        self.curvature_rate = 0.0
        self.forced_this_step = False
        self.tafc_vetoes = {}
        if ctrl is not None:
            ctrl.reset()

    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)


# Replace Diffusers model forward
FluxTransformer2DModel.forward = tafc_forward


# ============================================================================
# Cache state management
# ============================================================================

def configure_tafc(pipe, thresh, max_cache, num_steps, use_ret_steps=False, log_interval=10,
                   controller=None, calib_steps=3):
    """Attach TAFC state to the transformer and return (reset_fn, stats_fn).

    ``reset_fn(num_steps)`` clears the trajectory state before each image;
    ``stats_fn()`` reports the cache counters of the trajectory just finished.

    ``controller`` is a :class:`TAFCController` (or ``None`` for the original
    open-loop behaviour). It is reset per trajectory: each image recalibrates
    its own error baselines, since the intrinsic error scale depends on the
    prompt and seed as much as on the schedule.
    """
    tr = pipe.transformer
    tr.scheduler = pipe.scheduler
    tr.enable_tafc = True
    tr.tafc_thresh = float(thresh)
    tr.tafc_max_cache = float(max_cache)
    tr.cache_log_interval = int(log_interval)
    tr.tafc_ctrl = controller
    tr.tafc_calib_steps = int(calib_steps)

    def reset(steps=num_steps):
        tr.cnt = 0
        tr.num_steps = int(steps)
        tr.previous_residual = None
        tr.prev_prev_residual = None
        tr.residual_velocity = None
        tr.previous_t = None
        tr.cumulative_physical_time = 0.0
        tr.cache_skip_count = 0
        tr.cache_step_count = 0
        tr.last_cached_steps = None
        tr.last_total_steps = None
        tr.last_compute_step = -1
        tr.curv_ref_step = -1
        tr.prev_curvature = None
        tr.curvature_rate = 0.0
        tr.forced_this_step = False
        tr.tafc_vetoes = {}
        tr.last_tafc_diag = None
        if controller is not None:
            controller.reset()
        if use_ret_steps:
            tr.ret_steps = 5                      # force-compute the first 5 steps
            tr.cutoff_steps = int(steps)          # no extra cutoff region
        else:
            tr.ret_steps = 1
            tr.cutoff_steps = int(steps) - 1      # force-compute the final step

    def stats():
        # The forward pass clears the counters at the end of a trajectory, so
        # read the snapshot it left behind; fall back to the live counter if the
        # trajectory ended early (e.g. interrupted).
        cached = tr.last_cached_steps
        total = tr.last_total_steps
        if cached is None:
            cached, total = getattr(tr, "cache_skip_count", 0), tr.num_steps
        row = {
            "cached_steps": int(cached),
            "computed_steps": int(total) - int(cached),
            "total_steps": int(total),
            "cache_rate_pct": round(100.0 * cached / max(1, total), 2),
        }
        # Closed-loop diagnostics of the trajectory just finished: which gate did
        # the vetoing, and where the PID gain settled. Needed to tell "cached a
        # lot because the flow was smooth" from "cached a lot because the loop
        # went slack".
        diag = getattr(tr, "last_tafc_diag", None)
        if diag:
            row.update(diag)
        return row

    reset(num_steps)
    return reset, stats


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="FLUX + TAFC — Target-Anchored Flow Caching (DrawBench benchmark arm)"
    )
    add_common_args(parser)
    parser.add_argument("--tafc_thresh", type=float, default=0.3,
                        help="Base curvature threshold. Lower = stricter/better quality (0.1-0.2), "
                             "higher = more aggressive caching (0.3-0.5).")
    parser.add_argument("--tafc_max_cache", type=float, default=6.0,
                        help="Max consecutive cache budget at the LAST denoising step; scales linearly "
                             "from 1 across the trajectory.")
    parser.add_argument("--use_ret_steps", action="store_true", default=False,
                        help="Force-compute the first 5 steps (better quality warmup).")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="Print TAFC step stats every N steps (0 = silent, recommended for batch runs).")

    # -- closed-loop PID control -------------------------------------------
    pid = parser.add_argument_group(
        "closed-loop control",
        "Error-feedback control of the caching budget. The open-loop schedule "
        "relaxes the threshold as a fixed function of step index; this loop "
        "corrects it using the measured local truncation error of the "
        "extrapolations it authorised."
    )
    pid.add_argument("--tafc_no_pid", action="store_true", default=False,
                     help="Disable closed-loop control and run the original open-loop TAFC "
                          "(the A/B baseline for this feature).")
    pid.add_argument("--tafc_e_target", type=float, default=0.0,
                     help="Target a-posteriori extrapolation error. 0 = auto-calibrate from the "
                          "model's own one-step error during warmup (recommended: the scale is "
                          "model/resolution dependent).")
    pid.add_argument("--tafc_auto_tol", type=float, default=2.5,
                     help="In auto mode, target = auto_tol x median one-step error. Higher = "
                          "more error tolerated = more caching.")
    pid.add_argument("--tafc_reject", type=float, default=3.0,
                     help="A-priori LTE budget in units of one-step first-order correction. "
                          "Timestep-aware replacement for a fixed consecutive-cache count.")
    pid.add_argument("--tafc_kp", type=float, default=0.4,
                     help="Proportional gain (the exponent p of the tau update law).")
    pid.add_argument("--tafc_ki", type=float, default=0.05,
                     help="Integral gain; removes steady-state bias. 0 = PD only.")
    pid.add_argument("--tafc_kd", type=float, default=0.2,
                     help="Derivative gain; damps oscillation on a noisy error signal.")
    pid.add_argument("--tafc_gain_min", type=float, default=0.3,
                     help="Lower clamp on the closed-loop gain (hardest brake).")
    pid.add_argument("--tafc_gain_max", type=float, default=3.0,
                     help="Upper clamp on the closed-loop gain (most aggressive relaxation).")
    pid.add_argument("--tafc_brake_beta", type=float, default=2.0,
                     help="Curvature-spike brake strength. 0 disables the second-order brake.")
    pid.add_argument("--tafc_brake_floor", type=float, default=0.3,
                     help="Lower bound on the curvature brake, so one spike cannot stall caching.")
    pid.add_argument("--tafc_calib_steps", type=int, default=3,
                     help="Force single-step computes for the first N steps while the controller "
                          "still lacks an error baseline. Costs ~2 extra computes out of 50.")
    args = parser.parse_args()

    items, prompt_source = collect_prompts(args, parser)
    model_id, model_name = resolve_model(args)
    num_steps = args.num_inference_steps if args.num_inference_steps else (
        4 if model_name == "flux-schnell" else 50)

    pipe, device, torch_dtype = build_pipeline(args, model_id)
    # `None` selects the original open-loop behaviour, so the A/B baseline is
    # literally the absence of the controller rather than a neutered copy of it.
    controller = None if args.tafc_no_pid else TAFCController(
        kp=args.tafc_kp, ki=args.tafc_ki, kd=args.tafc_kd,
        gain_min=args.tafc_gain_min, gain_max=args.tafc_gain_max,
        target=args.tafc_e_target,
        auto_tol=args.tafc_auto_tol,
        reject=args.tafc_reject,
        brake_beta=args.tafc_brake_beta,
        brake_floor=args.tafc_brake_floor,
    )

    reset_fn, stats_fn = configure_tafc(
        pipe, args.tafc_thresh, args.tafc_max_cache, num_steps,
        use_ret_steps=args.use_ret_steps, log_interval=args.log_interval,
        controller=controller, calib_steps=args.tafc_calib_steps,
    )
    tr = pipe.transformer

    if args.tafc_no_pid:
        loop_desc = "open-loop (PID disabled)"
    else:
        tgt = "auto" if args.tafc_e_target <= 0 else f"{args.tafc_e_target:g}"
        loop_desc = (f"closed-loop | e_target={tgt} | reject={args.tafc_reject:g} | "
                     f"pid=({args.tafc_kp:g},{args.tafc_ki:g},{args.tafc_kd:g}) | "
                     f"gain=[{args.tafc_gain_min:g},{args.tafc_gain_max:g}] | "
                     f"brake_beta={args.tafc_brake_beta:g}")

    print(f"[{now_str()}] TAFC | steps={num_steps} | guidance={args.guidance} | "
          f"thresh={args.tafc_thresh} | max_cache={args.tafc_max_cache} | "
          f"ret_steps={tr.ret_steps} | cutoff_steps={tr.cutoff_steps} | "
          f"prompts={len(items)} | seed={args.seed} | dtype={torch_dtype}\n"
          f"[{now_str()}] TAFC | {loop_desc}", flush=True)

    generate_loop(
        pipe, args, model_name, model_id, device, num_steps, items, prompt_source,
        tag="TAFC",
        config_extra={
            "method": "tafc",
            "tafc_thresh": float(args.tafc_thresh),
            "tafc_max_cache": float(args.tafc_max_cache),
            "use_ret_steps": bool(args.use_ret_steps),
            "ret_steps": int(tr.ret_steps),
            "cutoff_steps": int(tr.cutoff_steps),
            # Closed-loop controller settings (so open- vs closed-loop runs are
            # distinguishable in the stats files during A/B evaluation).
            "pid_enabled": not bool(args.tafc_no_pid),
            "pid_e_target": float(args.tafc_e_target),
            "pid_auto_tol": float(args.tafc_auto_tol),
            "pid_reject": float(args.tafc_reject),
            "pid_kp": float(args.tafc_kp),
            "pid_ki": float(args.tafc_ki),
            "pid_kd": float(args.tafc_kd),
            "pid_gain_min": float(args.tafc_gain_min),
            "pid_gain_max": float(args.tafc_gain_max),
            "pid_brake_beta": float(args.tafc_brake_beta),
            "pid_brake_floor": float(args.tafc_brake_floor),
            "pid_calib_steps": int(args.tafc_calib_steps),
        },
        reset_fn=reset_fn,
        per_image_stats=stats_fn,
    )


if __name__ == "__main__":
    main()
