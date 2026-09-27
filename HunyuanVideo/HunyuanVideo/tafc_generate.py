#!/usr/bin/env python3
"""
HunyuanVideo + TAFC (Target-Anchored Flow Caching) Generation Script

Same method as the FLUX and Wan2.1 arms, ported to HunyuanVideo's
double-stream/single-stream DiT:

1. TRUE velocity: the cached quantity is the network output residual, not a
   modulated input, so the extrapolation lives on the ODE's own right-hand side
2. PHYSICAL timestep differences read from the scheduler, so a shifted
   (non-uniform) flow-matching schedule is handled correctly
3. Trajectory CURVATURE (magnitude change + direction change) as the gate
4. FIRST-ORDER extrapolation across the coasting horizon
5. CLOSED-LOOP error feedback (PID) over the open-loop schedule: the curvature
   derivative brakes on turbulence, and a local-truncation-error estimate steers
   the caching budget from measured extrapolation error instead of a fixed ramp

HunyuanVideo calls the transformer ONCE per denoising step (guidance is
embedded, and when classifier-free guidance is on the two branches arrive
batched in a single call), so one cached residual stream suffices -- unlike
Wan2.1, which needs an independent loop per CFG pass.
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union

import torch
from loguru import logger

from hyvideo.config import parse_args
from hyvideo.inference import HunyuanVideoSampler
from hyvideo.modules.attenion import get_cu_seqlens
from hyvideo.utils.file_utils import save_videos_grid

# NOTE: the curvature metric, the local-truncation-error estimators, the PID
# controller and the per-stream gate all live in `util_tafc.py`, shared verbatim
# with the FLUX and Wan2.1 arms so the three implement the same method.
from util_tafc import (
    TAFCBranch,
    TAFCController,
    get_physical_timestep,
    print_tafc_stats,
    print_tafc_summary,
    should_force_compute,
)


# ============================================================================
# TAFC Forward Pass for HunyuanVideo
# ============================================================================

def tafc_forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,  # Should be in range(0, 1000).
        text_states: torch.Tensor = None,
        text_mask: torch.Tensor = None,
        text_states_2: Optional[torch.Tensor] = None,  # Text embedding for modulation.
        freqs_cos: Optional[torch.Tensor] = None,
        freqs_sin: Optional[torch.Tensor] = None,
        guidance: torch.Tensor = None,  # Guidance for modulation, cfg_scale x 1000.
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Drop-in replacement for HYVideoDiffusionTransformer.forward with TAFC caching.

    The cached quantity is ``img - ori_img``, i.e. everything the double- and
    single-stream blocks contributed at this step. Skipping a step replaces that
    tensor with a first-order extrapolation over the physical time actually
    coasted since the last compute, and the closed loop scores that
    extrapolation the next time a real residual arrives.

    TAFC is computed ONLY on the conditional path, then the cache decision
    and strategy are replicated to the unconditional path to ensure identical
    cache rates.
    """
    out = {}
    img = x
    txt = text_states
    batch_size = x.shape[0]
    _, _, ot, oh, ow = x.shape
    tt, th, tw = (
        ot // self.patch_size[0],
        oh // self.patch_size[1],
        ow // self.patch_size[2],
    )

    # Detect if we have both conditional and unconditional in batch (CFG)
    # When CFG is enabled, batch contains [conditional, unconditional]
    has_cfg = batch_size == 2 and getattr(self, "enable_tafc", False)
    half_batch = batch_size // 2 if has_cfg else batch_size

    # Prepare modulation vectors.
    vec = self.time_in(t)

    # text modulation
    vec = vec + self.vector_in(text_states_2)

    # guidance modulation
    if self.guidance_embed:
        if guidance is None:
            raise ValueError(
                "Didn't get guidance strength for guidance distilled model."
            )

        # our timestep_embedding is merged into guidance_in(TimestepEmbedder)
        vec = vec + self.guidance_in(guidance)

    # Embed image and text.
    img = self.img_in(img)
    if self.text_projection == "linear":
        txt = self.txt_in(txt)
    elif self.text_projection == "single_refiner":
        txt = self.txt_in(txt, t, text_mask if self.use_attention_mask else None)
    else:
        raise NotImplementedError(
            f"Unsupported text_projection: {self.text_projection}"
        )

    txt_seq_len = txt.shape[1]
    img_seq_len = img.shape[1]

    # Compute cu_squlens and max_seqlen for flash attention
    cu_seqlens_q = get_cu_seqlens(text_mask, img_seq_len)
    cu_seqlens_kv = cu_seqlens_q
    max_seqlen_q = img_seq_len + txt_seq_len
    max_seqlen_kv = max_seqlen_q

    freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None

    # ========================================================================
    # TAFC Gating Logic (Conditional Path Only)
    # ========================================================================

    # Compute step index and physical timestep BEFORE the enable_tafc check
    # This matches FLUX implementation and ensures current_t is always available
    step_idx = getattr(self, "cnt", 0)
    num_steps = getattr(self, "num_steps", 50)
    current_t = get_physical_timestep(self.scheduler, step_idx, num_steps)
    normalized_time = step_idx / max(1, num_steps - 1)

    should_calc = True
    branch = getattr(self, "tafc_branch", None)
    cached_residual_cond = None  # Will store conditional residual for replication

    if getattr(self, "enable_tafc", False):
        # Force computation for warmup/cutoff steps. While the closed loop is
        # still uncalibrated we force as well, so the controller collects its
        # one-step error baselines before authorising any extrapolation. Costs
        # ~2 extra computes out of 50.
        forced = should_force_compute(
            step_idx, num_steps, self.ret_steps, self.cutoff_steps
        ) or (branch.needs_calibration
              and step_idx < getattr(self, "tafc_calib_steps", 3))
        branch.forced_this_step = forced

        veto = None
        if forced or not branch.has_history:
            should_calc = True
        else:
            should_calc, veto = branch.should_compute(
                step_idx, normalized_time,
                self.tafc_thresh, self.tafc_max_cache, current_t,
            )

        log_every = getattr(self, "cache_log_interval", 10)
        if log_every and step_idx > 0 and step_idx % log_every == 0:
            ctrl = branch.ctrl
            print_tafc_stats(
                step_idx, self.num_steps, branch.cache_skip_count,
                branch.last_curvature, branch.last_thresh, normalized_time,
                branch.last_max_consec, branch.cache_step_count,
                curvature_rate=branch.curvature_rate,
                e_current=branch.last_e_current,
                e_budget=branch.last_e_budget,
                gain=None if ctrl is None else ctrl.gain,
                veto=veto,
                brake=None if ctrl is None else ctrl.brake(branch.curvature_rate),
            )

    # ========================================================================
    # Main Block Compute / Skip
    # ========================================================================

    if getattr(self, "enable_tafc", False) and not should_calc:
        # Cache hit: extrapolate for conditional path
        extrapolated = branch.extrapolate(current_t)
        if has_cfg:
            # Apply same extrapolation to both conditional and unconditional
            img_cond = img[:half_batch] + extrapolated
            img_uncond = img[half_batch:] + extrapolated
            img = torch.cat([img_cond, img_uncond], dim=0)
        else:
            img = img + extrapolated
    else:
        ori_img = img.clone() if getattr(self, "enable_tafc", False) else None

        # --------------------- Pass through DiT blocks ------------------------
        for _, block in enumerate(self.double_blocks):
            double_block_args = [
                img,
                txt,
                vec,
                cu_seqlens_q,
                cu_seqlens_kv,
                max_seqlen_q,
                max_seqlen_kv,
                freqs_cis,
            ]

            img, txt = block(*double_block_args)

        # Merge txt and img to pass through single stream blocks.
        x = torch.cat((img, txt), 1)
        if len(self.single_blocks) > 0:
            for _, block in enumerate(self.single_blocks):
                single_block_args = [
                    x,
                    vec,
                    txt_seq_len,
                    cu_seqlens_q,
                    cu_seqlens_kv,
                    max_seqlen_q,
                    max_seqlen_kv,
                    (freqs_cos, freqs_sin),
                ]

                x = block(*single_block_args)

        img = x[:, :img_seq_len, ...]

        if getattr(self, "enable_tafc", False):
            residual = img - ori_img
            if has_cfg:
                # Only observe and store conditional path residual
                residual_cond = residual[:half_batch]
                branch.observe(residual_cond, current_t, step_idx)
                # Store for stats: unconditional uses same cache decision
                cached_residual_cond = residual_cond
            else:
                # No CFG, observe the full residual
                branch.observe(residual, current_t, step_idx)

    # ---------------------------- Final layer ------------------------------
    img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)

    img = self.unpatchify(img, tt, th, tw)

    # ========================================================================
    # Step accounting / end-of-trajectory reset
    # ========================================================================

    if getattr(self, "enable_tafc", False):
        self.cnt += 1
        if self.cnt >= self.num_steps:
            cached = branch.cache_skip_count
            # `run_tafc_eval.py`-style harnesses parse this exact line.
            print(f"[TAFC Summary] Total cached: {cached}/{self.num_steps} "
                  f"({cached / max(1, self.num_steps) * 100:.1f}%)")
            # Snapshot before the reset: the counters below are cleared for the
            # next video, but a benchmark harness reads them after predict().
            self.last_cached_steps = cached
            self.last_total_steps = self.num_steps
            diag = {f"veto_{k}": int(v) for k, v in branch.vetoes.items()}
            if branch.ctrl is not None:
                diag.update(branch.ctrl.diagnostics())
            self.last_tafc_diag = diag
            if getattr(self, "cache_log_interval", 10):
                print_tafc_summary(self.num_steps, cached,
                                   vetoes=dict(branch.vetoes),
                                   controller=branch.ctrl)
            # Closed-loop state is per-trajectory: the baselines and the gain
            # describe THIS video's flow, not the next prompt's.
            branch.reset()
            self.cnt = 0

    if return_dict:
        out["x"] = img
        return out
    return img


# ============================================================================
# Configuration
# ============================================================================

def _make_controller(args):
    """Build the closed-loop controller, or ``None`` for open-loop TAFC.

    ``None`` selects the original open-loop behaviour, so the A/B baseline is
    literally the absence of the controller rather than a neutered copy of it.
    """
    if args.tafc_no_pid:
        return None
    return TAFCController(
        kp=args.tafc_kp,
        ki=args.tafc_ki,
        kd=args.tafc_kd,
        gain_min=args.tafc_gain_min,
        gain_max=args.tafc_gain_max,
        target=args.tafc_e_target,
        auto_tol=args.tafc_auto_tol,
        reject=args.tafc_reject,
        brake_beta=args.tafc_brake_beta,
        brake_floor=args.tafc_brake_floor,
    )


def configure_tafc(sampler, args):
    """Attach TAFC state to the transformer and return ``stats_fn``.

    Uses INSTANCE attributes (like FLUX) to avoid class/instance attribute conflicts.
    The forward method is patched on the class, but all state lives on the instance.
    """
    tr = sampler.pipeline.transformer
    cls = tr.__class__

    # Patch the forward method on the class (shared across instances)
    cls.forward = tafc_forward

    # All state goes on the INSTANCE (like FLUX does it)
    tr.scheduler = sampler.pipeline.scheduler
    tr.enable_tafc = True
    tr.cnt = 0
    tr.num_steps = int(args.infer_steps)
    tr.tafc_thresh = float(args.tafc_thresh)
    tr.tafc_max_cache = float(args.tafc_max_cache)
    tr.tafc_calib_steps = int(args.tafc_calib_steps)
    tr.cache_log_interval = int(args.log_interval)
    tr.last_cached_steps = None
    tr.last_total_steps = None
    tr.last_tafc_diag = None

    if args.use_ret_steps:
        tr.ret_steps = 5                            # force the first 5 steps
        tr.cutoff_steps = int(args.infer_steps)     # no extra cutoff region
    else:
        tr.ret_steps = 1
        tr.cutoff_steps = int(args.infer_steps) - 1  # force the final step

    tr.tafc_branch = TAFCBranch("main", _make_controller(args))

    def stats():
        # The forward pass clears the counters at the end of a trajectory, so
        # read the snapshot it left behind; fall back to the live counter if the
        # trajectory ended early (e.g. interrupted).
        cached = tr.last_cached_steps
        total = tr.last_total_steps
        if cached is None:
            cached, total = tr.tafc_branch.cache_skip_count, tr.num_steps
        row = {
            "cached_steps": int(cached),
            "computed_steps": int(total) - int(cached),
            "total_steps": int(total),
            "cache_rate_pct": round(100.0 * cached / max(1, total), 2),
        }
        # Which gate did the vetoing, and where the PID gain settled. Needed to
        # tell "cached a lot because the flow was smooth" from "cached a lot
        # because the loop went slack".
        diag = getattr(tr, "last_tafc_diag", None)
        if diag:
            row.update(diag)
        return row

    return stats


def add_tafc_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """TAFC flags, with the same names and defaults as the FLUX/Wan2.1 arms."""
    parser.add_argument("--tafc_thresh", type=float, default=0.3,
                        help="Base curvature threshold. Lower = stricter/better quality "
                             "(0.1-0.2), higher = more aggressive caching (0.3-0.5).")
    parser.add_argument("--tafc_max_cache", type=float, default=6.0,
                        help="Max consecutive cache budget at the LAST denoising step; "
                             "scales linearly from 1 across the trajectory.")
    parser.add_argument("--use_ret_steps", action="store_true", default=False,
                        help="Force-compute the first 5 steps (better quality warmup).")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="Print TAFC diagnostics every N steps (0 disables).")

    pid = parser.add_argument_group(
        "closed-loop control",
        "Error-driven adaptive caching. The open-loop schedule above sets the "
        "nominal budget; these steer it from the measured extrapolation error, "
        "the way an adaptive ODE solver steers its step size.")
    pid.add_argument("--tafc_no_pid", action="store_true", default=False,
                     help="Disable the closed loop and run the original open-loop "
                          "schedule (A/B baseline).")
    pid.add_argument("--tafc_e_target", type=float, default=0.0,
                     help="Absolute target extrapolation error. 0 = auto-calibrate "
                          "from the measured one-step baseline.")
    pid.add_argument("--tafc_auto_tol", type=float, default=2.5,
                     help="With auto-calibration, tolerate this multiple of the "
                          "one-step error baseline.")
    pid.add_argument("--tafc_reject", type=float, default=3.0,
                     help="Hard rejection: veto a cache whose a-priori error "
                          "exceeds this multiple of the budget.")
    pid.add_argument("--tafc_kp", type=float, default=0.4,
                     help="Proportional gain of the log-space error controller.")
    pid.add_argument("--tafc_ki", type=float, default=0.05,
                     help="Integral gain (removes steady-state offset).")
    pid.add_argument("--tafc_kd", type=float, default=0.2,
                     help="Derivative gain (reacts to error growth, damps overshoot).")
    pid.add_argument("--tafc_gain_min", type=float, default=0.3,
                     help="Lower clamp on the controller's budget multiplier.")
    pid.add_argument("--tafc_gain_max", type=float, default=3.0,
                     help="Upper clamp on the controller's budget multiplier.")
    pid.add_argument("--tafc_brake_beta", type=float, default=2.0,
                     help="Feedforward brake strength on RISING curvature "
                          "(0 disables the brake).")
    pid.add_argument("--tafc_brake_floor", type=float, default=0.3,
                     help="Hardest the feedforward brake may squeeze the budget.")
    pid.add_argument("--tafc_calib_steps", type=int, default=3,
                     help="Force single-step computes for the first N steps while the "
                          "controller still lacks an error baseline. Costs ~2 extra "
                          "computes out of 50.")
    return parser


# ============================================================================
# Main
# ============================================================================

def main():
    # Initialize CUDA early and disable cuDNN to prevent CUDNN_STATUS_NOT_INITIALIZED
    # cuDNN has initialization issues with Conv3d in this environment
    if torch.cuda.is_available():
        torch.cuda.init()
        # CRITICAL: Disable cuDNN to avoid CUDNN_STATUS_NOT_INITIALIZED error
        # Conv3d works fine with native CUDA backend
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        # Set device early
        if 'CUDA_VISIBLE_DEVICES' not in os.environ:
            # If no specific GPU is set, use the first available one
            torch.cuda.set_device(0)
        torch.cuda.synchronize()
        logger.info("CUDA initialized with cuDNN disabled (using native CUDA backend)")

    # `hyvideo.config.parse_args` owns the model/inference flags; inject the TAFC
    # group into the same parser by pre-parsing our own flags out of argv.
    tafc_parser = argparse.ArgumentParser(add_help=False)
    add_tafc_args(tafc_parser)
    # `parse_args` builds its own parser and exits on -h, so print our section
    # first -- otherwise the TAFC flags would be undiscoverable from --help.
    if "-h" in sys.argv[1:] or "--help" in sys.argv[1:]:
        print(tafc_parser.format_help())
    tafc_args, remaining = tafc_parser.parse_known_args()
    argv_backup = sys.argv
    sys.argv = [argv_backup[0]] + remaining
    try:
        args = parse_args(namespace=tafc_args)
    finally:
        sys.argv = argv_backup
    print(args)

    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")

    # Create save folder to save the samples
    save_path = (args.save_path if args.save_path_suffix == ""
                 else f'{args.save_path}_{args.save_path_suffix}')
    if not os.path.exists(args.save_path):
        os.makedirs(save_path, exist_ok=True)

    # Load models
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(
        models_root_path, args=args)

    # Get the updated args
    args = hunyuan_video_sampler.args

    # Override video dimensions to ensure consistent output
    args.video_size = (480, 832)  # height=480, width=832
    args.video_length = 65  # 65 frames
    logger.info(f"[Video Config] Fixed dimensions: {args.video_size[0]}x{args.video_size[1]}, {args.video_length} frames")

    # TAFC
    stats_fn = configure_tafc(hunyuan_video_sampler, args)

    tr = hunyuan_video_sampler.pipeline.transformer
    mode = "open-loop" if args.tafc_no_pid else "CLOSED-LOOP (PID)"
    logger.info(
        f"[TAFC] {mode} | thresh={args.tafc_thresh} | max_cache={args.tafc_max_cache} "
        f"| steps={args.infer_steps} | ret_steps={tr.ret_steps} "
        f"| cutoff_steps={tr.cutoff_steps}")
    if not args.tafc_no_pid:
        tgt = ("auto" if args.tafc_e_target <= 0
               else f"{args.tafc_e_target:g}")
        logger.info(
            f"[TAFC] e_target={tgt} (auto_tol={args.tafc_auto_tol}) "
            f"| kp={args.tafc_kp} ki={args.tafc_ki} kd={args.tafc_kd} "
            f"| gain=[{args.tafc_gain_min}, {args.tafc_gain_max}] "
            f"| brake_beta={args.tafc_brake_beta} floor={args.tafc_brake_floor} "
            f"| reject={args.tafc_reject}x | calib_steps={args.tafc_calib_steps}")

    # Start sampling
    start = time.time()
    outputs = hunyuan_video_sampler.predict(
        prompt=args.prompt,
        height=args.video_size[0],
        width=args.video_size[1],
        video_length=args.video_length,
        seed=args.seed,
        negative_prompt=args.neg_prompt,
        infer_steps=args.infer_steps,
        guidance_scale=args.cfg_scale,
        num_videos_per_prompt=args.num_videos,
        flow_shift=args.flow_shift,
        batch_size=args.batch_size,
        embedded_guidance_scale=args.embedded_cfg_scale,
        flow_reverse=args.flow_reverse
    )
    elapsed = time.time() - start
    samples = outputs['samples']

    row = stats_fn()
    logger.info(f"[TAFC] latency={elapsed:.1f}s | " +
                " | ".join(f"{k}={v}" for k, v in row.items()))

    # Save samples
    if 'LOCAL_RANK' not in os.environ or int(os.environ['LOCAL_RANK']) == 0:
        for i, sample in enumerate(samples):
            sample = samples[i].unsqueeze(0)
            time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H:%M:%S")
            save_path_i = (f"{save_path}/{time_flag}_seed{outputs['seeds'][i]}_"
                           f"{outputs['prompts'][i][:100].replace('/','')}.mp4")
            save_videos_grid(sample, save_path_i, fps=24)
            logger.info(f'Sample save to: {save_path_i}')


if __name__ == "__main__":
    main()
