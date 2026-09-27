import argparse
from datetime import datetime
import logging
import os
import sys
import warnings

warnings.filterwarnings('ignore')

import torch, random
import torch.distributed as dist
from PIL import Image

import wan
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS, SUPPORTED_SIZES
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import cache_video, cache_image, str2bool

import gc
from contextlib import contextmanager
import torchvision.transforms.functional as TF
import torch.cuda.amp as amp
import numpy as np
import math
from wan.modules.model import sinusoidal_embedding_1d
from wan.utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                               get_sampling_sigmas, retrieve_timesteps)
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from tqdm import tqdm
from util_seacache import rel_l1, apply_sea_with_scheduler
from util_tafc import (
    TAFCBranch,
    TAFCController,
    get_physical_timestep,
    print_tafc_stats,
    print_tafc_summary,
)


EXAMPLE_PROMPT = {
    "t2v-1.3B": {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "t2v-14B": {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "t2i-14B": {
        "prompt": "一个朴素端庄的美人",
    },
    "i2v-14B": {
        "prompt":
            "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
        "image":
            "examples/i2v_input.JPG",
    },
}


# NOTE: the curvature metric, the drift estimator, the local-truncation-error
# estimators and the PID controller all live in `util_tafc.py`, shared verbatim
# with the FLUX and HunyuanVideo arms so the three implement the same method.


def t2v_generate(self,
                 input_prompt,
                 size=(1280, 720),
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True):
    r"""
    Generates video frames from text prompt using TAFC (Target-Anchored Flow Caching).
    
    This method uses velocity acceleration to predict target drift and make
    intelligent caching decisions based on trajectory curvature.
    """
    # preprocess
    F = frame_num
    target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
                    size[1] // self.vae_stride[1],
                    size[0] // self.vae_stride[2])

    seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                        (self.patch_size[1] * self.patch_size[2]) *
                        target_shape[1] / self.sp_size) * self.sp_size

    if n_prompt == "":
        n_prompt = self.sample_neg_prompt
    seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
    seed_g = torch.Generator(device=self.device)
    seed_g.manual_seed(seed)

    if not self.t5_cpu:
        self.text_encoder.model.to(self.device)
        context = self.text_encoder([input_prompt], self.device)
        context_null = self.text_encoder([n_prompt], self.device)
        if offload_model:
            self.text_encoder.model.cpu()
    else:
        context = self.text_encoder([input_prompt], torch.device('cpu'))
        context_null = self.text_encoder([n_prompt], torch.device('cpu'))
        context = [t.to(self.device) for t in context]
        context_null = [t.to(self.device) for t in context_null]

    noise = [
        torch.randn(
            target_shape[0],
            target_shape[1],
            target_shape[2],
            target_shape[3],
            dtype=torch.float32,
            device=self.device,
            generator=seed_g)
    ]

    @contextmanager
    def noop_no_sync():
        yield

    no_sync = getattr(self.model, 'no_sync', noop_no_sync)

    # evaluation mode
    with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():

        if sample_solver == 'unipc':
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            sample_scheduler.set_timesteps(
                sampling_steps, device=self.device, shift=shift)
            timesteps = sample_scheduler.timesteps
        elif sample_solver == 'dpm++':
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
            timesteps, _ = retrieve_timesteps(
                sample_scheduler,
                device=self.device,
                sigmas=sampling_sigmas)
        else:
            raise NotImplementedError("Unsupported solver.")
        self.model.scheduler = sample_scheduler

        # sample videos
        latents = noise

        arg_c = {'context': context, 'seq_len': seq_len}
        arg_null = {'context': context_null, 'seq_len': seq_len}

        for _, t in enumerate(tqdm(timesteps)):
            latent_model_input = latents
            timestep = [t]

            timestep = torch.stack(timestep)

            self.model.to(self.device)
            noise_pred_cond = self.model(
                latent_model_input, t=timestep, **arg_c)[0]
            noise_pred_uncond = self.model(
                latent_model_input, t=timestep, **arg_null)[0]

            noise_pred = noise_pred_uncond + guide_scale * (
                noise_pred_cond - noise_pred_uncond)

            temp_x0 = sample_scheduler.step(
                noise_pred.unsqueeze(0),
                t,
                latents[0].unsqueeze(0),
                return_dict=False,
                generator=seed_g)[0]
            latents = [temp_x0.squeeze(0)]

        x0 = latents
        if offload_model:
            self.model.cpu()
            torch.cuda.empty_cache()
        if self.rank == 0:
            videos = self.vae.decode(x0)

    del noise, latents
    del sample_scheduler
    if offload_model:
        gc.collect()
        torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()

    return videos[0] if self.rank == 0 else None



def i2v_generate(self,
                 input_prompt,
                 img,
                 max_area=720 * 1280,
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=40,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True):
    r"""
    Generates video frames from input image and text prompt using TAFC.
    """
    img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)

    F = frame_num
    h, w = img.shape[1:]
    aspect_ratio = h / w
    lat_h = round(
        np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] //
        self.patch_size[1] * self.patch_size[1])
    lat_w = round(
        np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] //
        self.patch_size[2] * self.patch_size[2])
    h = lat_h * self.vae_stride[1]
    w = lat_w * self.vae_stride[2]

    max_seq_len = ((F - 1) // self.vae_stride[0] + 1) * lat_h * lat_w // (
        self.patch_size[1] * self.patch_size[2])
    max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

    seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
    seed_g = torch.Generator(device=self.device)
    seed_g.manual_seed(seed)
    noise = torch.randn(
        self.vae.model.z_dim, 
        (F - 1) // self.vae_stride[0] + 1,
        lat_h,
        lat_w,
        dtype=torch.float32,
        generator=seed_g,
        device=self.device)

    msk = torch.ones(1, F, lat_h, lat_w, device=self.device)
    msk[:, 1:] = 0
    msk = torch.concat([
        torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
    ],
                       dim=1)
    msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
    msk = msk.transpose(1, 2)[0]

    if n_prompt == "":
        n_prompt = self.sample_neg_prompt

    # preprocess
    if not self.t5_cpu:
        self.text_encoder.model.to(self.device)
        context = self.text_encoder([input_prompt], self.device)
        context_null = self.text_encoder([n_prompt], self.device)
        if offload_model:
            self.text_encoder.model.cpu()
    else:
        context = self.text_encoder([input_prompt], torch.device('cpu'))
        context_null = self.text_encoder([n_prompt], torch.device('cpu'))
        context = [t.to(self.device) for t in context]
        context_null = [t.to(self.device) for t in context_null]

    self.clip.model.to(self.device)
    clip_context = self.clip.visual([img[:, None, :, :]])
    if offload_model:
        self.clip.model.cpu()

    y = self.vae.encode([
        torch.concat([
            torch.nn.functional.interpolate(
                img[None].cpu(), size=(h, w), mode='bicubic').transpose(
                    0, 1),
            torch.zeros(3, F-1, h, w)
        ],
                     dim=1).to(self.device)
    ])[0]
    y = torch.concat([msk, y])

    @contextmanager
    def noop_no_sync():
        yield

    no_sync = getattr(self.model, 'no_sync', noop_no_sync)

    # evaluation mode
    with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():

        if sample_solver == 'unipc':
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            sample_scheduler.set_timesteps(
                sampling_steps, device=self.device, shift=shift)
            timesteps = sample_scheduler.timesteps
        elif sample_solver == 'dpm++':
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
            timesteps, _ = retrieve_timesteps(
                sample_scheduler,
                device=self.device,
                sigmas=sampling_sigmas)
        else:
            raise NotImplementedError("Unsupported solver.")
        self.model.scheduler = sample_scheduler

        # sample videos
        latent = noise

        arg_c = {
            'context': context,
            'clip_fea': clip_context,
            'seq_len': max_seq_len,
            'y': [y],
        }

        arg_null = {
            'context': context_null,
            'clip_fea': clip_context,
            'seq_len': max_seq_len,
            'y': [y],
        }

        if offload_model:
            torch.cuda.empty_cache()

        self.model.to(self.device)
        for _, t in enumerate(tqdm(timesteps)):
            latent_model_input = [latent.to(self.device)]
            timestep = [t]

            timestep = torch.stack(timestep).to(self.device)

            noise_pred_cond = self.model(
                latent_model_input, t=timestep, **arg_c)[0].to(
                    torch.device('cpu') if offload_model else self.device)
            if offload_model:
                torch.cuda.empty_cache()
            noise_pred_uncond = self.model(
                latent_model_input, t=timestep, **arg_null)[0].to(
                    torch.device('cpu') if offload_model else self.device)
            if offload_model:
                torch.cuda.empty_cache()

            noise_pred = noise_pred_uncond + guide_scale * (
                noise_pred_cond - noise_pred_uncond)

            latent = latent.to(
                torch.device('cpu') if offload_model else self.device)

            temp_x0 = sample_scheduler.step(
                noise_pred.unsqueeze(0),
                t,
                latent.unsqueeze(0),
                return_dict=False,
                generator=seed_g)[0]
            latent = temp_x0.squeeze(0)

            x0 = [latent.to(self.device)]
            del latent_model_input, timestep

        if offload_model:
            self.model.cpu()
            torch.cuda.empty_cache()

        if self.rank == 0:
            videos = self.vae.decode(x0)

    del noise, latent
    del sample_scheduler
    if offload_model:
        gc.collect()
        torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()

    return videos[0] if self.rank == 0 else None



def tafc_forward(
    self,
    x,
    t,
    context,
    seq_len,
    clip_fea=None,
    y=None,
):
    r"""
    Forward pass through the diffusion model with TAFC (Target-Anchored Flow Caching).

    Base mechanism:
    1. Uses TRUE velocity (network output residual) not intermediate features
    2. Uses PHYSICAL timestep differences from ODE schedule, not integer skip counts
    3. Measures trajectory curvature with BOTH magnitude + direction (cosine similarity)
    4. Applies FIRST-ORDER extrapolation instead of zero-order hold

    Closed-loop error feedback on top of that (see `util_tafc.TAFCController`):
    5. CURVATURE DERIVATIVE brakes the budget when the flow is entering a
       turbulent region, instead of relaxing it on the open-loop ramp alone
    6. LOCAL TRUNCATION ERROR, from the first-order/zero-order embedded pair,
       gates the cache on a timestep-aware error estimate
    7. A PID loop steers a multiplicative gain from the *measured* a-posteriori
       error, so the budget follows evidence rather than a preset formula

    Wan calls the model TWICE per denoising step (conditional + unconditional
    for CFG). Those two passes are different trajectories, so each gets its own
    `TAFCBranch`: independent residual history, velocity, and PID controller.
    ``self.cnt`` counts model CALLS, so the step index is ``cnt // 2``.

    The physics intuition:
    - In Flow Matching, velocity v_t = model_output is the residual pointing toward x_0
    - Trajectory curvature combines magnitude change + direction change
    - Using cached velocity causes drift ≈ 0.5 * curvature * (physical_Δt)²
    - First-order extrapolation: x_new = x_cached + velocity_rate * Δt
    """
    if self.model_type == 'i2v':
        assert clip_fea is not None and y is not None

    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    # embeddings
    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack(
        [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len
    x = torch.cat([
        torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                  dim=1) for u in x
    ])

    # time embeddings
    with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).float())
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))
        assert e.dtype == torch.float32 and e0.dtype == torch.float32

    # context
    context_lens = None
    context = self.text_embedding(
        torch.stack([
            torch.cat(
                [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
            for u in context
        ]))

    if clip_fea is not None:
        context_clip = self.img_emb(clip_fea)
        context = torch.concat([context_clip, context], dim=1)

    # arguments
    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=self.freqs,
        context=context,
        context_lens=context_lens)

    # ========================================================================
    # TAFC gating
    # ========================================================================

    # `cnt` counts model CALLS (2 per denoising step under CFG), so the
    # denoising step -- and therefore the physical timestep -- is cnt // 2.
    total_steps = self.num_steps // 2
    current_step_idx = self.cnt // 2
    current_t = get_physical_timestep(self.scheduler, current_step_idx, total_steps)

    # Normalized position in the trajectory: 0.0 = early denoising, 1.0 = late.
    normalized_time = current_step_idx / max(1, total_steps - 1)

    should_calc = True

    if self.enable_tafc:
        self.is_even = (self.cnt % 2 == 0)
        branch = self.tafc_even if self.is_even else self.tafc_odd
        other_branch = self.tafc_odd if self.is_even else self.tafc_even

        # ====================================================================
        # CFG-AWARE FORCED COMPUTE LOGIC
        # ====================================================================
        # Force a compute during warmup/cutoff. While the closed loop is still
        # uncalibrated we force as well, so the controller collects its one-step
        # error baselines before authorising any extrapolation.
        #
        # CRITICAL: The forced decision must be synchronized across BOTH CFG
        # branches. If one branch needs calibration but the other doesn't,
        # we must force BOTH to compute, otherwise they diverge immediately.
        # ====================================================================

        # Check this branch's forced conditions
        forced = (self.cnt < self.ret_steps or self.cnt >= self.cutoff_steps or
                  (branch.needs_calibration and
                   current_step_idx < getattr(self, 'tafc_calib_steps', 3)))

        # CFG sync: check if the OTHER branch is forced
        if self.is_even:
            # First branch: check if second branch is forced
            forced_other = (self.cnt < self.ret_steps or self.cnt >= self.cutoff_steps or
                           (other_branch.needs_calibration and
                            current_step_idx < getattr(self, 'tafc_calib_steps', 3)))
            # If EITHER is forced, BOTH are forced
            forced = forced or forced_other
            self._cfg_forced = forced
        else:
            # Second branch: use the unified forced decision
            if hasattr(self, '_cfg_forced'):
                forced = self._cfg_forced

        branch.forced_this_step = forced

        veto = None
        if forced or not branch.has_history:
            should_calc = True
        else:
            should_calc, veto = branch.should_compute(
                current_step_idx, normalized_time,
                self.tafc_thresh, self.tafc_max_cache, current_t,
            )

            # ================================================================
            # CFG BRANCH SYNC (Solution to branch imbalance quality collapse)
            # ================================================================
            # Problem: In CFG, output = uncond + scale*(cond - uncond).
            # If cond and uncond branches cache at different rates, the
            # (cond - uncond) difference is computed from residuals at
            # different timesteps, causing severe quality collapse.
            #
            # Solution: STRICT SYNCHRONIZATION before individual decisions.
            # On the first branch (even/cond), we peek at what the second
            # branch (odd/uncond) WOULD decide, then apply OR logic to get
            # a unified decision that BOTH branches will use.
            #
            # Cost: ~5-10% lower cache rate, but completely prevents quality
            # collapse from branch imbalance.
            # ================================================================

            # On the first branch (even/cond), evaluate both branches and unify
            if self.is_even:
                if other_branch.has_history:
                    # Peek at what the other branch wants WITHOUT modifying its state
                    should_calc_other, veto_other = other_branch.should_compute(
                        current_step_idx, normalized_time,
                        self.tafc_thresh, self.tafc_max_cache, current_t,
                    )
                    # CRITICAL: OR logic - if EITHER wants compute, BOTH must compute
                    unified_decision = should_calc or should_calc_other

                    # Override this branch's decision
                    if unified_decision and not should_calc:
                        veto = "cfg_sync_from_uncond"
                        branch.vetoes[veto] = branch.vetoes.get(veto, 0) + 1
                    should_calc = unified_decision

                    # Store for the second branch to use
                    self._cfg_sync_decision = unified_decision
                    self._cfg_sync_veto_other = "cfg_sync_from_cond" if (unified_decision and not should_calc_other) else None
                else:
                    # Second branch has no history yet, just store our decision
                    self._cfg_sync_decision = should_calc
                    self._cfg_sync_veto_other = None

            # On the second branch (odd/uncond), use the unified decision
            else:
                if hasattr(self, '_cfg_sync_decision'):
                    # Force use of the unified decision made by first branch
                    if self._cfg_sync_decision != should_calc:
                        # Our natural decision differs from the unified one
                        if self._cfg_sync_veto_other:
                            veto = self._cfg_sync_veto_other
                            branch.vetoes[veto] = branch.vetoes.get(veto, 0) + 1
                    should_calc = self._cfg_sync_decision

        log_every = getattr(self, 'cache_log_interval', 20)
        if log_every and self.is_even and self.cnt and self.cnt % log_every == 0:
            ctrl = branch.ctrl
            print_tafc_stats(
                current_step_idx, total_steps, branch.cache_skip_count,
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
    # Main block compute / skip
    # ========================================================================

    if self.enable_tafc:
        branch = self.tafc_even if self.is_even else self.tafc_odd
        if not should_calc:
            # Reuse the cached residual, advanced to this step by first-order
            # extrapolation over the horizon we have actually been coasting.
            x = x + branch.extrapolate(current_t)
        else:
            ori_x = x.clone()
            for block in self.blocks:
                x = block(x, **kwargs)
            # Scores the extrapolation the loop authorised, updates the PID gain
            # and rolls the residual history forward.
            # CRITICAL: detach and clone the residual to avoid any potential
            # memory aliasing issues in video tensors with temporal dimension
            residual = (x - ori_x).detach().clone()
            branch.observe(residual, current_t, current_step_idx)
    else:
        for block in self.blocks:
            x = block(x, **kwargs)

    # head
    x = self.head(x, e)

    # unpatchify
    x = self.unpatchify(x, grid_sizes)
    self.cnt += 1

    if self.cnt >= self.num_steps:
        if self.enable_tafc:
            cached = self.tafc_even.cache_skip_count + self.tafc_odd.cache_skip_count
            # Keep the format run_tafc_eval.py parses: it regexes this line for
            # the per-video cache rate.
            print(f"[TAFC Summary] Total cached: {cached}/{self.num_steps} "
                  f"({cached / max(1, self.num_steps) * 100:.1f}%)")

            # Per-branch breakdown
            even_rate = self.tafc_even.cache_skip_count / max(1, self.num_steps // 2) * 100
            odd_rate = self.tafc_odd.cache_skip_count / max(1, self.num_steps // 2) * 100
            print(f"[TAFC Branch] cond:   {self.tafc_even.cache_skip_count}/{self.num_steps//2} ({even_rate:.1f}%)")
            print(f"[TAFC Branch] uncond: {self.tafc_odd.cache_skip_count}/{self.num_steps//2} ({odd_rate:.1f}%)")

            vetoes = {}
            for br in (self.tafc_even, self.tafc_odd):
                for k, v in br.vetoes.items():
                    vetoes[k] = vetoes.get(k, 0) + v
            self.last_tafc_diag = {"veto_" + k: v for k, v in vetoes.items()}
            self.last_tafc_diag.update(self.tafc_even.diagnostics("cond_"))
            self.last_tafc_diag.update(self.tafc_odd.diagnostics("uncond_"))
            if getattr(self, 'cache_log_interval', 20):
                print_tafc_summary(self.num_steps, cached, vetoes=vetoes,
                                   controller=self.tafc_even.ctrl)
            # Per-video state: the baselines and the gain describe THIS prompt's
            # flow, not the next one's.
            self.tafc_even.reset()
            self.tafc_odd.reset()
        self.cnt = 0
    return [u.float() for u in x]



def _validate_args(args):
    # Basic check
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"
    assert args.task in EXAMPLE_PROMPT, f"Unsupport task: {args.task}"

    # The default sampling steps are 40 for image-to-video tasks and 50 for text-to-video tasks.
    if args.sample_steps is None:
        args.sample_steps = 40 if "i2v" in args.task else 50

    if args.sample_shift is None:
        args.sample_shift = 5.0
        if "i2v" in args.task and args.size in ["832*480", "480*832"]:
            args.sample_shift = 3.0

    # The default number of frames are 1 for text-to-image tasks and 81 for other tasks.
    if args.frame_num is None:
        args.frame_num = 1 if "t2i" in args.task else 81

    # T2I frame_num check
    if "t2i" in args.task:
        assert args.frame_num == 1, f"Unsupport frame_num {args.frame_num} for task {args.task}"

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(
        0, sys.maxsize)
    # Size check
    assert args.size in SUPPORTED_SIZES[
        args.
        task], f"Unsupport size {args.size} for task {args.task}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a image or video from a text prompt or image using Wan with TAFC"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="t2v-14B",
        choices=list(WAN_CONFIGS.keys()),
        help="The task to run.")
    parser.add_argument(
        "--size",
        type=str,
        default="1280*720",
        choices=list(SIZE_CONFIGS.keys()),
        help="The area (width*height) of the generated video. For the I2V task, the aspect ratio of the output video will follow that of the input image."
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=None,
        help="How many frames to sample from a image or video. The number should be 4n+1"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="The path to the checkpoint directory.")
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage."
    )
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.")
    parser.add_argument(
        "--ring_size",
        type=int,
        default=1,
        help="The size of the ring attention parallelism in DiT.")
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5.")
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.")
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT.")
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="The file to save the generated image or video to.")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="The prompt to generate the image or video from.")
    parser.add_argument(
        "--use_prompt_extend",
        action="store_true",
        default=False,
        help="Whether to use prompt extend.")
    parser.add_argument(
        "--prompt_extend_method",
        type=str,
        default="local_qwen",
        choices=["dashscope", "local_qwen"],
        help="The prompt extend method to use.")
    parser.add_argument(
        "--prompt_extend_model",
        type=str,
        default=None,
        help="The prompt extend model to use.")
    parser.add_argument(
        "--prompt_extend_target_lang",
        type=str,
        default="ch",
        choices=["ch", "en"],
        help="The target language of prompt extend.")
    parser.add_argument(
        "--base_seed",
        type=int,
        default=-1,
        help="The seed to use for generating the image or video.")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="The image to generate the video from.")
    parser.add_argument(
        "--sample_solver",
        type=str,
        default='unipc',
        choices=['unipc', 'dpm++'],
        help="The solver used to sample.")
    parser.add_argument(
        "--sample_steps", type=int, default=None, help="The sampling steps.")
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument(
        "--sample_guide_scale",
        type=float,
        default=5.0,
        help="Classifier free guidance scale.")
    parser.add_argument(
        "--tafc_thresh",
        type=float,
        default=0.3,
        help="TAFC target drift threshold. Higher values = more aggressive caching. Recommended: 0.1 for ~2x speedup, 0.2 for ~3x speedup")
    parser.add_argument(
        "--tafc_max_cache",
        type=float,
        default=6.0,
        help="TAFC max consecutive cache budget at the LAST denoising step. "
             "Budget scales linearly 0 -> this value across denoising, so late "
             "stages cache up to ~max/(max+1) of steps. Higher = more caching late.")
    parser.add_argument(
        "--use_ret_steps",
        action="store_true",
        default=False,
        help="Using Retention Steps will result in faster generation speed and better generation quality.")
    parser.add_argument(
        "--log_interval",
        type=int,
        default=20,
        help="Print TAFC step stats every N model calls (0 = silent).")

    # -- closed-loop PID control (flag names match the FLUX arm) ------------
    pid = parser.add_argument_group(
        "closed-loop control",
        "Error-feedback control of the caching budget. The open-loop schedule "
        "relaxes the threshold as a fixed function of step index; this loop "
        "corrects it using the measured local truncation error of the "
        "extrapolations it authorised. The conditional and unconditional CFG "
        "branches are controlled independently.")
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
                     help="Force single-step computes for the first N denoising steps while the "
                          "controller still lacks an error baseline.")

    args = parser.parse_args()

    _validate_args(args)

    return args


def _init_logging(rank):
    # logging
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)



def _make_controller(args):
    """Build a TAFCController from CLI args, or None for open-loop TAFC."""
    if args.tafc_no_pid:
        return None
    return TAFCController(
        kp=args.tafc_kp, ki=args.tafc_ki, kd=args.tafc_kd,
        gain_min=args.tafc_gain_min, gain_max=args.tafc_gain_max,
        target=args.tafc_e_target,
        auto_tol=args.tafc_auto_tol,
        reject=args.tafc_reject,
        brake_beta=args.tafc_brake_beta,
        brake_floor=args.tafc_brake_floor,
    )


def _apply_tafc_hooks(pipe, generate_fn, args):
    """
    Apply TAFC (Target-Anchored Flow Caching) hooks to the pipeline.

    Base mechanism:
    1. TRUE velocity (network output residual)
    2. PHYSICAL timestep differences from ODE schedule
    3. Trajectory CURVATURE (magnitude + direction changes)
    4. FIRST-ORDER extrapolation

    Closed loop on top:
    5. Curvature derivative brake on turbulent regions
    6. Local-truncation-error gate (timestep-aware)
    7. PID control of the budget from measured extrapolation error

    The conditional and unconditional CFG passes are separate trajectories, so
    each gets its own branch with its own controller. Branch state is per
    INSTANCE (it holds residual tensors); only the config and the patched
    forward go on the class.
    """
    pipe.__class__.generate = generate_fn
    model = pipe.model
    model.__class__.enable_tafc = True
    model.__class__.forward = tafc_forward
    model.__class__.cnt = 0
    model.__class__.num_steps = args.sample_steps * 2
    model.__class__.tafc_thresh = args.tafc_thresh
    model.__class__.tafc_max_cache = args.tafc_max_cache
    model.__class__.tafc_calib_steps = args.tafc_calib_steps
    model.__class__.cache_log_interval = args.log_interval
    model.__class__.is_even = True
    model.__class__.last_tafc_diag = None

    # One independently-controlled cache per CFG branch.
    model.tafc_even = TAFCBranch("cond", _make_controller(args))
    model.tafc_odd = TAFCBranch("uncond", _make_controller(args))

    if args.use_ret_steps:
        model.__class__.ret_steps = 5 * 2
        model.__class__.cutoff_steps = args.sample_steps * 2
    else:
        model.__class__.ret_steps = 1 * 2
        model.__class__.cutoff_steps = args.sample_steps * 2 - 2


def generate(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(
            f"offload_model is not specified, set to {args.offload_model}.")
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)
    else:
        assert not (
            args.t5_fsdp or args.dit_fsdp
        ), f"t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (
            args.ulysses_size > 1 or args.ring_size > 1
        ), f"context parallel are not supported in non-distributed environments."

    if args.ulysses_size > 1 or args.ring_size > 1:
        assert args.ulysses_size * args.ring_size == world_size, f"The number of ulysses_size and ring_size should be equal to the world size."
        from xfuser.core.distributed import (initialize_model_parallel,
                                             init_distributed_environment)
        init_distributed_environment(
            rank=dist.get_rank(), world_size=dist.get_world_size())

        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=args.ring_size,
            ulysses_degree=args.ulysses_size,
        )

    if args.use_prompt_extend:
        if args.prompt_extend_method == "dashscope":
            prompt_expander = DashScopePromptExpander(
                model_name=args.prompt_extend_model, is_vl="i2v" in args.task)
        elif args.prompt_extend_method == "local_qwen":
            prompt_expander = QwenPromptExpander(
                model_name=args.prompt_extend_model,
                is_vl="i2v" in args.task,
                device=rank)
        else:
            raise NotImplementedError(
                f"Unsupport prompt_extend_method: {args.prompt_extend_method}")

    cfg = WAN_CONFIGS[args.task]
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0, f"`num_heads` must be divisible by `ulysses_size`."

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")
    logging.info("Using TAFC (Target-Anchored Flow Caching) with target drift estimation")
    if args.tafc_no_pid:
        logging.info("TAFC control: open-loop (PID disabled)")
    else:
        tgt = "auto" if args.tafc_e_target <= 0 else f"{args.tafc_e_target:g}"
        logging.info(
            f"TAFC control: closed-loop | e_target={tgt} | reject={args.tafc_reject:g} | "
            f"pid=({args.tafc_kp:g},{args.tafc_ki:g},{args.tafc_kd:g}) | "
            f"gain=[{args.tafc_gain_min:g},{args.tafc_gain_max:g}] | "
            f"brake_beta={args.tafc_brake_beta:g} | calib_steps={args.tafc_calib_steps}")

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    if "t2v" in args.task or "t2i" in args.task:
        if args.prompt is None:
            args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
        logging.info(f"Input prompt: {args.prompt}")
        if args.use_prompt_extend:
            logging.info("Extending prompt ...")
            if rank == 0:
                prompt_output = prompt_expander(
                    args.prompt,
                    tar_lang=args.prompt_extend_target_lang,
                    seed=args.base_seed)
                if prompt_output.status == False:
                    logging.info(
                        f"Extending prompt failed: {prompt_output.message}")
                    logging.info("Falling back to original prompt.")
                    input_prompt = args.prompt
                else:
                    input_prompt = prompt_output.prompt
                input_prompt = [input_prompt]
            else:
                input_prompt = [None]
            if dist.is_initialized():
                dist.broadcast_object_list(input_prompt, src=0)
            args.prompt = input_prompt[0]
            logging.info(f"Extended prompt: {args.prompt}")

        logging.info("Creating WanT2V pipeline.")
        wan_t2v = wan.WanT2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
            t5_cpu=args.t5_cpu,
        )

        # TAFC
        _apply_tafc_hooks(wan_t2v, t2v_generate, args)

        logging.info(
            f"Generating {'image' if 't2i' in args.task else 'video'} with TAFC...")
        video = wan_t2v.generate(
            args.prompt,
            size=SIZE_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model)

    else:
        if args.prompt is None:
            args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
        if args.image is None:
            args.image = EXAMPLE_PROMPT[args.task]["image"]
        logging.info(f"Input prompt: {args.prompt}")
        logging.info(f"Input image: {args.image}")

        img = Image.open(args.image).convert("RGB")
        if args.use_prompt_extend:
            logging.info("Extending prompt ...")
            if rank == 0:
                prompt_output = prompt_expander(
                    args.prompt,
                    tar_lang=args.prompt_extend_target_lang,
                    image=img,
                    seed=args.base_seed)
                if prompt_output.status == False:
                    logging.info(
                        f"Extending prompt failed: {prompt_output.message}")
                    logging.info("Falling back to original prompt.")
                    input_prompt = args.prompt
                else:
                    input_prompt = prompt_output.prompt
                input_prompt = [input_prompt]
            else:
                input_prompt = [None]
            if dist.is_initialized():
                dist.broadcast_object_list(input_prompt, src=0)
            args.prompt = input_prompt[0]
            logging.info(f"Extended prompt: {args.prompt}")

        logging.info("Creating WanI2V pipeline.")
        wan_i2v = wan.WanI2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
            t5_cpu=args.t5_cpu,
        )
        # TAFC
        _apply_tafc_hooks(wan_i2v, i2v_generate, args)

        logging.info("Generating video with TAFC...")
        video = wan_i2v.generate(
            args.prompt,
            img,
            max_area=MAX_AREA_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model)

    if rank == 0:
        if args.save_file is None:
            formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            formatted_prompt = args.prompt.replace(" ", "_").replace("/",
                                                                     "_")[:50]
            suffix = '.png' if "t2i" in args.task else '.mp4'
            args.save_file = f"{args.task}_{args.size}_{args.ulysses_size}_{args.ring_size}_{formatted_prompt}_{formatted_time}" + suffix

        if "t2i" in args.task:
            logging.info(f"Saving generated image to {args.save_file}")
            cache_image(
                tensor=video.squeeze(1)[None],
                save_file=args.save_file,
                nrow=1,
                normalize=True,
                value_range=(-1, 1))
        else:
            logging.info(f"Saving generated video to {args.save_file}")
            cache_video(
                tensor=video[None],
                save_file=args.save_file,
                fps=cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1))
    logging.info("Finished.")    


if __name__ == "__main__":
    args = _parse_args()
    generate(args)
