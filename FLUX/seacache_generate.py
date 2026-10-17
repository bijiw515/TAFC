#!/usr/bin/env python3
"""
FLUX + SeaCache generation (DrawBench benchmark arm).

Shares prompt order, seeds, resolution and file naming with flux_generate.py
(baseline) and tafc_generate.py, so all three configs are directly comparable.
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
from util_seacache import rel_l1, apply_sea_with_scheduler

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def seacache_forward(
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
    Drop-in replacement for FluxTransformer2DModel.forward with SeaCache gating.
    Logic mirrors the denoise_cache gating (accumulated rescaled relative L1).
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
        # Fallback: some pipelines may pass precomputed rotary or leave ids None in edge cases.
        image_rotary_emb = None

    # ---- SeaCache gating ----
    should_calc = True
    if getattr(self, "enable_seacache", False):
        inp = hidden_states
        temb_ = temb
        modulated_inp, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.transformer_blocks[0].norm1(inp, emb=temb_)
        # Track change based on the first block's norm1 modulation
        if self.cnt == 0 or self.cnt == self.num_steps - 1 or self.previous_modulated_input is None:
            should_calc = True
            self.accumulated_rel_l1_distance = 0.0
        else:
            # Apply SEA filtering before computing distance
            modulated_inp = modulated_inp.reshape(
                modulated_inp.shape[0],
                int(img_ids[:, 1].max().item() + 1),
                int(img_ids[:, 2].max().item() + 1),
                modulated_inp.shape[-1],
            )
            modulated_inp = apply_sea_with_scheduler(
                modulated_inp,
                self.scheduler,
                getattr(self, "cnt", 0),
                power_exp=2.0,
                dims=(-2, -3),
                norm_mode="mean",
            )
            modulated_inp = modulated_inp.reshape(modulated_inp.shape[0], -1, modulated_inp.shape[-1])
            self.accumulated_rel_l1_distance += rel_l1(modulated_inp, self.previous_modulated_input)

            if self.accumulated_rel_l1_distance < float(self.seacache_thresh):
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0.0

        self.previous_modulated_input = modulated_inp
        self.cnt += 1
        if self.cnt == self.num_steps:
            # End of a full trajectory: keep the totals for the benchmark
            # harness (which reads them after pipe() returns), then reset.
            # The final step always recomputes, so no skip is lost here.
            self.last_cached_steps = getattr(self, "cache_skip_count", 0)
            self.last_total_steps = self.num_steps
            self.cnt = 0
            self.cache_skip_count = 0

    # ---- Main block compute / skip ----
    if getattr(self, "enable_seacache", False) and not should_calc and (self.previous_residual is not None):
        # Reuse residual from the previous timestep
        hidden_states = hidden_states + self.previous_residual
        self.cache_skip_count = getattr(self, "cache_skip_count", 0) + 1
    else:
        ori_hidden_states = hidden_states
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

        if getattr(self, "enable_seacache", False):
            # Cache residual from the current step
            self.previous_residual = hidden_states - ori_hidden_states

    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    if USE_PEFT_BACKEND:
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)


# Replace Diffusers model forward
FluxTransformer2DModel.forward = seacache_forward


# ----------------------------
# Cache state management
# ----------------------------
def configure_seacache(pipe, thresh, num_steps):
    """Attach SeaCache state to the transformer; return (reset_fn, stats_fn)."""
    tr = pipe.transformer
    tr.scheduler = pipe.scheduler
    tr.enable_seacache = True
    tr.seacache_thresh = float(thresh)

    def reset(steps=num_steps):
        tr.cnt = 0
        tr.num_steps = int(steps)
        tr.accumulated_rel_l1_distance = 0.0
        tr.previous_modulated_input = None
        tr.previous_residual = None
        tr.cache_skip_count = 0
        tr.last_cached_steps = None
        tr.last_total_steps = None

    def stats():
        cached, total = tr.last_cached_steps, tr.last_total_steps
        if cached is None:
            cached, total = getattr(tr, "cache_skip_count", 0), tr.num_steps
        return {
            "cached_steps": int(cached),
            "computed_steps": int(total) - int(cached),
            "total_steps": int(total),
            "cache_rate_pct": round(100.0 * cached / max(1, total), 2),
        }

    reset(num_steps)
    return reset, stats


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="FLUX + SeaCache — DrawBench benchmark arm."
    )
    add_common_args(parser)
    parser.add_argument("--seacache_thresh", type=float, default=0.3,
                        help="SeaCache accumulated rel-L1 threshold. 0.3: ~2x, 0.6: ~3x speedup.")
    args = parser.parse_args()

    items, prompt_source = collect_prompts(args, parser)
    model_id, model_name = resolve_model(args)
    num_steps = args.num_inference_steps if args.num_inference_steps else (
        4 if model_name == "flux-schnell" else 50)

    pipe, device, torch_dtype = build_pipeline(args, model_id)
    reset_fn, stats_fn = configure_seacache(pipe, args.seacache_thresh, num_steps)

    print(f"[{now_str()}] SeaCache | steps={num_steps} | guidance={args.guidance} | "
          f"thresh={args.seacache_thresh} | prompts={len(items)} | seed={args.seed} | "
          f"dtype={torch_dtype}", flush=True)

    generate_loop(
        pipe, args, model_name, model_id, device, num_steps, items, prompt_source,
        tag="SeaCache",
        config_extra={"method": "seacache", "seacache_thresh": float(args.seacache_thresh)},
        reset_fn=reset_fn,
        per_image_stats=stats_fn,
    )


if __name__ == "__main__":
    main()
