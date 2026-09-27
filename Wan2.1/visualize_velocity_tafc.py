"""
在 TAFC 的完整 Wan T2V pipeline 上统计并可视化每个 timestep 的 velocity 特性。

统计量:
  - magnitude:  ||v_t||                                    (velocity 大小)
  - direction:  angle(v_t, v_{t+1})                        (相邻方向夹角, 度)
  - curvature:  ||v_{t+1} - v_t|| / dt                     (曲率/加速度)

同时支持对比 TAFC ON / OFF 两种模式下的 velocity 轨迹, 用于验证:
  "去噪前期 velocity 剧烈变化, 后期趋于稳定" 这一猜想, 并观察 TAFC
  跳过的步骤是否恰好落在 velocity 的稳定区间。

用法示例:
  # 单独分析 (TAFC OFF, 真实完整 velocity 轨迹)
  CUDA_VISIBLE_DEVICES=6 python visualize_velocity_tafc.py \
      --ckpt_dir Wan2.1-T2V-1.3B --task t2v-1.3B --size 832*480 --sample_steps 50

  # 对比 TAFC ON vs OFF
  CUDA_VISIBLE_DEVICES=6 python visualize_velocity_tafc.py \
      --ckpt_dir Wan2.1-T2V-1.3B --task t2v-1.3B --size 832*480 \
      --sample_steps 50 --compare --tafc_thresh 0.2 --use_ret_steps
"""
import argparse
import gc
import logging
import math
import os
import random
import sys
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.cuda.amp as amp
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import wan
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS
from wan.utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                                  get_sampling_sigmas, retrieve_timesteps)
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

# 复用 TAFC 的缓存 forward
from tafc_generate import tafc_forward
from util_tafc import TAFCBranch


# ----------------------------------------------------------------------------- #
#  velocity 采样: 在真实 pipeline 中跑一次去噪, 记录每一步的 velocity
# ----------------------------------------------------------------------------- #
def sample_and_record_velocity(pipe, prompt, size, frame_num, shift,
                               sample_solver, sampling_steps, guide_scale,
                               seed, device):
    """
    复刻 t2v_generate 的去噪循环, 但在每一步记录 CFG 后的 velocity (noise_pred)。

    返回:
      velocities: List[Tensor]  每步 velocity (CPU, float32)
      timesteps:  Tensor        对应的离散 timestep
      skip_flags: List[bool]    该步 TAFC 是否跳过了 DiT 计算 (条件分支)
    """
    cfg = pipe.config
    n_prompt = pipe.sample_neg_prompt

    F = frame_num
    target_shape = (pipe.vae.model.z_dim, (F - 1) // pipe.vae_stride[0] + 1,
                    size[1] // pipe.vae_stride[1],
                    size[0] // pipe.vae_stride[2])
    seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                        (pipe.patch_size[1] * pipe.patch_size[2]) *
                        target_shape[1] / pipe.sp_size) * pipe.sp_size

    seed_g = torch.Generator(device=device)
    seed_g.manual_seed(seed)

    # 文本编码
    pipe.text_encoder.model.to(device)
    context = pipe.text_encoder([prompt], device)
    context_null = pipe.text_encoder([n_prompt], device)

    noise = [torch.randn(target_shape[0], target_shape[1], target_shape[2],
                         target_shape[3], dtype=torch.float32, device=device,
                         generator=seed_g)]

    velocities = []
    skip_flags = []

    with amp.autocast(dtype=pipe.param_dtype), torch.no_grad():
        # scheduler
        if sample_solver == 'unipc':
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=pipe.num_train_timesteps,
                shift=1, use_dynamic_shifting=False)
            sample_scheduler.set_timesteps(sampling_steps, device=device, shift=shift)
            timesteps = sample_scheduler.timesteps
        elif sample_solver == 'dpm++':
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=pipe.num_train_timesteps,
                shift=1, use_dynamic_shifting=False)
            sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
            timesteps, _ = retrieve_timesteps(sample_scheduler, device=device,
                                              sigmas=sampling_sigmas)
        else:
            raise NotImplementedError("Unsupported solver.")

        # TAFC forward 需要 model.scheduler
        pipe.model.scheduler = sample_scheduler

        latents = noise
        arg_c = {'context': context, 'seq_len': seq_len}
        arg_null = {'context': context_null, 'seq_len': seq_len}

        pipe.model.to(device)
        for _, t in enumerate(tqdm(timesteps, desc="denoise")):
            timestep = torch.stack([t])

            # 记录该步(条件分支)是否会被 TAFC 跳过: cnt 为偶数 = 条件分支
            if getattr(pipe.model, 'enable_tafc', False):
                # 记录条件分支的实际跳过情况: 让 forward 自己决定, 事后读计数器,
                # 而不是在这里重算一遍门控逻辑 (那样必然和 forward 走偏)。
                skipped_before = pipe.model.tafc_even.cache_skip_count
            else:
                skipped_before = None

            noise_pred_cond = pipe.model(latents, t=timestep, **arg_c)[0]
            noise_pred_uncond = pipe.model(latents, t=timestep, **arg_null)[0]
            noise_pred = noise_pred_uncond + guide_scale * (
                noise_pred_cond - noise_pred_uncond)

            if skipped_before is None:
                skip_flags.append(False)
            else:
                skip_flags.append(
                    pipe.model.tafc_even.cache_skip_count > skipped_before)

            velocities.append(noise_pred.detach().clone().float().cpu())

            temp_x0 = sample_scheduler.step(
                noise_pred.unsqueeze(0), t, latents[0].unsqueeze(0),
                return_dict=False, generator=seed_g)[0]
            latents = [temp_x0.squeeze(0)]

    # 重置缓存状态, 便于第二次调用。forward 在轨迹末尾已经 reset 过一次, 这里
    # 兜住提前退出 (比如 solver 报错) 的情况。
    if getattr(pipe.model, 'enable_tafc', False):
        pipe.model.cnt = 0
        pipe.model.tafc_even.reset()
        pipe.model.tafc_odd.reset()

    del latents, noise
    gc.collect()
    torch.cuda.empty_cache()
    return velocities, timesteps.cpu(), skip_flags


# ----------------------------------------------------------------------------- #
#  统计量计算
# ----------------------------------------------------------------------------- #
def compute_velocity_stats(velocities, timesteps):
    ts = timesteps.numpy().astype(np.float64)
    # 排除最后一个 timestep 和对应的 velocity，避免最后一步的异常突起
    stats = {'timesteps': ts[:-1], 'magnitudes': [], 'direction_changes': [],
             'curvatures': []}

    # 只统计前 N-1 个 velocity 的 magnitude
    for v in velocities[:-1]:
        stats['magnitudes'].append(torch.norm(v).item())

    # 只计算到倒数第二对，避免最后一步的异常突起
    for i in range(len(velocities) - 2):
        v1 = velocities[i].flatten()
        v2 = velocities[i + 1].flatten()
        cos_sim = torch.nn.functional.cosine_similarity(
            v1.unsqueeze(0), v2.unsqueeze(0)).item()
        angle = np.degrees(np.arccos(np.clip(cos_sim, -1.0, 1.0)))
        stats['direction_changes'].append(angle)

        v_diff = velocities[i + 1] - velocities[i]
        dt = abs(ts[i + 1] - ts[i])
        curv = torch.norm(v_diff).item() / dt if dt > 0 else torch.norm(v_diff).item()
        stats['curvatures'].append(curv)

    for k in ('magnitudes', 'direction_changes', 'curvatures'):
        stats[k] = np.array(stats[k])
    return stats


# ----------------------------------------------------------------------------- #
#  可视化
# ----------------------------------------------------------------------------- #
C = {
    'mag': '#1f77b4', 'dir': '#d62728', 'curv': '#2ca02c',
    'mean': '#ff7f0e', 'skip': '#9467bd',
    'early': '#ffd5d5', 'late': '#d5f0d5',
}


def _style_ax(ax):
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1.0)
    ax.set_axisbelow(True)
    ax.invert_xaxis()  # t 从大(噪声) 到 小(干净)
    ax.tick_params(labelsize=11, width=1.5, length=5)
    for s in ax.spines.values():
        s.set_linewidth(1.5)
        s.set_edgecolor('#333333')


def _mark_skips(ax, ts_mid, skip_flags):
    """在方向/曲率图上用竖直标记标出 TAFC 跳过的步。"""
    if not any(skip_flags):
        return
    ymin, ymax = ax.get_ylim()
    labeled = False
    # skip_flags 与 velocities 对齐(每步一个); 方向/曲率用相邻两步, 取后一步的标记
    for i in range(1, len(skip_flags)):
        if skip_flags[i] and i - 1 < len(ts_mid):
            ax.axvline(ts_mid[i - 1], color=C['skip'], alpha=0.25,
                       linewidth=2.5, zorder=0,
                       label='TAFC skipped' if not labeled else None)
            labeled = True


def visualize(stats, save_path, skip_flags=None, title_suffix=""):
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
    plt.rcParams['axes.unicode_minus'] = False

    ts = stats['timesteps']
    # ts 已经是 N-1 个元素，ts_mid 需要再取前 N-2 个的中点
    ts_mid = (ts[:-1] + ts[1:]) / 2
    n = len(stats['magnitudes'])

    fig = plt.figure(figsize=(19, 13))
    gs = fig.add_gridspec(3, 2, hspace=0.33, wspace=0.24,
                          top=0.93, bottom=0.06, left=0.07, right=0.97)

    # (a) magnitude
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(ts, stats['magnitudes'], color=C['mag'], lw=3, marker='o', ms=6,
            markeredgecolor='white', markeredgewidth=1.5, label='||v(t)||')
    ax.axvspan(ts[0], ts[n // 3], alpha=0.5, color=C['early'], zorder=0)
    ax.axvspan(ts[2 * n // 3], ts[-1], alpha=0.5, color=C['late'], zorder=0)
    ymax = stats['magnitudes'].max()
    ax.text(ts[n // 6], ymax * 0.9, 'Early\n(High Noise)\nDrastic', ha='center',
            va='center', fontsize=11, weight='bold',
            bbox=dict(boxstyle='round,pad=0.5', fc='white', ec=C['dir'], lw=2))
    ax.text(ts[5 * n // 6], ymax * 0.9, 'Late\n(Low Noise)\nStable', ha='center',
            va='center', fontsize=11, weight='bold',
            bbox=dict(boxstyle='round,pad=0.5', fc='white', ec=C['curv'], lw=2))
    ax.set_xlabel('Timestep (t)', fontsize=14, weight='bold')
    ax.set_ylabel('Velocity Magnitude ||v||', fontsize=14, weight='bold')
    ax.set_title('(a) Velocity Magnitude', fontsize=16, weight='bold', pad=12)
    _style_ax(ax)
    ax.legend(fontsize=12, loc='upper right', framealpha=0.95, shadow=True)

    # (b) magnitude (log)
    ax = fig.add_subplot(gs[0, 1])
    ax.semilogy(ts, stats['magnitudes'], color=C['mag'], lw=3, marker='o', ms=6,
                markeredgecolor='white', markeredgewidth=1.5)
    ax.set_xlabel('Timestep (t)', fontsize=14, weight='bold')
    ax.set_ylabel('||v|| (log scale)', fontsize=14, weight='bold')
    ax.set_title('(b) Velocity Magnitude (Log Scale)', fontsize=16, weight='bold', pad=12)
    _style_ax(ax)

    # (c) direction change
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(ts_mid, stats['direction_changes'], color=C['dir'], lw=3, marker='s',
            ms=6, markeredgecolor='white', markeredgewidth=1.5,
            label='angle(v_t, v_{t+1})')
    mean_ang = stats['direction_changes'].mean()
    ax.axhline(mean_ang, color=C['mean'], ls='--', lw=2.5, alpha=0.8,
               label=f'Mean: {mean_ang:.1f} deg')
    if skip_flags:
        _mark_skips(ax, ts_mid, skip_flags)
    ax.set_xlabel('Timestep (t)', fontsize=14, weight='bold')
    ax.set_ylabel('Direction Change (deg)', fontsize=14, weight='bold')
    ax.set_title('(c) Velocity Direction Change', fontsize=16, weight='bold', pad=12)
    _style_ax(ax)
    ax.legend(fontsize=11, loc='best', framealpha=0.95, shadow=True)

    # (d) curvature
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(ts_mid, stats['curvatures'], color=C['curv'], lw=3, marker='^', ms=7,
            markeredgecolor='white', markeredgewidth=1.5,
            label='||v_{t+1}-v_t|| / dt')
    if skip_flags:
        _mark_skips(ax, ts_mid, skip_flags)
    ax.set_xlabel('Timestep (t)', fontsize=14, weight='bold')
    ax.set_ylabel('Curvature (Acceleration)', fontsize=14, weight='bold')
    ax.set_title('(d) Velocity Curvature', fontsize=16, weight='bold', pad=12)
    _style_ax(ax)
    ax.legend(fontsize=11, loc='best', framealpha=0.95, shadow=True)

    # (e) stage comparison bar
    ax = fig.add_subplot(gs[2, :])
    e, m, l = slice(0, n // 3), slice(n // 3, 2 * n // 3), slice(2 * n // 3, n)
    nd = len(stats['direction_changes'])
    ed, md, ld = slice(0, nd // 3), slice(nd // 3, 2 * nd // 3), slice(2 * nd // 3, nd)
    mag = [stats['magnitudes'][e].mean(), stats['magnitudes'][m].mean(), stats['magnitudes'][l].mean()]
    dirc = [stats['direction_changes'][ed].mean(), stats['direction_changes'][md].mean(), stats['direction_changes'][ld].mean()]
    curv = [stats['curvatures'][ed].mean(), stats['curvatures'][md].mean(), stats['curvatures'][ld].mean()]

    x = np.arange(3)
    w = 0.26
    # 归一化到各自 early 值, 便于同图对比趋势
    mag_n = np.array(mag) / (mag[0] + 1e-9)
    dir_n = np.array(dirc) / (dirc[0] + 1e-9)
    curv_n = np.array(curv) / (curv[0] + 1e-9)
    b1 = ax.bar(x - w, mag_n, w, label='Magnitude (norm.)', color=C['mag'], alpha=0.85, edgecolor='white', lw=2)
    b2 = ax.bar(x, dir_n, w, label='Direction (norm.)', color=C['dir'], alpha=0.85, edgecolor='white', lw=2)
    b3 = ax.bar(x + w, curv_n, w, label='Curvature (norm.)', color=C['curv'], alpha=0.85, edgecolor='white', lw=2)
    for bars, raw in zip((b1, b2, b3), (mag, dirc, curv)):
        for bar, rv in zip(bars, raw):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f'{rv:.3g}', ha='center', va='bottom', fontsize=10, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Early\n(High Noise)', 'Middle', 'Late\n(Low Noise)'],
                       fontsize=13, weight='bold')
    ax.set_ylabel('Normalized to Early Stage', fontsize=14, weight='bold')
    ax.set_title('(e) Stage-wise Comparison (bar label = raw mean)',
                 fontsize=16, weight='bold', pad=12)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=11)
    for s in ax.spines.values():
        s.set_linewidth(1.5)
        s.set_edgecolor('#333333')
    ax.legend(fontsize=12, loc='upper right', ncol=3, framealpha=0.95, shadow=True)

    fig.suptitle(f'Flow Matching Velocity Analysis: Wan T2V{title_suffix}',
                 fontsize=20, weight='bold', y=0.98)
    fig.savefig(save_path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[saved] {save_path}")


def visualize_compare(stats_off, stats_on, save_path, skip_flags_on):
    """TAFC OFF vs ON 三张曲线对比。"""
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
    plt.rcParams['axes.unicode_minus'] = False
    ts = stats_off['timesteps']
    # ts 已经是 N-1 个元素，ts_mid 需要再取前 N-2 个的中点
    ts_mid = (ts[:-1] + ts[1:]) / 2

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    panels = [
        ('magnitudes', ts, 'Velocity Magnitude ||v||', 'o'),
        ('direction_changes', ts_mid, 'Direction Change (deg)', 's'),
        ('curvatures', ts_mid, 'Curvature ||dv||/dt', '^'),
    ]
    for ax, (key, xs, ylab, mk) in zip(axes, panels):
        ax.plot(xs, stats_off[key], color='#1f77b4', lw=2.5, marker=mk, ms=5,
                label='TAFC OFF (full)', alpha=0.9)
        ax.plot(xs, stats_on[key], color='#d62728', lw=2.5, marker=mk, ms=5,
                label='TAFC ON', alpha=0.9, linestyle='--')
        _mark_skips(ax, ts_mid, skip_flags_on)
        ax.set_xlabel('Timestep (t)', fontsize=13, weight='bold')
        ax.set_ylabel(ylab, fontsize=13, weight='bold')
        _style_ax(ax)
        ax.legend(fontsize=11, loc='best', framealpha=0.95)
    fig.suptitle('TAFC ON vs OFF: Velocity Trajectory Comparison',
                 fontsize=18, weight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[saved] {save_path}")


def print_summary(stats, tag=""):
    n = len(stats['magnitudes'])
    nd = len(stats['direction_changes'])
    e, l = slice(0, n // 3), slice(2 * n // 3, n)
    ed, ld = slice(0, nd // 3), slice(2 * nd // 3, nd)

    def red(a, b):
        return (a - b) / (a + 1e-12) * 100

    em, lm = stats['magnitudes'][e].mean(), stats['magnitudes'][l].mean()
    ea, la = stats['direction_changes'][ed].mean(), stats['direction_changes'][ld].mean()
    ec, lc = stats['curvatures'][ed].mean(), stats['curvatures'][ld].mean()

    print("\n" + "=" * 72)
    print(f" Velocity Analysis Summary {tag}")
    print("=" * 72)
    print(f" Magnitude   early={em:.4f}  late={lm:.4f}  reduction={red(em, lm):5.1f}%")
    print(f" Direction   early={ea:.3f} deg  late={la:.3f} deg  reduction={red(ea, la):5.1f}%")
    print(f" Curvature   early={ec:.4f}  late={lc:.4f}  reduction={red(ec, lc):5.1f}%")
    print("=" * 72)


# ----------------------------------------------------------------------------- #
#  TAFC hook (与 tafc_generate._apply_tafc_hooks 一致)
# ----------------------------------------------------------------------------- #
def enable_tafc(pipe, sample_steps, tafc_thresh, tafc_max_cache, use_ret_steps,
                controller_factory=None, calib_steps=3, log_interval=0):
    """把 TAFC 挂到 model 上, 与 tafc_generate._apply_tafc_hooks 保持一致。

    ``controller_factory`` 为 ``None`` 时跑开环 TAFC（可视化默认如此, 这样图上
    的跳过标记只反映预设调度, 不含闭环增益的影响）; 传入一个返回
    ``TAFCController`` 的可调用对象即可打开闭环, 两个 CFG 分支各拿一个独立控制器。
    """
    m = pipe.model.__class__
    m.enable_tafc = True
    m.forward = tafc_forward
    m.cnt = 0
    m.num_steps = sample_steps * 2
    m.tafc_thresh = tafc_thresh
    m.tafc_max_cache = tafc_max_cache
    m.tafc_calib_steps = calib_steps
    m.cache_log_interval = log_interval
    m.is_even = True
    m.last_tafc_diag = None

    # 条件分支与无条件分支各自一条独立的缓存流 + 闭环。
    # 注意: 分支对象持有 residual 张量, 必须放在 instance 上, 不能放到 class 上。
    mk = controller_factory if controller_factory is not None else (lambda: None)
    pipe.model.tafc_even = TAFCBranch("cond", mk())
    pipe.model.tafc_odd = TAFCBranch("uncond", mk())

    if use_ret_steps:
        m.ret_steps = 5 * 2
        m.cutoff_steps = sample_steps * 2
    else:
        m.ret_steps = 1 * 2
        m.cutoff_steps = sample_steps * 2 - 2


def disable_tafc(pipe):
    pipe.model.__class__.enable_tafc = False


def main():
    ap = argparse.ArgumentParser(description="Visualize Wan flow-matching velocity with TAFC")
    ap.add_argument('--ckpt_dir', type=str, required=True)
    ap.add_argument('--task', type=str, default='t2v-1.3B', choices=list(WAN_CONFIGS.keys()))
    ap.add_argument('--size', type=str, default='832*480', choices=list(SIZE_CONFIGS.keys()))
    ap.add_argument('--frame_num', type=int, default=81)
    ap.add_argument('--sample_solver', type=str, default='unipc', choices=['unipc', 'dpm++'])
    ap.add_argument('--sample_steps', type=int, default=50)
    ap.add_argument('--sample_shift', type=float, default=5.0)
    ap.add_argument('--sample_guide_scale', type=float, default=5.0)
    ap.add_argument('--base_seed', type=int, default=42)
    ap.add_argument('--prompt', type=str,
                    default="Two anthropomorphic cats in comfy boxing gear and "
                            "bright gloves fight intensely on a spotlighted stage.")
    ap.add_argument('--compare', action='store_true',
                    help="额外跑一次 TAFC ON 并对比")
    ap.add_argument('--tafc_thresh', type=float, default=0.2,
                    help="TAFC curvature threshold")
    ap.add_argument('--tafc_max_cache', type=float, default=6.0,
                    help="TAFC max consecutive cache budget")
    ap.add_argument('--use_ret_steps', action='store_true', default=False)
    ap.add_argument('--out_prefix', type=str, default='velocity_tafc')
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(levelname)s: %(message)s",
                        handlers=[logging.StreamHandler(stream=sys.stdout)])

    device = 0  # 配合 CUDA_VISIBLE_DEVICES 使用
    size = SIZE_CONFIGS[args.size]
    cfg = WAN_CONFIGS[args.task]

    logging.info(f"Creating WanT2V pipeline from {args.ckpt_dir} ...")
    pipe = wan.WanT2V(config=cfg, checkpoint_dir=args.ckpt_dir, device_id=device,
                      rank=0, t5_fsdp=False, dit_fsdp=False, use_usp=False,
                      t5_cpu=False)

    # ---- Pass 1: TAFC OFF, 完整 velocity 轨迹 ----
    disable_tafc(pipe)
    logging.info("Pass 1/{}: TAFC OFF (full velocity) ...".format(2 if args.compare else 1))
    vels_off, ts, _ = sample_and_record_velocity(
        pipe, args.prompt, size, args.frame_num, args.sample_shift,
        args.sample_solver, args.sample_steps, args.sample_guide_scale,
        args.base_seed, device)
    stats_off = compute_velocity_stats(vels_off, ts)
    print_summary(stats_off, tag="(TAFC OFF)")
    visualize(stats_off, f"{args.out_prefix}_off.png", skip_flags=None,
              title_suffix=" (TAFC OFF)")

    if args.compare:
        # ---- Pass 2: TAFC ON ----
        enable_tafc(pipe, args.sample_steps, args.tafc_thresh,
                   args.tafc_max_cache, args.use_ret_steps)
        logging.info("Pass 2/2: TAFC ON ...")
        vels_on, ts2, skip_flags = sample_and_record_velocity(
            pipe, args.prompt, size, args.frame_num, args.sample_shift,
            args.sample_solver, args.sample_steps, args.sample_guide_scale,
            args.base_seed, device)
        stats_on = compute_velocity_stats(vels_on, ts2)
        print_summary(stats_on, tag="(TAFC ON)")
        n_skip = sum(skip_flags)
        logging.info(f"TAFC skipped {n_skip}/{len(skip_flags)} steps "
                     f"({100*n_skip/max(1,len(skip_flags)):.1f}%)")
        visualize(stats_on, f"{args.out_prefix}_on.png", skip_flags=skip_flags,
                  title_suffix=f" (TAFC ON, {n_skip} skipped)")
        visualize_compare(stats_off, stats_on, f"{args.out_prefix}_compare.png",
                          skip_flags)

    logging.info("Done.")


if __name__ == "__main__":
    main()
