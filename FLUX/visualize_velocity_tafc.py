#!/usr/bin/env python3
"""
在 TAFC 的完整 FLUX pipeline 上统计并可视化每个 timestep 的 velocity 特性。

统计量:
  - magnitude:  ||v_t||                                    (velocity 大小)
  - direction:  angle(v_t, v_{t+1})                        (相邻方向夹角, 度)
  - curvature:  ||v_{t+1} - v_t|| / dt                     (曲率/加速度)

与 Wan2.1 的差别: FLUX-dev 用的是蒸馏后的 embedded guidance, 每个 timestep 只有
一次 transformer 前向 (没有 cond/uncond 双分支), 所以 transformer 的输出本身就是
flow matching 的 velocity, 不需要再做 CFG 组合。

同时支持对比 TAFC ON / OFF 两种模式下的 velocity 轨迹, 用于验证:
  "去噪前期 velocity 剧烈变化, 后期趋于稳定" 这一猜想, 并观察 TAFC
  跳过的步骤是否恰好落在 velocity 的稳定区间。

用法示例:
  # 单独分析 (TAFC OFF, 真实完整 velocity 轨迹)
  CUDA_VISIBLE_DEVICES=6 python visualize_velocity_tafc.py --num_inference_steps 50

  # 对比 TAFC ON vs OFF
  CUDA_VISIBLE_DEVICES=6 python visualize_velocity_tafc.py \
      --num_inference_steps 50 --compare --tafc_thresh 0.3 --use_ret_steps
"""
import argparse
import gc
import logging
import os
import sys
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from diffusers.models import FluxTransformer2DModel

# tafc_generate 在 import 时就把 FluxTransformer2DModel.forward 换成了 tafc_forward,
# 所以必须先把原始 forward 存下来, 否则 TAFC OFF 那一趟拿不到 baseline 轨迹。
_STOCK_FLUX_FORWARD = FluxTransformer2DModel.forward

from flux_generate import build_pipeline, now_str, pipe_kwargs, resolve_model  # noqa: E402
from tafc_generate import configure_tafc, tafc_forward  # noqa: E402
from util_tafc import TAFCController  # noqa: E402

# ----------------------------------------------------------------------------- #
#  velocity 采样: 在真实 pipeline 中跑一次去噪, 记录每一步的 velocity
# ----------------------------------------------------------------------------- #
def sample_and_record_velocity(pipe, args, model_name, num_steps, prompt, device,
                               image_path=None):
    """
    跑一次完整的 FLUX 去噪, 用 forward hook 记录 transformer 每步输出的 velocity。

    FLUX-dev 每个 timestep 只调用 transformer 一次, 输出即 flow matching velocity,
    所以 hook 到的张量不需要再做 CFG 组合。

    返回:
      velocities: List[Tensor]  每步 velocity (CPU, float32, 已 flatten)
      timesteps:  Tensor        对应的离散 timestep (1000 尺度, 递减)
      skip_flags: List[bool]    该步 TAFC 是否跳过了 transformer blocks
    """
    tr = pipe.transformer
    velocities, skip_flags = [], []
    state = {'prev_skips': 0}

    def hook(module, inputs, output):
        vel = output[0] if isinstance(output, (tuple, list)) else output.sample
        velocities.append(vel.detach().float().flatten().cpu())

        if not getattr(module, 'enable_tafc', False):
            skip_flags.append(False)
            return
        # cache_skip_count 只在缓存步 +1, 所以增量就是"这一步被跳过了"。
        # 轨迹末尾 tafc_forward 把计数器清零 (cnt 也归 0), 那一步改用它留下的
        # last_cached_steps 快照倒推。
        if int(getattr(module, 'cnt', 0)) == 0:
            total = getattr(module, 'last_cached_steps', None)
            total = state['prev_skips'] if total is None else int(total)
            skip_flags.append(total > state['prev_skips'])
        else:
            cur = int(getattr(module, 'cache_skip_count', 0))
            skip_flags.append(cur > state['prev_skips'])
            state['prev_skips'] = cur

    handle = tr.register_forward_hook(hook)
    try:
        kwargs = pipe_kwargs(args, model_name, num_steps)
        generator = torch.Generator(device=device).manual_seed(args.seed)
        out = pipe(prompt=prompt, generator=generator, **kwargs)
    finally:
        handle.remove()

    if image_path is not None:
        out.images[0].save(image_path)
        print(f"[saved] {image_path}")

    ts = getattr(pipe.scheduler, 'timesteps', None)
    if ts is None or len(ts) != len(velocities):
        # 兜底: 均匀 1000 -> 0, 只影响曲率的 dt 归一化
        logging.warning("scheduler.timesteps 与记录步数不一致, 退化为均匀间隔")
        ts = torch.linspace(1000.0, 0.0, len(velocities) + 1)[:len(velocities)]
    timesteps = ts.detach().float().cpu()

    del out
    gc.collect()
    torch.cuda.empty_cache()
    return velocities, timesteps, skip_flags


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
        v1, v2 = velocities[i], velocities[i + 1]
        cos_sim = torch.nn.functional.cosine_similarity(
            v1.unsqueeze(0), v2.unsqueeze(0)).item()
        angle = np.degrees(np.arccos(np.clip(cos_sim, -1.0, 1.0)))
        stats['direction_changes'].append(angle)

        v_diff = v2 - v1
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
    if not skip_flags or not any(skip_flags):
        return
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

    fig.suptitle(f'Flow Matching Velocity Analysis: FLUX{title_suffix}',
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
#  TAFC 控制
# ----------------------------------------------------------------------------- #
def enable_tafc(pipe, args):
    """启用 TAFC, 复用 tafc_generate.configure_tafc 的逻辑。"""
    FluxTransformer2DModel.forward = tafc_forward
    configure_tafc(pipe, args.tafc_thresh, args.tafc_max_cache,
                   args.num_inference_steps, args.use_ret_steps)


def disable_tafc(pipe):
    """禁用 TAFC, 恢复原始 forward。"""
    FluxTransformer2DModel.forward = _STOCK_FLUX_FORWARD
    pipe.transformer.enable_tafc = False


def main():
    ap = argparse.ArgumentParser(description="Visualize FLUX flow-matching velocity with TAFC")
    ap.add_argument('--model_name', type=str, default='flux-dev',
                    choices=['flux-dev', 'flux-schnell'])
    ap.add_argument('--model_id', type=str, default=None,
                    help="Explicit HF model id; overrides --model_name")
    ap.add_argument('--width', type=int, default=1024)
    ap.add_argument('--height', type=int, default=1024)
    ap.add_argument('--num_inference_steps', type=int, default=50)
    ap.add_argument('--guidance', type=float, default=3.5)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--dtype', type=str, default='bf16', choices=['bf16', 'fp16'])
    ap.add_argument('--offload', action='store_true', help='Use CPU offload')
    ap.add_argument('--prompt', type=str,
                    default="Bzaseball galove.,Misspellings")
    ap.add_argument('--compare', action='store_true',
                    help="额外跑一次 TAFC ON 并对比")
    ap.add_argument('--tafc_thresh', type=float, default=0.3,
                    help="TAFC curvature threshold")
    ap.add_argument('--tafc_max_cache', type=float, default=1.0,
                    help="TAFC max consecutive cache budget")
    ap.add_argument('--use_ret_steps', action='store_true', default=False)
    ap.add_argument('--out_prefix', type=str, default='velocity_tafc')
    ap.add_argument('--save_images', action='store_true',
                    help="保存生成的图像 (默认只保存统计图)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(levelname)s: %(message)s",
                        handlers=[logging.StreamHandler(stream=sys.stdout)])

    model_id, model_name = resolve_model(args)

    logging.info(f"Building FLUX pipeline: {model_id} ...")
    pipe, device, torch_dtype = build_pipeline(args, model_id)

    # ---- Pass 1: TAFC OFF, 完整 velocity 轨迹 ----
    disable_tafc(pipe)
    logging.info("Pass 1/{}: TAFC OFF (full velocity) ...".format(2 if args.compare else 1))
    img_path = f"{args.out_prefix}_off.png" if args.save_images else None
    vels_off, ts, _ = sample_and_record_velocity(
        pipe, args, model_name, args.num_inference_steps, args.prompt, device,
        image_path=img_path)
    stats_off = compute_velocity_stats(vels_off, ts)
    print_summary(stats_off, tag="(TAFC OFF)")
    visualize(stats_off, f"{args.out_prefix}_stats_off.png", skip_flags=None,
              title_suffix=" (TAFC OFF)")

    if args.compare:
        # ---- Pass 2: TAFC ON ----
        enable_tafc(pipe, args)
        logging.info("Pass 2/2: TAFC ON ...")
        img_path = f"{args.out_prefix}_on.png" if args.save_images else None
        vels_on, ts2, skip_flags = sample_and_record_velocity(
            pipe, args, model_name, args.num_inference_steps, args.prompt, device,
            image_path=img_path)
        stats_on = compute_velocity_stats(vels_on, ts2)
        print_summary(stats_on, tag="(TAFC ON)")
        n_skip = sum(skip_flags)
        logging.info(f"TAFC skipped {n_skip}/{len(skip_flags)} steps "
                     f"({100*n_skip/max(1,len(skip_flags)):.1f}%)")
        visualize(stats_on, f"{args.out_prefix}_stats_on.png", skip_flags=skip_flags,
                  title_suffix=f" (TAFC ON, {n_skip} skipped)")
        visualize_compare(stats_off, stats_on, f"{args.out_prefix}_compare.png",
                          skip_flags)

    logging.info("Done.")


if __name__ == "__main__":
    main()
