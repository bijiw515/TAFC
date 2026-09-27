#!/usr/bin/env python3
"""
TAFC 工作机制完整可视化脚本 - 展示与 SeaCache 的对比优势

这个脚本创建一个多面板可视化，展示：
1. 缓存决策时间线（何时计算/何时缓存）
2. 轨迹曲率与阈值的动态变化
3. PID控制器的闭环反馈行为
4. 局部截断误差（LTE）预算管理
5. 与SeaCache的性能对比
6. 生成质量对比（通过与完整计算的误差）

设计亮点：
- 实时展示TAFC的自适应决策过程
- 可视化PID控制器如何根据实际误差调整缓存策略
- 对比TAFC与SeaCache在相同计算预算下的表现
- 展示TAFC的物理启发设计（曲率、误差估计）

用法示例：
    # 基础可视化（TAFC vs SeaCache，匹配NFE）
    CUDA_VISIBLE_DEVICES=6 python visualize_tafc_mechanism.py \
        --num_inference_steps 50 --seed 42 --match_nfe

    # 高分辨率，展示PID控制细节
    CUDA_VISIBLE_DEVICES=6 python visualize_tafc_mechanism.py \
        --num_inference_steps 50 --width 1024 --height 1024 \
        --tafc_thresh 0.04 --seed 123 --out_prefix tafc_comparison

    # 不匹配NFE，直接对比
    CUDA_VISIBLE_DEVICES=6 python visualize_tafc_mechanism.py \
        --num_inference_steps 50 --no_match_nfe --seacache_thresh 0.3
"""

import argparse
import gc
import logging
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings('ignore')

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec

from diffusers.models import FluxTransformer2DModel

# 保存原始 forward 用于完整计算参考
_STOCK_FLUX_FORWARD = FluxTransformer2DModel.forward

from flux_generate import build_pipeline, pipe_kwargs, resolve_model, now_str
from tafc_generate import configure_tafc, tafc_forward
from seacache_generate import configure_seacache, seacache_forward
from util_tafc import (
    TAFCController,
    compute_trajectory_curvature,
    adaptive_threshold_schedule,
    adaptive_max_cache_schedule
)
from util_seacache import rel_l1


# ============================================================================
# 轨迹记录器
# ============================================================================

class MethodRecorder:
    """记录缓存方法的完整运行轨迹"""

    def __init__(self, method_name: str):
        self.method_name = method_name
        self.reset()

    def reset(self):
        """重置所有记录"""
        self.steps = []
        self.computed_flags = []  # True = 计算, False = 缓存
        self.curvatures = []
        self.thresholds = []
        self.normalized_times = []
        self.cache_counts = []
        self.total_cached = 0

        # TAFC特有
        self.pid_gains = []
        self.lte_currents = []
        self.lte_budgets = []
        self.curvature_rates = []
        self.brake_factors = []
        self.veto_reasons = []

        # 误差记录（与完整计算对比）
        self.errors_vs_full = []

    def record_step(
        self,
        step_idx: int,
        computed: bool,
        normalized_time: float = None,
        curvature: float = None,
        threshold: float = None,
        pid_gain: float = None,
        lte_current: float = None,
        lte_budget: float = None,
        curvature_rate: float = None,
        brake_factor: float = None,
        veto_reason: str = None,
        error_vs_full: float = None,
    ):
        """记录一步的信息"""
        self.steps.append(step_idx)
        self.computed_flags.append(computed)
        self.normalized_times.append(normalized_time)
        self.curvatures.append(curvature)
        self.thresholds.append(threshold)
        self.pid_gains.append(pid_gain)
        self.lte_currents.append(lte_current)
        self.lte_budgets.append(lte_budget)
        self.curvature_rates.append(curvature_rate)
        self.brake_factors.append(brake_factor)
        self.veto_reasons.append(veto_reason if veto_reason else "")
        self.errors_vs_full.append(error_vs_full)

        if not computed:
            self.total_cached += 1
        self.cache_counts.append(self.total_cached)

    def get_cache_rate(self) -> float:
        """获取缓存率"""
        if len(self.steps) == 0:
            return 0.0
        return self.total_cached / len(self.steps) * 100

    def get_speedup(self) -> float:
        """估算加速比"""
        if len(self.steps) == 0:
            return 1.0
        computed_steps = sum(self.computed_flags)
        if computed_steps == 0:
            return float('inf')
        return len(self.steps) / computed_steps


# 全局记录器
_tafc_recorder = MethodRecorder("TAFC")
_seacache_recorder = MethodRecorder("SeaCache")
_full_velocities = []  # 完整计算的velocity参考


# ============================================================================
# 插桩的 TAFC forward
# ============================================================================

def tafc_forward_instrumented(
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
) -> Any:
    """TAFC forward with instrumentation for visualization"""

    # 调用原始TAFC forward
    output = tafc_forward(
        self,
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        pooled_projections=pooled_projections,
        timestep=timestep,
        img_ids=img_ids,
        txt_ids=txt_ids,
        guidance=guidance,
        joint_attention_kwargs=joint_attention_kwargs,
        controlnet_block_samples=controlnet_block_samples,
        controlnet_single_block_samples=controlnet_single_block_samples,
        return_dict=return_dict,
        controlnet_blocks_repeat=controlnet_blocks_repeat,
    )

    # 提取TAFC状态进行记录
    if hasattr(self, 'tafc_branch') and self.tafc_branch is not None:
        branch = self.tafc_branch
        ctrl = branch.ctrl

        step_idx = int(getattr(self, 'cnt', 0))
        computed = branch.last_compute_step == (step_idx - 1)

        _tafc_recorder.record_step(
            step_idx=step_idx,
            computed=computed,
            normalized_time=getattr(self, 'tafc_normalized_time', None),
            curvature=branch.last_curvature,
            threshold=branch.last_thresh,
            pid_gain=ctrl.gain if ctrl else None,
            lte_current=branch.last_e_current,
            lte_budget=branch.last_e_budget,
            curvature_rate=branch.curvature_rate if hasattr(branch, 'curvature_rate') else None,
            brake_factor=ctrl.brake(branch.curvature_rate) if ctrl and hasattr(branch, 'curvature_rate') else None,
            veto_reason=branch.vetoes.get(step_idx),
        )

    return output


# ============================================================================
# 插桩的 SeaCache forward
# ============================================================================

def seacache_forward_instrumented(
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
) -> Any:
    """SeaCache forward with instrumentation for visualization"""

    # 调用原始SeaCache forward
    output = seacache_forward(
        self,
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        pooled_projections=pooled_projections,
        timestep=timestep,
        img_ids=img_ids,
        txt_ids=txt_ids,
        guidance=guidance,
        joint_attention_kwargs=joint_attention_kwargs,
        controlnet_block_samples=controlnet_block_samples,
        controlnet_single_block_samples=controlnet_single_block_samples,
        return_dict=return_dict,
        controlnet_blocks_repeat=controlnet_blocks_repeat,
    )

    # 提取SeaCache状态
    step_idx = int(getattr(self, 'cnt', 0))
    computed = getattr(self, 'seacache_last_computed', True)

    _seacache_recorder.record_step(
        step_idx=step_idx,
        computed=computed,
        normalized_time=step_idx / getattr(self, 'tafc_num_steps', step_idx + 1),
    )

    return output


# ============================================================================
# 完整计算参考（用于误差计算）
# ============================================================================

def record_full_computation(pipe, args, model_name, prompt, device):
    """运行完整计算，记录每步velocity作为参考"""
    print("\n[Phase 1/3] 运行完整计算作为参考...")

    global _full_velocities
    _full_velocities = []

    # 临时hook记录velocity
    def hook(module, inputs, output):
        vel = output.sample if hasattr(output, 'sample') else output[0]
        _full_velocities.append(vel.detach().clone())

    tr = pipe.transformer
    FluxTransformer2DModel.forward = _STOCK_FLUX_FORWARD
    handle = tr.register_forward_hook(hook)

    try:
        kwargs = pipe_kwargs(args, model_name, args.num_inference_steps)
        generator = torch.Generator(device=device).manual_seed(args.seed)
        pipe(prompt=prompt, generator=generator, **kwargs)
    finally:
        handle.remove()

    print(f"   ✓ 记录了 {len(_full_velocities)} 步完整计算")


# ============================================================================
# 运行TAFC
# ============================================================================

def run_tafc(pipe, args, model_name, prompt, device):
    """运行TAFC并记录轨迹"""
    print("\n[Phase 2/3] 运行 TAFC...")

    _tafc_recorder.reset()

    # 创建PID控制器
    controller = TAFCController(
        kp=args.tafc_kp,
        ki=args.tafc_ki,
        kd=args.tafc_kd,
        gain_min=args.tafc_gain_min,
        gain_max=args.tafc_gain_max,
        target=None,  # auto-calibrate
        auto_tol=args.tafc_auto_tol,
        reject=args.tafc_reject,
    )

    # 配置TAFC - 使用正确的参数名
    configure_tafc(
        pipe,
        thresh=args.tafc_thresh,
        max_cache=args.tafc_max_cache,
        num_steps=args.num_inference_steps,
        use_ret_steps=args.use_ret_steps,
        log_interval=10,
        controller=controller,
        calib_steps=args.tafc_calib_steps,
    )

    # 使用插桩的forward
    FluxTransformer2DModel.forward = tafc_forward_instrumented

    kwargs = pipe_kwargs(args, model_name, args.num_inference_steps)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    out = pipe(prompt=prompt, generator=generator, **kwargs)

    cache_rate = _tafc_recorder.get_cache_rate()
    speedup = _tafc_recorder.get_speedup()
    print(f"   ✓ TAFC 完成: 缓存率 {cache_rate:.1f}%, 加速比 {speedup:.2f}x")

    # 保存图像
    tafc_img_path = f"{args.out_prefix}_tafc.png"
    out.images[0].save(tafc_img_path)
    print(f"   ✓ 图像保存到: {tafc_img_path}")

    return out.images[0]


# ============================================================================
# 运行SeaCache
# ============================================================================

def run_seacache(pipe, args, model_name, prompt, device, target_nfe=None):
    """运行SeaCache并记录轨迹"""
    print("\n[Phase 3/3] 运行 SeaCache...")

    _seacache_recorder.reset()

    # 如果需要匹配NFE，搜索合适的阈值
    if args.match_nfe and target_nfe is not None:
        print(f"   正在搜索SeaCache阈值以匹配NFE≈{target_nfe}...")
        seacache_thresh = search_seacache_threshold(
            pipe, args, model_name, prompt, device, target_nfe, args.nfe_tolerance
        )
        print(f"   ✓ 找到阈值: {seacache_thresh:.4f}")
    else:
        seacache_thresh = args.seacache_thresh
        print(f"   使用固定阈值: {seacache_thresh}")

    # 配置SeaCache - 使用正确的参数名
    configure_seacache(
        pipe,
        thresh=seacache_thresh,
        num_steps=args.num_inference_steps,
    )

    # 使用插桩的forward
    FluxTransformer2DModel.forward = seacache_forward_instrumented

    kwargs = pipe_kwargs(args, model_name, args.num_inference_steps)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    out = pipe(prompt=prompt, generator=generator, **kwargs)

    cache_rate = _seacache_recorder.get_cache_rate()
    speedup = _seacache_recorder.get_speedup()
    print(f"   ✓ SeaCache 完成: 缓存率 {cache_rate:.1f}%, 加速比 {speedup:.2f}x")

    # 保存图像
    seacache_img_path = f"{args.out_prefix}_seacache.png"
    out.images[0].save(seacache_img_path)
    print(f"   ✓ 图像保存到: {seacache_img_path}")

    return out.images[0]


def search_seacache_threshold(pipe, args, model_name, prompt, device, target_nfe, tolerance):
    """二分搜索SeaCache阈值以匹配目标NFE"""

    low, high = args.seacache_search_low, args.seacache_search_high
    best_thresh = args.seacache_thresh
    best_diff = float('inf')

    for iteration in range(args.seacache_search_iters):
        mid = (low + high) / 2.0

        # 测试这个阈值
        configure_seacache(pipe, thresh=mid, num_steps=args.num_inference_steps)
        FluxTransformer2DModel.forward = seacache_forward

        # 计数器
        compute_count = [0]
        def count_hook(module, inputs, output):
            if getattr(module, 'seacache_last_computed', True):
                compute_count[0] += 1

        tr = pipe.transformer
        handle = tr.register_forward_hook(count_hook)
        try:
            kwargs = pipe_kwargs(args, model_name, args.num_inference_steps)
            generator = torch.Generator(device=device).manual_seed(args.seed + 1000)
            pipe(prompt=prompt, generator=generator, **kwargs)
        finally:
            handle.remove()

        nfe = compute_count[0]
        diff = abs(nfe - target_nfe)

        if diff < best_diff:
            best_diff = diff
            best_thresh = mid

        if diff <= tolerance:
            break

        if nfe < target_nfe:
            # 需要更多计算，降低阈值
            high = mid
        else:
            # 需要更少计算，提高阈值
            low = mid

    return best_thresh


# ============================================================================
# 可视化函数
# ============================================================================

def create_visualization(args):
    """创建完整的对比可视化"""
    print("\n[可视化] 创建多面板对比图...")

    # 创建大图：3行2列
    fig = plt.figure(figsize=(16, 18))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.25)

    # Panel 1: 缓存决策时间线（TAFC）
    ax1 = fig.add_subplot(gs[0, 0])
    plot_cache_timeline(ax1, _tafc_recorder, "TAFC")

    # Panel 2: 缓存决策时间线（SeaCache）
    ax2 = fig.add_subplot(gs[0, 1])
    plot_cache_timeline(ax2, _seacache_recorder, "SeaCache")

    # Panel 3: TAFC 曲率与阈值
    ax3 = fig.add_subplot(gs[1, 0])
    plot_curvature_threshold(ax3, _tafc_recorder)

    # Panel 4: TAFC PID 控制
    ax4 = fig.add_subplot(gs[1, 1])
    plot_pid_control(ax4, _tafc_recorder)

    # Panel 5: 局部截断误差（LTE）
    ax5 = fig.add_subplot(gs[2, 0])
    plot_lte_budget(ax5, _tafc_recorder)

    # Panel 6: 性能对比总结
    ax6 = fig.add_subplot(gs[2, 1])
    plot_performance_summary(ax6, _tafc_recorder, _seacache_recorder, args)

    # 总标题
    fig.suptitle(
        f'TAFC vs SeaCache 机制对比 | {args.num_inference_steps} 步 | Seed {args.seed}',
        fontsize=16, fontweight='bold', y=0.995
    )

    # 保存
    output_path = f"{args.out_prefix}_mechanism_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ 可视化保存到: {output_path}")
    plt.close()


def plot_cache_timeline(ax, recorder: MethodRecorder, title: str):
    """绘制缓存决策时间线"""
    steps = np.array(recorder.steps)
    computed = np.array(recorder.computed_flags)

    if len(steps) == 0:
        ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f"{title} 缓存时间线")
        return

    # 使用颜色标记计算（红）和缓存（绿）
    colors = ['#e74c3c' if c else '#2ecc71' for c in computed]

    ax.scatter(steps, [0]*len(steps), c=colors, s=100, alpha=0.7, marker='|', linewidths=3)

    # 添加区域标记
    for i, (step, comp) in enumerate(zip(steps, computed)):
        if comp:
            ax.axvspan(step-0.5, step+0.5, alpha=0.15, color='red')

    ax.set_xlim(-1, max(steps)+1)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel('推理步骤', fontsize=11)
    ax.set_yticks([])
    ax.set_title(f"{title} 缓存决策时间线", fontsize=12, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)

    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', alpha=0.7, label=f'计算 ({sum(computed)} 步)'),
        Patch(facecolor='#2ecc71', alpha=0.7, label=f'缓存 ({len(steps)-sum(computed)} 步)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)


def plot_curvature_threshold(ax, recorder: MethodRecorder):
    """绘制轨迹曲率与阈值"""
    steps = np.array(recorder.steps)
    curvatures = np.array([c if c is not None else np.nan for c in recorder.curvatures])
    thresholds = np.array([t if t is not None else np.nan for t in recorder.thresholds])
    computed = np.array(recorder.computed_flags)

    if len(steps) == 0:
        ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
        return

    # 绘制曲率
    ax.plot(steps, curvatures, 'o-', color='#3498db', linewidth=2, markersize=4,
            label='轨迹曲率', alpha=0.8)

    # 绘制阈值
    ax.plot(steps, thresholds, 's-', color='#e67e22', linewidth=2, markersize=3,
            label='动态阈值 (PID调整)', alpha=0.8)

    # 标记计算步骤
    compute_steps = steps[computed]
    compute_curvs = curvatures[computed]
    ax.scatter(compute_steps, compute_curvs, c='red', s=80, marker='x',
               linewidths=2, label='触发计算', zorder=5)

    ax.set_xlabel('推理步骤', fontsize=11)
    ax.set_ylabel('曲率 / 阈值', fontsize=11)
    ax.set_title('TAFC 曲率监控与自适应阈值', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(steps.min()-1, steps.max()+1)


def plot_pid_control(ax, recorder: MethodRecorder):
    """绘制PID控制器行为"""
    steps = np.array(recorder.steps)
    gains = np.array([g if g is not None else np.nan for g in recorder.pid_gains])
    computed = np.array(recorder.computed_flags)

    if len(steps) == 0 or np.all(np.isnan(gains)):
        ax.text(0.5, 0.5, 'No PID Data', ha='center', va='center', transform=ax.transAxes)
        return

    # 绘制增益
    ax.plot(steps, gains, 'o-', color='#9b59b6', linewidth=2.5, markersize=5,
            label='PID 增益', alpha=0.8)

    # 标记增益范围
    gain_min = gains[~np.isnan(gains)].min() if not np.all(np.isnan(gains)) else 0.3
    gain_max = gains[~np.isnan(gains)].max() if not np.all(np.isnan(gains)) else 3.0
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, label='基准增益')
    ax.fill_between(steps, gain_min, gain_max, alpha=0.1, color='purple')

    # 标记计算步骤
    compute_steps = steps[computed]
    compute_gains = gains[computed]
    ax.scatter(compute_steps, compute_gains, c='red', s=60, marker='D',
               linewidths=1.5, label='测量点', zorder=5, alpha=0.7)

    ax.set_xlabel('推理步骤', fontsize=11)
    ax.set_ylabel('增益倍数', fontsize=11)
    ax.set_title('PID 闭环控制增益', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(steps.min()-1, steps.max()+1)


def plot_lte_budget(ax, recorder: MethodRecorder):
    """绘制局部截断误差预算"""
    steps = np.array(recorder.steps)
    lte_current = np.array([e if e is not None else np.nan for e in recorder.lte_currents])
    lte_budget = np.array([e if e is not None else np.nan for e in recorder.lte_budgets])
    computed = np.array(recorder.computed_flags)

    if len(steps) == 0:
        ax.text(0.5, 0.5, 'No LTE Data', ha='center', va='center', transform=ax.transAxes)
        return

    # 绘制当前误差估计
    valid_mask = ~np.isnan(lte_current)
    if valid_mask.any():
        ax.plot(steps[valid_mask], lte_current[valid_mask], 'o-',
                color='#e74c3c', linewidth=2, markersize=4,
                label='当前误差估计 (先验)', alpha=0.8)

    # 绘制预算
    valid_mask = ~np.isnan(lte_budget)
    if valid_mask.any():
        ax.plot(steps[valid_mask], lte_budget[valid_mask], 's-',
                color='#27ae60', linewidth=2, markersize=3,
                label='允许误差预算', alpha=0.8)

    # 填充安全区域
    if valid_mask.any():
        ax.fill_between(steps[valid_mask], 0, lte_budget[valid_mask],
                        alpha=0.15, color='green', label='安全区')

    # 标记超出预算的点（触发计算）
    exceeded = (lte_current > lte_budget) & computed
    if exceeded.any():
        ax.scatter(steps[exceeded], lte_current[exceeded],
                  c='darkred', s=100, marker='x', linewidths=2.5,
                  label='超预算→计算', zorder=5)

    ax.set_xlabel('推理步骤', fontsize=11)
    ax.set_ylabel('归一化误差', fontsize=11)
    ax.set_title('局部截断误差 (LTE) 预算管理', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(steps.min()-1, steps.max()+1)
    if valid_mask.any():
        ax.set_ylim(bottom=0)


def plot_performance_summary(ax, tafc_rec: MethodRecorder, sea_rec: MethodRecorder, args):
    """绘制性能对比总结"""
    ax.axis('off')

    # 标题
    ax.text(0.5, 0.95, '性能对比总结', ha='center', va='top',
            fontsize=14, fontweight='bold', transform=ax.transAxes)

    # 数据
    tafc_cache_rate = tafc_rec.get_cache_rate()
    tafc_speedup = tafc_rec.get_speedup()
    tafc_nfe = sum(tafc_rec.computed_flags)

    sea_cache_rate = sea_rec.get_cache_rate()
    sea_speedup = sea_rec.get_speedup()
    sea_nfe = sum(sea_rec.computed_flags)

    # 绘制表格
    y_start = 0.80
    row_height = 0.08

    headers = ['指标', 'TAFC', 'SeaCache', '优势']
    col_widths = [0.3, 0.2, 0.2, 0.3]
    col_x = [0.05, 0.35, 0.55, 0.75]

    # 表头
    for i, (header, x) in enumerate(zip(headers, col_x)):
        ax.text(x, y_start, header, ha='left', va='top', fontweight='bold',
                fontsize=11, transform=ax.transAxes,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.5))

    # 数据行
    rows = [
        ('缓存率', f'{tafc_cache_rate:.1f}%', f'{sea_cache_rate:.1f}%',
         f'+{tafc_cache_rate-sea_cache_rate:.1f}%' if tafc_cache_rate > sea_cache_rate else f'{tafc_cache_rate-sea_cache_rate:.1f}%'),
        ('加速比', f'{tafc_speedup:.2f}x', f'{sea_speedup:.2f}x',
         f'+{((tafc_speedup/sea_speedup-1)*100):.1f}%' if tafc_speedup > sea_speedup else f'{((tafc_speedup/sea_speedup-1)*100):.1f}%'),
        ('NFE', f'{tafc_nfe}', f'{sea_nfe}',
         f'{tafc_nfe-sea_nfe:+d}' if args.match_nfe else 'N/A'),
    ]

    for row_idx, row_data in enumerate(rows):
        y_pos = y_start - (row_idx + 1) * row_height
        for col_idx, (text, x) in enumerate(zip(row_data, col_x)):
            color = 'black'
            weight = 'normal'
            # 高亮优势列
            if col_idx == 3:
                if '+' in text or (row_idx == 2 and '-' in text and text != 'N/A'):
                    color = 'green'
                    weight = 'bold'
                elif '-' in text and text != 'N/A':
                    color = 'red'

            ax.text(x, y_pos, text, ha='left', va='top',
                   fontsize=10, color=color, fontweight=weight,
                   transform=ax.transAxes)

    # 关键优势总结
    y_summary = y_start - len(rows) * row_height - 0.12
    ax.text(0.05, y_summary, 'TAFC 核心优势:', ha='left', va='top',
            fontsize=12, fontweight='bold', transform=ax.transAxes)

    advantages = [
        '✓ 自适应阈值：PID控制器根据实际误差动态调整',
        '✓ 物理启发：基于轨迹曲率和误差传播理论',
        '✓ 多重门控：曲率、LTE、连续缓存上限三重保护',
        '✓ 一阶外推：比零阶保持更准确的缓存预测',
        f'✓ {tafc_speedup:.2f}x 加速 @ {tafc_cache_rate:.1f}% 缓存率',
    ]

    for idx, adv in enumerate(advantages):
        y_adv = y_summary - (idx + 1) * 0.06
        ax.text(0.08, y_adv, adv, ha='left', va='top',
               fontsize=9, transform=ax.transAxes, color='darkgreen')

    # 配置信息
    y_config = 0.08
    config_text = (
        f"配置: {args.num_inference_steps} 步 | "
        f"TAFC阈值={args.tafc_thresh} | "
        f"PID=(kp={args.tafc_kp}, ki={args.tafc_ki}, kd={args.tafc_kd}) | "
        f"Seed={args.seed}"
    )
    ax.text(0.5, y_config, config_text, ha='center', va='bottom',
           fontsize=8, style='italic', color='gray', transform=ax.transAxes)


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="可视化 TAFC 工作机制并与 SeaCache 对比"
    )

    # 模型参数
    parser.add_argument('--model_name', type=str, default='flux-dev',
                       choices=['flux-dev', 'flux-schnell'])
    parser.add_argument('--model_id', type=str, default=None)
    parser.add_argument('--width', type=int, default=1024)
    parser.add_argument('--height', type=int, default=1024)
    parser.add_argument('--num_inference_steps', type=int, default=50)
    parser.add_argument('--guidance', type=float, default=3.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dtype', type=str, default='bf16', choices=['bf16', 'fp16'])
    parser.add_argument('--offload', action='store_true')
    parser.add_argument('--prompt', type=str,
                       default="Bzaseball galove.,Misspellings")

    # TAFC参数
    parser.add_argument('--tafc_thresh', type=float, default=0.04)
    parser.add_argument('--tafc_max_cache', type=int, default=1)
    parser.add_argument('--tafc_calib_steps', type=int, default=3)
    parser.add_argument('--use_ret_steps', action='store_true', default=False)

    # PID参数
    parser.add_argument('--tafc_kp', type=float, default=0.4)
    parser.add_argument('--tafc_ki', type=float, default=0.05)
    parser.add_argument('--tafc_kd', type=float, default=0.2)
    parser.add_argument('--tafc_gain_min', type=float, default=0.3)
    parser.add_argument('--tafc_gain_max', type=float, default=3.0)
    parser.add_argument('--tafc_auto_tol', type=float, default=2.5)
    parser.add_argument('--tafc_reject', type=float, default=3.0)

    # SeaCache参数
    parser.add_argument('--seacache_thresh', type=float, default=0.3)
    parser.add_argument('--seacache_search_low', type=float, default=0.01)
    parser.add_argument('--seacache_search_high', type=float, default=1.5)
    parser.add_argument('--seacache_search_iters', type=int, default=10)
    parser.add_argument('--nfe_tolerance', type=int, default=1)

    # NFE匹配
    match_group = parser.add_mutually_exclusive_group()
    match_group.add_argument('--match_nfe', dest='match_nfe', action='store_true',
                            help="搜索SeaCache阈值以匹配TAFC的NFE（默认）")
    match_group.add_argument('--no_match_nfe', dest='match_nfe', action='store_false',
                            help="直接使用--seacache_thresh，不进行NFE匹配")
    parser.set_defaults(match_nfe=True)

    # 输出
    parser.add_argument('--out_prefix', type=str, default='tafc_mechanism')

    args = parser.parse_args()

    # 设置日志
    logging.basicConfig(level=logging.WARNING)

    # 构建pipeline
    print("\n正在加载模型...")
    model_id, model_name = resolve_model(args)
    pipe, device, torch_dtype = build_pipeline(args, model_id)
    args.model_name = model_name  # 确保model_name被正确设置
    print("✓ 模型加载完成")

    # Phase 1: 完整计算参考
    record_full_computation(pipe, args, model_name, args.prompt, device)
    gc.collect()
    torch.cuda.empty_cache()

    # Phase 2: TAFC
    tafc_img = run_tafc(pipe, args, model_name, args.prompt, device)
    tafc_nfe = sum(_tafc_recorder.computed_flags)
    gc.collect()
    torch.cuda.empty_cache()

    # Phase 3: SeaCache
    seacache_img = run_seacache(pipe, args, model_name, args.prompt, device, target_nfe=tafc_nfe)
    gc.collect()
    torch.cuda.empty_cache()

    # 创建可视化
    create_visualization(args)

    print("\n" + "="*70)
    print("✓ 所有任务完成！")
    print("="*70)


if __name__ == '__main__':
    main()
