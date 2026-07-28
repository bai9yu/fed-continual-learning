"""
================================================================================
可视化 - 绘图函数
================================================================================
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional

from config import (
    EXP_MARKERS,
    EXPERIMENT_TYPES,
    get_frame1_total_num_classes,
    get_frame2_total_num_classes,
    get_num_rounds,
)
from config import get_param_str


def _apply_round_xlim(ax, config=None, max_rounds=None) -> None:
    """横轴贴边：0 在最左，max_rounds 或 config 总轮数在最右，无左右留白。"""
    if max_rounds is not None:
        xmax = int(max_rounds)
    elif config:
        try:
            xmax = int(get_num_rounds(config))
        except (KeyError, ValueError, TypeError):
            xmax = 8000
    else:
        xmax = 8000
    ax.set_xlim(0, xmax)
    ax.margins(x=0)
from modules.fixmatch import pseudo_precision_total_from_unlabeled_agg


def legend_label(exp_name: str) -> str:
    """与 plot_from_json 图例一致；惰性导入避免与 plot_from_json 循环依赖。"""
    from plot_from_json import legend_label as _legend_label_impl
    return _legend_label_impl(exp_name)


def _unique_legend_label(exp_name: str, seen_labels: set) -> str:
    label = legend_label(exp_name)
    if label in seen_labels:
        return "_nolegend_"
    seen_labels.add(label)
    return label


def _strategy_figure_title(exp_name: str, title: str) -> str:
    """图总标题：仅策略简称 + 图表说明（不含 Clients/α/Threshold/阶段等超参）。"""
    return f"{legend_label(exp_name)}: {title}"


def _color_by_family(exp_name: str) -> str:
    """按策略族着色（与 plot_from_json 一致）；惰性导入避免循环依赖。"""
    from plot_from_json import _color_by_family as _fn
    return _fn(exp_name)


def _prepare_output_dir(output_dir: str) -> str:
    """本次实验输出目录（通常为 runs/.../exp_...）。"""
    if not output_dir:
        raise ValueError(
            "必须指定 output_dir（通常为 run_all_experiments 返回的 exp_dir）"
        )
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def continual_frame_boundary_x(metrics: Dict, config: Dict) -> Optional[float]:
    """
    单次运行仅包含一帧联邦时，不在此标注帧间分界（两帧请分两次实验）。
    """
    return None


def apply_continual_vlines_to_axes(axes, metrics: Dict, config: Dict) -> None:
    """在若干子图上绘制帧1/帧2分界竖线（当前实现恒不绘制，保留接口）。"""
    xb = continual_frame_boundary_x(metrics, config)
    if xb is None:
        return
    axes_flat = np.atleast_1d(axes).ravel()
    for j, ax in enumerate(axes_flat):
        kw = {'x': xb, 'color': 'gray', 'linestyle': ':', 'alpha': 0.75, 'linewidth': 1.5}
        if j == 0:
            kw['label'] = 'Frame 1 → Frame 2'
        ax.axvline(**kw)


def get_plot_num_classes(config: Dict) -> int:
    """
    绘图用类别维数：与当前训练阶段一致（帧1/帧1前预热→K2；帧2/帧2前预热→frame2 总类数）。
    用于 pseudo_per_class 等堆叠图；总类数由 get_frame1_total_num_classes / get_frame2_total_num_classes 推导。
    """
    stage = config.get('continual_run_stage')
    if stage == 'server_init_pre':
        return int(config["frame1_initial_num_classes"])
    if stage in ('frame1_pre',):
        return int(get_frame1_total_num_classes(config))
    if stage in ('frame2_pre',):
        return int(get_frame2_total_num_classes(config))
    if stage == 'frame1':
        return int(get_frame1_total_num_classes(config))
    if stage in ('frame2',):
        return int(get_frame2_total_num_classes(config))
    return int(get_frame1_total_num_classes(config))


def _series_optional_float(seq) -> np.ndarray:
    """None → nan；用于折线图。"""
    if seq is None:
        return np.array([])
    if not seq:
        return np.array([])
    out = []
    for x in seq:
        if x is None:
            out.append(float("nan"))
        else:
            out.append(float(x))
    return np.asarray(out, dtype=float)


def plot_experiment_metrics_suite(
    exp_name: str, metrics: Dict, config: Dict, output_dir: str
) -> List[str]:
    """
    单实验多图输出（调度客户端 + 训练步统计）：
      1) test_acc  2) test_loss  3) All/Old/New 测试准确率（continual_eval）
      4) All/Old/New 平均置信度（训练样本）  5) All/Old/New 伪标签精度（训练、按真标签分）
      6) 预测旧/新类样本数 + 预测旧/新类上的准确率  7) 真新→旧 误判（无标签打标签 vs 测试集）
    """
    od = _prepare_output_dir(output_dir)
    param_str = get_param_str(config) if config else ""
    base = f"{exp_name}_{param_str}"
    rounds = list(metrics.get("round") or [])
    if not rounds:
        return []

    markevery = max(1, len(rounds) // 10)
    color = _color_by_family(exp_name)
    marker = EXP_MARKERS.get(exp_name, "o")
    paths: List[str] = []

    # 1. Test accuracy only
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(
        _strategy_figure_title(exp_name, "Test Accuracy (global)"),
        fontsize=12,
        fontweight="bold",
    )
    ax.plot(
        rounds,
        metrics["test_accuracy"],
        color=color,
        marker=marker,
        markevery=markevery,
        markersize=5,
        linewidth=2,
    )
    ax.set_xlabel("Communication Round", fontsize=11)
    ax.set_ylabel("Test Accuracy (%)", fontsize=11)
    ax.grid(True, alpha=0.3, linestyle="--")
    _apply_round_xlim(ax, config)
    plt.tight_layout()
    p1 = os.path.join(od, f"{base}_test_acc.png")
    plt.savefig(p1, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    paths.append(p1)
    print(f"✅ {exp_name} → {p1}")

    # 2. Test loss only
    tls = metrics.get("test_loss") or []
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(
        _strategy_figure_title(exp_name, "Test Loss (CE, global)"),
        fontsize=12,
        fontweight="bold",
    )
    if isinstance(tls, (list, tuple)) and len(tls) == len(rounds) and len(tls) > 0:
        ax.plot(
            rounds,
            tls,
            color=color,
            marker=marker,
            markevery=markevery,
            markersize=5,
            linewidth=2,
        )
    else:
        ax.text(0.5, 0.5, "No test_loss in metrics", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Communication Round", fontsize=11)
    ax.set_ylabel("Test Loss (CE)", fontsize=11)
    ax.grid(True, alpha=0.3, linestyle="--")
    _apply_round_xlim(ax, config)
    plt.tight_layout()
    p2 = os.path.join(od, f"{base}_test_loss.png")
    plt.savefig(p2, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    paths.append(p2)
    print(f"✅ {exp_name} → {p2}")

    # 3–7：依赖训练侧统计字段；缺失则跳过对应子图
    has_train_split = bool(metrics.get("pseudo_train_pred_old_count_total"))

    if has_train_split:
        # 4. Mean confidence All / Old / New（训练样本）
        mc_all = _series_optional_float(metrics.get("pseudo_mean_confidence"))
        mc_o = _series_optional_float(metrics.get("pseudo_train_conf_mean_old"))
        mc_n = _series_optional_float(metrics.get("pseudo_train_conf_mean_new"))
        fig, ax = plt.subplots(figsize=(11, 5.5))
        fig.suptitle(
            _strategy_figure_title(
                exp_name,
                "Mean confidence on training samples (All / true-old / true-new)",
            ),
            fontsize=12,
            fontweight="bold",
        )
        ax.plot(rounds, mc_all, color=color, linewidth=1.8, label="All", markevery=markevery)
        ax.plot(rounds, mc_o, "b--", linewidth=1.6, label="True old", markevery=markevery)
        ax.plot(rounds, mc_n, "r--", linewidth=1.6, label="True new", markevery=markevery)
        ax.axhline(
            y=config.get("pseudo_threshold", 0.8),
            color="#C0392B",
            linestyle="--",
            alpha=0.8,
            label=f'τ={config.get("pseudo_threshold", 0.8)}',
        )
        ax.set_xlabel("Communication Round", fontsize=11)
        ax.set_ylabel("Mean confidence", fontsize=11)
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(True, alpha=0.3, linestyle="--")
        _apply_round_xlim(ax, config)
        plt.tight_layout()
        p4 = os.path.join(od, f"{base}_train_conf_means.png")
        plt.savefig(p4, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        paths.append(p4)
        print(f"✅ {exp_name} → {p4}")

        # 5. Precision All / Old / New（训练、按真标签）
        pr_a = _series_optional_float(metrics.get("pseudo_train_precision_all")) * 100.0
        pr_o = _series_optional_float(metrics.get("pseudo_train_precision_old")) * 100.0
        pr_n = _series_optional_float(metrics.get("pseudo_train_precision_new")) * 100.0
        fig, ax = plt.subplots(figsize=(11, 5.5))
        fig.suptitle(
            _strategy_figure_title(
                exp_name,
                "Pseudo-label precision on training samples (All / true-old / true-new)",
            ),
            fontsize=12,
            fontweight="bold",
        )
        ax.plot(rounds, pr_a, color=color, linewidth=1.8, label="All", markevery=markevery)
        ax.plot(rounds, pr_o, "b--", linewidth=1.6, label="True old", markevery=markevery)
        ax.plot(rounds, pr_n, "r--", linewidth=1.6, label="True new", markevery=markevery)
        ax.set_ylabel("Precision (%)", fontsize=11)
        ax.set_ylim(0, 105)
        ax.set_xlabel("Communication Round", fontsize=11)
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(True, alpha=0.3, linestyle="--")
        _apply_round_xlim(ax, config)
        plt.tight_layout()
        p5 = os.path.join(od, f"{base}_train_precision_split.png")
        plt.savefig(p5, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        paths.append(p5)
        print(f"✅ {exp_name} → {p5}")

        # 5b. 伪标签错误且真类 = 次高预测类 的比例（回合初、全客户端 mask 汇总）
        ru_r = metrics.get("pseudo_masked_wrong_runnerup_rate") or []
        if isinstance(ru_r, (list, tuple)) and len(ru_r) == len(rounds) and len(ru_r) > 0:
            ser = _series_optional_float(ru_r)
            if np.any(np.isfinite(ser)):
                ser_pct = ser * 100.0
                fig, ax = plt.subplots(figsize=(11, 5.5))
                fig.suptitle(
                    _strategy_figure_title(
                        exp_name,
                        "Among wrong pseudo (masked): true class equals 2nd-highest prob class (%)",
                    ),
                    fontsize=12,
                    fontweight="bold",
                )
                ax.plot(
                    rounds,
                    ser_pct,
                    color=color,
                    linewidth=1.8,
                    markevery=markevery,
                )
                ax.set_ylabel("Rate (%)", fontsize=11)
                ax.set_ylim(0, 105)
                ax.set_xlabel("Communication Round", fontsize=11)
                ax.grid(True, alpha=0.3, linestyle="--")
                _apply_round_xlim(ax, config)
                plt.tight_layout()
                p5b = os.path.join(od, f"{base}_pseudo_wrong_runnerup_rate.png")
                plt.savefig(p5b, dpi=150, bbox_inches="tight", facecolor="white")
                plt.close()
                paths.append(p5b)
                print(f"✅ {exp_name} → {p5b}")

        # 6. Left: pred old/new counts  Right: pred old/new accuracy
        c_old = _series_optional_float(metrics.get("pseudo_train_pred_old_count_total"))
        c_new = _series_optional_float(metrics.get("pseudo_train_pred_new_count_total"))
        a_po = _series_optional_float(metrics.get("pseudo_train_pred_old_accuracy")) * 100.0
        a_pn = _series_optional_float(metrics.get("pseudo_train_pred_new_accuracy")) * 100.0
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            _strategy_figure_title(
                exp_name,
                "Pred-old vs pred-new (counts & accuracy, scheduled devices)",
            ),
            fontsize=12,
            fontweight="bold",
        )
        axes[0].plot(rounds, c_old, color="tab:blue", linewidth=1.5, label="Pred as old (count)")
        axes[0].plot(rounds, c_new, color="tab:orange", linewidth=1.5, label="Pred as new (count)")
        axes[0].set_ylabel("Sample count", fontsize=11)
        axes[0].set_xlabel("Communication Round", fontsize=11)
        axes[0].legend(fontsize=9)
        axes[0].grid(True, alpha=0.3, linestyle="--")
        _apply_round_xlim(axes[0], config)
        axes[0].set_ylim(bottom=0)

        axes[1].plot(rounds, a_po, color="tab:blue", linewidth=1.5, label="Accuracy | pred old")
        axes[1].plot(rounds, a_pn, color="tab:orange", linewidth=1.5, label="Accuracy | pred new")
        axes[1].set_ylabel("Accuracy (%)", fontsize=11)
        axes[1].set_ylim(0, 105)
        axes[1].set_xlabel("Communication Round", fontsize=11)
        axes[1].legend(fontsize=9)
        axes[1].grid(True, alpha=0.3, linestyle="--")
        _apply_round_xlim(axes[1], config)
        plt.tight_layout()
        p6 = os.path.join(od, f"{base}_train_pred_split.png")
        plt.savefig(p6, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        paths.append(p6)
        print(f"✅ {exp_name} → {p6}")

    # 7. New→old misclassification: unlabeled (scheduled, mask) vs test
    ce_list = metrics.get("continual_eval") or []
    tm = _continual_eval_nested_series(ce_list, "test", "misclass_new_to_old_rate") * 100.0
    pm = _series_optional_float(metrics.get("pseudo_scheduled_misclass_new_to_old")) * 100.0
    if np.any(np.isfinite(tm)) or np.any(np.isfinite(pm)):
        fig, ax = plt.subplots(figsize=(11, 5.5))
        fig.suptitle(
            _strategy_figure_title(
                exp_name,
                "True-new → pred-old misclassification (unlabeled vs test)",
            ),
            fontsize=12,
            fontweight="bold",
        )
        ax.plot(rounds, pm, color="darkgreen", linewidth=1.8, label="Unlabeled pseudo (scheduled)", markevery=markevery)
        ax.plot(rounds, tm, color="purple", linewidth=1.8, linestyle="--", label="Test set", markevery=markevery)
        ax.set_ylabel("Rate (%)", fontsize=11)
        ax.set_ylim(0, 105)
        ax.set_xlabel("Communication Round", fontsize=11)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.3, linestyle="--")
        _apply_round_xlim(ax, config)
        plt.tight_layout()
        p7 = os.path.join(od, f"{base}_misclass_new_to_old.png")
        plt.savefig(p7, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        paths.append(p7)
        print(f"✅ {exp_name} → {p7}")

    return paths


def plot_single_experiment(exp_name: str, metrics: Dict, config: Dict, output_dir: str) -> List[str]:
    """
    单实验指标图（多文件）。兼容旧接口名；返回路径列表。
    """
    return plot_experiment_metrics_suite(exp_name, metrics, config, output_dir)


def plot_comparison(all_results: Dict, config: Dict, output_dir: str) -> str:
    """
    绘制所有实验的对比图：上 Test Accuracy，下 Test Loss（无 test_loss 则下图留空提示）。
    output_dir 通常为 runs/ 下本次实验文件夹。
    """
    from plot_from_json import _ordered_experiment_keys, _policy_linestyle

    fig, (ax_acc, ax_loss) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(
        "Comparison: Test Accuracy & Test Loss",
        fontsize=14,
        fontweight="bold",
    )

    keys = _ordered_experiment_keys(all_results)
    seen_acc_labels = set()
    seen_loss_labels = set()
    for exp_name in keys:
        metrics = all_results[exp_name]

        ax_acc.plot(
            metrics['round'],
            metrics['test_accuracy'],
            color=_color_by_family(exp_name),
            linestyle=_policy_linestyle(exp_name),
            linewidth=2,
            label=_unique_legend_label(exp_name, seen_acc_labels),
        )
        tl = metrics.get('test_loss') or []
        rounds = metrics['round']
        if isinstance(tl, (list, tuple)) and len(tl) == len(rounds) and len(tl) > 0:
            ax_loss.plot(
                rounds,
                tl,
                color=_color_by_family(exp_name),
                linestyle=_policy_linestyle(exp_name),
                linewidth=2,
                label=_unique_legend_label(exp_name, seen_loss_labels),
            )
    
    ref_metrics = next(iter(all_results.values()))
    xb = continual_frame_boundary_x(ref_metrics, config)
    if xb is not None:
        ax_acc.axvline(x=xb, color='gray', linestyle=':', alpha=0.75, linewidth=1.5, label='Frame 1 → Frame 2')
        ax_loss.axvline(x=xb, color='gray', linestyle=':', alpha=0.75, linewidth=1.5)
    
    ax_acc.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax_acc.legend(fontsize=10, loc='lower right', framealpha=0.9)
    ax_acc.grid(True, alpha=0.3, linestyle='--')
    _apply_round_xlim(ax_acc, config)
    ax_acc.set_title('Test Accuracy', fontsize=12, fontweight='bold')

    ax_loss.set_xlabel('Communication Round', fontsize=12)
    ax_loss.set_ylabel('Test Loss (CE)', fontsize=12)
    ax_loss.set_title('Test Loss', fontsize=12, fontweight='bold')
    if not ax_loss.lines:
        ax_loss.text(
            0.5,
            0.5,
            'No test_loss in metrics',
            ha='center',
            va='center',
            transform=ax_loss.transAxes,
            fontsize=11,
            color='gray',
        )
    else:
        ax_loss.legend(fontsize=10, loc='upper right', framealpha=0.9)
    ax_loss.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    
    _prepare_output_dir(output_dir)
    plot_filename = os.path.join(output_dir, f"comparison_{get_param_str(config)}.png")
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ 对比图已保存: {plot_filename}")
    
    return plot_filename


def plot_summary_bar(all_results: Dict, config: Dict, output_dir: str) -> str:
    """
    绘制平均准确率 vs 最高准确率柱状图（与 plot_from_json.plot_summary_bar 一致）。
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    exp_order = EXPERIMENT_TYPES
    exp_names = []
    avg_accs = []
    best_accs = []
    
    for exp_name in exp_order:
        if exp_name not in all_results:
            continue
        exp_names.append(exp_name)
        acc = np.asarray(all_results[exp_name]['test_accuracy'], dtype=float)
        avg_accs.append(float(np.nanmean(acc)) if acc.size else 0.0)
        best_accs.append(float(np.nanmax(acc)) if acc.size else 0.0)
    
    x = np.arange(len(exp_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, avg_accs, width, label='Average Accuracy',
                   color=[_color_by_family(e) for e in exp_names], alpha=0.7)
    bars2 = ax.bar(x + width/2, best_accs, width, label='Best Accuracy',
                   color=[_color_by_family(e) for e in exp_names], alpha=1.0)
    
    ax.set_xlabel('Experiment', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Average vs Best Accuracy', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([legend_label(e) for e in exp_names])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    _prepare_output_dir(output_dir)
    plot_filename = os.path.join(output_dir, f"summary_bar_{get_param_str(config)}.png")
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ 汇总柱状图已保存: {plot_filename}")
    
    return plot_filename


def plot_client_selection(all_results: Dict, config: Dict, output_dir: str) -> str:
    """绘制客户端选择频率图。output_dir 通常为 runs/ 下本次实验文件夹。"""
    fig, ax = plt.subplots(figsize=(10, 6))
    seen_labels = set()
    
    for exp_name in EXPERIMENT_TYPES:
        if exp_name not in all_results:
            continue
        
        client_counts = defaultdict(int)
        for selected in all_results[exp_name]['selected_clients']:
            for client_id in selected:
                client_counts[client_id] += 1
        
        clients = list(range(config['num_clients']))
        counts = [client_counts[c] for c in clients]
        
        ax.plot(clients, counts,
                color=_color_by_family(exp_name),
                marker=EXP_MARKERS[exp_name],
                markersize=8,
                linewidth=2,
                label=_unique_legend_label(exp_name, seen_labels),
                alpha=0.8)
    
    ax.axvline(x=0.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.text(0.7, ax.get_ylim()[1] * 0.9, 'Labeled', fontsize=9, color='gray')
    
    ax.set_xlabel('Device ID', fontsize=12)
    ax.set_ylabel('Selection Count', fontsize=12)
    ax.set_title('Device Selection Frequency', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    _prepare_output_dir(output_dir)
    plot_filename = os.path.join(output_dir, f"client_selection_{get_param_str(config)}.png")
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ 客户端选择频率图已保存: {plot_filename}")
    
    return plot_filename


def _continual_eval_nested_series(continual_eval: List, *path: str) -> np.ndarray:
    """从每轮 continual_eval 条目中按路径取标量，缺失则为 nan。"""
    out: List[float] = []
    for ce in continual_eval:
        if ce is None:
            out.append(float("nan"))
            continue
        cur = ce
        for k in path:
            if not isinstance(cur, dict):
                cur = None
                break
            cur = cur.get(k)
        if cur is None:
            out.append(float("nan"))
        else:
            out.append(float(cur))
    return np.asarray(out, dtype=float)


def plot_test_accuracy_all_old_new(
    exp_name: str, metrics: Dict, config: Dict, output_dir: str
) -> str:
    """
    单张图：Test Acc (all) / (old) / (new)。
    无 continual_eval 或 test 侧 old/new 均无有效数据时返回空串。
    """
    continual_eval = metrics.get("continual_eval") or []
    rounds = list(metrics.get("round") or [])
    n = min(len(rounds), len(continual_eval))
    if n == 0:
        return ""
    rounds = rounds[:n]
    continual_eval = continual_eval[:n]

    acc_all = np.asarray(metrics["test_accuracy"][:n], dtype=float)
    acc_old = _continual_eval_nested_series(continual_eval, "test", "test_acc_old")
    acc_new = _continual_eval_nested_series(continual_eval, "test", "test_acc_new")
    if not (np.any(np.isfinite(acc_old)) or np.any(np.isfinite(acc_new))):
        return ""

    num_points = n
    markevery = max(1, num_points // 10)
    color = _color_by_family(exp_name)
    marker = EXP_MARKERS.get(exp_name, "o")

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle(
        _strategy_figure_title(exp_name, "Test Accuracy (All / Old / New)"),
        fontsize=13,
        fontweight="bold",
    )

    ax.plot(
        rounds,
        acc_all,
        color=color,
        marker=marker,
        markevery=markevery,
        markersize=6,
        linewidth=2,
        label="Test Acc (all)",
    )
    ax.plot(
        rounds,
        acc_old,
        "b--",
        linewidth=1.8,
        alpha=0.9,
        label="Test Acc (old)",
        markevery=markevery,
    )
    ax.plot(
        rounds,
        acc_new,
        "r--",
        linewidth=1.8,
        alpha=0.9,
        label="Test Acc (new)",
        markevery=markevery,
    )

    xb = continual_frame_boundary_x(metrics, config)
    if xb is not None:
        ax.axvline(
            x=xb,
            color="gray",
            linestyle=":",
            alpha=0.75,
            linewidth=1.5,
            label="Frame 1 → Frame 2",
        )

    ax.set_xlabel("Communication Round", fontsize=12)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.legend(fontsize=10, loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")
    _apply_round_xlim(ax, config)

    plt.tight_layout()
    _prepare_output_dir(output_dir)
    plot_filename = os.path.join(
        output_dir, f"{exp_name}_{get_param_str(config)}_test_acc_split.png"
    )
    plt.savefig(plot_filename, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"✅ {exp_name} Test Acc 拆分图已保存: {plot_filename}")
    return plot_filename


def plot_continual_eval_curves(
    exp_name: str,
    metrics: Dict,
    config: Dict,
    output_dir: str,
    max_rounds: Optional[int] = None,
) -> str:
    """
    将 metrics['continual_eval'] 画成折线图：测试集 Acc_old / Acc_new、误判率，
    以及无标签侧伪标签旧/新精度与 new→old 误判率（与联邦第二帧每轮记录一致）。
    图内文案为英文；无有效数据时返回空字符串。
    """
    continual_eval = metrics.get("continual_eval")
    if not continual_eval or not any(x is not None for x in continual_eval):
        return ""

    rounds = list(metrics.get("round") or [])
    n = min(len(rounds), len(continual_eval))
    if n == 0:
        return ""
    rounds = rounds[:n]
    continual_eval = continual_eval[:n]

    if max_rounds is not None:
        idx = [i for i, r in enumerate(rounds) if r <= max_rounds]
        rounds = [rounds[i] for i in idx]
        continual_eval = [continual_eval[i] for i in idx]
        n = len(rounds)
        if n == 0:
            return ""

    acc_old = _continual_eval_nested_series(continual_eval, "test", "test_acc_old")
    acc_new = _continual_eval_nested_series(continual_eval, "test", "test_acc_new")
    m_n2o = _continual_eval_nested_series(continual_eval, "test", "misclass_new_to_old_rate") * 100.0
    m_o2n = _continual_eval_nested_series(continual_eval, "test", "misclass_old_to_new_rate") * 100.0
    p_old = _continual_eval_nested_series(continual_eval, "pseudo_unlabeled_agg", "pseudo_precision_old") * 100.0
    p_new = _continual_eval_nested_series(continual_eval, "pseudo_unlabeled_agg", "pseudo_precision_new") * 100.0
    p_tot_list: List[float] = []
    for ce in continual_eval:
        t = pseudo_precision_total_from_unlabeled_agg(
            ce.get("pseudo_unlabeled_agg") if ce else None
        )
        p_tot_list.append(float(t) * 100.0 if t is not None else float("nan"))
    p_tot = np.asarray(p_tot_list, dtype=float)
    p_n2o = _continual_eval_nested_series(
        continual_eval, "pseudo_unlabeled_agg", "pseudo_misclass_new_to_old_rate"
    ) * 100.0

    series_list = [acc_old, acc_new, m_n2o, m_o2n, p_old, p_new, p_tot, p_n2o]
    if not any(np.any(np.isfinite(s)) for s in series_list):
        return ""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    cfg = config or {}
    fig.suptitle(
        _strategy_figure_title(
            exp_name, "Continual Evaluation (Test & Unlabeled Pseudo)"
        ),
        fontsize=13,
        fontweight="bold",
    )
    markevery = max(1, n // 10)

    ax = axes[0, 0]
    ax.plot(
        rounds,
        acc_old,
        "b-o",
        markersize=4,
        linewidth=1.5,
        label="Test Acc (old classes)",
        markevery=markevery,
    )
    ax.plot(
        rounds,
        acc_new,
        "r-s",
        markersize=4,
        linewidth=1.5,
        label="Test Acc (new classes)",
        markevery=markevery,
    )
    ax.set_xlabel("Communication Round", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("(a) Test Accuracy (Old vs New)", fontsize=12, fontweight="bold")
    _apply_round_xlim(ax, config, max_rounds)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(
        rounds,
        m_n2o,
        "c-^",
        markersize=4,
        linewidth=1.5,
        label="True new → pred old (rate)",
        markevery=markevery,
    )
    ax.plot(
        rounds,
        m_o2n,
        "m-v",
        markersize=4,
        linewidth=1.5,
        label="True old → pred new (rate)",
        markevery=markevery,
    )
    ax.set_xlabel("Communication Round", fontsize=11)
    ax.set_ylabel("Misclassification rate (%)", fontsize=11)
    ax.set_title("(b) Test Misclassification Rates", fontsize=12, fontweight="bold")
    _apply_round_xlim(ax, config, max_rounds)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="best")

    ax = axes[1, 0]
    ax.plot(
        rounds,
        p_old,
        "g-o",
        markersize=4,
        linewidth=1.5,
        label="Pseudo precision (old)",
        markevery=markevery,
    )
    ax.plot(
        rounds,
        p_new,
        color="darkorange",
        linestyle="-",
        marker="s",
        markersize=4,
        linewidth=1.5,
        label="Pseudo precision (new)",
        markevery=markevery,
    )
    ax.plot(
        rounds,
        p_tot,
        color="0.35",
        linestyle="--",
        marker="^",
        markersize=4,
        linewidth=1.5,
        label="Pseudo precision (all selected)",
        markevery=markevery,
    )
    ax.set_xlabel("Communication Round", fontsize=11)
    ax.set_ylabel("Precision (%)", fontsize=11)
    ax.set_title("(c) Unlabeled Pseudo-Label Precision", fontsize=12, fontweight="bold")
    _apply_round_xlim(ax, config, max_rounds)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="best")

    ax = axes[1, 1]
    ax.plot(
        rounds,
        p_n2o,
        "k-D",
        markersize=4,
        linewidth=1.5,
        label="True new → pseudo old (rate)",
        markevery=markevery,
    )
    ax.set_xlabel("Communication Round", fontsize=11)
    ax.set_ylabel("Rate (%)", fontsize=11)
    ax.set_title("(d) Unlabeled Pseudo: New→Old Confusion", fontsize=12, fontweight="bold")
    _apply_round_xlim(ax, config, max_rounds)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="best")

    apply_continual_vlines_to_axes(axes, metrics, cfg)
    axes[0, 0].legend(fontsize=9, loc="best")

    plt.tight_layout()

    param_str = get_param_str(config) if config else ""
    base = f"{exp_name}_{param_str}_continual_eval.png" if param_str else f"{exp_name}_continual_eval.png"
    _prepare_output_dir(output_dir)
    plot_path = os.path.join(output_dir, base)

    plt.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"✅ {exp_name} continual_eval 折线图已保存: {plot_path}")
    return plot_path


def plot_per_experiment_figures_from_metrics(
    exp_name: str,
    metrics: Dict,
    config: Dict,
    output_dir: str,
    *,
    standard_plots: bool = True,
    continual_eval_long: bool = False,
    max_rounds: Optional[int] = None,
) -> List[str]:
    """
    单策略出图统一入口（main / plot_from_json 共用）。

    - standard_plots：test_acc、loss、train 拆分、misclass、test_acc_split 等；
      具体画哪些子图仅由 metrics 是否含对应字段决定（见 plot_experiment_metrics_suite 等）。
    - continual_eval_long：是否额外输出 continual_eval 四宫格长图。
    """
    paths: List[str] = []
    if standard_plots:
        paths.extend(plot_single_experiment(exp_name, metrics, config, output_dir))
        extra_acc = plot_test_accuracy_all_old_new(exp_name, metrics, config, output_dir)
        if extra_acc:
            paths.append(extra_acc)
    if continual_eval_long:
        extra_ce = plot_continual_eval_curves(
            exp_name,
            metrics,
            config,
            output_dir,
            max_rounds=max_rounds,
        )
        if extra_ce:
            paths.append(extra_ce)
    return paths


def plot_all_results(all_results: Dict, config: Dict, output_dir: str) -> List[str]:
    """
    绘制所有结果图（分开输出到 output_dir，通常为 runs/ 下本次实验目录）。
    """
    od = _prepare_output_dir(output_dir)
    plot_files = []
    
    # 1. 每个实验单独的图
    for exp_name, metrics in all_results.items():
        plot_files.extend(
            plot_per_experiment_figures_from_metrics(
                exp_name,
                metrics,
                config,
                od,
                standard_plots=True,
                continual_eval_long=True,
            )
        )
    
    # 2. 对比图（Test Accuracy）
    if len(all_results) > 1:
        plot_file = plot_comparison(all_results, config, od)
        plot_files.append(plot_file)
        
        # 3. 汇总柱状图
        plot_file = plot_summary_bar(all_results, config, od)
        plot_files.append(plot_file)
        
        # 4. 客户端选择频率图
        scheduling_exps = EXPERIMENT_TYPES
        if any(exp in all_results for exp in scheduling_exps):
            plot_file = plot_client_selection(all_results, config, od)
            plot_files.append(plot_file)
    
    # 打印结果摘要
    print_results_summary(all_results)
    
    return plot_files


def print_results_summary(all_results: Dict):
    """打印结果摘要"""
    exp_order = EXPERIMENT_TYPES
    
    print("\n" + "=" * 100)
    print("Final Results Summary")
    print("=" * 100)
    print(f"{'Experiment':<15} {'Final Acc':<12} {'Best Acc':<12} {'Best Round':<10}")
    print("-" * 100)
    
    for exp_name in exp_order:
        if exp_name not in all_results:
            continue
        metrics = all_results[exp_name]
        
        final_acc = metrics['test_accuracy'][-1]
        best_acc = max(metrics['test_accuracy'])
        best_round = metrics['test_accuracy'].index(best_acc)
        
        print(f"{exp_name:<15}{final_acc:>10.2f}% {best_acc:>10.2f}% {best_round:>10}")
    
    print("=" * 100)
