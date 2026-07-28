"""
================================================================================
从 JSON 文件夹读取并绘制跨实验对比图；单实验内的 accuracy/loss 等长图由训练时 plot_all_results 生成。
本脚本输出：test accuracy、包络线、每轮调度客户端数、达到目标精度所需轮数、continual 子指标对比、
summary 精度柱图、按策略族汇总的平均每轮客户端柱图等。
================================================================================
使用方法:
    1. 命令行运行:
       python plot_from_json.py runs/exp_c10_l1_a0.5_r3000_20250108_120000
    
    2. 在代码中调用:
       from plot_from_json import plot_from_folder
       plot_from_folder("runs/exp_xxx")
================================================================================
"""

import os
import re
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from config import (
    ABBREVIATIONS,
    COMPARISON_PLOT_STYLE,
    EXPERIMENT_TYPES,
    EXP_LINESTYLES,
    EXP_MARKERS,
    FAMILY_COLORS,
    get_path_abbrev,
)

from modules.fixmatch import pseudo_precision_total_from_unlabeled_agg
from visualization import (
    _apply_round_xlim,
    _continual_eval_nested_series,
    continual_frame_boundary_x,
    plot_per_experiment_figures_from_metrics,
)

from config import get_param_str as utils_get_param_str


def _metrics_last_train_precision_all(metrics: Dict) -> float:
    """训练侧总体伪标签精度（最后一轮）。"""
    s = metrics.get("pseudo_train_precision_all") or []
    if not s:
        return 0.0
    v = s[-1]
    return float(v) if v is not None else 0.0


def scheduling_suffix_key(exp_name: str) -> Optional[str]:
    """从长/短实验名解析出与 EXPERIMENT_TYPES 一致的后缀，供 linestyle / color 查找。"""
    for candidate in sorted(EXPERIMENT_TYPES, key=len, reverse=True):
        if exp_name.endswith(candidate):
            return candidate
    return None


def _folder_slug_from_long_prefix(prefix: str) -> Optional[str]:
    """长文件名前缀 exp_ACT_GT_svhn_... → ACT_GT（与 main 下 run 目录命名一致）。"""
    m = re.match(r"^exp_(.+?)_(?:svhn|cifar|mnist|c10|c100)", prefix)
    if m:
        return m.group(1)
    return None


# Random 的伪标签/选端变体：各自独立配色（缩写 R_PO / R_SN），勿与 R 共用绿色
_RANDOM_SCHED_VARIANTS = frozenset({
    'Random_PriorityOldNew',
    'PO_FPM',
    'PO_FPLN',
    'PO_MFLN',
    'Random_StickyNew',
    'Random_FLFL',
    'Random_FedLGMatch',
    'Random_FSSL_UC',
})


def _strategy_family_key(exp_name: str) -> str:
    """策略族键：exp_ACT_GT_... 与 ACT_GT 等均归为 ACT，用于同色。"""
    if exp_name in EXPERIMENT_TYPES:
        if exp_name in _RANDOM_SCHED_VARIANTS:
            return get_path_abbrev(exp_name)
        ab = get_path_abbrev(exp_name)
        if '_' in ab:
            return ab.split('_')[0]
        return ab
    suf = scheduling_suffix_key(exp_name)
    if not suf:
        return 'misc'
    prefix = exp_name[: -len(suf)].rstrip('_')
    slug = _folder_slug_from_long_prefix(prefix)
    if slug and '_' in slug:
        if slug in (
            'R_PO',
            'PO_FPM',
            'PO_FPLN',
            'PO_MFLN',
            'R_SN',
            'R_FLFL',
            'R_FedLG',
            'R_FSSL_UC',
        ):
            return slug
        return slug.split('_')[0]
    if slug:
        return slug
    if suf in _RANDOM_SCHED_VARIANTS:
        return get_path_abbrev(suf)
    ab = get_path_abbrev(suf)
    return ab.split('_')[0] if '_' in str(ab) else str(ab)


def _color_by_family(exp_name: str) -> str:
    return FAMILY_COLORS.get(_strategy_family_key(exp_name), FAMILY_COLORS['misc'])


def _comparison_figsize() -> Tuple[int, int]:
    return COMPARISON_PLOT_STYLE['figsize']


def _rounds_to_target_figsize() -> Tuple[int, int]:
    return COMPARISON_PLOT_STYLE.get('rounds_to_target_figsize', _comparison_figsize())


def _comparison_envelope_figsize() -> Tuple[int, int]:
    return COMPARISON_PLOT_STYLE.get('comparison_envelope_figsize', (14, 10))


def _continual_plot_figsize(fname_base: str) -> Tuple[int, int]:
    if fname_base in _CONTINUAL_ENVELOPE_PLOTS:
        return COMPARISON_PLOT_STYLE.get(
            'continual_pseudo_precision_new_figsize',
            _rounds_to_target_figsize(),
        )
    return _comparison_figsize()


def _continual_plot_save_pdf(fname_base: str) -> bool:
    return fname_base in _CONTINUAL_ENVELOPE_PLOTS


# 包络对比图：仅以下策略保留 raw 曲线 + fill_between 阴影
_ENVELOPE_SHADED_STRATEGIES = frozenset({'PO_MFLN', 'Random', 'Random_FSSL_UC'})


def _envelope_shading_enabled(exp_name: str) -> bool:
    suf = scheduling_suffix_key(exp_name)
    key = suf if suf is not None else exp_name
    return key in _ENVELOPE_SHADED_STRATEGIES


# continual 子图：原始曲线（淡）+ 包络线（实线）
_CONTINUAL_ENVELOPE_PLOTS = frozenset({
    'comparison_continual_pseudo_precision_new',
    'comparison_continual_misclass_new_to_old',
    'comparison_continual_pseudo_new_to_old',
})

# 越低越好（误分类/误判率）：包络取运行最小值
_CONTINUAL_ENVELOPE_MIN_PLOTS = frozenset({
    'comparison_continual_misclass_new_to_old',
    'comparison_continual_pseudo_new_to_old',
})


def _continual_envelope_series(vals_arr: np.ndarray, fname_base: str) -> np.ndarray:
    if fname_base in _CONTINUAL_ENVELOPE_MIN_PLOTS:
        return np.minimum.accumulate(vals_arr)
    return np.maximum.accumulate(vals_arr)


def _plot_continual_envelope_curve(
    ax,
    rounds,
    vals,
    color: str,
    exp_name: str,
    *,
    fname_base: str,
    label: Optional[str] = None,
) -> None:
    vals_arr = np.asarray(vals, dtype=float)
    envelope = _continual_envelope_series(vals_arr, fname_base)
    legend = label if label is not None else legend_label(exp_name)
    ax.plot(
        rounds,
        vals_arr,
        color=color,
        alpha=0.22,
        linewidth=0.75,
        zorder=1,
    )
    ax.plot(
        rounds,
        envelope,
        color=color,
        linewidth=_comparison_line_width(),
        linestyle=_exp_linestyle(exp_name),
        label=legend,
        zorder=2,
    )


def _comparison_line_width() -> float:
    return COMPARISON_PLOT_STYLE['line_width']


def _comparison_dpi() -> int:
    return COMPARISON_PLOT_STYLE['dpi']


def _apply_comparison_grid(ax) -> None:
    s = COMPARISON_PLOT_STYLE
    ax.set_facecolor(s['facecolor'])
    ax.grid(
        True,
        alpha=s['grid_alpha'],
        color=s['grid_color'],
        linestyle=s['grid_linestyle'],
        linewidth=s['grid_linewidth'],
    )


def _apply_comparison_axes(
    ax,
    xlabel: str,
    ylabel: str,
    *,
    config=None,
    max_rounds=None,
    apply_round_xlim: bool = True,
    legend_loc: str = 'lower right',
    legend_ncol: int = 1,
) -> None:
    s = COMPARISON_PLOT_STYLE
    ax.set_xlabel(xlabel, fontsize=s['axis_label_fontsize'])
    ax.set_ylabel(ylabel, fontsize=s['axis_label_fontsize'])
    ax.tick_params(axis='both', labelsize=s['tick_fontsize'])
    ax.legend(
        fontsize=s['legend_fontsize'],
        loc=legend_loc,
        framealpha=0.9,
        ncol=legend_ncol,
    )
    _apply_comparison_grid(ax)
    if apply_round_xlim:
        _apply_round_xlim(ax, config, max_rounds)


def _save_comparison_figure(fig, plot_path: str) -> None:
    fig.savefig(
        plot_path,
        dpi=_comparison_dpi(),
        bbox_inches='tight',
        facecolor=COMPARISON_PLOT_STYLE['facecolor'],
    )


def _save_comparison_figure_png_and_pdf(fig, png_path: str) -> Tuple[str, str]:
    """保存 PNG 与同名的 PDF（论文用矢量图）。"""
    _save_comparison_figure(fig, png_path)
    pdf_path = os.path.splitext(png_path)[0] + '.pdf'
    fig.savefig(
        pdf_path,
        dpi=COMPARISON_PLOT_STYLE.get('pdf_dpi', 600),
        bbox_inches='tight',
        facecolor=COMPARISON_PLOT_STYLE['facecolor'],
    )
    return png_path, pdf_path


def legend_label(exp_name: str) -> str:
    """图例：直接读 ABBREVIATIONS（可含 $FL^2$ 等）；长名组合路径 slug + 图例缩写。"""
    if not exp_name:
        return ""
    if exp_name in EXPERIMENT_TYPES:
        return ABBREVIATIONS.get(exp_name, exp_name)
    suf = scheduling_suffix_key(exp_name)
    if suf is None:
        if len(exp_name) > 52:
            return exp_name[:26] + "…" + exp_name[-22:]
        return exp_name
    sched = ABBREVIATIONS.get(suf, suf)
    prefix = exp_name[: -len(suf)].rstrip("_")
    slug = _folder_slug_from_long_prefix(prefix)
    path_slug = get_path_abbrev(suf) if suf in EXPERIMENT_TYPES else None
    if not slug:
        return sched
    if slug == sched or slug == path_slug:
        return sched
    return f"{slug}_{sched}"


def _unique_legend_label(exp_name: str, seen_labels: set) -> str:
    label = legend_label(exp_name)
    if label in seen_labels:
        return "_nolegend_"
    seen_labels.add(label)
    return label


def get_param_str(config: Dict) -> str:
    """与 config.get_param_str 一致（阶段+轮次+阈值，无 alpha/客户端数）。"""
    return utils_get_param_str(config)


# 图例/曲线顺序（与 matplotlib legend 绘制顺序一致）：
# Random 各变体 → R → BC → NCC → ACT → NCT（NCC 在 All 上方、图例倒数第三；R 在 BC 前）
_FAMILY_ORDER = {
    'R_PO': 0,
    'PO_FPM': 1,
    'PO_FPLN': 2,
    'PO_MFLN': 3,
    'R_SN': 4,
    'R_FLFL': 5,
    'R_FedLG': 6,
    'R_FSSL_UC': 7,
    'R': 8,
    'BC': 9,
    'NCC': 10,
    'ACT': 11,
    'NCT': 12,
}


def _family_abbrev_prefix(exp_name: str) -> str:
    """策略族排序键：短名用路径缩写；长名与 _strategy_family_key 一致。"""
    if exp_name in EXPERIMENT_TYPES:
        if exp_name in _RANDOM_SCHED_VARIANTS:
            return get_path_abbrev(exp_name)
        ab = get_path_abbrev(exp_name)
        return ab.split('_')[0]
    return _strategy_family_key(exp_name)


def _variant_tiebreak(suf: Optional[str]) -> int:
    """同类内：基线 → TrueLabel(GT) → FullTrueLabel(Gall)。须先判 Full，再判 True。"""
    if not suf:
        return 0
    if suf.endswith('_FullTrueLabel'):
        return 2
    if suf.endswith('_TrueLabel'):
        return 1
    return 0


def _policy_linestyle(exp_name: str) -> str:
    """GT(TrueLabel) 虚线；Gall(FullTrueLabel) 点划线；其余实线。"""
    suf = scheduling_suffix_key(exp_name)
    if suf:
        if suf.endswith('_FullTrueLabel'):
            return '-.'
        if suf.endswith('_TrueLabel'):
            return '--'
    return '-'


def _exp_linestyle(exp_name: str):
    """在 GT/Gall 线型规则之上，叠加 config.EXP_LINESTYLES（便于区分各调度）。"""
    pol = _policy_linestyle(exp_name)
    if pol != '-':
        return pol
    suf = scheduling_suffix_key(exp_name)
    if suf and suf in EXP_LINESTYLES:
        return EXP_LINESTYLES[suf]
    return '-'


def _line_marker_plot_kwargs(exp_name: str, color: str, n_points: int) -> Dict:
    """长序列对比曲线：线型 + 稀疏标记；Random_FSSL_UC 为空心圆。"""
    suf = scheduling_suffix_key(exp_name)
    marker = EXP_MARKERS.get(suf, 'o') if suf else 'o'
    if n_points <= 12:
        markevery = 1
    elif n_points <= 40:
        markevery = max(1, n_points // 10)
    else:
        markevery = max(1, n_points // 20)
    kw: Dict = {
        'linestyle': _exp_linestyle(exp_name),
        'marker': marker,
        'markersize': 4.2 if marker in ('s', 'D', 'd', 'h', 'p') else 4.8,
        'markevery': markevery,
    }
    if suf == 'Random_FSSL_UC':
        kw['markerfacecolor'] = 'none'
        kw['markeredgecolor'] = color
        kw['markeredgewidth'] = 1.35
    return kw


def _scatter_marker_plot_kwargs(exp_name: str, color: str) -> Dict:
    """离散点折线（如 rounds-to-target）：每点一个标记。"""
    suf = scheduling_suffix_key(exp_name)
    marker = EXP_MARKERS.get(suf, 'o') if suf else 'o'
    ms = COMPARISON_PLOT_STYLE.get('scatter_marker_size', 10)
    kw: Dict = {
        'linestyle': _exp_linestyle(exp_name),
        'marker': marker,
        'markersize': ms,
    }
    if suf == 'Random_FSSL_UC':
        kw['markerfacecolor'] = 'none'
        kw['markeredgecolor'] = color
        kw['markeredgewidth'] = max(1.8, ms * 0.2)
    return kw


def _ordered_experiment_keys(all_results: Dict) -> List[str]:
    """对比图/柱状图统一顺序：R_PO → PO_FPM/FPLN/MFLN → R_SN→…；同类内基线→GT→Gall。"""

    def sort_key(k: str):
        fam = _family_abbrev_prefix(k)
        bucket = _FAMILY_ORDER.get(fam, 99)
        suf = scheduling_suffix_key(k)
        var = _variant_tiebreak(suf)
        return (bucket, var, k)

    return sorted(all_results.keys(), key=sort_key)


# 基线调度（不含 TrueLabel / FullTrueLabel），用于 continual 子指标对比图
_BASELINE_SCHEDULING_SUFFIXES = frozenset({
    'NoClientTrain', 'AllClientsTrain', 'Random', 'BestChannel', 'NewClassClientsOnly',
    'Random_PriorityOldNew',
    'PO_FPM',
    'PO_FPLN',
    'PO_MFLN',
    'Random_StickyNew', 'Random_FLFL',
    'Random_FedLGMatch', 'Random_FSSL_UC',
})


def _is_baseline_exp(exp_name: str) -> bool:
    suf = scheduling_suffix_key(exp_name)
    if not suf:
        return False
    if suf.endswith('_FullTrueLabel') or suf.endswith('_TrueLabel'):
        return False
    return suf in _BASELINE_SCHEDULING_SUFFIXES


def _ordered_baseline_experiment_keys(all_results: Dict) -> List[str]:
    """R 变体→R→BC→NCC→ACT→NCT，且仅基线（无 GT/Gall）。"""
    return [k for k in _ordered_experiment_keys(all_results) if _is_baseline_exp(k)]


def _continual_eval_pct_series(
    metrics: Dict,
    max_rounds: Optional[int],
    *path: str,
) -> Optional[Tuple[List, np.ndarray]]:
    """从 continual_eval 取路径对应标量，转为百分比；与 plot_continual_eval_curves 一致。"""
    continual_eval = metrics.get('continual_eval')
    if not continual_eval or not any(x is not None for x in continual_eval):
        return None
    rounds = list(metrics.get('round') or [])
    n = min(len(rounds), len(continual_eval))
    if n == 0:
        return None
    rounds = rounds[:n]
    continual_eval = continual_eval[:n]

    if max_rounds is not None:
        idx = [i for i, r in enumerate(rounds) if r <= max_rounds]
        rounds = [rounds[i] for i in idx]
        continual_eval = [continual_eval[i] for i in idx]
        if not rounds:
            return None

    if path == ("pseudo_unlabeled_agg", "pseudo_precision_total"):
        vals = []
        for ce in continual_eval:
            if ce is None:
                vals.append(float("nan"))
                continue
            t = pseudo_precision_total_from_unlabeled_agg(ce.get("pseudo_unlabeled_agg"))
            vals.append(float(t) if t is not None else float("nan"))
        arr = np.asarray(vals, dtype=float) * 100.0
    else:
        arr = _continual_eval_nested_series(continual_eval, *path) * 100.0
    if not np.any(np.isfinite(arr)):
        return None
    return rounds, arr


def extract_exp_name(filename: str, folder_path: Optional[str] = None) -> str:
    """
    从文件名中提取实验名称
    
    支持格式:
        - {实验目录名}_{ExpName}.json（与 runs 下文件夹同名前缀）
        - ExpName_{frame*}_r*_t*_result.json（旧中间格式）
        - ExpName_c10_l1_a0.5_r3000_t0.95_result.json（更旧格式）
    
    参数:
        filename: JSON 文件名
        folder_path: 若提供，则按「目录名_」前缀解析实验名
    """
    name = filename.replace('.json', '')
    name = re.sub(r'_result$', '', name)

    if folder_path:
        fb = os.path.basename(os.path.normpath(folder_path))
        prefix = fb + '_'
        if name.startswith(prefix):
            return name[len(prefix):]

    # 无目录上下文时：frame 参数字串后缀
    name2 = re.sub(
        r'_(?:warmup|frame1|frame2|frame1_2)_r[0-9.]+_t[0-9.]+$',
        '',
        name,
    )
    if name2 != name:
        return name2

    if '_c' in name:
        return name.split('_c', 1)[0]

    return name


def _iter_metrics_json_tasks(folder_path: str) -> List[Tuple[str, str]]:
    """
    列出待加载的 (json_filename, extract_dir) 任务。

    - 若 folder_path 下直接有 *.json（除 config.json），则与现行为一致：extract_dir 为 folder_path。
    - 若当前目录没有任何结果 JSON（例如顶层目录下只有多个 ``exp_*`` 子文件夹、每个内含 JSON），
      则对每个**一层**子目录分别查找 JSON，extract_dir 为该子目录，以便 ``{子目录名}_Random_FSSL_UC.json``
      能正确解析出实验名。
    同名实验在多个子目录各有一份时，按子目录与文件名字典序**后出现的覆盖先前的**（通常 exp_all 在后，与其它单策略目录互补）。
    """
    names = [
        f
        for f in os.listdir(folder_path)
        if f.endswith('.json') and f != 'config.json'
    ]
    if names:
        return [(j, folder_path) for j in sorted(names)]

    tasks: List[Tuple[str, str]] = []
    for sub in sorted(os.listdir(folder_path)):
        sub_path = os.path.join(folder_path, sub)
        if not os.path.isdir(sub_path) or sub.startswith('.'):
            continue
        sub_json = [
            f
            for f in os.listdir(sub_path)
            if f.endswith('.json') and f != 'config.json'
        ]
        for j in sorted(sub_json):
            tasks.append((j, sub_path))
    return tasks


def load_results_from_folder(folder_path: str, verbose: bool = True) -> tuple:
    """
    从文件夹中读取所有JSON文件
    
    参数:
        folder_path: 文件夹路径
        verbose: 是否打印加载日志

    返回:
        all_results: 所有实验结果 {exp_name: metrics}
        config: 配置（从第一个文件读取）
    """
    all_results = {}
    config = None
    
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"文件夹不存在: {folder_path}")
    
    tasks = _iter_metrics_json_tasks(folder_path)
    if not tasks:
        raise ValueError(f"文件夹中没有JSON文件: {folder_path}")

    if verbose:
        print(f"📂 从文件夹读取: {folder_path}")
        print(f"   找到 {len(tasks)} 个JSON文件（含子目录聚合时跨目录计数）")

    for json_file, extract_dir in tasks:
        file_path = os.path.join(extract_dir, json_file)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            exp_name = extract_exp_name(json_file, extract_dir)

            metrics = data.get('metrics', data)
            file_config = data.get('config', {})

            if config is None:
                config = file_config

            if exp_name in all_results:
                if verbose:
                    print(
                        f"   ℹ️ 重复键 {exp_name!r}: 用 {os.path.relpath(file_path, folder_path)} 覆盖同键先前结果"
                    )
            all_results[exp_name] = metrics
            if verbose:
                rel = os.path.relpath(file_path, folder_path)
                print(f"   ✅ 已加载: {rel} -> {legend_label(exp_name)}")

        except Exception as e:
            if verbose:
                print(f"   ⚠️ 跳过 {file_path}: {e}")

    if not all_results:
        raise ValueError("没有成功加载任何实验结果")

    return all_results, config


# 达到固定测试精度所需轮数：折线图横轴（目标精度 %）；须在 ``plot_comparison_rounds_to_target_accuracy`` 默认参数之前定义。
DEFAULT_ROUNDS_TO_TARGET_ACCURACIES: Tuple[float, ...] = (81.0, 83.0, 85.0, 87.0, 89.0, 91.0)
#81.0, 83.0, 85.0, 87.0, 89.0, 91.0, 93.0


def _first_round_reaching_accuracy(
    rounds: List,
    acc: List,
    threshold: float,
) -> Optional[float]:
    """按轮次顺序，返回首次满足 test_accuracy >= threshold 的通信轮次；若从未达到则 None。"""
    for r, a in zip(rounds, acc):
        try:
            v = float(a)
        except (TypeError, ValueError):
            continue
        if np.isnan(v):
            continue
        if v >= threshold:
            try:
                return float(r)
            except (TypeError, ValueError):
                continue
    return None


def plot_comparison(all_results: Dict, config: Dict, output_dir: str,
                    max_rounds: int = None) -> Optional[str]:
    """绘制多实验 Test Accuracy 对比图（单面板）。
    
    max_rounds: 若指定，则 x 轴仅显示 0 到 max_rounds 的轮次范围
    """
    if len(all_results) <= 1:
        return None
    
    try:
        cfg = config or {}
        fig, ax = plt.subplots(figsize=_comparison_figsize())
        
        keys = _ordered_experiment_keys(all_results)
        seen_labels = set()

        for exp_name in keys:
            metrics = all_results[exp_name]
            rounds = list(metrics['round'])
            acc = list(metrics['test_accuracy'])
            if max_rounds is not None:
                idx = [j for j, r in enumerate(rounds) if r <= max_rounds]
                rounds = [rounds[j] for j in idx]
                acc = [acc[j] for j in idx]
            if not rounds:
                continue
            color = _color_by_family(exp_name)
            ax.plot(
                rounds,
                acc,
                color=color,
                linewidth=_comparison_line_width(),
                label=_unique_legend_label(exp_name, seen_labels),
                **_line_marker_plot_kwargs(exp_name, color, len(rounds)),
            )
        
        ref_metrics = next(iter(all_results.values()))
        xb = continual_frame_boundary_x(ref_metrics, cfg)
        if xb is not None:
            ax.axvline(x=xb, color='gray', linestyle=':', alpha=0.75, linewidth=1.5, label='Frame 1 → Frame 2')
        
        _apply_comparison_axes(
            ax, 'Communication Round', 'Test Accuracy (%)',
            config=cfg, max_rounds=max_rounds, legend_loc='lower right',
        )
        ax.set_ylim(50, 95)
        
        plt.tight_layout()
        
        # 文件名包含参数
        param_str = get_param_str(config) if config else ""
        if param_str:
            plot_filename = f"comparison_{param_str}.png"
        else:
            plot_filename = "comparison_plot.png"
        if max_rounds is not None:
            plot_filename = plot_filename.replace('.png', f'_r{max_rounds}.png')
        plot_path = os.path.join(output_dir, plot_filename)
        _save_comparison_figure(fig, plot_path)
        plt.close()
        
        print(f"   ✅ 对比图: {plot_path}")
        return plot_path
        
    except Exception as e:
        print(f"   ⚠️ 对比图绘制失败: {e}")
        return None

def plot_comparison_envelope(all_results: Dict, config: Dict, output_dir: str,
                             max_rounds: int = None) -> Optional[str]:
    """绘制对比包络线图（精度与运行最大值分离展示）
    
    max_rounds: 若指定，则 x 轴仅显示 0 到 max_rounds 的轮次范围
    """
    if len(all_results) <= 1:
        return None

    try:
        cfg = config or {}
        fig, ax = plt.subplots(figsize=_comparison_envelope_figsize())

        keys = _ordered_experiment_keys(all_results)
        seen_labels = set()

        for exp_name in keys:
            metrics = all_results[exp_name]
            rounds = metrics['round']
            acc = metrics['test_accuracy']
            if max_rounds is not None:
                idx = [j for j, r in enumerate(rounds) if r <= max_rounds]
                rounds = [rounds[j] for j in idx]
                acc = [acc[j] for j in idx]
            if not rounds:
                continue
            raw_acc = np.asarray(acc, dtype=float)
            envelope = np.maximum.accumulate(raw_acc)
            color = _color_by_family(exp_name)
            linestyle = _exp_linestyle(exp_name)

            show_shading = _envelope_shading_enabled(exp_name)

            if show_shading:
                ax.plot(
                    rounds,
                    raw_acc,
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.0,
                    alpha=0.09,
                    zorder=1,
                )
                ax.fill_between(
                    rounds,
                    raw_acc,
                    envelope,
                    color=color,
                    alpha=0.018,
                    linewidth=0,
                    zorder=0,
                )
            ax.plot(
                rounds,
                envelope,
                color=color,
                linestyle=linestyle,
                linewidth=_comparison_line_width(),
                label=_unique_legend_label(exp_name, seen_labels),
                zorder=2,
            )

        ref_metrics = next(iter(all_results.values()))
        xb = continual_frame_boundary_x(ref_metrics, cfg)
        if xb is not None:
            ax.axvline(x=xb, color='gray', linestyle=':', alpha=0.75, linewidth=1.5, label='Frame 1 → Frame 2')

        _apply_comparison_axes(
            ax, 'Communication Round', 'Test Accuracy (%)',
            config=cfg, max_rounds=max_rounds, legend_loc='lower right',
        )
        ax.set_ylim(75, 95)

        plt.tight_layout()

        param_str = get_param_str(config) if config else ""
        if param_str:
            plot_filename = f"comparison_envelope_{param_str}.png"
        else:
            plot_filename = "comparison_envelope_plot.png"
        if max_rounds is not None:
            plot_filename = plot_filename.replace('.png', f'_r{max_rounds}.png')
        plot_path = os.path.join(output_dir, plot_filename)
        _, pdf_path = _save_comparison_figure_png_and_pdf(fig, plot_path)
        plt.close()

        print(f"   ✅ 包络线对比图: {plot_path}")
        print(f"   ✅ 包络线对比图 (PDF): {pdf_path}")
        return plot_path

    except Exception as e:
        print(f"   ⚠️ 包络线对比图绘制失败: {e}")
        return None


def plot_comparison_clients_per_round(
    all_results: Dict,
    config: Dict,
    output_dir: str,
    max_rounds: int = None,
) -> Optional[str]:
    """多实验折线图：横轴通信轮次，纵轴该轮参与的调度客户端数量 len(selected_clients[i])。

    NoClientTrain 无客户端调度，不画曲线。与 plot_comparison 相同图例顺序与线型。
    """
    if len(all_results) <= 1:
        return None

    try:
        cfg = config or {}
        fig, ax = plt.subplots(figsize=_comparison_figsize())

        keys = _ordered_experiment_keys(all_results)
        seen_labels = set()
        n_lines = 0
        for exp_name in keys:
            if _strategy_family_key(exp_name) == 'NCT':
                continue
            metrics = all_results[exp_name]
            sc = metrics.get('selected_clients')
            if not sc:
                continue
            rounds = list(metrics.get('round') or [])
            n = min(len(rounds), len(sc))
            if n == 0:
                continue
            rounds = rounds[:n]
            counts = [
                len(sc[i]) if sc[i] is not None else 0 for i in range(n)
            ]
            if max_rounds is not None:
                idx = [j for j, r in enumerate(rounds) if r <= max_rounds]
                rounds = [rounds[j] for j in idx]
                counts = [counts[j] for j in idx]
            if not rounds:
                continue

            color = _color_by_family(exp_name)
            ax.plot(
                rounds,
                counts,
                color=color,
                linewidth=_comparison_line_width(),
                label=_unique_legend_label(exp_name, seen_labels),
                **_line_marker_plot_kwargs(exp_name, color, len(rounds)),
            )
            n_lines += 1

        if n_lines == 0:
            plt.close()
            return None

        ref_metrics = next(iter(all_results.values()))
        xb = continual_frame_boundary_x(ref_metrics, cfg)
        if xb is not None:
            ax.axvline(
                x=xb,
                color='gray',
                linestyle=':',
                alpha=0.75,
                linewidth=1.5,
                label='Frame 1 → Frame 2',
            )

        _apply_comparison_axes(
            ax,
            'Communication Round',
            'Number of devices scheduled',
            config=cfg,
            max_rounds=max_rounds,
            legend_loc='best',
        )
        ax.set_ylim(bottom=0)

        plt.tight_layout()

        param_str = get_param_str(config) if config else ""
        if param_str:
            plot_filename = f"comparison_clients_per_round_{param_str}.png"
        else:
            plot_filename = "comparison_clients_per_round_plot.png"
        if max_rounds is not None:
            plot_filename = plot_filename.replace('.png', f'_r{max_rounds}.png')
        plot_path = os.path.join(output_dir, plot_filename)
        _save_comparison_figure(fig, plot_path)
        plt.close()

        print(f"   ✅ 每轮调度客户端数对比图: {plot_path}")
        return plot_path

    except Exception as e:
        print(f"   ⚠️ 每轮调度客户端数对比图绘制失败: {e}")
        return None


def plot_comparison_rounds_to_target_accuracy(
    all_results: Dict,
    config: Dict,
    output_dir: str,
    max_rounds: int = None,
    targets: Tuple[float, ...] = DEFAULT_ROUNDS_TO_TARGET_ACCURACIES,
) -> Optional[str]:
    """
    多实验折线图：横轴为目标测试精度（默认 80/82/84/86/88/90），纵轴为首次达到该精度所需的通信轮次。
    与 plot_comparison 使用相同的曲线顺序、颜色与线型。
    """
    if len(all_results) <= 1:
        return None

    try:
        cfg = config or {}
        xs = list(targets)
        fig, ax = plt.subplots(figsize=_rounds_to_target_figsize())

        keys = _ordered_experiment_keys(all_results)
        seen_labels = set()
        ymax_candidates: List[float] = []

        for exp_name in keys:
            metrics = all_results[exp_name]
            rounds = list(metrics['round'])
            acc = list(metrics['test_accuracy'])
            if max_rounds is not None:
                idx = [j for j, r in enumerate(rounds) if r <= max_rounds]
                rounds = [rounds[j] for j in idx]
                acc = [acc[j] for j in idx]
            if not rounds:
                continue

            ys: List[float] = []
            for thr in targets:
                r_first = _first_round_reaching_accuracy(rounds, acc, thr)
                if r_first is not None:
                    ys.append(r_first)
                    ymax_candidates.append(r_first)
                else:
                    ys.append(float('nan'))

            if np.all(np.isnan(ys)):
                continue

            color = _color_by_family(exp_name)
            ax.plot(
                xs,
                ys,
                color=color,
                linewidth=_comparison_line_width(),
                label=_unique_legend_label(exp_name, seen_labels),
                **_scatter_marker_plot_kwargs(exp_name, color),
            )

        # 若没有任何曲线被绘制（理论上 keys 非空时很少发生）
        if not ax.get_lines():
            plt.close()
            return None

        _apply_comparison_axes(
            ax,
            'Target test accuracy (%)',
            'Communication round (first round ≥ target)',
            apply_round_xlim=False,
            legend_loc='upper left',
        )
        ax.set_xticks(xs)
        ax.set_xticklabels([f'{int(t)}' if t == int(t) else str(t) for t in xs])
        ax.set_xlim(min(xs), max(xs))
        ax.margins(x=0)
        ax.set_ylim(bottom=0)
        if ymax_candidates:
            y_hi = max(ymax_candidates) * 1.08
            ax.set_ylim(top=max(y_hi, 1.0))
        elif max_rounds is not None:
            ax.set_ylim(0, float(max_rounds))

        plt.tight_layout()

        param_str = get_param_str(config) if config else ""
        if param_str:
            plot_filename = f"comparison_rounds_to_target_{param_str}.png"
        else:
            plot_filename = "comparison_rounds_to_target_plot.png"
        if max_rounds is not None:
            plot_filename = plot_filename.replace('.png', f'_r{max_rounds}.png')
        plot_path = os.path.join(output_dir, plot_filename)
        _, pdf_path = _save_comparison_figure_png_and_pdf(fig, plot_path)
        plt.close()

        print(f"   ✅ 达到目标精度所需轮数对比图: {plot_path}")
        print(f"   ✅ 达到目标精度所需轮数对比图 (PDF): {pdf_path}")
        return plot_path

    except Exception as e:
        print(f"   ⚠️ 达到目标精度所需轮数对比图绘制失败: {e}")
        return None


def plot_comparison_continual_bcd(
    all_results: Dict,
    config: Dict,
    output_dir: str,
    max_rounds: int = None,
) -> List[str]:
    """
    continual_eval 中 (b)(c)(c2)(d) 的跨实验对比，各输出一张图：
      (b) True new → pred old（不测 True old → pred new）
      (c) Pseudo precision (new)（仅真标签为新类的过阈样本）
      (c2) Pseudo precision total（全体无标签客户端：过阈旧+新合并的总体精度）
      (d) True new → pseudo old (rate)
    仅含 5 种基线调度（无 GT/Gall）；颜色与 FAMILY_COLORS 一致；线型/标记与主对比图一致（便于区分 R_FLFL / R_FSSL_UC 等）。
    """
    keys = _ordered_baseline_experiment_keys(all_results)
    if not keys:
        return []

    specs = [
        (
            ('test', 'misclass_new_to_old_rate'),
            'Comparison: (b) Test Misclassification — True new → pred old (rate)',
            'Misclassification rate (%)',
            'comparison_continual_misclass_new_to_old',
            None,
        ),
        (
            ('pseudo_unlabeled_agg', 'pseudo_precision_new'),
            'Comparison: (c) Unlabeled Pseudo-Label Precision — pseudo precision (new)',
            'Precision (%)',
            'comparison_continual_pseudo_precision_new',
            (20, 105),
        ),
        (
            ('pseudo_unlabeled_agg', 'pseudo_precision_total'),
            'Comparison: (c2) Unlabeled Pseudo-Label Precision — total (old + new, all unlabeled devices)',
            'Precision (%)',
            'comparison_continual_pseudo_precision_total',
            (80, 100),
        ),
        (
            ('pseudo_unlabeled_agg', 'pseudo_misclass_new_to_old_rate'),
            'Comparison: (d) Unlabeled Pseudo: New→Old — True new → pseudo old (rate)',
            'Rate (%)',
            'comparison_continual_pseudo_new_to_old',
            (0, 105),
        ),
    ]

    cfg = config or {}
    param_str = get_param_str(config) if config else ''
    out_paths: List[str] = []

    for path_keys, title_line, ylabel, fname_base, ylim in specs:
        try:
            fig, ax = plt.subplots(figsize=_continual_plot_figsize(fname_base))
            use_envelope = fname_base in _CONTINUAL_ENVELOPE_PLOTS
            seen_labels = set()
            n_plotted = 0
            for exp_name in keys:
                metrics = all_results[exp_name]
                ser = _continual_eval_pct_series(metrics, max_rounds, *path_keys)
                if ser is None:
                    continue
                rounds, vals = ser
                color = _color_by_family(exp_name)
                if use_envelope:
                    _plot_continual_envelope_curve(
                        ax,
                        rounds,
                        vals,
                        color,
                        exp_name,
                        fname_base=fname_base,
                        label=_unique_legend_label(exp_name, seen_labels),
                    )
                else:
                    ax.plot(
                        rounds,
                        vals,
                        color=color,
                        linewidth=_comparison_line_width(),
                        label=_unique_legend_label(exp_name, seen_labels),
                        **_line_marker_plot_kwargs(exp_name, color, len(rounds)),
                    )
                n_plotted += 1

            if n_plotted == 0:
                plt.close()
                continue

            ref_metrics = all_results[keys[0]]
            xb = continual_frame_boundary_x(ref_metrics, cfg)
            if xb is not None:
                ax.axvline(
                    x=xb,
                    color='gray',
                    linestyle=':',
                    alpha=0.75,
                    linewidth=1.5,
                    label='Frame 1 → Frame 2',
                )

            _apply_comparison_axes(
                ax, 'Communication Round', ylabel,
                config=cfg, max_rounds=max_rounds, legend_loc='best',
            )
            if ylim is not None:
                ax.set_ylim(ylim)

            plt.tight_layout()

            if param_str:
                plot_filename = f'{fname_base}_{param_str}.png'
            else:
                plot_filename = f'{fname_base}.png'
            if max_rounds is not None:
                plot_filename = plot_filename.replace('.png', f'_r{max_rounds}.png')
            plot_path = os.path.join(output_dir, plot_filename)
            if _continual_plot_save_pdf(fname_base):
                _, pdf_path = _save_comparison_figure_png_and_pdf(fig, plot_path)
                plt.close()
                print(f'   ✅ continual 子指标对比: {plot_path}')
                print(f'   ✅ continual 子指标对比 (PDF): {pdf_path}')
            else:
                _save_comparison_figure(fig, plot_path)
                plt.close()
                print(f'   ✅ continual 子指标对比: {plot_path}')
            out_paths.append(plot_path)
        except Exception as e:
            print(f'   ⚠️ continual 对比图 ({fname_base}) 失败: {e}')

    return out_paths


def plot_summary_bar(all_results: Dict, config: Dict, output_dir: str) -> Optional[str]:
    """绘制柱状图"""
    if len(all_results) <= 1:
        return None

    try:
        fig, ax = plt.subplots(figsize=_comparison_figsize())
        s = COMPARISON_PLOT_STYLE

        keys = _ordered_experiment_keys(all_results)
        exp_names: List[str] = []
        avg_accs: List[float] = []
        best_accs: List[float] = []
        colors: List = []

        for exp_name in keys:
            acc_list = all_results[exp_name]['test_accuracy']
            arr = np.asarray(acc_list, dtype=float)
            if arr.size == 0:
                continue
            avg_v = float(np.nanmean(arr))
            best_v = float(np.nanmax(arr))
            if not np.isfinite(avg_v):
                avg_v = 0.0
            if not np.isfinite(best_v):
                best_v = 0.0
            avg_accs.append(avg_v)
            best_accs.append(best_v)
            exp_names.append(exp_name)
            colors.append(_color_by_family(exp_name))

        if not exp_names:
            return None

        x = np.arange(len(exp_names))
        width = 0.35

        bars1 = ax.bar(x - width / 2, avg_accs, width, label='Average Accuracy',
                       color=colors, alpha=0.8)
        bars2 = ax.bar(x + width / 2, best_accs, width, label='Best Accuracy',
                       color=colors, alpha=0.5, hatch='//')

        ax.set_xlabel('Experiment', fontsize=s['axis_label_fontsize'])
        ax.set_ylabel('Test Accuracy (%)', fontsize=s['axis_label_fontsize'])
        ax.set_xticks(x)
        ax.set_xticklabels([legend_label(e) for e in exp_names], rotation=0, ha='center')
        ax.tick_params(axis='both', labelsize=s['tick_fontsize'])
        ax.legend(fontsize=s['legend_fontsize'])
        _apply_comparison_grid(ax)
        ax.set_ylim(75, 100)

        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=s['annotate_fontsize'])
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=s['annotate_fontsize'])

        plt.tight_layout()

        param_str = get_param_str(config) if config else ""
        if param_str:
            plot_filename = f"summary_bar_{param_str}.png"
        else:
            plot_filename = "summary_bar_plot.png"
        plot_path = os.path.join(output_dir, plot_filename)
        _save_comparison_figure(fig, plot_path)
        plt.close()

        print(f"   ✅ 柱状图: {plot_path}")
        return plot_path

    except Exception as e:
        print(f"   ⚠️ 柱状图绘制失败: {e}")
        return None


def _ordered_strategy_families(families: List[str]) -> List[str]:
    """avg_clients 图：策略族展示顺序（ACT/BC/NCC/R 等）。"""
    priority = [
        'ACT', 'BC', 'NCC', 'R',
        'R_PO',
        'PO_FPM',
        'PO_FPLN',
        'PO_MFLN',
        'R_SN',
        'misc',
    ]

    def _sort_key(f: str) -> tuple:
        try:
            return (0, priority.index(f))
        except ValueError:
            return (1, f)

    return sorted(families, key=_sort_key)


def _collect_avg_clients_per_round(all_results: Dict) -> tuple:
    """从 metrics['selected_clients'] 计算各策略族「全轮次平均每轮客户端数」。

    每个策略族（ACT/BC/NCC/R…）只出现一条柱：GT/Gall 等变体合并，数值为同族各实验均值的平均。
    NoClientTrain（NCT）无客户端调度，不参与本图。
    """
    fam_values: Dict[str, List[float]] = defaultdict(list)
    keys = _ordered_experiment_keys(all_results)
    for exp_name in keys:
        fam = _strategy_family_key(exp_name)
        if fam == 'NCT':
            continue
        metrics = all_results[exp_name]
        sc = metrics.get('selected_clients')
        if not sc:
            continue
        counts = [len(x) for x in sc]
        if not counts:
            continue
        fam_values[fam].append(float(np.mean(counts)))
    if not fam_values:
        return [], [], []
    families = _ordered_strategy_families(list(fam_values.keys()))
    means = [float(np.mean(fam_values[f])) for f in families]
    colors = [FAMILY_COLORS.get(f, FAMILY_COLORS['misc']) for f in families]
    return families, means, colors


def plot_avg_selected_clients_bar(all_results: Dict, config: Dict, output_dir: str) -> Optional[str]:
    """
    柱状图：各**策略族**（ACT/BC/NCC/R）在全部通信轮次上的「平均每轮选中客户端数量」。
    GT/Gall 等变体合并为一条柱，数值为同族各实验均值的平均。
    数据来自 metrics['selected_clients']。
    """
    family_labels, means, colors = _collect_avg_clients_per_round(all_results)
    if not family_labels:
        return None
    try:
        fig, ax = plt.subplots(figsize=_comparison_figsize())
        s = COMPARISON_PLOT_STYLE
        x = np.arange(len(family_labels))
        bars = ax.bar(
            x,
            means,
            color=colors,
            alpha=0.85,
            edgecolor='white',
            linewidth=0.8,
            label='Average selected devices per round (all rounds)',
        )
        ax.set_xlabel('Scheduling policy', fontsize=s['axis_label_fontsize'])
        ax.set_ylabel('Average number of devices per round', fontsize=s['axis_label_fontsize'])
        ax.set_xticks(x)
        ax.set_xticklabels(family_labels, rotation=0, ha='center')
        ax.tick_params(axis='both', labelsize=s['tick_fontsize'])
        ax.legend(loc='upper right', fontsize=s['legend_fontsize'], framealpha=0.9)
        _apply_comparison_grid(ax)
        y_max = max(means) if means else 1.0
        ax.set_ylim(0, max(y_max * 1.12, 0.5))

        for bar, v in zip(bars, means):
            h = bar.get_height()
            ax.annotate(
                f'{v:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords='offset points',
                ha='center',
                va='bottom',
                fontsize=s['annotate_fontsize'],
            )

        plt.tight_layout()
        param_str = get_param_str(config) if config else ""
        plot_filename = (
            f"avg_clients_per_round_{param_str}.png" if param_str else "avg_clients_per_round.png"
        )
        plot_path = os.path.join(output_dir, plot_filename)
        _save_comparison_figure(fig, plot_path)
        plt.close()
        print(f"   ✅ 平均每轮客户端数柱状图: {plot_path}")
        return plot_path
    except Exception as e:
        print(f"   ⚠️ 平均每轮客户端数柱状图绘制失败: {e}")
        return None


def _dict_client_get(d: Optional[Dict], cid: int):
    """JSON 反序列化后客户端 id 可能为 int 或 str。"""
    if not d:
        return None
    if cid in d:
        return d[cid]
    return d.get(str(cid))


def _is_no_client_train_schedule(exp_name: str) -> bool:
    """NoClientTrain 无客户端伪标签流程，不参与 pseudo 全零诊断与汇总。"""
    if exp_name == "NoClientTrain":
        return True
    return scheduling_suffix_key(exp_name) == "NoClientTrain"


def _format_round_list(round_nums: List[int], max_inline: int = 120) -> str:
    """将轮次列表格式化为可读的紧凑字符串（过长则截断提示）。"""
    if not round_nums:
        return "[]"
    if len(round_nums) <= max_inline:
        return str(round_nums)
    head = round_nums[:max_inline]
    return f"{head} ... (共 {len(round_nums)} 个，此处仅列前 {max_inline})"


def print_pseudo_zero_rounds_report(all_results: Dict, config: Optional[Dict] = None) -> None:
    """
    从已保存的 metrics JSON 中，找出「pseudo_mean_confidence==0 且 pseudo_train_precision_all==0」的轮次，
    按策略输出对应通信轮次，并可选打印每轮客户端级明细。
    """
    cfg = config or {}
    le = int(cfg.get("local_epochs_unlabeled", cfg.get("local_epochs", 5)) or 5)

    # 各策略 → 全零轮次列表，用于文末汇总
    strategy_to_zero_rounds: List[Tuple[str, List[int]]] = []

    for exp_name, metrics in sorted(all_results.items(), key=lambda x: legend_label(x[0])):
        if _is_no_client_train_schedule(exp_name):
            continue
        rounds = metrics.get("round") or []
        pm = metrics.get("pseudo_mean_confidence") or []
        pp = metrics.get("pseudo_train_precision_all") or []
        sc_all = metrics.get("selected_clients") or []
        ps_sel = metrics.get("pseudo_selected") or []
        ppc_all = metrics.get("pseudo_per_class_per_client") or []

        n = min(len(rounds), len(pm), len(pp))
        if n == 0:
            continue

        bad_idx = []
        for i in range(n):
            try:
                a, b = float(pm[i]), float(pp[i])
            except (TypeError, ValueError):
                continue
            if abs(a) < 1e-15 and abs(b) < 1e-15:
                bad_idx.append(i)

        if not bad_idx:
            continue

        lab = legend_label(exp_name)
        zero_comm_rounds = [int(rounds[i]) for i in bad_idx]
        strategy_to_zero_rounds.append((lab, zero_comm_rounds))

        print("\n" + "=" * 100)
        print(f"[pseudo 全零轮次] 策略: {lab}")
        print(f"  ★ 本策略对应通信轮次 (共 {len(zero_comm_rounds)} 个): {_format_round_list(zero_comm_rounds)}")
        print(
            f"  配置参考: pseudo_threshold={cfg.get('pseudo_threshold', '?')}, "
            f"clients_per_round={cfg.get('clients_per_round', '?')}, "
            f"local_epochs_unlabeled={le}"
        )
        print(f"  满足「平均置信度≈0 且 平均精度≈0」的轮次数: {len(bad_idx)} / {n}")

        detail_cap = 40
        for k, i in enumerate(bad_idx):
            if k >= detail_cap:
                rest = [rounds[j] for j in bad_idx[detail_cap:]]
                print(f"  ... 其余 {len(bad_idx) - detail_cap} 轮 round 编号: {rest[:30]}{' ...' if len(rest) > 30 else ''}")
                break

            r = rounds[i]
            sel = sc_all[i] if i < len(sc_all) else []
            total_pseudo = ps_sel[i] if i < len(ps_sel) else None
            round_ppc = ppc_all[i] if i < len(ppc_all) else {}

            print(f"  --- round={r} (index={i}) | 选中客户端({len(sel)}个)={sel} | "
                  f"pseudo_selected_sum={total_pseudo}")
            if not sel:
                print("      → 本轮无选中设备（如 NoClientTrain / 带宽未接入任何端），指标均值按空集为 0。")
                continue

            for cid in sel:
                pc = _dict_client_get(round_ppc, int(cid))
                n_pass = int(np.sum(pc)) if pc is not None else -1
                print(
                    f"      device {cid}: 预打伪标签通过样本数(各类之和)={n_pass}"
                )
                if n_pass == 0:
                    print(
                        f"        → 该客户端在本轮全量预打伪标签后无样本过阈，拉低本轮平均置信度/精度。"
                    )

        print("=" * 100)

    if strategy_to_zero_rounds:
        print("\n" + "-" * 100)
        print("[pseudo 全零轮次] 各策略 → 通信轮次 汇总（按策略一行，便于检索）")
        print("-" * 100)
        for lab, zr in strategy_to_zero_rounds:
            print(f"  {lab}")
            print(f"    → 轮次 ({len(zr)} 个): {_format_round_list(zr, max_inline=200)}")
        print("-" * 100)


def print_selected_clients_stats(all_results: Dict) -> None:
    """打印各调度在全部轮次上的「平均每轮客户端数」及标准差、最小、最大。"""
    rows: List[tuple] = []
    seen = set()

    def _row(exp_name: str, metrics: Dict) -> None:
        sc = metrics.get('selected_clients')
        if sc is None:
            return
        counts = [len(x) for x in sc]
        if not counts:
            return
        arr = np.asarray(counts, dtype=float)
        rows.append((
            legend_label(exp_name),
            float(np.mean(arr)),
            int(np.min(arr)),
            int(np.max(arr)),
            len(counts),
        ))

    for exp_name in EXPERIMENT_TYPES:
        if exp_name not in all_results or exp_name in seen:
            continue
        _row(exp_name, all_results[exp_name])
        seen.add(exp_name)

    for exp_name, metrics in sorted(all_results.items()):
        if exp_name in seen:
            continue
        _row(exp_name, metrics)
        seen.add(exp_name)

    if not rows:
        return

    _lw = 40
    print("\n" + "=" * 100)
    print("每轮选中设备数量（全轮次统计，基于 selected_clients）")
    print("=" * 100)
    print(f"{'实验':<{_lw}} {'平均':>10} {'最小':>8} {'最大':>8} {'总轮数':>8}")
    print("-" * 100)
    for name, mean_c, mn, mx, n_rounds in rows:
        disp = name if len(name) <= _lw else name[: _lw - 1] + "…"
        print(f"{disp:<{_lw}} {mean_c:>10.3f} {mn:>8d} {mx:>8d} {n_rounds:>8d}")
    print("=" * 100)


def print_results_summary(all_results: Dict, config: Optional[Dict] = None):
    """打印结果汇总表格；并基于 JSON 输出 pseudo 置信度/精度均为 0 的轮次诊断。"""
    _lw = 20
    print("\n" + "=" * 100)
    print("实验结果汇总")
    print("=" * 100)
    print(f"{'实验':<{_lw}} {'平均准确率':>12} {'最高准确率':>12} {'伪标签精度':>12}")
    print("-" * 100)

    for exp_name in EXPERIMENT_TYPES:
        if exp_name not in all_results:
            continue
        metrics = all_results[exp_name]
        acc_arr = np.asarray(metrics['test_accuracy'], dtype=float)
        mean_acc = float(np.nanmean(acc_arr)) if acc_arr.size else 0.0
        best_acc = float(np.nanmax(acc_arr)) if acc_arr.size else 0.0
        pseudo_prec = _metrics_last_train_precision_all(metrics)
        lab = legend_label(exp_name)
        disp = lab if len(lab) <= _lw else lab[: _lw - 1] + "…"
        print(f"{disp:<{_lw}} {mean_acc:>12f}% {best_acc:>12f}% {pseudo_prec:>12f}")

    for exp_name, metrics in all_results.items():
        if exp_name not in EXPERIMENT_TYPES:
            acc_arr = np.asarray(metrics['test_accuracy'], dtype=float)
            mean_acc = float(np.nanmean(acc_arr)) if acc_arr.size else 0.0
            best_acc = float(np.nanmax(acc_arr)) if acc_arr.size else 0.0
            pseudo_prec = _metrics_last_train_precision_all(metrics)
            lab = legend_label(exp_name)
            disp = lab if len(lab) <= _lw else lab[: _lw - 1] + "…"
            print(f"{disp:<{_lw}} {mean_acc:>12f}% {best_acc:>12f}% {pseudo_prec:>12f}")

    print("=" * 100)
    print_selected_clients_stats(all_results)
    print_pseudo_zero_rounds_report(all_results, config)


def plot_from_folder(
    folder_path: str,
    output_dir: str = None,
    plot_comparison_flag: bool = True,
    plot_bar_flag: bool = True,
    plot_avg_clients_bar_flag: bool = True,
    plot_per_experiment_standard: bool = False,
    max_rounds: int = None,
) -> List[str]:
    """
    从文件夹读取 JSON 并绘图：默认先为每个实验生成与 ``plot_all_results`` 一致的
    test_acc / loss / continual_eval 等单实验图；随后在存在多个 JSON 时绘制对比类图表。

    参数:
        folder_path: 包含 JSON 文件的文件夹路径
        output_dir: 输出目录（默认与输入文件夹相同）
        plot_comparison_flag: 是否绘制对比图组（test accuracy、包络线、每轮调度客户端数、
            达到目标精度所需轮数、continual 子指标基线对比）
        plot_bar_flag: 是否绘制各实验平均/最高精度柱状图（summary_bar_*.png）
        plot_avg_clients_bar_flag: 是否绘制按策略族汇总的平均每轮客户端数柱状图（avg_clients_per_round_*.png）
        plot_per_experiment_standard: 是否为每个 JSON 生成单实验标准曲线（与 ``visualization.plot_per_experiment_figures_from_metrics`` 一致）
        max_rounds: 对比图 x 轴最大轮次（如 200），不指定则显示全部

    返回:
        保存的图片文件路径列表
    """
    # 加载数据
    all_results, config = load_results_from_folder(folder_path)
    
    # 设置输出目录
    if output_dir is None:
        output_dir = folder_path
    os.makedirs(output_dir, exist_ok=True)
    
    plot_files = []
    
    print(f"\n📊 开始绘图，输出目录: {output_dir}")

    if plot_per_experiment_standard:
        for exp_name, metrics in all_results.items():
            plot_files.extend(
                plot_per_experiment_figures_from_metrics(
                    exp_name,
                    metrics,
                    config or {},
                    output_dir,
                    standard_plots=True,
                    continual_eval_long=True,
                    max_rounds=max_rounds,
                )
            )

    # 对比图（需至少 2 个实验 JSON）
    if plot_comparison_flag:
        file = plot_comparison(all_results, config, output_dir, max_rounds=max_rounds)
        if file:
            plot_files.append(file)
        # 2b. 对比包络线图（单独输出）
        env_file = plot_comparison_envelope(all_results, config, output_dir, max_rounds=max_rounds)
        if env_file:
            plot_files.append(env_file)
        clients_line = plot_comparison_clients_per_round(
            all_results, config, output_dir, max_rounds=max_rounds
        )
        if clients_line:
            plot_files.append(clients_line)
        rtt_file = plot_comparison_rounds_to_target_accuracy(
            all_results, config, output_dir, max_rounds=max_rounds
        )
        if rtt_file:
            plot_files.append(rtt_file)
        for p in plot_comparison_continual_bcd(
            all_results, config, output_dir, max_rounds=max_rounds
        ):
            plot_files.append(p)

    if plot_bar_flag:
        file = plot_summary_bar(all_results, config, output_dir)
        if file:
            plot_files.append(file)

    if plot_avg_clients_bar_flag:
        file = plot_avg_selected_clients_bar(all_results, config, output_dir)
        if file:
            plot_files.append(file)

    # 打印汇总
    print(f"\n📊 绘图完成！共生成 {len(plot_files)} 个图表")
    print_results_summary(all_results, config)
    
    return plot_files


if __name__ == "__main__":
    import sys

    _raw = sys.argv[1:]
    argv = [
        a
        for a in _raw
        if a not in ("--avg-clients", "--client-stats-only")
    ]
    stats_only = "--avg-clients" in _raw or "--client-stats-only" in _raw

    if len(argv) < 1:
        print("使用方法: python plot_from_json.py <文件夹路径> [输出目录] [最大轮次]")
        print("示例: python plot_from_json.py runs/nopre1000")
        print("示例: python plot_from_json.py runs/nopre1000 . 200   # 仅显示前200轮")
        print("示例: python plot_from_json.py runs/exp_all --avg-clients   # 打印统计并生成设备折线/柱图")
        sys.exit(1)

    folder_path = argv[0]
    output_dir = argv[1] if len(argv) > 1 else None
    max_rounds = None
    if len(argv) > 2:
        try:
            max_rounds = int(argv[2])
        except ValueError:
            pass
    elif len(argv) == 2 and argv[1].isdigit():
        try:
            max_rounds = int(argv[1])
            output_dir = None
        except ValueError:
            pass

    if stats_only:
        all_results, cfg = load_results_from_folder(folder_path)
        out = folder_path
        os.makedirs(out, exist_ok=True)
        print_selected_clients_stats(all_results)
        print_pseudo_zero_rounds_report(all_results, cfg)
        line_path = plot_comparison_clients_per_round(all_results, cfg or {}, out)
        if line_path:
            print(f"\n已保存每轮调度设备数折线图: {line_path}")
        bar_path = plot_avg_selected_clients_bar(all_results, cfg or {}, out)
        if bar_path:
            print(f"已保存平均每轮设备数柱状图: {bar_path}")
        sys.exit(0)

    plot_files = plot_from_folder(
        folder_path,
        output_dir,
        max_rounds=max_rounds,
    )

    print("\n生成的图片文件:")
    for f in plot_files:
        print(f"  - {f}")
        
