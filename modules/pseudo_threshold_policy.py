"""
================================================================================
伪标签阈值策略（priority_old_new / sticky_new）
================================================================================
策略 1: priority_old_new — 旧类 τ_old 固定；新类 τ_new 随全局轮数从初值线性升到终值后保持；
       若 max(新类 softmax) ≥ τ_new，优先采用新类伪标签；否则若 max(旧类) ≥ τ_old 采用旧类。
策略 2: sticky_new — 单一固定阈值 τ；若某样本曾被「预测为新类且置信度≥τ」则置位，之后不再赋予旧类伪标签。

阈值与 ramp 等数值仅来自运行时的 config（须已合并 config.DEFAULT_CONFIG 或等价字段）。
================================================================================
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


def tau_new_linear_schedule(
    global_round: int,
    ramp_start: int,
    ramp_end: int,
    tau_start: float,
    tau_end: float,
) -> float:
    """
    τ_new：在 [ramp_start, ramp_end] 内从 tau_start 线性增至 tau_end；
    之前恒为 tau_start，之后恒为 tau_end。
    """
    global_round = int(global_round)
    r0, r1 = int(ramp_start), int(ramp_end)
    if r1 <= r0:
        return float(tau_end) if global_round >= r0 else float(tau_start)
    if global_round <= r0:
        return float(tau_start)
    if global_round >= r1:
        return float(tau_end)
    t = (global_round - r0) / float(r1 - r0)
    return float(tau_start + t * (tau_end - tau_start))


def tau_new_from_config(global_round: int, config: Optional[Dict]) -> float:
    cfg = config or {}
    return tau_new_linear_schedule(
        global_round,
        int(cfg["pseudo_threshold_new_ramp_start_round"]),
        int(cfg["pseudo_threshold_new_ramp_end_round"]),
        float(cfg["pseudo_threshold_new_initial"]),
        float(cfg["pseudo_threshold_new_final"]),
    )


def tau_old_from_config(config: Optional[Dict]) -> float:
    return float((config or {})["pseudo_threshold_old"])


def sticky_threshold_from_config(config: Optional[Dict]) -> float:
    return float((config or {})["pseudo_sticky_threshold"])


def apply_priority_old_new(
    probs: np.ndarray,
    num_old_classes: int,
    tau_old: float,
    tau_new: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    策略 1：优先新类（新类最大概率 ≥ τ_new），否则在旧类分支上要求 ≥ τ_old。

    参数:
        probs: (N, C) softmax
        num_old_classes: k_old，类别 0..k_old-1 为旧类
    返回:
        preds: (N,) 伪标签（未选中可为 -1）
        mask: (N,) bool
        max_conf: (N,) 选中样本在对应伪标签上的置信度
    """
    k_old = int(num_old_classes)
    if k_old <= 0:
        raise ValueError("num_old_classes 须为正整数")
    N, C = probs.shape
    if C <= k_old:
        raise ValueError("类别数须大于 num_old_classes")

    old_p = probs[:, :k_old]
    new_p = probs[:, k_old:]
    p_old_max = old_p.max(axis=1)
    p_new_max = new_p.max(axis=1)
    pred_old = old_p.argmax(axis=1).astype(np.int64)
    pred_new = k_old + new_p.argmax(axis=1).astype(np.int64)

    assign_new = p_new_max >= tau_new
    assign_old = (~assign_new) & (p_old_max >= tau_old)

    preds = np.full(N, -1, dtype=np.int64)
    mask = np.zeros(N, dtype=bool)
    max_conf = np.zeros(N, dtype=np.float64)

    preds[assign_new] = pred_new[assign_new]
    mask[assign_new] = True
    max_conf[assign_new] = p_new_max[assign_new]

    preds[assign_old] = pred_old[assign_old]
    mask[assign_old] = True
    max_conf[assign_old] = p_old_max[assign_old]

    return preds, mask, max_conf


def apply_sticky_new(
    probs: np.ndarray,
    num_old_classes: int,
    tau: float,
    ever_predicted_new: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    策略 2：固定阈值 τ；若 (argmax 为新类且置信度≥τ) 则置 ever_predicted_new；
    已置位样本不再赋予旧类伪标签（仅当预测为新类且仍≥τ 时选中）。
    """
    k_old = int(num_old_classes)
    N, C = probs.shape
    preds = probs.argmax(axis=1).astype(np.int64)
    max_conf = probs[np.arange(N), preds]

    if ever_predicted_new.shape != (N,):
        raise ValueError("ever_predicted_new 形状须为 (N,)")

    ever = ever_predicted_new.astype(bool).copy()
    ever |= (preds >= k_old) & (max_conf >= tau)
    mask = np.where(ever, (preds >= k_old) & (max_conf >= tau), max_conf >= tau)
    preds_out = preds.copy()
    preds_out[~mask] = -1
    return preds_out, mask, max_conf, ever
