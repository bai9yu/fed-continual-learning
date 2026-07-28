"""
================================================================================
客户端本地训练 - FixMatch 训练逻辑
================================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import copy
from typing import Tuple, Optional, Dict, List, Any

from modules.pseudo_threshold_policy import (
    apply_priority_old_new,
    apply_sticky_new,
    tau_new_from_config,
    tau_old_from_config,
    sticky_threshold_from_config,
)


def masked_pseudo_wrong_true_equals_runner_up_counts(
    probs: np.ndarray,
    pseudo_assigned: np.ndarray,
    mask: np.ndarray,
    true_labels: np.ndarray,
) -> Tuple[int, int]:
    """
    在已采用伪标签 (mask=True) 的样本上，统计：
    - 伪标签与真实标签不一致的数量 n_wrong；
    - 其中真实标签等于模型输出第二大概率对应的类别「次预测类」(runner-up) 的数量。

    probs: (N, C) softmax；pseudo_assigned: (N,) 当前采用的伪标签类 id，
    mask 外样本的 pseudo_assigned 值不参与（需满足 mask 处为有效类 id >=0）。
    """
    m = np.asarray(mask, dtype=bool)
    pred = np.asarray(pseudo_assigned, dtype=np.int64)
    y = np.asarray(true_labels, dtype=np.int64)
    p = np.asarray(probs, dtype=np.float64)
    wrong = m & (pred >= 0) & (pred != y)
    n_wrong = int(wrong.sum())
    if n_wrong == 0:
        return 0, 0
    # 按概率降序的第二类（与 argmax 不同的「次高概率」类）
    sorted_cls = np.argsort(-p, axis=1, kind="stable")
    runner_up = sorted_cls[:, 1]
    n_runner = int((wrong & (runner_up == y)).sum())
    return n_wrong, n_runner


def cosine_lr_for_global_round(
    global_round: int,
    num_rounds: int,
    lr_max: float,
    lr_min: float = 0.0,
) -> float:
    """
    与 ``CosineAnnealingLR(..., T_max=num_rounds, eta_min=lr_min)`` 在 ``last_epoch=global_round`` 时的 LR 一致
    （``global_round`` 为 0..num_rounds-1 的全局联邦轮下标）。
    联邦侧固定 ``lr_min=0``，仅 ``lr_max`` 来自 ``lr_unlabeled``。
    """
    if num_rounds < 1:
        return float(lr_max)
    g = min(max(int(global_round), 0), num_rounds - 1)
    return float(
        lr_min + (lr_max - lr_min) * (1.0 + math.cos(math.pi * g / float(num_rounds))) / 2.0
    )


def generate_pseudo_labels_fixmatch(model: nn.Module, client_dataset,
                                    pseudo_threshold: float, device: str,
                                    class_thresholds: Optional[Dict[int, float]] = None,
                                    pseudo_label_policy: str = "default",
                                    num_old_classes: Optional[int] = None,
                                    global_round: int = 0,
                                    config: Optional[Dict] = None) -> Tuple:
    """
    生成伪标签（FixMatch方式）。返回 8 元组末尾两项为回合级诊断：
    n_wrong_masked、n_masked_wrong_with_true_equals_runner_up。

    pseudo_label_policy:
        - default: 单一 pseudo_threshold（或 class_thresholds 按真标签查表，与旧行为一致）
        - priority_old_new: 旧/新分支阈值 + 新类优先（需 num_old_classes、config 中 ramp）
        - sticky_new: 固定阈值 + ever_predicted_new（使用 client_dataset.ever_predicted_new）
    """
    model.eval()

    size = len(client_dataset)
    all_true_labels = np.full(size, -1, dtype=int)

    original_aug_state = client_dataset.use_strong_aug
    client_dataset.set_strong_augmentation(False)

    temp_loader = DataLoader(client_dataset, batch_size=128, shuffle=False)

    num_classes = None
    full_probs = None

    with torch.no_grad():
        for data, true_labels, indices in temp_loader:
            data = data.to(device)
            outputs = model(data)
            probs_t = F.softmax(outputs, dim=1)
            if num_classes is None:
                num_classes = int(probs_t.shape[1])
                full_probs = np.zeros((size, num_classes), dtype=np.float32)
            idx_np = indices.numpy()
            true_np = true_labels.numpy()
            all_true_labels[idx_np] = true_np
            full_probs[idx_np] = probs_t.cpu().numpy()

    client_dataset.set_strong_augmentation(original_aug_state)
    true_labels = all_true_labels
    cfg = config or {}

    if pseudo_label_policy == "default":
        all_preds = full_probs.argmax(axis=1).astype(np.int64)
        all_max_probs = full_probs[np.arange(size), all_preds].astype(np.float64)
        if class_thresholds is None:
            mask = all_max_probs >= pseudo_threshold
        else:
            thresholds = np.array([
                class_thresholds.get(int(lbl), pseudo_threshold)
                for lbl in true_labels
            ])
            mask = all_max_probs >= thresholds
    elif pseudo_label_policy == "priority_old_new":
        k_old = int(num_old_classes) if num_old_classes is not None else 0
        if k_old <= 0:
            raise ValueError("priority_old_new 需要有效的 num_old_classes")
        tau_old = tau_old_from_config(cfg)
        tau_new = tau_new_from_config(global_round, cfg)
        all_preds, mask, all_max_probs = apply_priority_old_new(
            full_probs, k_old, tau_old, tau_new
        )
        all_preds = all_preds.astype(np.int64)
        mask = mask.astype(bool)
    elif pseudo_label_policy == "sticky_new":
        k_old = int(num_old_classes) if num_old_classes is not None else 0
        if k_old <= 0:
            raise ValueError("sticky_new 需要有效的 num_old_classes")
        tau = sticky_threshold_from_config(cfg)
        if not hasattr(client_dataset, "ever_predicted_new") or client_dataset.ever_predicted_new is None:
            client_dataset.ever_predicted_new = np.zeros(size, dtype=bool)
        ever = client_dataset.ever_predicted_new
        all_preds, mask, all_max_probs, ever_new = apply_sticky_new(
            full_probs, k_old, tau, ever
        )
        client_dataset.ever_predicted_new = ever_new
        all_preds = all_preds.astype(np.int64)
        mask = mask.astype(bool)
    else:
        raise ValueError(f"未知的 pseudo_label_policy: {pseudo_label_policy}")

    num_selected = int(mask.sum())
    mean_confidence = float(all_max_probs[mask].mean()) if num_selected > 0 else 0.0
    if num_selected > 0:
        correct = (all_preds[mask] == true_labels[mask]).sum()
        precision = float(correct / num_selected)
    else:
        precision = 0.0

    n_wrong, n_runner = masked_pseudo_wrong_true_equals_runner_up_counts(
        full_probs, all_preds, mask, true_labels
    )

    return (
        all_preds,
        mask,
        all_max_probs,
        mean_confidence,
        precision,
        num_selected,
        n_wrong,
        n_runner,
    )


def _weak_strong_tensors_for_index(client_dataset, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """无标签 ClientDataset：同一原图弱/强增强。"""
    real_idx = int(client_dataset.indices[idx])
    image, _ = client_dataset.base_dataset[real_idx]
    weak = client_dataset.augmentor.weak_augment(image)
    strong = client_dataset.augmentor.strong_augment(image)
    return weak, strong


def _pseudo_mask_preds_from_probs(
    probs: np.ndarray,
    pseudo_label_policy: str,
    pseudo_threshold: float,
    num_old_classes: Optional[int],
    global_round: int,
    config: Dict,
    idx_np: np.ndarray,
    ever_full: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """对一批 softmax 应用与 generate_pseudo_labels_fixmatch 一致的策略；sticky 时原地更新 ever_full。"""
    cfg = config or {}
    N = probs.shape[0]
    if pseudo_label_policy == "default":
        preds = probs.argmax(axis=1).astype(np.int64)
        max_conf = probs[np.arange(N), preds].astype(np.float64)
        mask = max_conf >= float(pseudo_threshold)
        return preds, mask, preds
    if pseudo_label_policy == "priority_old_new":
        k_old = int(num_old_classes) if num_old_classes is not None else 0
        if k_old <= 0:
            raise ValueError("priority_old_new 需要有效的 num_old_classes")
        tau_old = tau_old_from_config(cfg)
        tau_new = tau_new_from_config(global_round, cfg)
        preds, mask, _mc = apply_priority_old_new(probs, k_old, tau_old, tau_new)
        return preds.astype(np.int64), mask, preds
    if pseudo_label_policy == "sticky_new":
        k_old = int(num_old_classes) if num_old_classes is not None else 0
        if k_old <= 0:
            raise ValueError("sticky_new 需要有效的 num_old_classes")
        tau = sticky_threshold_from_config(cfg)
        if ever_full is None:
            raise ValueError("sticky_new 需要 ever_predicted_new 数组")
        ever_slice = ever_full[idx_np].copy()
        preds_out, mask, _mc, ever_new = apply_sticky_new(probs, k_old, tau, ever_slice)
        ever_full[idx_np] = ever_new
        return preds_out.astype(np.int64), mask, preds_out
    raise ValueError(f"未知的 pseudo_label_policy: {pseudo_label_policy}")


def _local_training_fixmatch_fill_batch(
    global_model: nn.Module,
    client_id: int,
    client_dataset,
    is_labeled: bool,
    local_epochs: int,
    lr: float,
    momentum: float,
    device: str,
    weight_decay: float,
    config: Dict,
    global_round: int,
    pseudo_threshold: float,
    pseudo_label_policy: str,
    num_old_classes: Optional[int],
    num_classes: int,
    optimizer_state: Optional[dict] = None,
    client_model_state_dict: Optional[dict] = None,
) -> Tuple[Optional[nn.Module], int, List, float, Optional[dict], Dict[str, float]]:
    """
    每 local epoch：按标签批（batch_size）流式打伪标签，过阈样本入 buffer；
    满 batch_size 则取前 batch_size 个做一步更新；若扫完数据仍不足 batch_size 则用当前 buffer 内全部样本做一步更新；buffer 为空则本 epoch 不训练。
    local_epochs 即「最多 local_epochs 次单步更新」。
    学习率由联邦按全局轮次余弦算好传入 ``lr``；本地仅 SGD + 策略 B 的 ``optimizer.state_dict()``。

    返回第 6 项为参与反向传播的伪标签样本统计：``n``、``sum_conf``、``n_correct``（用于全局平均置信度 / 精度）。
    """
    empty_stats = _empty_train_pl_stats()
    if is_labeled or client_dataset.supervision_mode != "pseudo":
        return None, 0, [], 0.0, optimizer_state, empty_stats
    cfg = config or {}
    bs = int(cfg.get("batch_size", 64))
    n = len(client_dataset)
    if n == 0:
        return None, 0, [], 0.0, optimizer_state, empty_stats

    if pseudo_label_policy == "sticky_new":
        if not hasattr(client_dataset, "ever_predicted_new") or client_dataset.ever_predicted_new is None:
            client_dataset.ever_predicted_new = np.zeros(n, dtype=bool)
        elif len(client_dataset.ever_predicted_new) != n:
            client_dataset.ever_predicted_new = np.zeros(n, dtype=bool)

    local_model = copy.deepcopy(global_model)
    if client_model_state_dict is not None:
        local_model.load_state_dict(client_model_state_dict, strict=False)
    local_model.train()
    optimizer = optim.SGD(
        local_model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    criterion = nn.CrossEntropyLoss()

    seed = int(cfg.get("seed", 42))

    k1 = int(num_old_classes) if num_old_classes is not None else int(cfg["frame1_initial_num_classes"])
    k = int(num_classes)

    step_losses: List[float] = []
    true_labels_np = client_dataset.labels.numpy()
    train_n = 0
    train_sum_conf = 0.0
    train_correct = 0
    n_true_old = 0
    sum_conf_true_old = 0.0
    n_correct_true_old = 0
    n_true_new = 0
    sum_conf_true_new = 0.0
    n_correct_true_new = 0
    n_pred_old = 0
    n_correct_if_pred_old = 0
    n_pred_new = 0
    n_correct_if_pred_new = 0

    for epoch in range(local_epochs):
        rng = np.random.RandomState(seed + client_id * 10007 + epoch * 1009)
        order = rng.permutation(n)
        pos = 0
        buffer: List[Dict[str, Any]] = []

        while len(buffer) < bs and pos < n:
            end = min(pos + bs, n)
            idx_chunk = order[pos:end]
            pos = end

            weak_list: List[torch.Tensor] = []
            strong_list: List[torch.Tensor] = []
            for i in idx_chunk:
                w, s = _weak_strong_tensors_for_index(client_dataset, int(i))
                weak_list.append(w)
                strong_list.append(s)
            weak_bt = torch.stack(weak_list).to(device)

            local_model.eval()
            with torch.no_grad():
                logits = local_model(weak_bt)
                probs = F.softmax(logits, dim=1).cpu().numpy()
            local_model.train()

            ever_arr = (
                client_dataset.ever_predicted_new
                if pseudo_label_policy == "sticky_new"
                else None
            )
            preds, mask, pl_use = _pseudo_mask_preds_from_probs(
                probs,
                pseudo_label_policy,
                pseudo_threshold,
                num_old_classes,
                global_round,
                cfg,
                np.asarray(idx_chunk, dtype=np.int64),
                ever_arr,
            )

            for j in range(len(idx_chunk)):
                if mask[j]:
                    buffer.append(
                        {
                            "strong": strong_list[j],
                            "pl": int(pl_use[j]),
                            "idx": int(idx_chunk[j]),
                            "conf": float(np.max(probs[j])),
                        }
                    )

        if len(buffer) > 0:
            take = min(len(buffer), bs)
            chunk = buffer[:take]
            strong_bt = torch.stack([c["strong"] for c in chunk]).to(device)
            pl_bt = torch.tensor([int(c["pl"]) for c in chunk], device=device, dtype=torch.long)
            for c in chunk:
                ii = int(c["idx"])
                pl = int(c["pl"])
                conf = float(c["conf"])
                ty = int(true_labels_np[ii])
                ok = pl == ty
                train_n += 1
                train_sum_conf += conf
                if ok:
                    train_correct += 1
                is_old_true = (ty < k1) and (ty < k)
                is_new_true = (ty >= k1) and (ty < k)
                is_pred_old = pl < k1
                is_pred_new = (pl >= k1) and (pl < k)
                if is_old_true:
                    n_true_old += 1
                    sum_conf_true_old += conf
                    if ok:
                        n_correct_true_old += 1
                if is_new_true:
                    n_true_new += 1
                    sum_conf_true_new += conf
                    if ok:
                        n_correct_true_new += 1
                if is_pred_old:
                    n_pred_old += 1
                    if ok:
                        n_correct_if_pred_old += 1
                if is_pred_new:
                    n_pred_new += 1
                    if ok:
                        n_correct_if_pred_new += 1
            optimizer.zero_grad()
            out = local_model(strong_bt)
            loss_u = criterion(out, pl_bt)
            step_losses.append(float(loss_u.detach()))
            loss_u.backward()
            optimizer.step()

    mean_loss = float(np.mean(step_losses)) if step_losses else 0.0
    train_stats = _pack_train_pl_stats(
        train_n,
        train_sum_conf,
        train_correct,
        n_true_old,
        sum_conf_true_old,
        n_correct_true_old,
        n_true_new,
        sum_conf_true_new,
        n_correct_true_new,
        n_pred_old,
        n_correct_if_pred_old,
        n_pred_new,
        n_correct_if_pred_new,
    )
    return (
        local_model,
        0,
        [],
        mean_loss,
        optimizer.state_dict(),
        train_stats,
    )


def _empty_train_pl_stats() -> Dict[str, float]:
    return {
        "n": 0.0,
        "sum_conf": 0.0,
        "n_correct": 0.0,
        "n_true_old": 0.0,
        "sum_conf_true_old": 0.0,
        "n_correct_true_old": 0.0,
        "n_true_new": 0.0,
        "sum_conf_true_new": 0.0,
        "n_correct_true_new": 0.0,
        "n_pred_old": 0.0,
        "n_correct_if_pred_old": 0.0,
        "n_pred_new": 0.0,
        "n_correct_if_pred_new": 0.0,
    }


def _pack_train_pl_stats(
    train_n: int,
    train_sum_conf: float,
    train_correct: int,
    n_true_old: int,
    sum_conf_true_old: float,
    n_correct_true_old: int,
    n_true_new: int,
    sum_conf_true_new: float,
    n_correct_true_new: int,
    n_pred_old: int,
    n_correct_if_pred_old: int,
    n_pred_new: int,
    n_correct_if_pred_new: int,
) -> Dict[str, float]:
    return {
        "n": float(train_n),
        "sum_conf": float(train_sum_conf),
        "n_correct": float(train_correct),
        "n_true_old": float(n_true_old),
        "sum_conf_true_old": float(sum_conf_true_old),
        "n_correct_true_old": float(n_correct_true_old),
        "n_true_new": float(n_true_new),
        "sum_conf_true_new": float(sum_conf_true_new),
        "n_correct_true_new": float(n_correct_true_new),
        "n_pred_old": float(n_pred_old),
        "n_correct_if_pred_old": float(n_correct_if_pred_old),
        "n_pred_new": float(n_pred_new),
        "n_correct_if_pred_new": float(n_correct_if_pred_new),
    }


def local_training_fixmatch(
    global_model: nn.Module,
    client_id: int,
    data_loader: Optional[DataLoader],
    client_dataset,
    is_labeled: bool,
    local_epochs: int,
    lr: float,
    momentum: float,
    device: str,
    weight_decay: float = 0.0,
    config: Optional[Dict] = None,
    global_round: int = 0,
    pseudo_threshold: float = 0.8,
    pseudo_label_policy: str = "default",
    num_old_classes: Optional[int] = None,
    num_classes: int = 10,
    optimizer_state: Optional[dict] = None,
    client_model_state_dict: Optional[dict] = None,
) -> Tuple[Optional[nn.Module], int, List, float, Optional[dict], Dict[str, float]]:
    """
    FixMatch 本地训练。

    supervision_mode=pseudo 时：按标签批流式打伪标签，每 local epoch 至多一步更新（满 batch 用满批，否则用已有过阈样本）；
    masked_true / full_true 仍走 DataLoader 路径。

    返回第 4 项为本地无监督损失均值；第 5 项为策略 B 的 ``optimizer.state_dict()``；
    第 6 项为当轮本地训练中实际参与反向传播的伪标签样本统计（``n`` / ``sum_conf`` / ``n_correct``），
    供联邦按调度客户端聚合为全局平均置信度与精度。
    学习率 ``lr`` 由联邦按全局轮次余弦计算后传入。
    """
    cfg = config or {}
    if not is_labeled and getattr(client_dataset, "supervision_mode", "pseudo") == "pseudo":
        return _local_training_fixmatch_fill_batch(
            global_model,
            client_id,
            client_dataset,
            is_labeled,
            local_epochs,
            lr,
            momentum,
            device,
            weight_decay,
            cfg,
            global_round,
            pseudo_threshold,
            pseudo_label_policy,
            num_old_classes,
            num_classes,
            optimizer_state,
            client_model_state_dict,
        )

    empty_stats = _empty_train_pl_stats()

    if data_loader is None or len(data_loader) == 0:
        return None, 0, [], 0.0, optimizer_state, empty_stats

    local_model = copy.deepcopy(global_model)
    if client_model_state_dict is not None:
        local_model.load_state_dict(client_model_state_dict, strict=False)
    local_model.train()

    optimizer = optim.SGD(local_model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    criterion = nn.CrossEntropyLoss()

    k1 = int(num_old_classes) if num_old_classes is not None else int(cfg["frame1_initial_num_classes"])
    k = int(num_classes)

    step_losses: List[float] = []
    labels_np = client_dataset.labels.numpy()
    pc_conf = getattr(client_dataset, "pseudo_confidence", None)
    train_n = 0
    train_sum_conf = 0.0
    train_correct = 0
    n_true_old = 0
    sum_conf_true_old = 0.0
    n_correct_true_old = 0
    n_true_new = 0
    sum_conf_true_new = 0.0
    n_correct_true_new = 0
    n_pred_old = 0
    n_correct_if_pred_old = 0
    n_pred_new = 0
    n_correct_if_pred_new = 0

    for epoch in range(local_epochs):
        for batch in data_loader:
            try:
                weak_data, strong_data, pseudo_labels, pseudo_mask, idx_batch = batch
            except Exception:
                continue

            strong_data = strong_data.to(device)
            pseudo_labels = pseudo_labels.to(device).long()
            pseudo_mask = pseudo_mask.to(device).bool()

            optimizer.zero_grad()

            strong_outputs = local_model(strong_data)

            if pseudo_mask.any():
                selected_logits = strong_outputs[pseudo_mask]
                selected_targets = pseudo_labels[pseudo_mask]
                loss_u = criterion(selected_logits, selected_targets)
                step_losses.append(float(loss_u.detach()))
                pm = pseudo_mask.detach().cpu().numpy()
                idx_np = idx_batch.detach().cpu().numpy()
                pl_np = pseudo_labels.detach().cpu().numpy()
                for j in range(pm.shape[0]):
                    if not pm[j]:
                        continue
                    ii = int(idx_np[j])
                    pl = int(pl_np[j])
                    ty = int(labels_np[ii])
                    ok = pl == ty
                    if pc_conf is not None:
                        c = pc_conf[ii]
                        conf = float(c.item() if torch.is_tensor(c) else c)
                    else:
                        conf = 1.0
                    train_n += 1
                    train_sum_conf += conf
                    if ok:
                        train_correct += 1
                    is_old_true = (ty < k1) and (ty < k)
                    is_new_true = (ty >= k1) and (ty < k)
                    is_pred_old = pl < k1
                    is_pred_new = (pl >= k1) and (pl < k)
                    if is_old_true:
                        n_true_old += 1
                        sum_conf_true_old += conf
                        if ok:
                            n_correct_true_old += 1
                    if is_new_true:
                        n_true_new += 1
                        sum_conf_true_new += conf
                        if ok:
                            n_correct_true_new += 1
                    if is_pred_old:
                        n_pred_old += 1
                        if ok:
                            n_correct_if_pred_old += 1
                    if is_pred_new:
                        n_pred_new += 1
                        if ok:
                            n_correct_if_pred_new += 1
            else:
                loss_u = (strong_outputs * 0).sum()

            loss_u.backward()
            optimizer.step()
            break

    mean_loss = float(np.mean(step_losses)) if step_losses else 0.0
    train_stats = _pack_train_pl_stats(
        train_n,
        train_sum_conf,
        train_correct,
        n_true_old,
        sum_conf_true_old,
        n_correct_true_old,
        n_true_new,
        sum_conf_true_new,
        n_correct_true_new,
        n_pred_old,
        n_correct_if_pred_old,
        n_pred_new,
        n_correct_if_pred_new,
    )
    return (
        local_model,
        0,
        [],
        mean_loss,
        optimizer.state_dict(),
        train_stats,
    )


def finetune_global_model(
    global_model: nn.Module,
    data_loader,
    finetune_epochs: int,
    lr: float,
    momentum: float,
    device: str,
    weight_decay: float = 0.0,
    optimizer_state: Optional[dict] = None,
) -> Optional[dict]:
    """
    使用有标签数据对全局模型进行微调（原地修改）。

    策略 B：每轮新建 ``SGD``，``load_state_dict(optimizer_state)`` 后强制 ``param_groups['lr']=lr``；
    学习率 ``lr`` 由联邦按全局轮次余弦计算。训毕返回 ``optimizer.state_dict()``。
    """
    if data_loader is None or len(data_loader) == 0:
        return optimizer_state

    global_model.train()
    optimizer = optim.SGD(global_model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    for pg in optimizer.param_groups:
        pg["lr"] = lr

    criterion = nn.CrossEntropyLoss()

    for epoch in range(finetune_epochs):
        for data, labels, _ in data_loader:
            data, labels = data.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = global_model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            break

    return optimizer.state_dict()
