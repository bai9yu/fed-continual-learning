"""
FedLGMatch 对比实验（Knowledge-Based Systems 2025 思想 + 用户给定超参）。

- 回合初：全局与上一轮本地在弱视图上的联合 softmax 伪标签（权重 ``fedlg_joint_alpha``）。
- 本地：过阈子集上强视图；可选 Batch Mixup（``fedlg_mixup_alpha``, ``fedlg_mixup_lambda``）。
- 调度与 Random 组合见 ``FEDLGMATCH_STRATEGY_OVERRIDES``。
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Beta
from torch.utils.data import DataLoader, Dataset

from local_update import (
    _empty_train_pl_stats,
    _pack_train_pl_stats,
    _weak_strong_tensors_for_index,
    masked_pseudo_wrong_true_equals_runner_up_counts,
)


def _cfg_val(cfg: Optional[Dict], key: str, defaults: Dict[str, Any]) -> Any:
    c = cfg or {}
    if key in c and c[key] is not None:
        return c[key]
    return defaults[key]


# 与用户 itemize 一致：τ=0.95，Mixup α=0.75，λ_m=1.0，E=5（E 在 overrides 里写 local_epochs_unlabeled）
FEDLGMATCH_DEFAULTS: Dict[str, Any] = {
    "pseudo_threshold": 0.95,
    "local_epochs_unlabeled": 5,
    "fedlg_joint_alpha": 0.5,
    "fedlg_mixup_alpha": 0.75,
    "fedlg_mixup_lambda": 1.0,
}

FEDLGMATCH_STRATEGY_OVERRIDES: Dict[str, Any] = {
    "pseudo_label_policy": "fedlgmatch",
    **FEDLGMATCH_DEFAULTS,
}


class _IndexedUnlabeledSubset(Dataset):
    def __init__(self, client_dataset, indices: np.ndarray, pseudo: np.ndarray):
        self.client_dataset = client_dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.pseudo = np.asarray(pseudo, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, i: int):
        j = int(self.indices[i])
        weak, strong = _weak_strong_tensors_for_index(self.client_dataset, j)
        pl = int(self.pseudo[j])
        return weak, strong, pl, j


@torch.no_grad()
def fedlgmatch_prepare_round_pseudo(
    global_model: nn.Module,
    client_dataset,
    device: str,
    config: Dict,
    prev_local_state_dict: Optional[dict],
    pseudo_threshold: float,
) -> Tuple[int, int]:
    joint_a = float(_cfg_val(config, "fedlg_joint_alpha", FEDLGMATCH_DEFAULTS))
    joint_a = min(max(joint_a, 0.0), 1.0)

    was_training = global_model.training
    global_model.eval()

    loc = copy.deepcopy(global_model)
    if prev_local_state_dict is not None:
        loc.load_state_dict(prev_local_state_dict, strict=False)
    loc.eval()
    loc.to(device)

    n = len(client_dataset)
    if n == 0:
        if was_training:
            global_model.train()
        client_dataset.set_pseudo_labels(
            np.full(0, -1, dtype=np.int64),
            np.zeros(0, dtype=bool),
            confidence=np.zeros(0, dtype=np.float32),
        )
        del loc
        return 0, 0

    bs = int((config or {}).get("batch_size", 64))
    idx_all = np.arange(n, dtype=np.int64)
    preds = np.full(n, -1, dtype=np.int64)
    conf = np.zeros(n, dtype=np.float32)
    sel = np.zeros(n, dtype=bool)
    joint_blocks: List[np.ndarray] = []

    for i in range(0, n, bs):
        chunk = idx_all[i : i + bs]
        weak_list = []
        for j in chunk:
            w, _ = _weak_strong_tensors_for_index(client_dataset, int(j))
            weak_list.append(w)
        x = torch.stack(weak_list).to(device)
        pg = F.softmax(global_model(x), dim=-1)
        pl = F.softmax(loc(x), dim=-1)
        pjoint = joint_a * pg + (1.0 - joint_a) * pl
        pjoint = pjoint / pjoint.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        joint_blocks.append(pjoint.detach().cpu().numpy())
        mprob, midx = pjoint.max(dim=-1)
        mprob_np = mprob.cpu().numpy()
        midx_np = midx.cpu().numpy().astype(np.int64)
        for t, j in enumerate(chunk):
            jj = int(j)
            cmt = float(mprob_np[t])
            if cmt >= float(pseudo_threshold):
                preds[jj] = int(midx_np[t])
                conf[jj] = cmt
                sel[jj] = True
            else:
                preds[jj] = -1
                conf[jj] = cmt
                sel[jj] = False

    if was_training:
        global_model.train()
    del loc

    full_probs = np.concatenate(joint_blocks, axis=0)
    truth_arr = np.asarray(client_dataset.labels.numpy(), dtype=np.int64)
    w_wrong, ru = masked_pseudo_wrong_true_equals_runner_up_counts(
        full_probs, preds, sel, truth_arr,
    )

    client_dataset.set_pseudo_labels(preds, sel, confidence=conf)
    return int(w_wrong), int(ru)


def _soft_cross_entropy(logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    log_p = F.log_softmax(logits, dim=-1)
    return -(soft_targets * log_p).sum(dim=-1).mean()


def local_training_fedlgmatch(
    global_model: nn.Module,
    client_id: int,
    client_dataset,
    local_epochs: int,
    lr: float,
    momentum: float,
    device: str,
    weight_decay: float,
    config: Dict,
    global_round: int,
    num_old_classes: Optional[int],
    num_classes: int,
    optimizer_state: Optional[dict],
    client_model_state_dict: Optional[dict],
) -> Tuple[Optional[nn.Module], int, List, float, Optional[dict], Dict[str, float]]:
    mask = getattr(client_dataset, "pseudo_mask", None)
    pseudo_full = getattr(client_dataset, "pseudo_labels", None)
    if mask is None or pseudo_full is None or not np.any(mask):
        local_model = copy.deepcopy(global_model)
        if client_model_state_dict is not None:
            local_model.load_state_dict(client_model_state_dict, strict=False)
        return (
            local_model,
            0,
            [],
            0.0,
            optimizer_state,
            _empty_train_pl_stats(),
        )

    cfg = config or {}
    subset = np.where(np.asarray(mask, dtype=bool))[0].astype(np.int64)
    ds = _IndexedUnlabeledSubset(client_dataset, subset, np.asarray(pseudo_full))
    bs = int(cfg.get("batch_size", 64))
    nw = int(cfg.get("num_workers", 0))
    seed = int(cfg.get("seed", 42))
    g = torch.Generator()
    g.manual_seed(seed + client_id * 10007 + int(global_round))
    loader = DataLoader(
        ds,
        batch_size=bs,
        shuffle=True,
        num_workers=nw,
        drop_last=False,
        generator=g,
    )

    mix_a = float(_cfg_val(cfg, "fedlg_mixup_alpha", FEDLGMATCH_DEFAULTS))
    lam_m = float(_cfg_val(cfg, "fedlg_mixup_lambda", FEDLGMATCH_DEFAULTS))
    nc = int(num_classes)

    k1 = int(num_old_classes) if num_old_classes is not None else int(cfg["frame1_initial_num_classes"])
    k = int(num_classes)
    labels_np = client_dataset.labels.numpy()
    pc_full = getattr(client_dataset, "pseudo_confidence", None)

    local_model = copy.deepcopy(global_model)
    if client_model_state_dict is not None:
        local_model.load_state_dict(client_model_state_dict, strict=False)
    local_model.train()
    optimizer = optim.SGD(
        local_model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    if optimizer_state is not None:
        try:
            optimizer.load_state_dict(optimizer_state)
        except Exception:
            pass
    for pg in optimizer.param_groups:
        pg["lr"] = lr

    beta = Beta(float(mix_a), float(mix_a))

    step_losses: List[float] = []
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

    for _epoch in range(local_epochs):
        try:
            _, strong_b, pl_b, idx_b = next(iter(loader))
        except StopIteration:
            break
        strong_b = strong_b.to(device)
        pl_b = pl_b.to(device).long()
        bsz = strong_b.shape[0]
        perm = torch.randperm(bsz, device=device)
        lam = beta.sample((bsz,)).to(device).clamp(1e-3, 1.0 - 1e-3)
        if strong_b.dim() == 4:
            lam_img = lam.view(-1, 1, 1, 1)
        else:
            lam_img = lam.view(-1, *([1] * (strong_b.dim() - 1)))
        strong_mix = lam_img * strong_b + (1.0 - lam_img) * strong_b[perm]
        one_a = F.one_hot(pl_b, num_classes=nc).float()
        one_b = F.one_hot(pl_b[perm], num_classes=nc).float()
        soft_t = lam.view(-1, 1) * one_a + (1.0 - lam.view(-1, 1)) * one_b

        logits_s = local_model(strong_mix)
        loss = lam_m * _soft_cross_entropy(logits_s, soft_t)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(local_model.parameters(), 1.0)
        optimizer.step()
        step_losses.append(float(loss.detach()))

        idx_np = idx_b.numpy()
        pl_np = pl_b.detach().cpu().numpy()
        for j in range(pl_b.shape[0]):
            ii = int(idx_np[j])
            pl = int(pl_np[j])
            ty = int(labels_np[ii])
            ok = pl == ty
            if pc_full is not None:
                c = pc_full[ii]
                cfn = float(c.item() if torch.is_tensor(c) else c)
            else:
                cfn = 1.0
            train_n += 1
            train_sum_conf += cfn
            if ok:
                train_correct += 1
            is_old_true = (ty < k1) and (ty < k)
            is_new_true = (ty >= k1) and (ty < k)
            is_pred_old = pl < k1
            is_pred_new = (pl >= k1) and (pl < k)
            if is_old_true:
                n_true_old += 1
                sum_conf_true_old += cfn
                if ok:
                    n_correct_true_old += 1
            if is_new_true:
                n_true_new += 1
                sum_conf_true_new += cfn
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

    mean_loss = float(np.mean(step_losses)) if step_losses else 0.0
    stats = _pack_train_pl_stats(
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
        stats,
    )
