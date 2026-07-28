"""
FSSL-UC 对比实验（Chaddad et al., IEEE IoT Journal 2026；式 (4)(7)(8)、Algorithm 1 思想）。

- 高置信：λ·一致性项（此处为弱 argmax 伪标签的强视图 CE）+ λ_l·EML
- 低置信：λ_l·KL(强‖弱)
- 与用户 itemize 一致：τ=0.95，λ=0.01，λ_l=0.01，EML top-k=2，E=5
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from local_update import _empty_train_pl_stats, _pack_train_pl_stats, _weak_strong_tensors_for_index


def _cfg_val(cfg: Optional[Dict], key: str, defaults: Dict[str, Any]) -> Any:
    c = cfg or {}
    if key in c and c[key] is not None:
        return c[key]
    return defaults[key]


FSSL_UC_DEFAULTS: Dict[str, Any] = {
    "pseudo_threshold": 0.95,
    "local_epochs_unlabeled": 5,
    "fssl_uc_lambda": 0.01,
    "fssl_uc_lambda_l": 0.01,
    "fssl_uc_eml_top_nontarget": 2,
}

FSSL_UC_STRATEGY_OVERRIDES: Dict[str, Any] = {
    "pseudo_label_policy": "fssl_uc",
    **FSSL_UC_DEFAULTS,
}


class _AllUnlabeledIndicesDataset(Dataset):
    def __init__(self, client_dataset):
        self.client_dataset = client_dataset

    def __len__(self) -> int:
        return int(len(self.client_dataset))

    def __getitem__(self, i: int):
        weak, strong = _weak_strong_tensors_for_index(self.client_dataset, int(i))
        return weak, strong, int(i)


def eml_loss_fssl_uc(
    logits_s: torch.Tensor,
    pseudo_class: torch.Tensor,
    k_nt: int,
) -> torch.Tensor:
    if logits_s.shape[0] == 0:
        return logits_s.sum() * 0.0
    p = F.softmax(logits_s, dim=-1)
    bsz, _ = p.shape
    dev = logits_s.device
    batch_idx = torch.arange(bsz, device=dev)
    p_g = p[batch_idx, pseudo_class]
    denom = float(max(int(k_nt) - 1, 1))
    a = (1.0 - p_g + 1e-7) / denom
    p_masked = p.clone()
    p_masked[batch_idx, pseudo_class] = -1.0
    k_take = min(int(k_nt), p.shape[1] - 1)
    if k_take <= 0:
        return logits_s.sum() * 0.0
    topv, _ = p_masked.topk(k_take, dim=-1)
    bjk = topv.clamp(min=1e-10, max=1.0 - 1e-7)
    one_m = (1.0 - a.unsqueeze(1)).clamp(min=1e-7, max=1.0 - 1e-7)
    bracket = a.unsqueeze(1) * torch.log(bjk) + one_m * torch.log(1.0 - bjk)
    return (-bracket.sum(dim=1)).mean()


def local_training_fssl_uc(
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
    pseudo_threshold: float,
) -> Tuple[Optional[nn.Module], int, List, float, Optional[dict], Dict[str, float]]:
    cfg = config or {}
    lam = float(_cfg_val(cfg, "fssl_uc_lambda", FSSL_UC_DEFAULTS))
    lam_l = float(_cfg_val(cfg, "fssl_uc_lambda_l", FSSL_UC_DEFAULTS))
    k_nt = int(_cfg_val(cfg, "fssl_uc_eml_top_nontarget", FSSL_UC_DEFAULTS))

    n = len(client_dataset)
    if n == 0:
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

    ds = _AllUnlabeledIndicesDataset(client_dataset)
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

    k1 = int(num_old_classes) if num_old_classes is not None else int(cfg["frame1_initial_num_classes"])
    k = int(num_classes)
    labels_np = client_dataset.labels.numpy()

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

    tau = float(pseudo_threshold)
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
            weak_b, strong_b, idx_b = next(iter(loader))
        except StopIteration:
            break
        weak_b = weak_b.to(device)
        strong_b = strong_b.to(device)

        logits_w = local_model(weak_b)
        logits_s = local_model(strong_b)
        p_w = F.softmax(logits_w, dim=-1)
        conf, pred = p_w.max(dim=-1)
        high = conf >= tau
        low = ~high

        loss_parts: List[torch.Tensor] = []
        if high.any():
            h = high.nonzero(as_tuple=True)[0]
            loss_parts.append(lam * F.cross_entropy(logits_s[h], pred[h]))
            loss_parts.append(lam_l * eml_loss_fssl_uc(logits_s[h], pred[h], k_nt))
        if low.any():
            lidx = low.nonzero(as_tuple=True)[0]
            log_ps = F.log_softmax(logits_s[lidx], dim=-1)
            pw = p_w[lidx].detach()
            loss_parts.append(lam_l * F.kl_div(log_ps, pw, reduction="batchmean"))

        if not loss_parts:
            continue
        loss = sum(loss_parts)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(local_model.parameters(), 1.0)
        optimizer.step()
        step_losses.append(float(loss.detach()))

        idx_np = idx_b.numpy()
        pred_np = pred.detach().cpu().numpy()
        conf_np = conf.detach().cpu().numpy()
        for j in range(pred.shape[0]):
            ii = int(idx_np[j])
            pl = int(pred_np[j])
            cfn = float(conf_np[j])
            ty = int(labels_np[ii])
            ok = pl == ty
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
