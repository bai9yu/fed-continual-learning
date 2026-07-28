"""
FLFL 对比实验（labels-at-server）：全局模型生成本轮伪标签 + SAT 子集、
本地 La（回合初 SAT 子集上的伪标签 CE，与参考 FLFL 一致，不做 step 内 sat_masking）
+ SACR（τ_f 固定高置信 + ASAM + 与教师 KL）、
服务器 LSAA 聚合（β_k ∝ 1−τ_t^k）。
global_ft=1：客户端训练 → LSAA → 服务器有标签微调（由 federated 主循环保证）。

本文件集中存放 FLFL/ASAM 默认超参与 ``Random_FLFL`` 策略覆盖项（见 ``FLFL_STRATEGY_OVERRIDES``）。
"""

from __future__ import annotations

import copy
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from local_update import (
    _empty_train_pl_stats,
    _pack_train_pl_stats,
    masked_pseudo_wrong_true_equals_runner_up_counts,
)

# ---------------------------------------------------------------------------
# FLFL 默认超参（与全局 DEFAULT_CONFIG 解耦；运行时 cfg 中同名键可覆盖）
# ---------------------------------------------------------------------------
FLFL_HPARAM_DEFAULTS: Dict[str, Any] = {
    "flfl_tau_fix": 0.95,
    "flfl_T": 1.0,
    "flfl_use_weak_teacher": False,
    "flfl_sam_rho": 0.1,
    "flfl_sam_eta": 0.01,
    "flfl_lambda_u": 1.0,
    "flfl_lsaa_lr": None,
}

# 供 scheduler.STRATEGY_EXPERIMENT_OVERRIDES["Random_FLFL"] 使用
FLFL_STRATEGY_OVERRIDES: Dict[str, Any] = {
    "pseudo_label_policy": "flfl",
    **FLFL_HPARAM_DEFAULTS,
}


def _flfl(cfg: Optional[Dict], key: str) -> Any:
    c = cfg or {}
    if key in c and c[key] is not None:
        return c[key]
    return FLFL_HPARAM_DEFAULTS[key]


# ---------------------------------------------------------------------------
# ASAM（仅保留 ASAM，与 FLFL 参考实现一致）
# ---------------------------------------------------------------------------
class ASAM:
    def __init__(
        self,
        model: nn.Module,
        rho: float = 0.5,
        eta: float = 0.01,
    ):
        self.model = model
        self.rho = float(rho)
        self.eta = float(eta)
        self.state = defaultdict(dict)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        wgrads = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            t_w = self.state[p].get("eps")
            if t_w is None:
                t_w = torch.clone(p).detach()
                self.state[p]["eps"] = t_w
            if "weight" in n:
                t_w[...] = p[...]
                t_w.abs_().add_(self.eta)
                p.grad.mul_(t_w)
            wgrads.append(torch.norm(p.grad, p=2))
        wgrad_norm = torch.norm(torch.stack(wgrads), p=2) + 1.0e-16
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            t_w = self.state[p].get("eps")
            if "weight" in n:
                p.grad.mul_(t_w)
            eps = t_w
            eps[...] = p.grad[...]
            eps.mul_(self.rho / wgrad_norm)
            p.add_(eps)

        if zero_grad:
            self.model.zero_grad()

    @torch.no_grad()
    def second_step(self) -> None:
        for _, p in self.model.named_parameters():
            if "eps" not in self.state[p]:
                continue
            p.sub_(self.state[p]["eps"])


def _ce_masked(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not mask.any():
        return logits.sum() * 0.0
    return F.cross_entropy(logits[mask], targets[mask])


def kldiv_with_mask(
    logits_s: torch.Tensor,
    logits_t: torch.Tensor,
    mask: torch.Tensor,
    T: float,
) -> torch.Tensor:
    """与参考 criterion.kldiv_with_mask 一致：仅对学生 logits 除以 T，教师为 softmax(logits_t)，不乘 T^2。"""
    if not mask.any():
        return logits_s.sum() * 0.0
    logits_s = F.log_softmax(logits_s[mask] / T, dim=-1)
    logits_w = F.softmax(logits_t[mask].detach(), dim=-1)
    return F.kl_div(logits_s, logits_w, reduction="batchmean")


class _FlflSubsetDataset(Dataset):
    def __init__(
        self,
        client_dataset,
        subset_idx: np.ndarray,
        pseudo: np.ndarray,
        fix_mask: np.ndarray,
    ):
        self.client_dataset = client_dataset
        self.subset = np.asarray(subset_idx, dtype=np.int64)
        self.pseudo = pseudo
        self.fix_mask = fix_mask

    def __len__(self) -> int:
        return int(self.subset.shape[0])

    def __getitem__(self, i: int):
        from local_update import _weak_strong_tensors_for_index

        j = int(self.subset[i])
        weak, strong = _weak_strong_tensors_for_index(self.client_dataset, j)
        pl = int(self.pseudo[j])
        fix = bool(self.fix_mask[j])
        return weak, strong, pl, fix, j


@torch.no_grad()
def flfl_prepare_round_pseudo(
    model: nn.Module,
    client_dataset,
    device: str,
    config: Dict,
) -> Tuple[float, int, int]:
    """
    用当前全局模型在客户端全体无标签上打伪标签；SAT 门控子集写入
    ``client_dataset.flfl_subset_indices``；并 ``set_pseudo_labels`` 供指标统计。
    返回 τ_t^k（max prob 全局均值），以及 (n_wrong_masked, n_true_is_runner_up) 供伪标签诊断。
    """
    cfg = config or {}
    tau_fix = float(_flfl(config, "flfl_tau_fix"))
    model.eval()
    n = len(client_dataset)
    if n == 0:
        client_dataset.flfl_subset_indices = np.zeros(0, dtype=np.int64)
        client_dataset.flfl_pseudo_full = np.full(0, -1, dtype=np.int64)
        client_dataset.flfl_fix_full = np.zeros(0, dtype=bool)
        return 0.0, 0, 0

    bs = int(cfg.get("batch_size", 64))
    idx_all = np.arange(n, dtype=np.int64)
    weak_list: List[torch.Tensor] = []
    for i in range(0, n, bs):
        chunk = idx_all[i : i + bs]
        batch_w = []
        for j in chunk:
            from local_update import _weak_strong_tensors_for_index

            w, _ = _weak_strong_tensors_for_index(client_dataset, int(j))
            batch_w.append(w)
        x = torch.stack(batch_w).to(device)
        logits = model(x)
        weak_list.append(logits.cpu())

    logits_all = torch.cat(weak_list, dim=0)
    probs = torch.softmax(logits_all, dim=-1)
    max_probs, max_idx = probs.max(dim=-1)
    global_t = float(max_probs.mean().item())
    local_t = probs.mean(dim=0).numpy()

    mod = local_t / max(local_t.max(), 1e-12)
    mask_sat = (max_probs.numpy() >= global_t * mod[max_idx.numpy()]).astype(bool)
    fix_mask = (max_probs.numpy() >= tau_fix).astype(bool)
    pseudo = max_idx.numpy().astype(np.int64)
    pseudo_full = np.full(n, -1, dtype=np.int64)
    pseudo_full[mask_sat] = pseudo[mask_sat]

    subset = np.where(mask_sat)[0].astype(np.int64)
    client_dataset.flfl_subset_indices = subset
    client_dataset.flfl_pseudo_full = pseudo_full.copy()
    client_dataset.flfl_fix_full = fix_mask

    conf = np.zeros(n, dtype=np.float32)
    conf[:] = max_probs.numpy().astype(np.float32)
    sel = np.zeros(n, dtype=bool)
    sel[subset] = True
    client_dataset.set_pseudo_labels(pseudo_full, sel, confidence=conf)

    probs_np = probs.numpy().astype(np.float64)
    truth_arr = np.asarray(client_dataset.labels.numpy(), dtype=np.int64)
    w_wrong, ru = masked_pseudo_wrong_true_equals_runner_up_counts(
        probs_np, pseudo_full, sel, truth_arr,
    )
    return global_t, int(w_wrong), int(ru)


def local_training_flfl(
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
    cfg = config or {}
    dev = torch.device(device)
    subset = getattr(client_dataset, "flfl_subset_indices", None)
    if subset is None or len(subset) == 0:
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

    pseudo_full = client_dataset.flfl_pseudo_full
    fix_full = client_dataset.flfl_fix_full
    ds = _FlflSubsetDataset(client_dataset, subset, pseudo_full, fix_full)
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

    num_c = int(num_classes)
    k1 = int(num_old_classes) if num_old_classes is not None else int(cfg["frame1_initial_num_classes"])
    k = int(num_classes)
    labels_np = client_dataset.labels.numpy()
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

    asam = ASAM(
        local_model,
        rho=float(_flfl(cfg, "flfl_sam_rho")),
        eta=float(_flfl(cfg, "flfl_sam_eta")),
    )
    flfl_T = float(_flfl(cfg, "flfl_T"))
    use_weak_teacher = bool(_flfl(cfg, "flfl_use_weak_teacher"))
    lam_u = float(_flfl(cfg, "flfl_lambda_u"))

    for _epoch in range(local_epochs):
        # 每个 local epoch 只训练一个 batch
        try:
            weak_b, strong_b, pl_b, fix_b, idx_b = next(iter(loader))
        except StopIteration:
            break

        weak_b = weak_b.to(device)
        strong_b = strong_b.to(device)
        pl_b = pl_b.to(device).long()
        fix_b = fix_b.to(device).bool()

        logits_w = local_model(weak_b)
        logits_s = local_model(strong_b)
        with torch.no_grad():
            logits_teacher = logits_w if use_weak_teacher else logits_s.detach()

        if fix_b.any():
            pl_fix = torch.where(fix_b, pl_b, torch.full_like(pl_b, -1))
            loss_lp = lam_u * _ce_masked(logits_s, pl_fix, fix_b)
            local_model.zero_grad(set_to_none=True)
            loss_lp.backward()
            asam.first_step(zero_grad=True)
            logits_s_hat = local_model(strong_b)
            kl = kldiv_with_mask(logits_s_hat, logits_teacher, fix_b, flfl_T)
            kl_grad = torch.autograd.grad(
                kl,
                local_model.parameters(),
                allow_unused=True,
            )
            asam.second_step()
            kl_term = kl.detach()
        else:
            kl_grad = tuple(None for _ in local_model.parameters())
            kl_term = torch.zeros((), device=logits_s.device)

        # La：子集内伪标签 CE。若上面已对 logits_s 做过 loss_lp.backward()，原 logits_s 的计算图已释放，
        # 必须在 ASAM second_step 恢复参数后重新 forward(strong_b)，否则 loss_la.backward() 会二次反传报错。
        mask_la = pl_b >= 0
        logits_s_la = local_model(strong_b)
        loss_la = lam_u * _ce_masked(logits_s_la, pl_b, mask_la)

        local_model.zero_grad(set_to_none=True)
        loss_la.backward()
        for p, gkl in zip(local_model.parameters(), kl_grad):
            if gkl is None:
                continue
            if p.grad is None:
                p.grad = gkl.clone()
            else:
                p.grad = p.grad + gkl
        torch.nn.utils.clip_grad_norm_(local_model.parameters(), 1.0)
        optimizer.step()

        step_losses.append(float(loss_la.detach() + kl_term))

        idx_np = idx_b.numpy()
        ml = mask_la.detach().cpu().numpy()
        pl_np = pl_b.detach().cpu().numpy()
        for j in range(ml.shape[0]):
            if not ml[j]:
                continue
            ii = int(idx_np[j])
            pl = int(pl_np[j])
            ty = int(labels_np[ii])
            ok = pl == ty
            if pc_full is not None:
                c = pc_full[ii]
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


def flfl_lsaa_aggregate(
    global_model: nn.Module,
    local_models: List[nn.Module],
    weights: List[float],
    lr: float,
    momentum: float,
    weight_decay: float,
    optimizer_state: Optional[dict],
) -> Optional[dict]:
    """
    SemiFL 式 LSAA：对 weight/bias 参数设 grad = θ − Σ β_k θ_k，再 SGD 一步。
    weights 须已归一化且与 local_models 顺序一致。
    """
    if not local_models:
        return optimizer_state
    opt = optim.SGD(
        global_model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    if optimizer_state is not None:
        try:
            opt.load_state_dict(optimizer_state)
        except Exception:
            pass
    for pg in opt.param_groups:
        pg["lr"] = lr
    opt.zero_grad(set_to_none=True)
    w_arr = [float(w) for w in weights]
    s = sum(w_arr)
    if s <= 0:
        w_arr = [1.0 / len(w_arr)] * len(w_arr)
    else:
        w_arr = [w / s for w in w_arr]

    targets: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name, p in global_model.named_parameters():
            suf = name.split(".")[-1]
            if "weight" not in suf and "bias" not in suf:
                continue
            acc = torch.zeros_like(p.data)
            for lm, beta in zip(local_models, w_arr):
                acc.add_(lm.state_dict()[name], alpha=beta)
            targets[name] = acc

    for name, p in global_model.named_parameters():
        if name in targets:
            p.grad = (p.data - targets[name]).detach()
    opt.step()
    return opt.state_dict()
