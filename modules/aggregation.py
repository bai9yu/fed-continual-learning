"""
================================================================================
联邦聚合 - FedAvg 及其他聚合算法
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List


@torch.no_grad()
def federated_averaging(global_model: nn.Module, local_models: List[nn.Module],
                        client_sizes: List[int]) -> None:
    """
    FedAvg聚合算法 - 按数据量加权平均

    参数:
        global_model: 全局模型（原地更新）
        local_models: 本地模型列表
        client_sizes: 每个客户端的数据量
    """
    if not local_models:
        return
    tmpl = global_model.state_dict()
    total_size = float(sum(client_sizes))
    aggregated: Dict[str, torch.Tensor] = {}
    lm0_sd = local_models[0].state_dict()

    for key in tmpl.keys():
        if not torch.is_floating_point(tmpl[key]):
            aggregated[key] = lm0_sd[key].clone()
            continue
        acc = torch.zeros_like(tmpl[key])
        for model, size in zip(local_models, client_sizes):
            w = float(size) / total_size
            acc.add_(model.state_dict()[key], alpha=w)
        aggregated[key] = acc
    global_model.load_state_dict(aggregated, strict=True)


def evaluate_model(
    model: nn.Module,
    test_loader,
    device: str,
    num_eval_classes: int = None,
) -> float:
    """
    评估模型在测试集上的准确率
    
    参数:
        model: 待评估的模型
        test_loader: 测试数据加载器
        device: 计算设备
        num_eval_classes: 若指定，仅统计真实标签 < num_eval_classes 的样本（持续学习第一帧等）
        
    返回:
        准确率（百分比）
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            if num_eval_classes is not None:
                mask = labels < num_eval_classes
                if not mask.any():
                    continue
                data = data[mask]
                labels = labels[mask]
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    if total == 0:
        return 0.0
    accuracy = 100 * correct / total
    return accuracy


def evaluate_model_loss(
    model: nn.Module,
    test_loader,
    device: str,
    num_eval_classes: int = None,
) -> float:
    """
    测试集平均交叉熵（与 ``evaluate_model`` 使用相同样本子集：``num_eval_classes`` 为真时仅 label < K）。
    """
    model.eval()
    total_loss = 0.0
    total = 0
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            if num_eval_classes is not None:
                mask = labels < num_eval_classes
                if not mask.any():
                    continue
                data = data[mask]
                labels = labels[mask]
            outputs = model(data)
            total_loss += float(F.cross_entropy(outputs, labels, reduction="sum").item())
            total += labels.size(0)
    if total == 0:
        return 0.0
    return total_loss / total


def evaluate_model_continual_metrics(
    model: nn.Module,
    test_loader,
    device: str,
    num_old_classes: int,
) -> Dict[str, Any]:
    """
    持续学习评估：旧类/新类准确率，以及新类被判为旧类、旧类被判为新类的比例。

    num_old_classes: 基类数 K1（标签 0..K1-1 为旧类，>=K1 为新类）。

    返回字典字段均为 Python 浮点或 None（JSON 可序列化）；比例为 [0,1]。
    """
    k1 = int(num_old_classes)
    if k1 <= 0:
        return {
            "test_acc_all": float(
                evaluate_model(model, test_loader, device, num_eval_classes=None)
            ),
            "test_acc_old": None,
            "test_acc_new": None,
            "n_test_old": 0,
            "n_test_new": 0,
            "misclass_new_to_old_rate": None,
            "misclass_old_to_new_rate": None,
        }

    model.eval()
    total_all = total_old = total_new = 0
    correct_all = correct_old = correct_new = 0
    new_true_pred_old = new_true_total = 0
    old_true_pred_new = old_true_total = 0

    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            predicted = outputs.argmax(1)
            correct = predicted.eq(labels)

            total_all += labels.size(0)
            correct_all += correct.sum().item()

            old_mask = labels < k1
            new_mask = labels >= k1

            to = old_mask.sum().item()
            tn = new_mask.sum().item()
            total_old += to
            total_new += tn
            if to:
                correct_old += correct[old_mask].sum().item()
            if tn:
                correct_new += correct[new_mask].sum().item()
                new_true_pred_old += (predicted[new_mask] < k1).sum().item()
                new_true_total += tn
            if to:
                old_true_pred_new += (predicted[old_mask] >= k1).sum().item()
                old_true_total += to

    acc_all = 100.0 * correct_all / total_all if total_all else 0.0
    acc_old = 100.0 * correct_old / total_old if total_old else None
    acc_new = 100.0 * correct_new / total_new if total_new else None
    r_new_to_old = (
        new_true_pred_old / new_true_total if new_true_total else None
    )
    r_old_to_new = (
        old_true_pred_new / old_true_total if old_true_total else None
    )

    return {
        "test_acc_all": float(acc_all),
        "test_acc_old": float(acc_old) if acc_old is not None else None,
        "test_acc_new": float(acc_new) if acc_new is not None else None,
        "n_test_old": int(total_old),
        "n_test_new": int(total_new),
        "misclass_new_to_old_rate": float(r_new_to_old) if r_new_to_old is not None else None,
        "misclass_old_to_new_rate": float(r_old_to_new) if r_old_to_new is not None else None,
    }
