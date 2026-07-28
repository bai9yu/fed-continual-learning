"""
================================================================================
预热训练模块 - Warmup Training
================================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, Optional, Tuple

from modules.models import build_model
from config import (
    get_device,
    DEFAULT_CONFIG,
    set_seed,
    get_frame1_total_num_classes,
    normalize_continual_settings,
)
from modules.aggregation import evaluate_model, evaluate_model_continual_metrics

# 模型统一放在项目根下 pre/：serverinit、frame1pre 为预热产物
PRE_ROOT = "pre"
PRE_SERVERINIT = os.path.join(PRE_ROOT, "serverinit")
PRE_FRAME1PRE = os.path.join(PRE_ROOT, "frame1pre")


# ================================================================================
# 用于匹配的关键配置参数
# ================================================================================
WARMUP_CONFIG_KEYS = [
    'server_init_pre_warmup_rounds',
    'server_init_pre_warmup_local_epochs',
    'frame1_pre_warmup_rounds',
    'frame1_pre_warmup_local_epochs',
    'lr_unlabeled',
    'server_batch_size',
    'momentum',
    'weight_decay',
    'frame1_initial_num_classes',
    'frame1_new_num_classes',
    'frame2_new_num_classes',
    'labeled_per_class',
    'alpha',
    'dataset_name',
    'model_name',
    'seed',
]


# ================================================================================
# 配置工具函数
# ================================================================================
def _warmup_param_tag(config: Dict, kind: str) -> str:
    """
    写入预热文件名的参数段：数据集、轮数、本地 epoch、lr/bs/momentum、lpc、seed、model。
    server_init_pre：仅 ``kold``（K_old），来自 ``server_init_pre_warmup_model_path`` 解析规则。
    frame1_pre：仅 ``k2``（frame1 总类数）；文件名中不出现 k1。
    缩写与数字直接相连（如 kold6、k210、r500、e5），中间无下划线。
    """
    ds = str(config.get("dataset_name", "dataset")).replace(" ", "").lower()
    lr = float(config.get("lr_unlabeled", 0.01))
    bs = int(config.get("server_batch_size", 32))
    mom = float(config.get("momentum", 0.9))
    md = str(config.get("model_name", "cnn")).lower()
    seed = int(config.get("seed", 42))
    lr_s = f"{lr:g}"
    mom_s = f"{mom:g}"
    if kind == "server_init_pre":
        k_old = int(config["frame1_initial_num_classes"])
        r = int(config["server_init_pre_warmup_rounds"])
        e = int(config["server_init_pre_warmup_local_epochs"])
        lpc = int(config.get("labeled_per_class", 100))
        return (
            f"{ds}_kold{k_old}_r{r}_e{e}_lr{lr_s}_bs{bs}_lpc{lpc}_mom{mom_s}_{md}_s{seed}"
        )
    if kind == "frame1_pre":
        k2 = int(get_frame1_total_num_classes(config))
        r = int(config["frame1_pre_warmup_rounds"])
        e = int(config["frame1_pre_warmup_local_epochs"])
        lpc = int(config.get("labeled_per_class", 100))
        return (
            f"{ds}_k2{k2}_r{r}_e{e}_lr{lr_s}_bs{bs}_lpc{lpc}_mom{mom_s}_{md}_s{seed}"
        )
    raise ValueError(f"_warmup_param_tag: kind 须为 server_init_pre|frame1_pre，收到: {kind!r}")


def get_warmup_config(config: Dict) -> Dict:
    """
    从完整配置中提取预热相关的配置参数
    
    参数:
        config: 完整配置字典
        
    返回:
        warmup_config: 预热相关配置
    """
    return {key: config[key] for key in WARMUP_CONFIG_KEYS if key in config}


def get_server_init_pre_model_path(base_path: str, config: Dict) -> str:
    """
    server_init_pre：文件名含 kold、r、e、lr、bs、lpc、mom、model、seed。

    若 ``server_init_pre_warmup_model_path`` 指向**已存在**的文件，则直接使用（便于与
    ``server_init_pre_warmup_rounds`` 等标签不一致时仍加载旧检查点）；否则按当前超参生成
    ``pre/serverinit/serverinit_warmup_{tag}.pth``。
    """
    if base_path and os.path.isfile(base_path):
        return os.path.normpath(base_path)
    tag = _warmup_param_tag(config, "server_init_pre")
    _, ext = os.path.splitext(base_path)
    ext = ext or ".pth"
    os.makedirs(PRE_SERVERINIT, exist_ok=True)
    return os.path.join(PRE_SERVERINIT, f"serverinit_warmup_{tag}{ext}")


def get_frame1_pre_k2_warmup_model_path(
    base_path: str, config: Dict, *, for_save: bool = False
) -> str:
    """
    frame1_pre 检查点路径：

    - ``for_save=True``（仅 frame1_pre 训练保存）：**始终**按当前超参生成
      ``pre/frame1pre/frame1pre_k2_warmup_{tag}.pth``，不因 ``frame1_pre_warmup_model_path`` 已存在而写到旧文件。

    - ``for_save=False``（联邦 frame1 等**加载**解析）：若 ``base_path``（``frame1_pre_warmup_model_path``）
      指向**已存在**的文件则直接返回该路径；否则按当前超参生成 tag 路径。
    """
    if not for_save and base_path and os.path.isfile(base_path):
        return os.path.normpath(base_path)
    tag = _warmup_param_tag(config, "frame1_pre")
    _, ext = os.path.splitext(base_path or "")
    ext = ext or ".pth"
    os.makedirs(PRE_FRAME1PRE, exist_ok=True)
    return os.path.join(PRE_FRAME1PRE, f"frame1pre_k2_warmup_{tag}{ext}")


def get_warmup_plot_path(model_path: str, metric_tag: str = "acc") -> str:
    """根据模型路径生成对应的预热曲线图保存路径"""
    root, _ = os.path.splitext(model_path)
    return f"{root}_{metric_tag}.png"


def get_warmup_json_path(model_path: str) -> str:
    """根据模型路径生成对应的预热 JSON 保存路径"""
    root, _ = os.path.splitext(model_path)
    return f"{root}.json"


def _to_serializable(value):
    """将 numpy/torch 标量与容器递归转换为 JSON 可序列化对象。"""
    if isinstance(value, dict):
        return {key: _to_serializable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        if value.dim() == 0:
            return value.item()
        return value.detach().cpu().tolist()
    return value


def save_warmup_json(json_path: str, metrics: Dict = None,
                     config: Dict = None,
                     model_path: str = None,
                     plot_path: str = None) -> str:
    """保存预热阶段配置与指标到 JSON 文件。"""
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    json_payload = {
        'metadata': {
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_path': model_path,
            'plot_path': plot_path,
        },
        'config': _to_serializable(config or {}),
        'metrics': _to_serializable(metrics or {}),
    }

    with open(json_path, 'w', encoding='utf-8') as file:
        json.dump(json_payload, file, indent=2, ensure_ascii=False)

    print(f"✅ 预热 JSON 已保存: {json_path}")
    return json_path


def configs_match(saved_config: Dict, current_config: Dict) -> Tuple[bool, list]:
    """
    检查保存的配置是否与当前配置匹配
    
    参数:
        saved_config: 保存的配置
        current_config: 当前配置
        
    返回:
        (is_match, mismatched_keys): 是否匹配，不匹配的键列表
    """
    if saved_config is None:
        return False, ['config_missing']
    
    mismatched = []
    for key in WARMUP_CONFIG_KEYS:
        if key not in saved_config:
            continue
        saved_val = saved_config.get(key)
        current_val = current_config.get(key)
        if saved_val != current_val:
            mismatched.append(f"{key}: saved={saved_val}, current={current_val}")
    
    return len(mismatched) == 0, mismatched


def plot_warmup_accuracy(metrics: Dict, save_path: str) -> Optional[str]:
    """绘制并保存预热阶段的准确率曲线"""
    rounds = metrics.get('round') or metrics.get('epoch', [])
    if not metrics or not rounds:
        print("⚠️ 预热指标为空，跳过绘图")
        return None
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    train_acc = metrics.get('train_accuracy', [])
    test_acc = metrics.get('test_accuracy', [])

    plt.figure(figsize=(8, 5))
    plt.plot(rounds, train_acc, label='Train Acc', linewidth=2, alpha=0.7)
    plt.plot(rounds, test_acc, label='Test Acc', linewidth=2)
    plt.xlabel('Round')
    plt.ylabel('Accuracy (%)')
    plt.title('Warm-up Accuracy')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ 预热准确率曲线已保存: {save_path}")
    return save_path


# ================================================================================
# 预热训练主函数
# ================================================================================


def warmup_training(data_loader, num_classes: int, warmup_rounds: int, 
                    warmup_local_epochs: int, lr: float, momentum: float,
                    save_path: str = None,
                    config: Dict = None,
                    device: Optional[str] = None, verbose: bool = True,
                    save_path_kind: str = 'server_init_pre') -> Dict:
    """
    save_path_kind:
        ``server_init_pre``：仅服务器、初始旧类 0..K_old-1（pre/serverinit/）；
        ``frame1_pre``：K2 全类服务器预热（pre/frame1pre/）；先加载 ``server_init_pre_warmup_model_path``
        解析后的检查点（K_old），再扩类至 K2 后训练。
    """
    if save_path is None:
        if save_path_kind == "frame1_pre":
            save_path = os.path.join(PRE_FRAME1PRE, "frame1pre_k2_warmup_model.pth")
        else:
            save_path = os.path.join(PRE_SERVERINIT, "serverinit_warmup_model.pth")
    device = device or get_device(config)
    
    print("\n" + "=" * 70)
    if save_path_kind == "frame1_pre":
        print("frame1_pre：K2 全类服务器预热（K2=frame1 总类数）")
    elif save_path_kind == "server_init_pre":
        print("server_init_pre：仅服务器、初始旧类（全类别前的基类子集）")
    else:
        print("预热训练 (Warm-up Stage)")
    print("=" * 70)
    server_loader = getattr(data_loader, 'server_loader', None)
    server_dataset = getattr(data_loader, 'server_dataset', None)
    if server_loader is None or server_dataset is None:
        raise RuntimeError("未找到服务器端有标注数据，请检查 FederatedDataLoader 是否构建 server_loader/server_dataset")

    server_size = len(server_dataset) if hasattr(server_dataset, '__len__') else 0
    rounds_to_run = max(1, warmup_rounds)
    if config:
        if save_path_kind == "frame1_pre":
            actual_save_path = get_frame1_pre_k2_warmup_model_path(
                save_path, config, for_save=True
            )
        elif save_path_kind == "server_init_pre":
            actual_save_path = get_server_init_pre_model_path(save_path, config)
        else:
            actual_save_path = get_server_init_pre_model_path(save_path, config)
    else:
        actual_save_path = save_path

    print(f"  - 训练数据: 服务器有标注集，样本数={server_size}")
    print(f"  - 通信轮次: {warmup_rounds}")
    print(f"  - 最终保存路径: {actual_save_path}")
    print("=" * 70 + "\n")
    
    # 初始化模型
    model_name = (config or {}).get('model_name', 'cnn')
    if save_path_kind == "frame1_pre" and config is not None:
        k_old = int(config["frame1_initial_num_classes"])
        init_path = get_server_init_pre_model_path(
            config["server_init_pre_warmup_model_path"], config
        )
        if not os.path.isfile(init_path):
            raise RuntimeError(
                f"frame1_pre 需从 server_init 预热权重初始化，文件不存在: {init_path}\n"
                "请先运行 continual_run_stage=server_init_pre 生成检查点，或检查 server_init_pre_warmup_model_path。"
            )
        if num_classes < k_old:
            raise ValueError(
                f"frame1_pre 全类数 K={num_classes} 小于旧类数 K_old={k_old}，无法从 server_init 加载。"
            )
        global_model = build_model(model_name, num_classes=k_old).to(device)
        load_warmup_model(
            global_model,
            init_path,
            current_config=config,
            strict_match=False,
            skip_config_check=True,
            device=device,
        )
        if num_classes > k_old:
            global_model.expand_classifier(num_classes, freeze_old_logits=False)
        print(f"  - 已从 server_init 加载并准备 K={num_classes}: {init_path}\n")
    else:
        global_model = build_model(model_name, num_classes=num_classes).to(device)
    
    # 直接在服务器有标注数据上预训练（不经过客户端聚合）
    nc_eval = num_classes
    cfg_w = config or {}
    local_model, warmup_metrics = _warmup_local_training(
        global_model,
        server_loader,
        data_loader.test_loader,
        rounds_to_run,
        warmup_local_epochs,
        lr,
        momentum,
        device,
        num_eval_classes=nc_eval,
        verbose=verbose,
        weight_decay=float(cfg_w.get("weight_decay", 0.0)),
    )
    global_model.load_state_dict(local_model.state_dict())
    
    # 最终评估（server_init_pre：nc_eval=K_old，仅统计旧类子集测试样本）
    final_acc = evaluate_model(
        global_model, data_loader.test_loader, device,
        num_eval_classes=nc_eval,
    )
    
    print(f"\n{'='*70}")
    print(f"预热训练完成！")
    print(f"  - 最终测试准确率: {final_acc:.2f}%")
    print(f"  - 最佳测试准确率: {max(warmup_metrics['test_accuracy']):.2f}%")
    if save_path_kind == "frame1_pre" and config is not None:
        k1 = int(config["frame1_initial_num_classes"])
        clm = evaluate_model_continual_metrics(
            global_model,
            data_loader.test_loader,
            device,
            num_old_classes=k1,
        )
        warmup_metrics["continual_eval_test"] = clm
        def _fmt(x):
            return f"{x:.3f}" if x is not None else "n/a"
        t = clm
        print(
            f"  - [第二帧前预热·测试集] Acc_old={_fmt(t.get('test_acc_old'))} "
            f"Acc_new={_fmt(t.get('test_acc_new'))} | "
            f"新→旧误判={_fmt(t.get('misclass_new_to_old_rate'))} "
            f"旧→新误判={_fmt(t.get('misclass_old_to_new_rate'))}"
        )
    print(f"{'='*70}\n")
    
    # 提取预热相关配置
    warmup_config = get_warmup_config(config) if config else None
    if warmup_config is not None and config is not None:
        if save_path_kind == "frame1_pre":
            k2 = int(get_frame1_total_num_classes(config))
            warmup_config["frame1_total_num_classes"] = k2
            warmup_config["num_classes"] = k2
            warmup_config["warmup_stage"] = "frame1_pre"
        elif save_path_kind == "server_init_pre":
            k_old = int(config["frame1_initial_num_classes"])
            warmup_config["frame1_initial_num_classes"] = k_old
            warmup_config["num_classes"] = k_old
            warmup_config["warmup_stage"] = "server_init_pre"
        else:
            k1w = int(config["frame1_initial_num_classes"])
            warmup_config['frame1_initial_num_classes'] = k1w
            warmup_config['num_classes'] = k1w
    
    # 保存模型
    save_warmup_model(global_model, actual_save_path, warmup_metrics, warmup_config)

    # 绘制预热准确率曲线
    plot_path = get_warmup_plot_path(actual_save_path)
    plot_warmup_accuracy(warmup_metrics, plot_path)
    warmup_metrics['plot_path'] = plot_path

    json_path = get_warmup_json_path(actual_save_path)
    save_warmup_json(
        json_path=json_path,
        metrics=warmup_metrics,
        config=warmup_config,
        model_path=actual_save_path,
        plot_path=plot_path,
    )
    warmup_metrics['json_path'] = json_path
    warmup_metrics['saved_model_path'] = actual_save_path

    return warmup_metrics


def _warmup_local_training(global_model: nn.Module, data_loader, test_loader,
                           warmup_rounds: int, warmup_local_epochs: int,
                           lr: float, momentum: float, device: str,
                           num_eval_classes: Optional[int] = None,
                           verbose: bool = True,
                           weight_decay: float = 0.0) -> tuple:
    """
    预热阶段服务器监督训练：SGD + CosineAnnealingLR，T_max 为整段预热总步数
    （warmup_rounds × warmup_local_epochs，每步对应一个 server batch）。
    """
    import copy

    local_model = copy.deepcopy(global_model)
    local_model.train()

    optimizer = optim.SGD(
        local_model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )
    total_steps = max(1, warmup_rounds * warmup_local_epochs)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    criterion = nn.CrossEntropyLoss()
    
    metrics = {
        'round': [],
        'train_accuracy': [],
        'test_accuracy': []
    }
    
    for round_idx in range(warmup_rounds):
        # 每个通信轮次内进行若干本地 epoch 训练
        round_loss = []
        round_correct = 0
        round_total = 0

        for epoch_idx in range(warmup_local_epochs):
            local_model.train()
            epoch_loss = []
            epoch_correct = 0
            epoch_total = 0

            for data, labels, _ in data_loader:
                data, labels = data.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = local_model(data)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()

                epoch_loss.append(loss.item())
                _, predicted = torch.max(outputs.data, 1)
                epoch_total += labels.size(0)
                epoch_correct += (predicted == labels).sum().item()
                break  # 每 epoch 仅训练一个 batch（刻意保留）

            round_loss.append(np.mean(epoch_loss))
            round_correct += epoch_correct
            round_total += epoch_total

        train_loss = np.mean(round_loss)
        train_acc = 100 * round_correct / round_total

        # 评估测试集（与第一帧类别数一致时仅统计 label < num_eval_classes）
        local_model.eval()
        test_acc = evaluate_model(
            local_model, test_loader, device,
            num_eval_classes=num_eval_classes,
        )

        metrics['train_accuracy'].append(train_acc)
        metrics['test_accuracy'].append(test_acc)
        metrics['round'].append(round_idx + 1)

        if verbose and (round_idx % 10 == 0 or round_idx == warmup_rounds - 1):
            print(f"  Round {round_idx+1:3d}/{warmup_rounds} | "
                  f"Epochs: {warmup_local_epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Train Acc: {train_acc:.2f}% | "
                  f"Test Acc: {test_acc:.2f}%")
    
    return local_model, metrics


def save_warmup_model(model: nn.Module, save_path: str, metrics: Dict = None,
                      config: Dict = None):
    """
    保存预热后的模型
    
    参数:
        model: 预热后的模型
        save_path: 保存路径
        metrics: 预热训练指标（可选）
        config: 预热相关配置参数（可选）
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'metrics': metrics,
        'config': config,
    }

    torch.save(checkpoint, save_path)
    print(f"✅ 预热模型已保存: {save_path}")
    if config:
        print(f"   配置参数: {config}")


def load_warmup_model(model: nn.Module, load_path: str,
                      current_config: Dict = None,
                      strict_match: bool = True,
                      skip_config_check: bool = False,
                      device: str = None) -> Tuple[Dict, bool]:
    """
    加载预热后的模型
    
    参数:
        model: 待加载的模型
        load_path: 模型路径
        current_config: 当前配置（用于验证）
        strict_match: 是否严格要求配置匹配
        skip_config_check: 是否跳过配置匹配检查
        device: 计算设备
        
    返回:
        metrics: 预热训练指标（如果有的话）
        config_matched: 配置是否匹配
        
    异常:
        FileNotFoundError: 模型文件不存在
        ValueError: 配置不匹配且strict_match=True
    """
    device = device or get_device(current_config)

    if not os.path.exists(load_path):
        raise FileNotFoundError(f"预热模型文件不存在: {load_path}")
    
    checkpoint = torch.load(load_path, map_location=device)
    saved_config = checkpoint.get('config')

    # 验证配置
    config_matched = True
    if not skip_config_check and current_config is not None and saved_config is not None:
        current_warmup_config = get_warmup_config(current_config)
        is_match, mismatched = configs_match(saved_config, current_warmup_config)

        if not is_match:
            config_matched = False
            warning_msg = (f"⚠️ 预热模型配置不匹配!\n"
                          f"   模型路径: {load_path}\n"
                          f"   不匹配的参数:\n")
            for m in mismatched:
                warning_msg += f"     - {m}\n"

            if strict_match:
                raise ValueError(warning_msg + "   请重新训练预热模型或设置 strict_match=False")
            else:
                print(warning_msg)
                print("   继续加载模型（strict_match=False）...")
    
    model.load_state_dict(checkpoint['model_state_dict'])

    if config_matched:
        print(f"✅ 预热模型已加载: {load_path}")
        if saved_config:
            print(f"   配置参数: {saved_config}")
    
    return checkpoint.get('metrics', {}), config_matched


def _ensure_warmup_defaults(cfg: Dict) -> Dict:
    """填充预热需要的关键默认值，避免缺失字段导致报错"""
    cfg = cfg.copy()
    dc = DEFAULT_CONFIG
    if 'lr_unlabeled' not in cfg:
        cfg['lr_unlabeled'] = float(dc.get('lr_unlabeled', 0.03))
    for k in (
        'server_init_pre_warmup_rounds', 'server_init_pre_warmup_local_epochs',
        'server_init_pre_warmup_model_path',
        'frame1_pre_warmup_rounds', 'frame1_pre_warmup_local_epochs', 'frame1_pre_warmup_model_path',
    ):
        if k not in cfg and k in dc:
            cfg[k] = dc[k]
    return cfg


def run_server_init_pre_only(config: Dict = None) -> Dict:
    """server_init_pre：仅服务器在初始旧类 0..K_old-1 上训练；有标签数量与默认 frame1 表一致，仅截取旧类条目。"""
    from modules.fixmatch import build_federated_loader_from_config
    from main import apply_dataset_overrides

    cfg = DEFAULT_CONFIG.copy()
    if config:
        cfg.update(config)

    cfg = apply_dataset_overrides(cfg)
    cfg["continual_run_stage"] = "server_init_pre"
    cfg = normalize_continual_settings(cfg)
    cfg = _ensure_warmup_defaults(cfg)

    set_seed(cfg.get('seed', 42), cfg)

    k_old = int(cfg["frame1_initial_num_classes"])
    # 与默认 frame1 表一致，仅保留旧类 0..k_old-1（partition=k1 + server_only）
    slc = cfg.get("server_labeled_per_class") or {}
    cfg["server_labeled_per_class"] = {
        int(k): int(v)
        for k, v in slc.items()
        if int(k) < k_old and int(v) > 0
    }

    data_loader = build_federated_loader_from_config(
        cfg,
        partition="k1",
        seed=cfg.get("seed", 42),
        server_only=True,
    )

    cfg['num_clients'] = data_loader.num_clients
    cfg['alpha'] = getattr(data_loader, 'alpha', cfg.get('alpha'))
    cfg['dataset_name'] = getattr(data_loader, 'dataset_name', cfg.get('dataset_name', 'dataset'))

    metrics = warmup_training(
        data_loader=data_loader,
        num_classes=k_old,
        warmup_rounds=int(cfg["server_init_pre_warmup_rounds"]),
        warmup_local_epochs=int(cfg["server_init_pre_warmup_local_epochs"]),
        lr=cfg['lr_unlabeled'],
        momentum=cfg['momentum'],
        save_path=cfg["server_init_pre_warmup_model_path"],
        config=cfg,
        device=get_device(cfg),
        verbose=True,
        save_path_kind="server_init_pre",
    )

    print("server_init_pre 结束，模型与曲线已生成。")
    return metrics


def run_frame1_pre_warmup_only(config: Dict = None) -> Dict:
    """frame1_pre：K2 全类预热；先加载 server_init 再扩类。保存路径恒为当前超参 tag；联邦 frame1 加载见 ``frame1_pre_warmup_model_path``。"""
    from modules.fixmatch import build_federated_loader_from_config
    from main import apply_dataset_overrides

    cfg = DEFAULT_CONFIG.copy()
    if config:
        cfg.update(config)
    cfg = apply_dataset_overrides(cfg)
    cfg["continual_run_stage"] = "frame1_pre"
    cfg = normalize_continual_settings(cfg)
    cfg = _ensure_warmup_defaults(cfg)
    set_seed(cfg.get("seed", 42), cfg)

    seed = cfg.get("seed", 42)
    data_loader = build_federated_loader_from_config(cfg, partition="full_k", seed=seed)
    cfg["num_clients"] = data_loader.num_clients
    cfg["alpha"] = getattr(data_loader, "alpha", cfg.get("alpha"))
    cfg["dataset_name"] = getattr(
        data_loader, "dataset_name", cfg.get("dataset_name", "dataset")
    )

    k2 = int(get_frame1_total_num_classes(cfg))
    base = cfg["frame1_pre_warmup_model_path"]
    wr = int(cfg["frame1_pre_warmup_rounds"])
    we = int(cfg["frame1_pre_warmup_local_epochs"])
    metrics = warmup_training(
        data_loader=data_loader,
        num_classes=k2,
        warmup_rounds=wr,
        warmup_local_epochs=we,
        lr=cfg["lr_unlabeled"],
        momentum=cfg["momentum"],
        save_path=base,
        config=cfg,
        device=get_device(cfg),
        verbose=True,
        save_path_kind="frame1_pre",
    )
    print(
        "frame1_pre（K2 全类预热）结束，模型与曲线已生成；"
        f"检查点: {metrics.get('saved_model_path', '')}"
    )
    return metrics


if __name__ == "__main__":
    import sys

    # 显式子命令优先；否则与 config.DEFAULT_CONFIG['continual_run_stage'] 一致（避免改了配置仍跑 server_init_pre）
    if len(sys.argv) > 1 and sys.argv[1] in (
        "frame1_pre",
        "frame1-pre",
        "frame2_pre",
        "frame2-pre",
        "--frame2-pre",
    ):
        run_frame1_pre_warmup_only()
    elif len(sys.argv) > 1 and sys.argv[1] in (
        "server_init_pre",
        "serverinit",
        "server-init",
    ):
        run_server_init_pre_only()
    else:
        stage = DEFAULT_CONFIG.get("continual_run_stage", "server_init_pre")
        if stage == "frame1_pre":
            run_frame1_pre_warmup_only()
        elif stage == "server_init_pre":
            run_server_init_pre_only()
        else:
            print(
                f"continual_run_stage={stage!r} 不是仅预热阶段（请用 main.py）；"
                f"或传参: python warmup.py frame1_pre | server_init_pre"
            )
            sys.exit(1)
