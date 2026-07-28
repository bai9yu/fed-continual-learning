"""
================================================================================
联邦半监督学习实验 - FixMatch
================================================================================
主程序入口

实验类型：
    - NoClientTrain:  无客户端训练，仅服务器微调
    - AllClientsTrain: 所有无标签客户端参与训练
    - Random:         随机选择若干客户端
================================================================================
"""


import argparse
import traceback
import warnings
warnings.filterwarnings('ignore')

import matplotlib

matplotlib.use("Agg")

import numpy as np
from typing import Any, Dict, List, Optional, Tuple
import os
import json
from datetime import datetime

from config import (
    ABBREVIATIONS,
    DEFAULT_CONFIG,
    EXPERIMENT_TYPES,
    build_default_server_labeled_per_class,
    get_file_stage_tag,
    get_frame1_total_num_classes,
    get_num_rounds,
    get_result_filename,
    normalize_continual_settings,
    set_seed,
    get_device,
)
from modules.fixmatch import build_federated_loader_from_config
from federated import FederatedLearning
from visualization import plot_all_results, plot_single_experiment
from warmup import get_server_init_pre_model_path, get_frame1_pre_k2_warmup_model_path


# 数据集默认超参数（学习率、批次大小）
DATASET_DEFAULTS = {
    'svhn': {
        'lr_unlabeled': 0.03,
        'batch_size': 64,
        'local_epochs_unlabeled': 5,
        'labeled_per_class': 25,
    },
    'cifar10': {
        'lr_unlabeled': 0.03,
        'batch_size': 64,
        'local_epochs_unlabeled': 5,
        'labeled_per_class': 25,
    },
    'cifar100': {
        'lr_unlabeled': 0.03,
        'batch_size': 64,
        'local_epochs_unlabeled': 5,
        'labeled_per_class': 25,
    },
}


# ========== JSON序列化辅助函数 ===========
def convert_to_serializable(obj):
    """将NumPy类型转换为Python原生类型，以便JSON序列化"""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


def get_configured_keys(user_config: Dict = None) -> set:
    """
    获取「已配置」的参数名集合。
    仅包含 DEFAULT_CONFIG 中的键，以及用户显式传入的键。
    用于过滤 config.json 输出，排除 apply_dataset_overrides 等自动填充的未配置参数。
    """
    keys = set(DEFAULT_CONFIG.keys())
    if user_config:
        keys |= set(user_config.keys())
    keys |= {'experiment_label', 'num_runs'}  # 运行时设置的键
    return keys


def filter_config_for_save(full_config: Dict, configured_keys: set) -> Dict:
    """仅保留已配置的参数，用于 config.json 输出"""
    return {k: v for k, v in full_config.items() if k in configured_keys}


def _merge_continual_eval_runs(exp_runs: List[Dict]) -> Optional[List[Any]]:
    """
    合并多次联邦 run 的 metrics['continual_eval']。
    单次运行：原样返回；多次：按轮对齐，对 test / pseudo_unlabeled_agg 内数值字段取平均。
    """
    if not exp_runs:
        return None
    if len(exp_runs) == 1:
        return exp_runs[0].get("continual_eval")
    ce_lists = [r.get("continual_eval") for r in exp_runs]
    if not any(ce_lists):
        return None
    valid = [c for c in ce_lists if c]
    if not valid:
        return None
    n = min(len(c) for c in valid)
    merged: List[Any] = []
    for i in range(n):
        entries = [c[i] for c in valid if i < len(c)]
        if all(e is None for e in entries):
            merged.append(None)
            continue
        non_none = [e for e in entries if e is not None]
        if len(non_none) == 1:
            merged.append(non_none[0])
            continue
        k1 = non_none[0].get("k1")
        k2 = non_none[0].get("k2")
        test_dicts = [e["test"] for e in non_none if e.get("test") is not None]
        pss_dicts = [
            e["pseudo_unlabeled_agg"]
            for e in non_none
            if e.get("pseudo_unlabeled_agg") is not None
        ]
        merged_test = _mean_scalar_subdicts(test_dicts)
        merged_test.pop("test_acc_all", None)
        merged.append({
            "k1": k1,
            "k2": k2,
            "test": merged_test,
            "pseudo_unlabeled_agg": _mean_scalar_subdicts(pss_dicts),
        })
    return merged


def _mean_scalar_subdicts(dicts: List[Dict]) -> Dict[str, Any]:
    """对结构相同的若干 dict 做逐键平均（整型/浮点键；否则取首个出现的值）。"""
    if not dicts:
        return {}
    keys = set()
    for d in dicts:
        keys |= set(d.keys())
    out: Dict[str, Any] = {}
    for k in keys:
        vals = []
        for d in dicts:
            if k not in d or d[k] is None:
                continue
            v = d[k]
            if isinstance(v, (int, float, np.integer, np.floating)):
                vals.append(float(v))
        if vals:
            out[k] = float(np.mean(vals))
        else:
            for d in dicts:
                if k in d:
                    out[k] = d[k]
                    break
    return out


def merge_default_config(user_config: Dict = None) -> Dict:
    """DEFAULT_CONFIG 与用户配置合并（副本，不修改传入的 dict）。"""
    cfg = DEFAULT_CONFIG.copy()
    if user_config:
        cfg.update(user_config)
    return cfg


def load_config_from_exp_dir(exp_dir: str) -> Dict:
    """从已有实验目录读取 config.json（用于断点续训）。"""
    path = os.path.join(exp_dir, "config.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"未找到实验配置: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_config_for_exp_dir(
    exp_dir: str,
    user_config: Dict = None,
    *,
    no_resume: bool = False,
) -> Dict:
    """合并 exp_dir/config.json 与可选覆盖项，并写入 exp_dir 供检查点使用。"""
    saved = load_config_from_exp_dir(exp_dir)
    cfg = merge_default_config(saved)
    if user_config:
        cfg.update(user_config)
    cfg = apply_dataset_overrides(cfg)
    cfg["exp_dir"] = exp_dir
    cfg["output_dir"] = exp_dir
    if cfg.get("checkpoint_dir"):
        cfg["checkpoint_dir"] = os.path.abspath(cfg["checkpoint_dir"])
    if no_resume:
        cfg["federated_auto_resume"] = False
    return cfg


def save_metrics_json(path: str, metrics: Dict, config: Dict) -> None:
    """将 metrics + config 写入 JSON（与原先 json.dump 内容一致）。"""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(
            convert_to_serializable({'metrics': metrics, 'config': config}),
            f,
            indent=2,
            ensure_ascii=False,
        )


def apply_dataset_overrides(config: Dict) -> Dict:
    """按照数据集名称填充/覆盖默认超参数"""
    cfg = config.copy()
    dataset_key = cfg.get('dataset_name', 'cifar10').lower()
    defaults = DATASET_DEFAULTS.get(dataset_key, DATASET_DEFAULTS['cifar10'])

    # 学习率与批次大小：仅在未显式提供时使用数据集默认
    cfg['lr_unlabeled'] = cfg.get('lr_unlabeled', defaults['lr_unlabeled'])
    cfg['batch_size'] = cfg.get('batch_size', defaults['batch_size'])
    cfg['labeled_per_class'] = cfg.get('labeled_per_class', defaults['labeled_per_class'])
    cfg['server_batch_size'] = cfg.get('server_batch_size', defaults.get('server_batch_size', 10))
    # 本地轮次
    cfg['local_epochs_unlabeled'] = cfg.get('local_epochs_unlabeled', defaults['local_epochs_unlabeled'])
    cfg['finetune_epochs'] = int(cfg.get('finetune_epochs', cfg['local_epochs_unlabeled']))

    # 无线信道（可选，与 modules.wireless_channel 一致）
    cfg['total_bandwidth_mhz'] = float(cfg.get('total_bandwidth_mhz', 20.0))
    cfg['tx_power_dbm'] = float(cfg.get('tx_power_dbm', 23.0))
    cfg['upload_deadline'] = float(cfg.get('upload_deadline', 1.6))
    cfg['cell_radius'] = float(cfg.get('cell_radius', 250.0))
    cfg['comp_delay_a_i'] = float(cfg.get('comp_delay_a_i', 0.5e-3))
    cfg['comp_delay_mu_i'] = float(cfg.get('comp_delay_mu_i', 2.0e3))

    cfg['seed'] = int(cfg.get('seed', 42))
    cfg['use_wireless_scheduling'] = bool(cfg.get('use_wireless_scheduling', True))

    cfg['frame1_initial_num_classes'] = int(cfg.get('frame1_initial_num_classes', 6))
    cfg['frame1_new_num_classes'] = int(cfg.get('frame1_new_num_classes', 4))
    cfg['frame2_new_num_classes'] = int(cfg.get('frame2_new_num_classes', 0))
    if cfg.get('frame1_total_num_classes') is not None:
        cfg['frame1_total_num_classes'] = int(cfg['frame1_total_num_classes'])
    if cfg.get('frame2_total_num_classes') is not None:
        cfg['frame2_total_num_classes'] = int(cfg['frame2_total_num_classes'])
    cfg['frame1_federated_rounds'] = max(1, int(cfg.get('frame1_federated_rounds', 1000)))
    cfg['frame2_federated_rounds'] = max(1, int(cfg.get('frame2_federated_rounds', 1000)))

    f1_total = get_frame1_total_num_classes(cfg)
    cfg['server_labeled_per_class'] = build_default_server_labeled_per_class(
        cfg, num_classes=f1_total
    )

    cfg = normalize_continual_settings(cfg)
    return cfg


# ========== 自动创建实验文件夹 ===========
def resolve_exp_dir(
    config: Dict,
    configured_keys: set = None,
    *,
    experiment_label: str = None,
) -> Tuple[str, Dict]:
    """
    解析实验目录并返回 (exp_dir, merged_config)。

    - config.output_dir 非空：使用固定目录；已有 config.json 则合并 saved 配置（当前 config 覆盖项优先）
    - 否则：create_exp_dir 带时间戳新建目录
    """
    output_dir = config.get("output_dir")
    if output_dir:
        exp_dir = os.path.abspath(output_dir)
        os.makedirs(exp_dir, exist_ok=True)
        cfg_path = os.path.join(exp_dir, "config.json")
        if os.path.isfile(cfg_path):
            saved = load_config_from_exp_dir(exp_dir)
            merged = merge_default_config(saved)
            merged.update(config)
            merged["output_dir"] = exp_dir
            merged = apply_dataset_overrides(merged)
            if experiment_label:
                merged["experiment_label"] = experiment_label
            print(f"📂 使用固定输出目录（续训）: {exp_dir}")
            return exp_dir, merged

        merged = config.copy()
        merged["output_dir"] = exp_dir
        if experiment_label:
            merged["experiment_label"] = experiment_label
        to_save = filter_config_for_save(merged, configured_keys) if configured_keys else merged
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(to_save, f, indent=2, ensure_ascii=False)
        print(f"✅ 固定输出目录已创建: {exp_dir}")
        return exp_dir, merged

    merged = config.copy()
    if experiment_label:
        merged["experiment_label"] = experiment_label
    exp_dir = create_exp_dir(merged, configured_keys=configured_keys)
    return exp_dir, merged


def create_exp_dir(config, base_dir='runs', configured_keys: set = None):
    """
    根据主要参数和时间戳创建实验文件夹，并保存config。
    仅保存已配置的参数（configured_keys），排除自动填充的未配置参数。
    返回实验文件夹路径。
    命名格式：调度名称_数据集_lpc_阶段_localepoch_总轮次_学习率_batchsize_阈值_上行时限_时间（不含 alpha、客户端总数、每轮客户端数）
    """
    timestamp = datetime.now().strftime('%m%d_%H%M%S')  # 不含年
    label = config.get('experiment_label', 'exp')
    dataset_name = config.get('dataset_name', 'cifar10')
    lpc = config.get('labeled_per_class', 0)
    stage = get_file_stage_tag(config)
    le = config.get('local_epochs_unlabeled', 1)
    rounds = get_num_rounds(config)
    lr = config.get('lr_unlabeled')
    bs = config.get('batch_size', 64)
    pt = config.get('pseudo_threshold', 0.95)
    ud = config.get('upload_deadline', 1.0)
    s = config.get('seed', 42)
    extra = []
    if config.get('server_labeled_per_class'):
        extra.append('simb')
    extra_s = ('_' + '_'.join(extra)) if extra else ''
    exp_name = (
        f"exp_{label}_{dataset_name}_lpc{lpc}_{stage}_le{le}_r{rounds}_lr{lr}_bs{bs}_pt{pt}_ud{ud}_s{s}_{timestamp}"
    )
    exp_dir = os.path.join(base_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    # 保存config（仅包含已配置的参数）
    to_save = filter_config_for_save(config, configured_keys) if configured_keys else config
    with open(os.path.join(exp_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(to_save, f, indent=2, ensure_ascii=False)
    print(f"✅ 实验文件夹已创建: {exp_dir}")
    return exp_dir


def build_runner(config, seed):
    """构建数据加载器和联邦学习实例，供多处复用"""
    cfg = apply_dataset_overrides(config)

    stage = cfg['continual_run_stage']
    if stage == 'frame1':
        data_loader = build_federated_loader_from_config(cfg, partition="full_k", seed=seed)
        fed_num_classes = get_frame1_total_num_classes(cfg)
    elif stage == 'frame2':
        raise NotImplementedError(
            "continual_run_stage=frame2 仅 CIFAR-100 预留，尚未实现；SVHN/CIFAR-10 请使用 frame1。"
        )
    else:
        raise ValueError(
            f"continual_run_stage={stage!r} 为仅预热阶段，请用 run_single_experiment 的预热分支，"
            "不要调用 build_runner。"
        )

    fed_learning = FederatedLearning(
        local_epochs_unlabeled=cfg['local_epochs_unlabeled'],
        lr_unlabeled=cfg['lr_unlabeled'],
        momentum=cfg['momentum'],
        pseudo_threshold=cfg['pseudo_threshold'],
        num_classes=fed_num_classes,
        frame1_pre_warmup_model_path=cfg['frame1_pre_warmup_model_path'],
        finetune_epochs=cfg['finetune_epochs'],
        weight_decay=cfg.get('weight_decay', 0.0),
        config=cfg
    )

    return data_loader, fed_learning, cfg


def get_experiment_abbrev(exp_type: str) -> str:
    """返回实验类型的简称（若无则返回简化的安全字符串）"""
    from config import get_path_abbrev
    if exp_type in ABBREVIATIONS:
        return get_path_abbrev(exp_type)
    # 简化并去掉空格/特殊字符作为后备
    return ''.join(ch for ch in exp_type if ch.isalnum())[:8]


def _run_all_warmup_only_branch(
    base_config: Dict,
    exp_dir: str,
    stage: str,
) -> Tuple[Dict, Dict, List[str], str]:
    """
    run_all_experiments 在 continual_run_stage 为 server_init_pre / frame1_pre 时的分支：
    仅运行预热、写 JSON、返回（不进入联邦多实验循环）。
    """
    print(f"\nℹ️ continual_run_stage={stage}：仅运行预热，跳过联邦实验类型列表。\n")
    set_seed(42, base_config)
    if stage == 'server_init_pre':
        from warmup import run_server_init_pre_only
        wm = run_server_init_pre_only(base_config)
    elif stage == 'frame1_pre':
        from warmup import run_frame1_pre_warmup_only
        wm = run_frame1_pre_warmup_only(base_config)
    else:
        raise ValueError(f"未知的预热-only 阶段: {stage!r}")
    jp = os.path.join(exp_dir, f"warmup_{stage}_result.json")
    save_metrics_json(jp, wm, base_config)
    print(f"✅ 预热结果已保存: {jp}")
    return {stage: wm}, base_config, [jp], exp_dir, []


def run_all_experiments(config: Dict = None, num_runs: int = 1,
                        experiment_types: List[str] = None,
                        exp_dir: str = None,
                        plot_results: bool = True) -> Tuple[Dict, Dict, List[str], str, List[str]]:
    """
    运行所有实验
    
    参数:
        config: 实验配置（如果为None则使用默认配置）
        num_runs: 每个实验运行次数
        experiment_types: 要运行的实验类型列表（默认为全部）
        exp_dir: 实验文件夹路径（如果为None则自动创建）
        plot_results: 是否在结束后将曲线图保存到 exp_dir（默认 True；仅联邦阶段会绘图，预热-only 分支不绘图）
        
    返回:
        all_results: 所有实验结果
        config: 使用的配置
        json_files: 保存的JSON文件列表
        exp_dir: 实验文件夹路径
        plot_files: 生成的图片路径列表（plot_results=False 或未绘图时为空列表）
    """
    # 实验列表
    if experiment_types is None:
        experiment_types = EXPERIMENT_TYPES
    
    base_config = merge_default_config(config)
    configured_keys = get_configured_keys(config)
    base_config = apply_dataset_overrides(base_config)
    base_config['num_runs'] = num_runs

    # 创建或解析实验文件夹
    if exp_dir is None:
        labels = [get_experiment_abbrev(e) for e in experiment_types]
        combined_label = '+'.join(labels)
        exp_dir, base_config = resolve_exp_dir(
            base_config,
            configured_keys=configured_keys,
            experiment_label=combined_label,
        )
    base_config["exp_dir"] = exp_dir
    if base_config.get("output_dir"):
        base_config["output_dir"] = exp_dir
    if base_config.get("checkpoint_dir"):
        base_config["checkpoint_dir"] = os.path.abspath(base_config["checkpoint_dir"])
    
    stage_early = base_config.get('continual_run_stage')
    if stage_early in ('server_init_pre', 'frame1_pre'):
        return _run_all_warmup_only_branch(base_config, exp_dir, stage_early)

    server_init_path = get_server_init_pre_model_path(
        base_config['server_init_pre_warmup_model_path'], base_config
    )
    frame1_pre_k2_path = get_frame1_pre_k2_warmup_model_path(
        base_config['frame1_pre_warmup_model_path'],
        base_config,
        for_save=False,
    )

    # 打印实验设置
    print("=" * 70)
    print("联邦半监督学习实验 - FixMatch + 智能调度")
    print("=" * 70)
    print("\n实验设置:")
    print(f"  - NoClientTrain:        无客户端训练，仅服务器微调")
    print(f"  - AllClientsTrain:      所有无标签客户端参与训练")
    print(f"  - Random / BestChannel / NewClassClientsOnly（仅 continual_new_class_client_ids，默认 18–29）")
    print(f"\n配置参数:")
    print(f"  - 数据集: {base_config.get('dataset_name', 'unknown')}")
    print(f"  - 客户端数量: {base_config['num_clients']}")
    print(f"  - Dirichlet α: {base_config['alpha']}")
    print(f"  - 伪标签阈值: {base_config['pseudo_threshold']}")
    print(f"  - 学习率 lr_unlabeled: {base_config['lr_unlabeled']}")
    print(f"  - 本地 epoch (unlabeled 与服务器等共用): {base_config['local_epochs_unlabeled']}")
    print(f"  - 微调轮数 finetune_epochs: {base_config.get('finetune_epochs', base_config['local_epochs_unlabeled'])}")
    print(f"  - 通信轮次: {get_num_rounds(base_config)}")
    print(f"  - 计算设备: {get_device(base_config)} (cuda_device={base_config.get('cuda_device')})")
    crs = base_config['continual_run_stage']
    print(f"\n持续学习: continual_run_stage={crs}")
    print("  (server_init_pre → frame1_pre → frame1；frame2_* 仅 CIFAR-100 预留)")
    print(
        f"  - server_init_pre（仅初始旧类 K_old，非 frame1 全类 K2）: r={base_config['server_init_pre_warmup_rounds']}, "
        f"e={base_config['server_init_pre_warmup_local_epochs']}, "
        f"labeled_per_class={base_config.get('labeled_per_class', 100)}（文件名 lpc 段）| → {server_init_path}"
    )
    print(
        f"  - frame1_pre（K2 全类预热）: r={base_config['frame1_pre_warmup_rounds']}, "
        f"e={base_config['frame1_pre_warmup_local_epochs']} | → {frame1_pre_k2_path}"
    )
    print("  联邦 frame1：从 frame1_pre（K2）检查点加载")
    print("=" * 70)
    
    all_results = {}
    json_files = []
    
    for exp_type in experiment_types:
        print(f"\n{'='*70}")
        print(f"运行 {exp_type} ({num_runs} 次)")
        print(f"{'='*70}")
        
        exp_runs = []
        exp_config = base_config.copy()
        # 无专用有标签客户端配置，统一保持默认（0）
        
        for run in range(num_runs):
            print(f"\n第 {run + 1}/{num_runs} 次运行")
            set_seed(42 + run, exp_config)
            exp_config["exp_dir"] = exp_dir

            data_loader, fed_learning, exp_config = build_runner(exp_config, seed=42 + run)
            
            # 运行实验
            metrics = fed_learning.run_experiment(exp_type, data_loader)
            exp_runs.append(metrics)

        if num_runs == 1:
            # 单次运行：保留 federated 完整 metrics，否则绘图缺 pseudo_train_* / misclass 等字段
            merged_metrics = exp_runs[0]
        else:
            # 多次运行：对逐轮标量取平均，并合并 continual_eval
            avg_metrics: Dict[str, Any] = {
                'round': exp_runs[0]['round'],
                'test_accuracy': [],
                'test_loss': [],
                'pseudo_mean_confidence': [],
                'pseudo_train_precision_all': [],
                'pseudo_selected': [],
                'selected_clients': exp_runs[0]['selected_clients'],
                'pseudo_per_class_per_client': exp_runs[0].get('pseudo_per_class_per_client', []),
            }
            _opt_float_keys = (
                'pseudo_train_conf_mean_old',
                'pseudo_train_conf_mean_new',
                'pseudo_train_precision_old',
                'pseudo_train_precision_new',
                'pseudo_train_pred_old_accuracy',
                'pseudo_train_pred_new_accuracy',
                'pseudo_scheduled_misclass_new_to_old',
                'pseudo_precision_old_scheduled',
                'pseudo_precision_new_scheduled',
                'pseudo_precision_total_scheduled',
                'pseudo_masked_wrong_runnerup_rate',
            )
            _int_round_keys = (
                'pseudo_train_pred_old_count_total',
                'pseudo_train_pred_new_count_total',
                'pseudo_masked_wrong_total',
                'pseudo_masked_wrong_runner_hit_total',
            )
            for _k in _opt_float_keys:
                avg_metrics[_k] = []
            for _k in _int_round_keys:
                avg_metrics[_k] = []

            for i in range(len(exp_runs[0]['round'])):
                avg_metrics['test_accuracy'].append(
                    np.mean([run['test_accuracy'][i] for run in exp_runs])
                )
                tls_vals = []
                for run in exp_runs:
                    tls = run.get('test_loss') or []
                    if i < len(tls):
                        tls_vals.append(tls[i])
                avg_metrics['test_loss'].append(float(np.mean(tls_vals)) if tls_vals else 0.0)
                avg_metrics['pseudo_mean_confidence'].append(
                    np.mean([run['pseudo_mean_confidence'][i] for run in exp_runs])
                )
                pta_vals = []
                for run in exp_runs:
                    seq = run.get('pseudo_train_precision_all') or []
                    if i < len(seq) and seq[i] is not None:
                        pta_vals.append(float(seq[i]))
                avg_metrics['pseudo_train_precision_all'].append(
                    float(np.mean(pta_vals)) if pta_vals else None
                )
                avg_metrics['pseudo_selected'].append(
                    np.mean([run['pseudo_selected'][i] for run in exp_runs])
                )
                for _k in _opt_float_keys:
                    _vals = []
                    for run in exp_runs:
                        seq = run.get(_k) or []
                        if i < len(seq) and seq[i] is not None:
                            _vals.append(float(seq[i]))
                    avg_metrics[_k].append(float(np.mean(_vals)) if _vals else None)
                for _k in _int_round_keys:
                    _vals = []
                    for run in exp_runs:
                        seq = run.get(_k) or []
                        if i < len(seq) and seq[i] is not None:
                            _vals.append(int(seq[i]))
                    avg_metrics[_k].append(int(round(float(np.mean(_vals)))) if _vals else 0)

            avg_metrics['pseudo_train_pred_new_count_per_client'] = exp_runs[0].get(
                'pseudo_train_pred_new_count_per_client', []
            )
            avg_metrics['pseudo_train_pred_old_count_per_client'] = exp_runs[0].get(
                'pseudo_train_pred_old_count_per_client', []
            )
            merged_metrics = avg_metrics

        merged_ce = _merge_continual_eval_runs(exp_runs)
        if merged_ce is not None:
            merged_metrics['continual_eval'] = merged_ce

        # 训练时间统计（多次运行取平均；单次为当次 total_time）
        total_times = [r.get('total_time_seconds', 0) for r in exp_runs if 'total_time_seconds' in r]
        merged_metrics['total_time_seconds'] = float(np.mean(total_times)) if total_times else 0
        if merged_metrics['total_time_seconds'] > 0:
            m, s = divmod(int(merged_metrics['total_time_seconds']), 60)
            fmt = f"{m}m {s}s" if m < 60 else f"{m // 60}h {m % 60}m {s}s"
            merged_metrics['total_time_formatted'] = fmt
        else:
            merged_metrics['total_time_formatted'] = '-'

        all_results[exp_type] = merged_metrics

        # 每个实验完成后立即保存JSON到实验文件夹（文件名包含参数）
        json_filename = get_result_filename(exp_type, exp_config, 'json', exp_dir=exp_dir)
        json_path = os.path.join(exp_dir, json_filename)
        save_metrics_json(json_path, merged_metrics, exp_config)
        json_files.append(json_path)
        print(f"✅ {exp_type} 结果已保存: {json_path}")
        if merged_metrics.get('total_time_formatted'):
            print(f"⏱️ 总训练时间: {merged_metrics['total_time_formatted']}")
    
    plot_files: List[str] = []
    if plot_results and all_results:
        try:
            plot_files = plot_all_results(all_results, base_config, exp_dir)
        except Exception as exc:
            print(f"\n⚠️ 绘图失败（JSON 已保存，可用 plot_from_json 补画）：{exc}")
            traceback.print_exc()
    return all_results, base_config, json_files, exp_dir, plot_files


def run_single_experiment(experiment_type: str, config: Dict = None,
                          exp_dir: str = None,
                          plot_results: bool = True) -> Tuple[Dict, Dict, str, str, List[str]]:
    """
    运行单个实验
    
    参数:
        experiment_type: 实验类型
        config: 实验配置
        exp_dir: 实验文件夹路径（如果为None则自动创建）
        plot_results: 联邦阶段结束后是否将曲线图保存到 exp_dir（默认 True；预热-only 不绘图）
        
    返回:
        metrics: 实验指标
        config: 使用的配置
        json_file: 保存的JSON文件路径
        exp_dir: 实验文件夹路径
        plot_files: 生成的图片路径列表（联邦阶段且 plot_results=True 时非空；预热-only 为空）
    """
    cfg = merge_default_config(config)
    configured_keys = get_configured_keys(config)
    cfg = apply_dataset_overrides(cfg)

    stage = cfg.get('continual_run_stage')
    if stage in ('server_init_pre', 'frame1_pre'):
        if exp_dir is None:
            exp_dir, cfg = resolve_exp_dir(
                cfg,
                configured_keys=configured_keys,
                experiment_label=get_experiment_abbrev(experiment_type),
            )
        set_seed(42, cfg)
        if stage == 'server_init_pre':
            from warmup import run_server_init_pre_only
            metrics = run_server_init_pre_only(cfg)
        else:
            from warmup import run_frame1_pre_warmup_only
            metrics = run_frame1_pre_warmup_only(cfg)
        json_filename = get_result_filename(experiment_type, cfg, 'json', exp_dir=exp_dir)
        json_path = os.path.join(exp_dir, json_filename)
        save_metrics_json(json_path, metrics, cfg)
        print(f"✅ {experiment_type}（{stage}）结果已保存: {json_path}")
        return metrics, cfg, json_path, exp_dir, []

    # 创建或解析实验文件夹
    if exp_dir is None:
        exp_dir, cfg = resolve_exp_dir(
            cfg,
            configured_keys=configured_keys,
            experiment_label=get_experiment_abbrev(experiment_type),
        )
    cfg["exp_dir"] = exp_dir
    if cfg.get("output_dir"):
        cfg["output_dir"] = exp_dir
    if cfg.get("checkpoint_dir"):
        cfg["checkpoint_dir"] = os.path.abspath(cfg["checkpoint_dir"])

    set_seed(42, cfg)
    
    data_loader, fed_learning, cfg = build_runner(cfg, seed=42)
    
    # 运行实验
    metrics = fed_learning.run_experiment(experiment_type, data_loader)
    
    # 保存JSON到实验文件夹（文件名包含参数）
    json_filename = get_result_filename(experiment_type, cfg, 'json', exp_dir=exp_dir)
    json_path = os.path.join(exp_dir, json_filename)
    save_metrics_json(json_path, metrics, cfg)

    print(f"✅ {experiment_type} 结果已保存: {json_path}")
    
    plot_files: List[str] = []
    if plot_results:
        try:
            plot_files = plot_all_results({experiment_type: metrics}, cfg, exp_dir)
        except Exception as exc:
            print(f"\n⚠️ 绘图失败（JSON 已保存，可用 plot_from_json 补画）：{exc}")
            traceback.print_exc()
    return metrics, cfg, json_path, exp_dir, plot_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Federated experiments. Pass one strategy per invocation (e.g. one tmux window per strategy)."
        )
    )
    parser.add_argument(
        "experiment_types",
        nargs="*",
        metavar="EXP",
        help=(
            "Experiment type(s), e.g. NoClientTrain AllClientsTrain. "
            "If omitted, runs a single default: AllClientsTrain."
        ),
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
        dest="num_runs",
        help="Repeat each experiment this many times (metrics averaged). Default: 1.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        metavar="DIR",
        help=(
            "检查点与 *_checkpoint_metrics.json 输出目录；"
            "最终 JSON/图表仍写入 runs/（或 --output-dir）。"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        metavar="DIR",
        help=(
            "固定实验输出目录；目录已存在则合并 config.json 并自动加载最新 checkpoint 续训。"
        ),
    )
    parser.add_argument(
        "--resume-dir",
        type=str,
        default=None,
        metavar="DIR",
        help=(
            "同 --output-dir（兼容旧参数）：读取 DIR/config.json，"
            "并自动加载 DIR/checkpoints/<策略>/ 下最新 .pth（若存在）。"
        ),
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="与 --output-dir / --resume-dir 联用时忽略已有检查点，从 frame1_pre 重新开始。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子（写入 config 并用于数据划分/调度）。",
    )
    parser.add_argument(
        "--upload-deadline",
        type=float,
        default=None,
        dest="upload_deadline",
        help="上行时隙占周期比例上限 upload_deadline（写入 config）。",
    )
    args = parser.parse_args()
    exp_list = list(args.experiment_types) if args.experiment_types else ["AllClientsTrain"]
    unknown = [e for e in exp_list if e not in EXPERIMENT_TYPES]
    if unknown:
        raise SystemExit(
            f"Unknown experiment type(s): {unknown}\n"
            f"Valid names include: {', '.join(EXPERIMENT_TYPES)}"
        )

    target_dir = args.output_dir or args.resume_dir
    user_cfg: Dict = {}
    if target_dir:
        user_cfg["output_dir"] = target_dir
    if args.checkpoint_dir:
        user_cfg["checkpoint_dir"] = os.path.abspath(args.checkpoint_dir)
    if args.no_resume:
        user_cfg["federated_auto_resume"] = False
    if args.seed is not None:
        user_cfg["seed"] = args.seed
    if args.upload_deadline is not None:
        user_cfg["upload_deadline"] = args.upload_deadline

    if target_dir:
        if os.path.isdir(target_dir) and os.path.isfile(os.path.join(target_dir, "config.json")):
            resume_cfg = build_config_for_exp_dir(
                target_dir,
                user_config=user_cfg or None,
                no_resume=args.no_resume,
            )
        else:
            resume_cfg = merge_default_config(user_cfg or None)
            resume_cfg = apply_dataset_overrides(resume_cfg)
        resume_cfg["experiment_label"] = "+".join(get_experiment_abbrev(e) for e in exp_list)
        all_results, config, json_files, exp_dir, plot_files = run_all_experiments(
            config=resume_cfg,
            experiment_types=exp_list,
            num_runs=args.num_runs,
            exp_dir=os.path.abspath(target_dir) if os.path.isdir(target_dir) else None,
        )
    else:
        all_results, config, json_files, exp_dir, plot_files = run_all_experiments(
            config=user_cfg or None,
            experiment_types=exp_list,
            num_runs=args.num_runs,
        )

    print("\n" + "=" * 70)
    print("实验完成！")
    for exp_type, res in all_results.items():
        t = res.get('total_time_formatted', '-')
        print(f"  - {exp_type}: 训练时间 {t}")
    print(f"\n所有结果已保存到 {exp_dir} 文件夹:")
    print("\nJSON 文件:")
    for f in json_files:
        print(f"  - {f}")
    print("\n图片文件:")
    for pf in plot_files:
        print(f"  - {pf}")
    print("=" * 70)
    