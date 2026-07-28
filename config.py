"""
================================================================================
配置文件 - 实验参数、样式配置、随机种子设置
================================================================================
"""

import os
from typing import Dict, Optional

import torch
import numpy as np
import random

# 持续学习运行阶段：server_init_pre / frame1_pre / frame1 / frame2_pre / frame2
# 类别按「帧」书写（示例 CIFAR-100：frame1 初始 30 类 + 新 20 类 = 50；frame2 再 +20 → 70）。
# SVHN/CIFAR-10：frame1 初始 6 + 新 4 = 10；frame2_new=0。默认值见 defaults_continual_frame() / DEFAULT_CONFIG
VALID_CONTINUAL_RUN_STAGES = (
    "server_init_pre",
    "frame1_pre",
    "frame1",
    "frame2_pre",
    "frame2",
)
# 客户端手工 Non-IID：总样本恒为 800。
# ——以下整段为上一版分配（已弃用，保留备查）—————————————————————————
# # 0–17：恰好 2 类、仅 0–5；按类别编号：奇数类(1,3,5)上样本总数 = 偶数类(0,2,4)上样本总数的三倍（即各 600 vs 200，合计 800）。
# # 18–29：每客户端恰好 1 个旧类（0–5）；新类（6–9）1 个或 2 个（共 2 类或 3 类/人）；总 800；与 continual_new_class_client_ids 一致。
MANUAL_ALLOCATION_FRAME1_OLD = {
    0: {0: 200, 1: 600},
    1: {0: 200, 3: 600},
    2: {0: 200, 5: 600},
    3: {2: 200, 1: 600},
    4: {2: 200, 3: 600},
    5: {2: 200, 5: 600},
    6: {4: 200, 1: 600},
    7: {4: 200, 3: 600},
    8: {4: 200, 5: 600},
    9: {2: 200, 1: 600},
    10: {4: 200, 5: 600},
    11: {0: 200, 5: 600},
    12: {0: 200, 3: 600},
    13: {4: 200, 1: 600},
    14: {4: 200, 3: 600},
    15: {2: 200, 5: 600},
    16: {2: 200, 3: 600},
    17: {0: 200, 1: 600},
    18: {0: 250, 7: 550},
    19: {1: 300, 8: 500},
    20: {2: 400, 9: 400},
    21: {3: 350, 6: 450},
    22: {4: 200, 7: 600},
    23: {5: 450, 8: 350},
    24: {0: 200, 6: 250, 9: 350},
    25: {1: 220, 7: 280, 8: 300},
    26: {2: 300, 6: 200, 7: 300},
    27: {3: 150, 8: 400, 9: 250},
    28: {4: 280, 6: 320, 9: 200},
    29: {5: 200, 7: 400, 8: 200},
}
# ——现役分配—————————————————————————————————————————————————————
# 30 客户端 × 800 = 24000；全局每类总样本数：
#   - 0,2,4,6,7,8,9 各 1500（偶数旧类与新类彼此一致）；
#   - 1,3,5 各 4500（奇数旧类彼此一致），且为奇数同类样本数 = 3 × {0/2/4 或 6/7/8/9 任一类的全局样本数}；
#     等价于全局「奇数三类合计」= 3 × 「偶数旧三类合计」，且 = 3 × 「新四类合计」。
# 0–17：仍以「两两类」覆盖 0–5，每客户端 150（偶数位类）+ 650（奇数位类）。
# 18–29：（旧类与单/双新类组合及 client id 对与 continual_new_class_client_ids / 代码一致）。
#       在每对「单客户端+双客户端」上旧类分摊和为 600，且两端旧样本量约在 [150,450]，
#       双端新类内拆分均 ≥50，避免病态 Non-IID；具体数值由约束下整数可行解给定。
MANUAL_ALLOCATION_FRAME1 = {
    0: {0: 150, 1: 650},
    1: {0: 150, 3: 650},
    2: {0: 150, 5: 650},
    3: {2: 150, 1: 650},
    4: {2: 150, 3: 650},
    5: {2: 150, 5: 650},
    6: {4: 150, 1: 650},
    7: {4: 150, 3: 650},
    8: {4: 150, 5: 650},
    9: {2: 150, 1: 650},
    10: {4: 150, 5: 650},
    11: {0: 150, 5: 650},
    12: {0: 150, 3: 650},
    13: {4: 150, 1: 650},
    14: {4: 150, 3: 650},
    15: {2: 150, 5: 650},
    16: {2: 150, 3: 650},
    17: {0: 150, 1: 650},
    18: {0: 450, 7: 350},
    19: {1: 450, 8: 350},
    20: {2: 450, 9: 350},
    21: {3: 450, 6: 350},
    22: {4: 350, 7: 450},
    23: {5: 350, 8: 450},
    24: {0: 150, 6: 50, 9: 600},
    25: {1: 150, 7: 350, 8: 300},
    26: {2: 150, 6: 600, 7: 50},
    27: {3: 150, 8: 150, 9: 500},
    28: {4: 250, 6: 500, 9: 50},
    29: {5: 250, 7: 300, 8: 250},
}


def get_frame1_total_num_classes(cfg: Dict) -> int:
    """frame1 阶段参与训练的类别总数：显式 frame1_total_num_classes，否则 initial + new。"""
    v = cfg.get("frame1_total_num_classes")
    if v is not None and v != "":
        return int(v)
    return int(cfg.get("frame1_initial_num_classes", 0)) + int(cfg.get("frame1_new_num_classes", 0))


def get_frame2_total_num_classes(cfg: Dict) -> int:
    """frame2 阶段总类别数：显式 frame2_total_num_classes，否则 frame1 总类数 + frame2_new。"""
    v = cfg.get("frame2_total_num_classes")
    if v is not None and v != "":
        return int(v)
    return get_frame1_total_num_classes(cfg) + int(cfg.get("frame2_new_num_classes", 0))


def build_default_server_labeled_per_class(cfg: Dict, num_classes: Optional[int] = None) -> Dict[int, int]:
    """
    服务器有标签每类数量：frame1 全 K；标签 0..frame1_initial-1 各 1000；其余至 K-1 各 labeled_per_class。
    num_classes 默认由 get_frame1_total_num_classes(cfg) 得到。
    """
    lpc = int(cfg.get('labeled_per_class', 100))
    old_split = int(cfg["frame1_initial_num_classes"])
    if num_classes is None:
        num_classes = get_frame1_total_num_classes(cfg)
        if num_classes <= 0:
            raise ValueError(
                "frame1 总类数无效：请设置 frame1_initial_num_classes + frame1_new_num_classes，"
                "或显式 frame1_total_num_classes"
            )
    out: Dict[int, int] = {}
    for c in range(num_classes):
        out[c] = 1000 if c < old_split else lpc
    return out


def normalize_continual_settings(cfg: Dict) -> Dict: 
    """校验 continual_run_stage。"""
    out = dict(cfg)
    stage = out.get("continual_run_stage")
    if stage is None or stage not in VALID_CONTINUAL_RUN_STAGES:
        raise ValueError(f"continual_run_stage 须为 {VALID_CONTINUAL_RUN_STAGES}，收到: {stage!r}")
    ds = str(out.get("dataset_name", "")).lower()
    if stage in ("frame2_pre", "frame2"):
        if ds != "cifar100":
            raise ValueError(
                f"continual_run_stage={stage!r} 仅用于 CIFAR-100，当前 dataset_name={out.get('dataset_name')!r}"
            )
        raise NotImplementedError("frame2_pre / frame2 尚未实现。")
    return out


# ================================================================================
# 默认实验配置（分块仅返回 dict，彼此不调用；最后一次性合并）
# ================================================================================
def defaults_client_and_data() -> Dict:
    return {
        "num_clients": 30,  # 无标签客户端总数
        "clients_per_round": 10,  # 每轮参与训练的客户端数上限
        "model_name": "cnn",  # 全局模型：cnn / resnet18
        "dataset_name": "svhn",  # 数据集：svhn / cifar10 / cifar100
        "labeled_per_class": 100,  # 服务器端每类有标签样本数（旧类在 build_default_server_labeled_per_class 中可覆盖为 1000）
        "alpha": 0.5,  # Dirichlet 划分参数（若数据管线使用）
        "batch_size": 64,  # 客户端无标签训练批大小
        "num_workers": 0,  # DataLoader 进程数（Windows 建议 0）
        "local_epochs_unlabeled": 5,  # 每轮每个客户端本地无标签训练 epoch 数
        "finetune_epochs": 5,  # 每轮服务器在有标签数据上微调的 epoch 数
        "lr_unlabeled": 0.005,  # 客户端无标签 SGD 学习率上限（联邦按轮次余弦衰减）
        "momentum": 0.9,  # SGD 动量（客户端与服务器微调共用）
        "weight_decay": 0.005,  # 权重衰减
        "server_batch_size": 32,  # 服务器微调 DataLoader 批大小
        # 每轮聚合后 gc + empty_cuda_cache，缓解碎片化导致的显存持续上涨观感（设为 False 可略省一点时间）
        "cuda_empty_cache_each_round": True,
    }


def defaults_continual_frame() -> Dict:
    return {
        "frame1_initial_num_classes": 6,  # 第一帧初始类别数（旧类）
        "frame1_new_num_classes": 4,  # 第一帧新增类别数
        "frame2_new_num_classes": 0,  # 第二帧新增类别数（CIFAR-100 预留；SVHN/CIFAR-10 常为 0）
        "frame1_federated_rounds": 8000,  # frame1 联邦通信总轮数
        "frame2_federated_rounds": 8000,  # frame2 联邦轮数（预留）
        "manual_allocation_frame1": MANUAL_ALLOCATION_FRAME1_OLD,  # 各客户端 train 样本按类别的手工 Non-IID 表
        "continual_new_class_client_ids": list(range(18, 30)),  # 含新类数据的客户端 id 列表（与手工表一致）
    }


def defaults_training_and_pseudo() -> Dict:
    pseudo_fixmatch_default = {
        "pseudo_threshold": 0.80,  # default 策略：伪标签最大置信度阈值（FixMatch）
    }
    pseudo_priority_old_new = {
        "pseudo_threshold_old": 0.95,  # 旧类分支置信度阈值（priority_old_new）
        "pseudo_threshold_new_initial": 0.5,  # 新类 τ_new 线性 ramp 起点
        "pseudo_threshold_new_final": 0.80,  # 新类 τ_new 线性 ramp 终点
        "pseudo_threshold_new_ramp_start_round": 0,  # τ_new 开始上升的联邦轮次
        "pseudo_threshold_new_ramp_end_round": 8000,  # τ_new 升至终值的联邦轮次
    }
    pseudo_sticky_new = {
        "pseudo_sticky_threshold": 0.80,  # sticky_new：判「高置信新类」与选伪标签的固定阈值 τ
    }
    return {
        **pseudo_fixmatch_default,
        **pseudo_priority_old_new,
        **pseudo_sticky_new,
    }


def defaults_warmup() -> Dict:
    return {
        "server_init_pre_warmup_rounds": 800,  # server_init_pre 阶段训练轮数（或迭代设计下的等价轮次）
        "server_init_pre_warmup_local_epochs": 5,  # 上述每轮内本地 epoch 数
        "server_init_pre_warmup_model_path": (  # 已保存的 server_init 预热权重路径（存在则直接加载）
            "pre/serverinit/serverinit_warmup_svhn_kold6_r800_e5_lr0.01_bs32_lpc100_mom0.9_cnn_s42.pth"
        ),
        "frame1_pre_warmup_rounds": 100,  # frame1_pre（全 K 或指定 K）预热轮数
        "frame1_pre_warmup_local_epochs": 5,  # frame1_pre 每轮本地 epoch 数
        "frame1_pre_warmup_model_path": (  # frame1 联邦入口使用的预热模型路径
            "pre/frame1pre/frame1pre_k2_warmup_svhn_k210_r100_e5_lr0.01_bs32_lpc100_mom0.9_cnn_s42.pth"
        ),
    }


def defaults_stage_wireless_and_meta() -> Dict:
    return {
        "continual_run_stage": "frame1",  # 当前流程阶段：server_init_pre / frame1_pre / frame1 / …
        "frame1_freeze_old_logits": False,  # True 时仅更新新类 logits 行（持续学习防旧类漂移）
        "total_bandwidth_mhz": 20.0,  # 仿真：上行总带宽（MHz）
        "tx_power_dbm": 23.0,  # 仿真：发射功率（dBm）
        "upload_deadline": 0.9,  # 仿真：上传时隙占周期比例上限
        "cell_radius": 250.0,  # 仿真：小区半径（米）
        "comp_delay_a_i": 0.5e-3,  # 仿真：客户端计算延迟模型参数 a_i
        "comp_delay_mu_i": 2.0e3,  # 仿真：客户端计算延迟模型参数 μ_i
        "seed": 100,  # 随机种子（调度、信道等）
        "use_wireless_scheduling": True,  # True：Random 等策略在排序后按带宽贪心截断接入
        # Random_PriorityOldNew 无线侧「伪标签快照」加权排序（策略 2/3 用到 λ）；策略 1 不使用 λ。
        "r_po_wireless_sched_lambda": 1.0,
        # 避免出现全零或未定义时出现排序截断/并列退化，略大于 0 即可。
        "r_po_wireless_sched_score_eps": 1e-8,
        "num_runs": 1,  # 每种实验类型重复运行次数
        # 联邦 frame1 断点续训：每 N 轮写入 exp_dir/checkpoints/<策略>/latest.pth；中断后同目录重跑可自动续上
        "federated_checkpoint_enabled": True,
        "federated_checkpoint_every": 10,
        "federated_auto_resume": True,
        # 每个策略目录下最多保留的 round_XXXXX.pth 数量（latest.pth 始终保留）；0 表示不限制
        "federated_checkpoint_max_keep": 3,
        # 非空时固定实验输出目录（覆盖 runs/exp_* 时间戳目录）；同目录重跑自动加载最新 checkpoint
        "output_dir": None,
        # 非空时检查点与 *_checkpoint_metrics.json 写入此目录（可与 output_dir / runs 分离）
        "checkpoint_dir": None,
        # 使用的 GPU 逻辑索引（对当前进程可见的编号；仅用 CUDA_VISIBLE_DEVICES 单卡时需写 0）
        "cuda_device": 3,
    }


DEFAULT_CONFIG = {
    **defaults_client_and_data(),
    **defaults_continual_frame(),
    **defaults_training_and_pseudo(),
    **defaults_warmup(),
    **defaults_stage_wireless_and_meta(),
}
DEFAULT_CONFIG["server_labeled_per_class"] = build_default_server_labeled_per_class(DEFAULT_CONFIG)

# ================================================================================
# 设备配置
# ================================================================================


def _resolve_cuda_device_index(config: Optional[Dict] = None) -> int:
    """读取配置中的 cuda_device；未提供时用 DEFAULT_CONFIG 的值（当前默认 2）。"""
    cfg = DEFAULT_CONFIG if config is None else config
    # 合并自 merge_default_config 的 cfg 必定含 cuda_device；仅手工 dict 可能缺省
    raw = cfg.get("cuda_device", DEFAULT_CONFIG["cuda_device"])
    idx = int(raw)
    if idx < 0:
        raise ValueError(f"cuda_device 须为非负整数，收到: {raw!r}")
    return idx


def get_device(config: Optional[Dict] = None) -> str:
    """
    返回本实验使用的设备字符串：与配置项 cuda_device 一致；无 CUDA 时为 cpu。
    调用时尽量传入 merge 后的 config，与 set_seed(config=...) 使用同一份。
    """
    if not torch.cuda.is_available():
        return "cpu"
    idx = _resolve_cuda_device_index(config)
    n = torch.cuda.device_count()
    if idx >= n:
        raise ValueError(
            f"配置 cuda_device={idx}，但当前进程仅可见 {n} 张 GPU（逻辑索引 0..{n-1}）。"
            "若使用 CUDA_VISIBLE_DEVICES，请改为仅暴露目标卡并把 cuda_device 设为 0（或对应逻辑编号）。"
        )
    return f"cuda:{idx}"


# ================================================================================
# 实验类型定义
EXPERIMENT_TYPES = [
    'NoClientTrain',         # 无客户端训练，仅服务器微调
    'AllClientsTrain',       # 所有客户端参与训练
    'Random',                # 随机选择若干客户端
    'BestChannel',           # 按信道增益降序，best-effort 带宽接入（或关闭无线时取前 clients_per_round）
    'NewClassClientsOnly',   # 仅从 continual_new_class_client_ids（默认 18..29）中选客户端；行为同 Random 但限制候选池
    # 调度同 Random，伪标签/选端策略由实验名单独指定（见 scheduler.STRATEGY_EXPERIMENT_OVERRIDES）
    'Random_PriorityOldNew',       # 旧/新分阈值 + 新类优先（tau_old=0.95，tau_new 线性升至 0.8）
    # 与 R_PO 相同伪标签；无线侧先按伪标签统计排序，再带宽贪心。PO_FPM=过阈比例+过阈均置信；PO_FPLN=ρ+λ×新类过阈均置信；PO_MFLN=过阈均置信+λ×新类在过阈内占比。
    'PO_FPM',
    'PO_FPLN',
    'PO_MFLN',
    'Random_StickyNew',            # 固定阈值；曾高置信判为新类则不再赋旧类伪标签
    'Random_FLFL',                 # FLFL：SAT 子集 + La/SACR + LSAA；调度同 Random（见 scheduler 覆盖项）
    'Random_FedLGMatch',           # FedLGMatch 对比：联合伪标签 + 本地 CE；调度同 Random
    'Random_FSSL_UC',              # FSSL-UC：高置信 CE+EML，低置信 KL；调度同 Random
    # # 与上对应基线调度相同；同一过阈值 mask，监督目标换为真实类别（伪标签仍每轮生成）
    # 'AllClientsTrain_TrueLabel',
    # 'Random_TrueLabel',
    # 'BestChannel_TrueLabel',
    # 'NewClassClientsOnly_TrueLabel',
    # # 与上对应基线调度相同；全样本真实监督，len(client) 聚合，不生成伪标签
    # 'AllClientsTrain_FullTrueLabel',
    # 'Random_FullTrueLabel',
    # 'BestChannel_FullTrueLabel',
    # 'NewClassClientsOnly_FullTrueLabel',
]

# 图例显示名（论文用完整描述；可含 matplotlib 数学模式如 r'$FL^2$'）
ABBREVIATIONS = {
    'NoClientTrain': 'No Device Training',
    'AllClientsTrain': 'All Devices Training',
    'Random': 'Random',
    'BestChannel': 'Best Channel',
    'NewClassClientsOnly': 'New-Class Devices Only',
    'Random_PriorityOldNew': 'Priority Old–New Thresholds',
    'PO_FPM': 'Pass-Ratio and Mean-Confidence Order',
    'PO_FPLN': 'Pass-Ratio and New-Class Confidence Order',
    'PO_MFLN': 'CATS-Semi',
    'Random_StickyNew': 'Sticky New-Class Pseudo-Labels',
    'Random_FLFL': r'$FL^2$',
    'Random_FedLGMatch': 'FedLGMatch',
    'Random_FSSL_UC': 'FSSL-UC',
    # 'AllClientsTrain_TrueLabel': 'All Clients Training (True Label)',
    # 'Random_TrueLabel': 'Random Selection (True Label)',
    # 'BestChannel_TrueLabel': 'Best Channel (True Label)',
    # 'NewClassClientsOnly_TrueLabel': 'New-Class Clients Only (True Label)',
    # 'AllClientsTrain_FullTrueLabel': 'All Clients Training (Full True Label)',
    # 'Random_FullTrueLabel': 'Random Selection (Full True Label)',
    # 'BestChannel_FullTrueLabel': 'Best Channel (Full True Label)',
    # 'NewClassClientsOnly_FullTrueLabel': 'New-Class Clients Only (Full True Label)',
}

# 目录 / checkpoint / 配色族键用短缩写（勿改图例时动这里即可）
PATH_ABBREVIATIONS = {
    'NoClientTrain': 'NCT',
    'AllClientsTrain': 'ACT',
    'Random': 'R',
    'BestChannel': 'BC',
    'NewClassClientsOnly': 'NCC',
    'Random_PriorityOldNew': 'R_PO',
    'PO_FPM': 'PO_FPM',
    'PO_FPLN': 'PO_FPLN',
    'PO_MFLN': 'PO_MFLN',
    'Random_StickyNew': 'R_SN',
    'Random_FLFL': 'R_FLFL',
    'Random_FedLGMatch': 'R_FedLG',
    'Random_FSSL_UC': 'R_FSSL_UC',
}


def get_path_abbrev(exp_type: str) -> str:
    """实验目录名、checkpoint 文件名、配色族键等路径安全缩写。"""
    if exp_type in PATH_ABBREVIATIONS:
        return PATH_ABBREVIATIONS[exp_type]
    return ''.join(ch for ch in exp_type if ch.isalnum())[:12] or exp_type


# ================================================================================
# 实验样式配置（线型/标记按实验类型；颜色按策略族 FAMILY_COLORS）
# ================================================================================
FAMILY_COLORS = {
    'NCT': '#757575',      # 灰
    'ACT': '#D32F2F',      # 红
    'BC': '#F9A825',       # 黄
    'NCC': '#1976D2',      # 蓝
    'R': '#388E3C',        # 绿
    'R_PO': '#7B1FA2',     # 紫
    # PO_* 无线加权变体：与 R_PO 分色系（青/橙/玫红），避免与紫/绿/蓝等基线混淆
    'PO_FPM': '#00ACC1',   # 青
    'PO_FPLN': '#FF7043',  # 深橙
    'PO_MFLN': '#E91E63',  # 玫红
    'R_SN': '#EC407A',     # 粉
    'R_FLFL': '#00796B',   # 青绿/深 teal（避免与 NCC 纯蓝 #1976D2 混淆）
    'R_FedLG': '#5D4037',  # 棕
    'R_FSSL_UC': '#E65100',  # 深橙（与 R_PO 紫、R 绿区分）
    'misc': '#424242',
}

EXP_LINESTYLES = {
    'NoClientTrain': '--', 'AllClientsTrain': '--', 'Random': ':',
    'BestChannel': ':',
    'NewClassClientsOnly': ':',
    'Random_PriorityOldNew': '-',
    'PO_FPM': '-',
    'PO_FPLN': '-',
    'PO_MFLN': '-',
    'Random_StickyNew': '-',
    'Random_FLFL': '-.',
    'Random_FedLGMatch': '-.',
    'Random_FSSL_UC': '-.',
    # 'AllClientsTrain_TrueLabel': '-.',
    # 'Random_TrueLabel': '-.',
    # 'BestChannel_TrueLabel': '-.',
    # 'NewClassClientsOnly_TrueLabel': '-.',
    # 'AllClientsTrain_FullTrueLabel': ':',
    # 'Random_FullTrueLabel': ':',
    # 'BestChannel_FullTrueLabel': ':',
    # 'NewClassClientsOnly_FullTrueLabel': ':',
}

EXP_MARKERS = {
    'NoClientTrain': 'o', 'AllClientsTrain': 'o', 'Random': 's',
    'BestChannel': 's',
    'NewClassClientsOnly': 's',
    'Random_PriorityOldNew': 'p',
    'PO_FPM': 'P',
    'PO_FPLN': '^',
    'PO_MFLN': 'v',
    'Random_StickyNew': 'h',
    'Random_FLFL': 'D',
    'Random_FedLGMatch': 'd',
    'Random_FSSL_UC': 'o',
    # 'AllClientsTrain_TrueLabel': '^',
    # 'Random_TrueLabel': 's',
    # 'BestChannel_TrueLabel': 'X',
    # 'NewClassClientsOnly_TrueLabel': 'D',
    # 'AllClientsTrain_FullTrueLabel': 'P',
    # 'Random_FullTrueLabel': 'h',
    # 'BestChannel_FullTrueLabel': '8',
    # 'NewClassClientsOnly_FullTrueLabel': 'D',
}

EXP_DESCRIPTIONS = {
    'NoClientTrain': 'No client-side training, only server fine-tune',
    'AllClientsTrain': 'All unlabeled clients participate each round',
    'Random': 'Random selection (with finetune)',
    'BestChannel': 'Best channel gain first, best-effort bandwidth',
    'NewClassClientsOnly': 'Random-like selection restricted to continual_new_class_client_ids (default clients 18–29)',
    'Random_PriorityOldNew': 'Random scheduling; old/new pseudo thresholds with new-class priority (see pseudo_threshold_policy)',
    'PO_FPM': 'Same pseudo as R_PO; ordering s = ρ_pass + mean_conf(pass) + eps; ties broken by channel gain; then bandwidth-greedy.',
    'PO_FPLN': 'Same pseudo as R_PO; s = ρ_pass + λ·mean_conf(pass ∧ pred-new) + eps (pred-new vs frame1_initial_num_classes); ties by gain; then bandwidth-greedy.',
    'PO_MFLN': 'Same pseudo as R_PO; s = mean_conf(pass) + λ·frac(pred-new | pass) + eps; ties by gain; then bandwidth-greedy.',
    'Random_StickyNew': 'Random scheduling; sticky new-class pseudo policy (no old pseudo after high-conf new)',
    'Random_FLFL': 'Random scheduling; FLFL (global pseudo + SAT subset, La + SACR, LSAA β∝1−τ_t); see scheduler STRATEGY_EXPERIMENT_OVERRIDES',
    'Random_FedLGMatch': 'Random; FedLGMatch-style: see modules/fedlgmatch.py',
    'Random_FSSL_UC': 'Random; FSSL-UC: see modules/fssl_uc.py',
    # 'AllClientsTrain_TrueLabel': 'Same as AllClientsTrain: same threshold mask; supervision uses GT class on masked samples',
    # 'Random_TrueLabel': 'Same as Random: same threshold mask; supervision uses GT class on masked samples',
    # 'BestChannel_TrueLabel': 'Same as BestChannel: same threshold mask; supervision uses GT class on masked samples',
    # 'NewClassClientsOnly_TrueLabel': 'Same as NewClassClientsOnly: same threshold mask; GT on masked samples',
    # 'AllClientsTrain_FullTrueLabel': 'Same scheduling as AllClientsTrain; all local samples supervised with GT; len() aggregation; no pseudo',
    # 'Random_FullTrueLabel': 'Same scheduling as Random; all local samples supervised with GT; len() aggregation; no pseudo',
    # 'BestChannel_FullTrueLabel': 'Same scheduling as BestChannel; all local samples supervised with GT; len() aggregation; no pseudo',
    # 'NewClassClientsOnly_FullTrueLabel': 'Same scheduling as NewClassClientsOnly; GT on all local samples; len() aggregation; no pseudo',
}


# ================================================================================
# 实验目录 / 结果文件命名
# ================================================================================
def get_file_stage_tag(config: Dict) -> str:
    """文件名片段：server_init_pre | frame1_pre | frame1 联邦 | frame2（预留）等。"""
    if not config:
        return "frame1"
    stage = config.get("continual_run_stage")
    if stage == "server_init_pre":
        return "server_init"
    if stage == "frame1_pre":
        return "frame1_pre"
    if stage == "frame2_pre":
        return "frame2_pre"
    if stage == "frame1":
        return "frame1"
    if stage == "frame2":
        return "frame2"
    raise ValueError(f"未知的 continual_run_stage: {stage!r}")


def _rounds_for_stage_tag(config: Dict, tag: str) -> int:
    if tag == "server_init":
        return int(config["server_init_pre_warmup_rounds"])
    if tag == "frame1_pre":
        return int(config["frame1_pre_warmup_rounds"])
    if tag == "frame2_pre":
        return int(config["frame1_pre_warmup_rounds"])
    if tag == "frame1":
        return int(config["frame1_federated_rounds"])
    if tag == "frame2":
        raise ValueError("continual_run_stage=frame2 未实现，无法取轮数")
    raise ValueError(f"未知的阶段标签: {tag!r}")


def get_num_rounds(config: Dict) -> int:
    """当前 continual_run_stage 对应的通信轮数（与 get_param_str 中的 r 段一致）。"""
    tag = get_file_stage_tag(config)
    return _rounds_for_stage_tag(config, tag)


def get_param_str(config: Dict, *, stage: Optional[str] = None) -> str:
    """生成用于文件命名的参数字符串：阶段 + 轮次 + 阈值（不含 alpha / 客户端数）。"""
    if stage is not None:
        tag = stage
    else:
        tag = get_file_stage_tag(config)
    rounds = _rounds_for_stage_tag(config, tag)
    threshold = float(config.get("pseudo_threshold", 0.95))
    return f"{tag}_r{rounds}_t{threshold}"


def get_result_filename(
    exp_name: str,
    config: Dict,
    ext: str = "json",
    exp_dir: Optional[str] = None,
) -> str:
    """生成结果文件名。"""
    if exp_dir:
        base = os.path.basename(exp_dir.rstrip(os.sep))
        return f"{base}_{exp_name}.{ext}"
    param_str = get_param_str(config)
    return f"{exp_name}_{param_str}_result.{ext}"


# ================================================================================
# 论文对比图样式（与 magazine_fcl/cifar100_premodel/plot_visualization.py 一致）
# ================================================================================
COMPARISON_PLOT_STYLE = {
    "figsize": (14, 8),
    "rounds_to_target_figsize": (14, 10),
    "continual_pseudo_precision_new_figsize": (14, 10),
    "scatter_marker_size": 10,
    "axis_label_fontsize": 20,
    "tick_fontsize": 20,
    "legend_fontsize": 20,
    "title_fontsize": 20,
    "annotate_fontsize": 14,
    "line_width": 1.5,
    "dpi": 300,
    "pdf_dpi": 600,
    "facecolor": "#ffffff",
    "grid_alpha": 0.3,
    "grid_color": "grey",
    "grid_linestyle": "-",
    "grid_linewidth": 0.5,
}


# ================================================================================
# 随机种子设置
# ================================================================================
def set_seed(seed: int = 42, config: Optional[Dict] = None) -> None:
    """
    设置随机种子。仅对配置中的 cuda_device 设 CUDA RNG，并 torch.cuda.set_device，
    避免 manual_seed_all 在所有可见 GPU 上建上下文。
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        idx = _resolve_cuda_device_index(config)
        n = torch.cuda.device_count()
        if idx >= n:
            raise ValueError(
                f"set_seed: cuda_device={idx} 超出可见 GPU 数量 {n}，请检查配置与 CUDA_VISIBLE_DEVICES。"
            )
        torch.cuda.set_device(idx)
        torch.cuda.manual_seed(seed)
