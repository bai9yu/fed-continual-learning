"""
================================================================================
联邦学习主框架 - FederatedLearning 类
================================================================================
"""


import copy
import gc
import hashlib
import json
import os
import time
import torch
import numpy as np
from typing import Any, Dict, List, Optional

from modules.models import build_model, _set_continual_class_state
from modules.fixmatch import (
    aggregate_pseudo_split_stats,
    aggregate_pseudo_split_stats_for_clients,
    build_federated_loader_from_config,
)
from scheduler import (
    select_clients_by_experiment,
    parse_client_supervision_suffix,
    get_experiment_strategy_overrides,
)
from local_update import (
    cosine_lr_for_global_round,
    generate_pseudo_labels_fixmatch,
    local_training_fixmatch,
    finetune_global_model,
)
from modules.flfl import (
    flfl_prepare_round_pseudo,
    flfl_lsaa_aggregate,
    local_training_flfl,
)
from modules.fedlgmatch import (
    fedlgmatch_prepare_round_pseudo,
    local_training_fedlgmatch,
)
from modules.fssl_uc import local_training_fssl_uc
from modules.aggregation import (
    federated_averaging,
    evaluate_model,
    evaluate_model_loss,
    evaluate_model_continual_metrics,
)
from modules.wireless_channel import calculate_model_size_bits
from config import get_device, get_frame1_total_num_classes, ABBREVIATIONS
from warmup import (
    load_warmup_model,
    get_frame1_pre_k2_warmup_model_path,
)


def _clone_model_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """快照存 CPU，避免每台客户端 + 服务器的 state_dict 在 GPU 各留一份拷贝（显存成倍且易持续上涨）。"""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _state_dict_tensors_to_cpu(state_dict: Optional[Dict[str, torch.Tensor]]) -> Optional[Dict[str, torch.Tensor]]:
    if state_dict is None:
        return None
    return {k: v.detach().cpu().clone() for k, v in state_dict.items()}


def _checkpoint_nested_to_cpu(obj: Any) -> Any:
    """仅用于写入磁盘检查点；与训练过程中的 optimizer 快照策略无关。"""
    if obj is None:
        return None
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {k: _checkpoint_nested_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_checkpoint_nested_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_checkpoint_nested_to_cpu(v) for v in obj)
    return obj


def _resolve_checkpoint_root(cfg: Dict) -> Optional[str]:
    root = cfg.get("checkpoint_dir") or cfg.get("exp_dir")
    if not root:
        return None
    return os.path.abspath(root)


def _federated_checkpoint_dir(checkpoint_root: str, experiment_type: str) -> str:
    return os.path.join(checkpoint_root, "checkpoints", experiment_type)


def _federated_checkpoint_path(checkpoint_root: str, experiment_type: str, name: str = "latest") -> str:
    return os.path.join(_federated_checkpoint_dir(checkpoint_root, experiment_type), f"{name}.pth")


def _atomic_torch_save(payload: Dict, path: str) -> None:
    tmp_path = path + ".tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def _data_loader_fingerprint(data_loader, cfg: Dict) -> Dict[str, Any]:
    ma = cfg.get("manual_allocation_frame1")
    if ma is not None:
        ma_hash = hashlib.md5(
            json.dumps(ma, sort_keys=True, default=str).encode()
        ).hexdigest()
    else:
        ma_hash = None
    return {
        "seed": int(getattr(data_loader, "seed", cfg.get("seed", 42))),
        "num_clients": int(getattr(data_loader, "num_clients", cfg.get("num_clients", 0))),
        "alpha": float(getattr(data_loader, "alpha", cfg.get("alpha", 0))),
        "manual_allocation_hash": ma_hash,
    }


def _serialize_optional_array(arr) -> Optional[List]:
    if arr is None:
        return None
    return np.asarray(arr).tolist()


def _serialize_client_dataset_state(ds) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "ever_predicted_new": np.asarray(
            getattr(ds, "ever_predicted_new", np.zeros(len(ds), dtype=bool)), dtype=bool
        ).tolist(),
        "supervision_mode": getattr(ds, "supervision_mode", "pseudo"),
        "pseudo_labels": _serialize_optional_array(getattr(ds, "pseudo_labels", None)),
        "pseudo_mask": _serialize_optional_array(getattr(ds, "pseudo_mask", None)),
        "pseudo_confidence": _serialize_optional_array(getattr(ds, "pseudo_confidence", None)),
    }
    flfl_subset = getattr(ds, "flfl_subset_indices", None)
    if flfl_subset is not None:
        out["flfl_subset_indices"] = np.asarray(flfl_subset).tolist()
    flfl_pseudo = getattr(ds, "flfl_pseudo_full", None)
    if flfl_pseudo is not None:
        out["flfl_pseudo_full"] = _serialize_optional_array(flfl_pseudo)
    flfl_fix = getattr(ds, "flfl_fix_full", None)
    if flfl_fix is not None:
        out["flfl_fix_full"] = _serialize_optional_array(flfl_fix)
    if hasattr(ds, "flfl_lsaa_tau"):
        tau = getattr(ds, "flfl_lsaa_tau", None)
        if tau is not None:
            out["flfl_lsaa_tau"] = float(tau)
    return out


def _restore_client_dataset_state(ds, cdata: Dict[str, Any]) -> None:
    n = len(ds)
    ever = cdata.get("ever_predicted_new")
    if ever is not None:
        ever_arr = np.asarray(ever, dtype=bool)
        if len(ever_arr) != n:
            raise RuntimeError(
                f"检查点 ever_predicted_new 长度 {len(ever_arr)} 与数据集 {n} 不一致"
            )
        ds.ever_predicted_new = ever_arr

    sm = cdata.get("supervision_mode")
    if sm is not None:
        ds.set_supervision_mode(sm)

    pl = cdata.get("pseudo_labels")
    ds.pseudo_labels = None if pl is None else np.asarray(pl)
    pm = cdata.get("pseudo_mask")
    ds.pseudo_mask = None if pm is None else np.asarray(pm, dtype=bool)
    pc = cdata.get("pseudo_confidence")
    ds.pseudo_confidence = None if pc is None else np.asarray(pc, dtype=np.float32)

    if "flfl_subset_indices" in cdata:
        ds.flfl_subset_indices = np.asarray(cdata["flfl_subset_indices"], dtype=np.int64)
    if "flfl_pseudo_full" in cdata:
        ds.flfl_pseudo_full = np.asarray(cdata["flfl_pseudo_full"])
    if "flfl_fix_full" in cdata:
        ds.flfl_fix_full = np.asarray(cdata["flfl_fix_full"])
    if cdata.get("flfl_lsaa_tau") is not None:
        ds.flfl_lsaa_tau = float(cdata["flfl_lsaa_tau"])


def _validate_data_loader_fingerprint(
    ckpt_fp: Optional[Dict[str, Any]],
    current_fp: Dict[str, Any],
) -> None:
    if not ckpt_fp:
        return
    mismatches = []
    for key in ("seed", "num_clients", "alpha", "manual_allocation_hash"):
        if ckpt_fp.get(key) != current_fp.get(key):
            mismatches.append(f"{key}: ckpt={ckpt_fp.get(key)!r} current={current_fp.get(key)!r}")
    if mismatches:
        raise RuntimeError(
            "检查点 data_loader_fingerprint 与当前配置不一致，拒绝续训:\n  "
            + "\n  ".join(mismatches)
        )


def _find_latest_checkpoint_path(checkpoint_root: str, experiment_type: str) -> Optional[str]:
    """扫描 checkpoints/<策略>/*.pth，按 next_round 最大选取；并列时取 mtime 最新。"""
    ckpt_dir = _federated_checkpoint_dir(checkpoint_root, experiment_type)
    if not os.path.isdir(ckpt_dir):
        return None
    best_path: Optional[str] = None
    best_round = -1
    best_mtime = 0.0
    for name in os.listdir(ckpt_dir):
        if not name.endswith(".pth") or name.endswith(".tmp"):
            continue
        path = os.path.join(ckpt_dir, name)
        if not os.path.isfile(path):
            continue
        try:
            ckpt = torch.load(path, map_location="cpu")
            nr = int(ckpt.get("next_round", -1))
        except Exception:
            continue
        mtime = os.path.getmtime(path)
        if nr > best_round or (nr == best_round and mtime > best_mtime):
            best_round = nr
            best_mtime = mtime
            best_path = path
    return best_path


def _list_round_checkpoint_paths(ckpt_dir: str) -> List[tuple]:
    """返回 (next_round, path) 列表，仅含 round_XXXXX.pth。"""
    out: List[tuple] = []
    if not os.path.isdir(ckpt_dir):
        return out
    prefix, suffix = "round_", ".pth"
    for name in os.listdir(ckpt_dir):
        if not name.startswith(prefix) or not name.endswith(suffix) or name.endswith(".tmp"):
            continue
        mid = name[len(prefix): -len(suffix)]
        try:
            nr = int(mid)
        except ValueError:
            continue
        path = os.path.join(ckpt_dir, name)
        if os.path.isfile(path):
            out.append((nr, path))
    return out


def _prune_round_checkpoints(ckpt_dir: str, max_keep: int) -> None:
    """保留 next_round 最大的 max_keep 个 round_*.pth，删除更旧的；latest.pth 不受影响。"""
    if max_keep <= 0:
        return
    rounds = _list_round_checkpoint_paths(ckpt_dir)
    if len(rounds) <= max_keep:
        return
    rounds.sort(key=lambda x: x[0], reverse=True)
    for _, path in rounds[max_keep:]:
        try:
            os.remove(path)
            print(f"🗑️ 已删除旧检查点: {os.path.basename(path)}")
        except OSError as exc:
            print(f"⚠️ 删除旧检查点失败 {path}: {exc}")


class FederatedClient:
    def __init__(self, client_id: int):
        self.client_id = client_id
        self.model_state_dict: Optional[Dict[str, torch.Tensor]] = None
        self.optimizer_state_dict: Optional[dict] = None
        self.active = False


class FederatedServer:
    def __init__(self) -> None:
        self.model_state_dict: Optional[Dict[str, torch.Tensor]] = None
        self.optimizer_state_dict: Optional[dict] = None
        # FLFL / LSAA：与主服务器微调优化器分离的聚合用 SGD 状态
        self.flfl_lsaa_optimizer_state: Optional[dict] = None


def _frame1_num_old_classes(cfg: Dict) -> int:
    return int(cfg["frame1_initial_num_classes"])


def _require_federated_training_stage(cfg: Dict) -> None:
    """联邦训练仅允许 continual_run_stage=frame1（与命名一致，不使用内部「frame2」指代数据划分）。"""
    s = cfg.get("continual_run_stage")
    if s in ("server_init_pre", "frame1_pre", "frame2_pre"):
        raise ValueError(
            "continual_run_stage 为 server_init_pre / frame1_pre / frame2_pre 时不应进入联邦训练，请使用预热入口。"
        )
    if s == "frame2":
        raise NotImplementedError(
            "continual_run_stage=frame2 仅 CIFAR-100 预留，尚未实现；SVHN/CIFAR-10 请使用 continual_run_stage=frame1。"
        )
    if s != "frame1":
        raise ValueError(f"联邦训练要求 continual_run_stage 为 frame1，收到: {s!r}")


def _format_duration(seconds: float) -> str:
    """将秒数格式化为可读字符串，如 '1h 23m 45s'"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m {s}s"


class FederatedLearning:
    """联邦学习框架 - FixMatch + 智能调度"""
    
    def __init__(self,
                 local_epochs_unlabeled: int,
                 lr_unlabeled: float, momentum: float,
                 pseudo_threshold: float,
                 num_classes: int,
                 frame1_pre_warmup_model_path: str,
                 finetune_epochs: int = 5,
                 weight_decay: float = 0.0005,
                 config: Dict = None,
                 device: Optional[str] = None):
        self.local_epochs_unlabeled = local_epochs_unlabeled
        self.lr_unlabeled = lr_unlabeled
        self.momentum = momentum
        self.pseudo_threshold = pseudo_threshold
        self.weight_decay = weight_decay
        self.device = device or get_device(config)
        self.num_classes = num_classes
        
        self.frame1_pre_warmup_model_path = frame1_pre_warmup_model_path
        
        self.finetune_epochs = finetune_epochs
        self.config = config
        self.model_name = (config or {}).get('model_name', 'cnn')
        
        self.global_model = build_model(self.model_name, num_classes=num_classes).to(self.device)
        self.metrics = self._init_metrics()
        self.warmup_metrics = {}
        self.clients: Dict[int, FederatedClient] = {}
        self.server = FederatedServer()
        self._training_elapsed_before = 0.0
        self._session_train_start: Optional[float] = None

    def _federated_checkpoint_enabled(self, cfg: Dict) -> bool:
        return bool(cfg.get("federated_checkpoint_enabled", True)) and bool(_resolve_checkpoint_root(cfg))

    def _save_federated_checkpoint(
        self,
        checkpoint_root: str,
        experiment_type: str,
        data_loader,
        next_round: int,
        num_rounds_total: int,
    ) -> None:
        cfg = self.config or {}
        ckpt_dir = _federated_checkpoint_dir(checkpoint_root, experiment_type)
        os.makedirs(ckpt_dir, exist_ok=True)
        path = _federated_checkpoint_path(checkpoint_root, experiment_type)

        clients_out: Dict[str, Dict[str, Any]] = {}
        for cid, fc in self.clients.items():
            if fc.model_state_dict is None and fc.optimizer_state_dict is None:
                continue
            clients_out[str(cid)] = {
                "model_state_dict": _state_dict_tensors_to_cpu(fc.model_state_dict),
                "optimizer_state_dict": _checkpoint_nested_to_cpu(fc.optimizer_state_dict),
            }

        client_datasets_out: Dict[str, Dict[str, Any]] = {}
        ever_out: Dict[str, List[bool]] = {}
        for cid in range(data_loader.num_clients):
            ds = data_loader.client_datasets[cid]
            ds_state = _serialize_client_dataset_state(ds)
            client_datasets_out[str(cid)] = ds_state
            ever = ds_state.get("ever_predicted_new")
            if ever is not None:
                ever_out[str(cid)] = ever

        elapsed = float(self._training_elapsed_before)
        if self._session_train_start is not None:
            elapsed += time.perf_counter() - self._session_train_start

        payload = {
            "version": 2,
            "experiment_type": experiment_type,
            "next_round": int(next_round),
            "num_rounds_total": int(num_rounds_total),
            "metrics": copy.deepcopy(self.metrics),
            "global_model_state_dict": _clone_model_state_dict(self.global_model),
            "server": {
                "model_state_dict": _state_dict_tensors_to_cpu(self.server.model_state_dict),
                "optimizer_state_dict": _checkpoint_nested_to_cpu(self.server.optimizer_state_dict),
                "flfl_lsaa_optimizer_state": _checkpoint_nested_to_cpu(
                    self.server.flfl_lsaa_optimizer_state
                ),
            },
            "clients": clients_out,
            "client_datasets": client_datasets_out,
            "client_ever_predicted_new": ever_out,
            "data_loader_fingerprint": _data_loader_fingerprint(data_loader, cfg),
            "training_elapsed_seconds": elapsed,
            "num_classes": int(self.num_classes),
            "model_name": self.model_name,
        }

        _atomic_torch_save(payload, path)
        round_path = _federated_checkpoint_path(
            checkpoint_root, experiment_type, name=f"round_{int(next_round):05d}"
        )
        _atomic_torch_save(payload, round_path)

        max_keep = int(cfg.get("federated_checkpoint_max_keep", 3))
        _prune_round_checkpoints(ckpt_dir, max_keep)

        from config import get_path_abbrev
        abbrev = get_path_abbrev(experiment_type)
        metrics_path = os.path.join(checkpoint_root, f"{abbrev}_checkpoint_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "next_round": next_round,
                    "num_rounds_total": num_rounds_total,
                    "completed_rounds": len(self.metrics.get("round", [])),
                    "metrics": self.metrics,
                },
                f,
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        print(
            f"💾 检查点已保存: {path} / {os.path.basename(round_path)} "
            f"（下一轮 {next_round}/{num_rounds_total}）"
        )

    def _try_load_federated_checkpoint(
        self,
        checkpoint_root: str,
        experiment_type: str,
        data_loader,
        k1: int,
        k2: int,
    ) -> Optional[Dict[str, Any]]:
        cfg = self.config or {}
        if not cfg.get("federated_auto_resume", True):
            return None
        path = _find_latest_checkpoint_path(checkpoint_root, experiment_type)
        if path is None:
            return None

        ckpt = torch.load(path, map_location=self.device)
        next_round = int(ckpt.get("next_round", -1))
        print(f"\n📂 发现检查点 round_{next_round:05d}，尝试恢复: {path}")
        if ckpt.get("experiment_type") != experiment_type:
            raise RuntimeError(
                f"检查点策略 {ckpt.get('experiment_type')!r} 与当前 {experiment_type!r} 不一致"
            )
        if int(ckpt.get("num_classes", k2)) != k2:
            raise RuntimeError("检查点 num_classes 与当前配置不一致，无法续训")

        current_fp = _data_loader_fingerprint(data_loader, cfg)
        _validate_data_loader_fingerprint(ckpt.get("data_loader_fingerprint"), current_fp)

        self.global_model.load_state_dict(ckpt["global_model_state_dict"])
        _set_continual_class_state(self.global_model, k1, k2)
        freeze_old = bool(cfg.get("frame1_freeze_old_logits", True))
        self.global_model.freeze_old_classes(freeze_old)

        srv = ckpt.get("server") or {}
        self.server.model_state_dict = srv.get("model_state_dict")
        self.server.optimizer_state_dict = srv.get("optimizer_state_dict")
        self.server.flfl_lsaa_optimizer_state = srv.get("flfl_lsaa_optimizer_state")

        self._ensure_clients(data_loader.num_clients)
        for cid_str, cdata in (ckpt.get("clients") or {}).items():
            cid = int(cid_str)
            self._ensure_clients(cid + 1)
            fc = self.clients[cid]
            fc.model_state_dict = cdata.get("model_state_dict")
            fc.optimizer_state_dict = cdata.get("optimizer_state_dict")

        ckpt_version = int(ckpt.get("version", 1))
        if ckpt_version >= 2 and ckpt.get("client_datasets"):
            for cid_str, ds_state in ckpt["client_datasets"].items():
                cid = int(cid_str)
                _restore_client_dataset_state(data_loader.client_datasets[cid], ds_state)
        else:
            for cid_str, ever_list in (ckpt.get("client_ever_predicted_new") or {}).items():
                cid = int(cid_str)
                ds = data_loader.client_datasets[cid]
                ds.ever_predicted_new = np.asarray(ever_list, dtype=bool)

        self.metrics = ckpt.get("metrics") or self._init_metrics()
        self._training_elapsed_before = float(ckpt.get("training_elapsed_seconds", 0.0))
        next_round = int(ckpt.get("next_round", len(self.metrics.get("round", []))))
        ckpt["next_round"] = next_round
        print(
            f"✅ 已恢复至第 {next_round} 轮（已完成 {len(self.metrics.get('round', []))} 轮记录）"
        )
        return ckpt

    def _ensure_clients(self, num_clients: int) -> None:
        for cid in range(num_clients):
            if cid not in self.clients:
                self.clients[cid] = FederatedClient(cid)

    def _sync_server_from_global(self) -> None:
        self.server.model_state_dict = _clone_model_state_dict(self.global_model)

    def distribute(self, client_ids: List[int]) -> None:
        if self.server.model_state_dict is None:
            self._sync_server_from_global()
        sd = self.server.model_state_dict
        for cid in client_ids:
            c = self.clients[cid]
            c.model_state_dict = copy.deepcopy(sd)
            c.active = True

    def _init_metrics(self) -> Dict:
        return {
            'test_accuracy': [],
            # 每轮：全局模型在测试集上的平均 CE（与 evaluate_model 同子集）
            'test_loss': [],
            'pseudo_mean_confidence': [],  # 训练样本全体平均置信度（调度客户端聚合）
            'pseudo_selected': [],  # 参与本地训练的伪标签样本总数（与 mask 上过阈总数一致口径）
            'pseudo_train_conf_mean_old': [],  # 真标签为旧类的训练样本平均置信度
            'pseudo_train_conf_mean_new': [],
            'pseudo_train_precision_old': [],  # 真标签为旧/新/全体 的训练样本精度
            'pseudo_train_precision_new': [],
            'pseudo_train_precision_all': [],
            'pseudo_train_pred_old_accuracy': [],  # 预测为旧类/新类的训练样本上，与真标签一致的比例
            'pseudo_train_pred_new_accuracy': [],
            'pseudo_train_pred_old_count_total': [],  # 调度客户端合计：预测为旧/新类的训练样本数
            'pseudo_train_pred_new_count_total': [],
            'pseudo_scheduled_misclass_new_to_old': [],  # 无标签侧：真新→伪旧 比例（调度+过阈）
            'round': [],
            'selected_clients': [],
            'pseudo_per_class_per_client': [],  # 每轮每个客户端各类别通过阈值的样本数
            'pseudo_train_pred_new_count_per_client': [],  # 各客户端训练步中预测为新类的样本数
            'pseudo_train_pred_old_count_per_client': [],
            'pseudo_precision_old_scheduled': [],  # 过阈+调度：按真标签旧/新/全体（数据集 mask）
            'pseudo_precision_new_scheduled': [],
            'pseudo_precision_total_scheduled': [],
            # 回合初、全客户端无标签：mask 上与真标签不符的伪标签中，真实类 = softmax 次高类的比例
            'pseudo_masked_wrong_total': [],
            'pseudo_masked_wrong_runner_hit_total': [],
            'pseudo_masked_wrong_runnerup_rate': [],
            # 每轮一条：含测试集旧/新类与误判率、无标签侧伪标签旧/新精度等（k1=0 时存 None）
            'continual_eval': [],
        }

    def _handle_warmup(self, data_loader) -> Dict:
        """
        处理预热训练逻辑
        """
        # 补充data_loader相关的配置（如果config存在）
        if self.config is not None:
            self.config['num_clients'] = data_loader.num_clients
            if 'alpha' not in self.config or self.config['alpha'] is None:
                self.config['alpha'] = getattr(data_loader, 'alpha', None)
            if 'dataset_name' not in self.config or self.config['dataset_name'] is None:
                self.config['dataset_name'] = getattr(data_loader, 'dataset_name', None)

        # 仅加载预热模型（预热需提前独立运行）
        load_path = self.frame1_pre_warmup_model_path
        
        if self.config is not None:
            print("\nℹ️ 将直接加载配置中的预热模型路径")
        
        print(f"\n📂 加载预热模型: {load_path}")
        try:
            warmup_metrics, config_matched = load_warmup_model(
                self.global_model,
                load_path,
                current_config=self.config,
                strict_match=False,
                skip_config_check=True,
                device=self.device
            )
            if not config_matched:
                print("⚠️ 预热模型配置与当前实验不完全匹配，可能影响结果")
            return warmup_metrics
        except FileNotFoundError:
            raise RuntimeError(
                f"未找到预热模型: {load_path}\n"
                "请先运行 warmup.py 独立完成预热训练并保存模型。"
            )
        except ValueError as e:
            raise RuntimeError(
                f"预热模型配置不匹配: {e}\n"
                "请重新运行预热训练或检查配置。"
            )

    def _print_training_header(
        self,
        experiment_type: str,
        num_rounds: int,
        supervision_mode: str,
        wireless_ch,
        use_wireless: bool,
        base_exp: str,
        continual_tag: str = "",
    ) -> None:
        print("\n" + "=" * 70)
        print("半监督联邦学习 (Semi-supervised Stage)" + continual_tag)
        print("=" * 70)
        print(f"  - 实验类型: {experiment_type}")
        print(f"  - 通信轮次: {num_rounds}")
        print(f"  - 本地训练 epoch / 微调: local_epochs_unlabeled={self.local_epochs_unlabeled}, finetune_epochs={self.finetune_epochs}")
        print(
            f"  - 学习率: 全局轮次余弦（T=通信轮次）, η_max=lr_unlabeled={self.lr_unlabeled}, η_min=0；"
            " 每轮客户端与服务器微调共用同一标量 lr"
        )
        print(f"  - 伪标签阈值: {self.pseudo_threshold}")
        pol = (self.config or {}).get("pseudo_label_policy", "default")
        if pol != "default":
            print(f"  - 伪标签策略 pseudo_label_policy: {pol}")
        if pol == "flfl":
            print(
                "  - FLFL：全局伪标签+SAT 子集、La+SACR(ASAM+KL)、"
                "LSAA 聚合(β∝1−τ_t)、global_ft=1（先 LSAA 再有标签微调）"
            )
        if pol == "fedlgmatch":
            print(
                "  - FedLGMatch 对比：弱视图下全局+上轮本地联合 softmax 伪标签，"
                "本地每 epoch 单 batch CE（聚合 FedAvg+服务端微调）"
            )
        if pol == "fssl_uc":
            print(
                "  - FSSL-UC 对比(IoTJ'26)：高置信 CE+EML，低置信 KL(强‖弱)；"
                "回合初仍写全局 FixMatch 伪标签仅用于统计/画曲线"
            )
        if supervision_mode == "masked_true":
            print(
                "  - 客户端本地训练: 与伪标签基线相同的过阈值 mask 与 mask.sum() 聚合；"
                "监督目标为真实类别"
            )
        elif supervision_mode == "full_true":
            print(
                "  - 客户端本地训练: 全样本真实类别监督，聚合权重 len(client)；不生成伪标签"
            )
        if wireless_ch is not None:
            print(
                f"  - 无线: use_wireless_scheduling={use_wireless}；"
                f"BestChannel=按增益排序；其余策略在开启无线时按序带宽贪心"
            )
        print("=" * 70 + "\n")

    def _setup_wireless_channel(self, data_loader, base_exp: str):
        cfg = self.config or {}
        use_wireless = cfg.get("use_wireless_scheduling", True)
        need_channel = (base_exp == "BestChannel") or (
            use_wireless
            and base_exp not in ("NoClientTrain", "AllClientsTrain")
        )
        wireless_ch = None
        if need_channel:
            from modules.wireless_channel import WirelessChannel, wireless_args_from_config

            seed = int(cfg.get("seed", 42))
            wargs = wireless_args_from_config(cfg, seed=seed, num_users=data_loader.num_clients)
            wireless_ch = WirelessChannel(wargs, model=self.global_model)
        return wireless_ch, use_wireless

    def _run_training_rounds(
        self,
        data_loader,
        experiment_type: str,
        num_rounds: int,
        round_offset: int,
        wireless_ch,
        base_exp: str,
        supervision_mode: str,
        continual_run_stage_tag: str,
        start_round: int = 0,
    ) -> None:
        cfg = self.config or {}
        cfg_eff = {**cfg, **get_experiment_strategy_overrides(experiment_type)}
        use_wireless = cfg.get("use_wireless_scheduling", True)
        is_flfl = str(cfg_eff.get("pseudo_label_policy", "")) == "flfl"
        is_fedlg = str(cfg_eff.get("pseudo_label_policy", "")) == "fedlgmatch"
        is_fssl_uc = str(cfg_eff.get("pseudo_label_policy", "")) == "fssl_uc"
        exp_dir = cfg.get("exp_dir")
        checkpoint_root = _resolve_checkpoint_root(cfg)
        ckpt_enabled = self._federated_checkpoint_enabled(cfg)
        ckpt_every = max(1, int(cfg.get("federated_checkpoint_every", 500)))

        if start_round > 0:
            print(f"ℹ️ 从第 {start_round} 轮继续训练（共 {num_rounds} 轮）")
        self._session_train_start = time.perf_counter()

        for round_num in range(start_round, num_rounds):
            self._ensure_clients(data_loader.num_clients)
            class_thresholds = None
            g_round = int(round_offset + round_num)
            k_old = _frame1_num_old_classes(cfg_eff)
            tau_eff = float(cfg_eff.get("pseudo_threshold", self.pseudo_threshold))
            local_ep = int(cfg_eff.get("local_epochs_unlabeled", self.local_epochs_unlabeled))

            if supervision_mode == "full_true":
                for client_id in range(data_loader.num_clients):
                    data_loader.client_datasets[client_id].set_supervision_mode("full_true")
            else:
                sm = "masked_true" if supervision_mode == "masked_true" else "pseudo"
                pl_policy = str(cfg_eff.get("pseudo_label_policy", "default"))
                round_masked_wrong = 0
                round_runner_hit = 0
                if is_flfl:
                    for client_id in range(data_loader.num_clients):
                        client_dataset = data_loader.client_datasets[client_id]
                        tau_k, w_loc, ru_loc = flfl_prepare_round_pseudo(
                            self.global_model,
                            client_dataset,
                            self.device,
                            cfg_eff,
                        )
                        round_masked_wrong += w_loc
                        round_runner_hit += ru_loc
                        client_dataset.flfl_lsaa_tau = float(tau_k)
                        client_dataset.set_supervision_mode(sm)
                elif is_fedlg:
                    for client_id in range(data_loader.num_clients):
                        client_dataset = data_loader.client_datasets[client_id]
                        fc_pre = self.clients[client_id]
                        w_loc, ru_loc = fedlgmatch_prepare_round_pseudo(
                            self.global_model,
                            client_dataset,
                            self.device,
                            cfg_eff,
                            prev_local_state_dict=fc_pre.model_state_dict,
                            pseudo_threshold=tau_eff,
                        )
                        round_masked_wrong += w_loc
                        round_runner_hit += ru_loc
                        client_dataset.set_supervision_mode(sm)
                elif is_fssl_uc:
                    for client_id in range(data_loader.num_clients):
                        client_dataset = data_loader.client_datasets[client_id]
                        (
                            all_preds,
                            mask,
                            all_max_probs,
                            _,
                            __,
                            ___,
                            w_loc,
                            ru_loc,
                        ) = generate_pseudo_labels_fixmatch(
                            self.global_model,
                            client_dataset,
                            tau_eff,
                            self.device,
                            class_thresholds=class_thresholds,
                            pseudo_label_policy="default",
                            num_old_classes=None,
                            global_round=g_round,
                            config=cfg_eff,
                        )
                        round_masked_wrong += w_loc
                        round_runner_hit += ru_loc
                        client_dataset.set_pseudo_labels(all_preds, mask, confidence=all_max_probs)
                        client_dataset.set_supervision_mode(sm)
                else:
                    for client_id in range(data_loader.num_clients):
                        client_dataset = data_loader.client_datasets[client_id]
                        (
                            all_preds,
                            mask,
                            all_max_probs,
                            _,
                            __,
                            ___,
                            w_loc,
                            ru_loc,
                        ) = generate_pseudo_labels_fixmatch(
                            self.global_model,
                            client_dataset,
                            tau_eff,
                            self.device,
                            class_thresholds=class_thresholds,
                            pseudo_label_policy=pl_policy,
                            num_old_classes=k_old if pl_policy != "default" else None,
                            global_round=g_round,
                            config=cfg_eff,
                        )
                        round_masked_wrong += w_loc
                        round_runner_hit += ru_loc
                        client_dataset.set_pseudo_labels(all_preds, mask, confidence=all_max_probs)
                        client_dataset.set_supervision_mode(sm)

            seed = int(cfg_eff.get("seed", 42))
            channel_gains = None
            comp_delays = None
            if wireless_ch is not None:
                channel_gains = wireless_ch.calculate_channel_gains(round_offset + round_num, seed)
                ul_ids = list(
                    range(data_loader.num_clients)
                )
                comp_delays = wireless_ch.generate_computation_delays(
                    round_offset + round_num, ul_ids, seed
                )

            selected_clients = select_clients_by_experiment(
                experiment_type,
                round_offset + round_num,
                data_loader,
                clients_per_round=cfg_eff.get("clients_per_round", 5),
                config=cfg_eff,
                wireless_channel=wireless_ch,
                channel_gains=channel_gains,
                comp_delays=comp_delays,
                use_wireless_scheduling=use_wireless
                and base_exp not in ("NoClientTrain", "AllClientsTrain"),
            )

            self._ensure_clients(data_loader.num_clients)
            self._sync_server_from_global()
            self.distribute(selected_clients)

            local_models = []
            client_sizes = []

            round_train_pl_n = 0
            round_train_pl_sum_conf = 0.0
            round_train_pl_correct = 0
            round_n_true_old = 0
            round_sum_conf_true_old = 0.0
            round_n_correct_true_old = 0
            round_n_true_new = 0
            round_sum_conf_true_new = 0.0
            round_n_correct_true_new = 0
            round_n_pred_old = 0
            round_n_correct_if_pred_old = 0
            round_n_pred_new = 0
            round_n_correct_if_pred_new = 0
            round_unlabeled_mean_losses: List[float] = []

            round_pseudo_per_class = {}
            for client_id in range(data_loader.num_clients):
                client_dataset = data_loader.client_datasets[client_id]
                per_class = client_dataset.get_pseudo_per_class_counts(self.num_classes)
                round_pseudo_per_class[client_id] = per_class.tolist()

            pred_new_count_pc: Dict[int, int] = {
                cid: 0 for cid in range(data_loader.num_clients)
            }
            pred_old_count_pc: Dict[int, int] = {
                cid: 0 for cid in range(data_loader.num_clients)
            }
            p_sched = aggregate_pseudo_split_stats_for_clients(
                data_loader.client_datasets,
                selected_clients,
                k_old,
                self.num_classes,
            )

            lr_round = cosine_lr_for_global_round(
                g_round, num_rounds, self.lr_unlabeled, 0.0
            )

            for client_id in selected_clients:
                client_dataset = data_loader.client_datasets[client_id]

                client_loader = data_loader.get_client_loader(
                    client_id, include_pseudo=True, use_strong_aug=True
                )

                fc = self.clients[client_id]
                opt_in = fc.optimizer_state_dict
                if is_flfl:
                    (
                        local_model,
                        _,
                        _,
                        mean_u_loss,
                        opt_out,
                        train_pl_stats,
                    ) = local_training_flfl(
                        self.global_model,
                        client_id,
                        client_dataset,
                        local_epochs=local_ep,
                        lr=lr_round,
                        momentum=self.momentum,
                        device=self.device,
                        weight_decay=self.weight_decay,
                        config=cfg_eff,
                        global_round=g_round,
                        num_old_classes=k_old,
                        num_classes=self.num_classes,
                        optimizer_state=opt_in,
                        client_model_state_dict=fc.model_state_dict,
                    )
                elif is_fedlg:
                    (
                        local_model,
                        _,
                        _,
                        mean_u_loss,
                        opt_out,
                        train_pl_stats,
                    ) = local_training_fedlgmatch(
                        self.global_model,
                        client_id,
                        client_dataset,
                        local_epochs=local_ep,
                        lr=lr_round,
                        momentum=self.momentum,
                        device=self.device,
                        weight_decay=self.weight_decay,
                        config=cfg_eff,
                        global_round=g_round,
                        num_old_classes=k_old,
                        num_classes=self.num_classes,
                        optimizer_state=opt_in,
                        client_model_state_dict=fc.model_state_dict,
                    )
                elif is_fssl_uc:
                    (
                        local_model,
                        _,
                        _,
                        mean_u_loss,
                        opt_out,
                        train_pl_stats,
                    ) = local_training_fssl_uc(
                        self.global_model,
                        client_id,
                        client_dataset,
                        local_epochs=local_ep,
                        lr=lr_round,
                        momentum=self.momentum,
                        device=self.device,
                        weight_decay=self.weight_decay,
                        config=cfg_eff,
                        global_round=g_round,
                        num_old_classes=k_old,
                        num_classes=self.num_classes,
                        optimizer_state=opt_in,
                        client_model_state_dict=fc.model_state_dict,
                        pseudo_threshold=tau_eff,
                    )
                else:
                    (
                        local_model,
                        _,
                        _,
                        mean_u_loss,
                        opt_out,
                        train_pl_stats,
                    ) = local_training_fixmatch(
                        self.global_model, client_id, client_loader, client_dataset,
                        is_labeled=False,
                        local_epochs=local_ep,
                        lr=lr_round, momentum=self.momentum,
                        device=self.device,
                        weight_decay=self.weight_decay,
                        config=cfg_eff,
                        global_round=g_round,
                        pseudo_threshold=tau_eff,
                        pseudo_label_policy=str(cfg_eff.get("pseudo_label_policy", "default")),
                        num_old_classes=k_old,
                        num_classes=self.num_classes,
                        optimizer_state=opt_in,
                        client_model_state_dict=fc.model_state_dict,
                    )
                if opt_out is not None:
                    fc.optimizer_state_dict = copy.deepcopy(opt_out)

                ts = train_pl_stats or {}
                round_train_pl_n += int(ts.get("n", 0))
                round_train_pl_sum_conf += float(ts.get("sum_conf", 0.0))
                round_train_pl_correct += int(ts.get("n_correct", 0))
                round_n_true_old += int(ts.get("n_true_old", 0))
                round_sum_conf_true_old += float(ts.get("sum_conf_true_old", 0.0))
                round_n_correct_true_old += int(ts.get("n_correct_true_old", 0))
                round_n_true_new += int(ts.get("n_true_new", 0))
                round_sum_conf_true_new += float(ts.get("sum_conf_true_new", 0.0))
                round_n_correct_true_new += int(ts.get("n_correct_true_new", 0))
                round_n_pred_old += int(ts.get("n_pred_old", 0))
                round_n_correct_if_pred_old += int(ts.get("n_correct_if_pred_old", 0))
                round_n_pred_new += int(ts.get("n_pred_new", 0))
                round_n_correct_if_pred_new += int(ts.get("n_correct_if_pred_new", 0))
                pred_new_count_pc[client_id] = int(ts.get("n_pred_new", 0))
                pred_old_count_pc[client_id] = int(ts.get("n_pred_old", 0))

                round_unlabeled_mean_losses.append(float(mean_u_loss))

                if local_model is not None:
                    fc.model_state_dict = _clone_model_state_dict(local_model)
                    local_models.append(local_model)
                    # FedAvg 按客户端加权：曾为 full_true→len；FLFL→SAT 子集；FSSL-UC→len；FixMatch/FedLG 等→过阈 mask.sum()。
                    # 现统一为「该客户端无标签全集样本数」，与经典 FedAvg 按数据量加权一致。
                    # if supervision_mode == "full_true":
                    #     client_sizes.append(len(client_dataset))
                    # elif is_flfl:
                    #     sub = getattr(client_dataset, "flfl_subset_indices", None)
                    #     client_sizes.append(
                    #         int(len(sub)) if sub is not None else 0
                    #     )
                    # elif is_fssl_uc:
                    #     client_sizes.append(len(client_dataset))
                    # else:
                    #     mask = client_dataset.pseudo_mask
                    #     client_sizes.append(mask.sum() if mask is not None else 0)
                    client_sizes.append(len(client_dataset))

            if local_models:
                if is_flfl:
                    lsaa_weights = []
                    for cid in selected_clients:
                        tau = float(
                            getattr(
                                data_loader.client_datasets[cid],
                                "flfl_lsaa_tau",
                                0.0,
                            )
                        )
                        lsaa_weights.append(max(1e-8, 1.0 - tau))
                    agg_lr = cfg_eff.get("flfl_lsaa_lr")
                    if agg_lr is None:
                        agg_lr = lr_round
                    st_lsaa = flfl_lsaa_aggregate(
                        self.global_model,
                        local_models,
                        lsaa_weights,
                        lr=float(agg_lr),
                        momentum=self.momentum,
                        weight_decay=self.weight_decay,
                        optimizer_state=self.server.flfl_lsaa_optimizer_state,
                    )
                    self.server.flfl_lsaa_optimizer_state = copy.deepcopy(st_lsaa)
                else:
                    federated_averaging(self.global_model, local_models, client_sizes)

            local_models.clear()
            gc.collect()
            if bool(cfg_eff.get("cuda_empty_cache_each_round", True)) and torch.cuda.is_available():
                dev_cur = torch.device(self.device)
                if dev_cur.type == "cuda":
                    with torch.cuda.device(dev_cur):
                        torch.cuda.empty_cache()

            if (
                data_loader.server_loader is not None
                and self.finetune_epochs > 0
            ):
                opt_s = self.server.optimizer_state_dict
                opt_state_out = finetune_global_model(
                    self.global_model,
                    data_loader.server_loader,
                    finetune_epochs=self.finetune_epochs,
                    lr=lr_round,
                    momentum=self.momentum,
                    device=self.device,
                    weight_decay=self.weight_decay,
                    optimizer_state=opt_s,
                )
                self.server.optimizer_state_dict = copy.deepcopy(opt_state_out)

            self.server.model_state_dict = _clone_model_state_dict(self.global_model)

            for c in self.clients.values():
                c.active = False

            test_acc = evaluate_model(
                self.global_model, data_loader.test_loader, self.device,
                num_eval_classes=data_loader.num_classes,
            )
            test_loss = evaluate_model_loss(
                self.global_model, data_loader.test_loader, self.device,
                num_eval_classes=data_loader.num_classes,
            )

            self.metrics['test_accuracy'].append(test_acc)
            self.metrics['test_loss'].append(float(test_loss))

            # 旧/新类测试指标与无标签伪标签分层（新→旧误判等）：联邦 continual_run_stage=frame1 时记录
            k1_eval = int(cfg["frame1_initial_num_classes"])
            record_old_new = k1_eval > 0 and cfg.get("continual_run_stage") == "frame1"
            if record_old_new:
                clm = evaluate_model_continual_metrics(
                    self.global_model,
                    data_loader.test_loader,
                    self.device,
                    num_old_classes=k1_eval,
                )
                pss = aggregate_pseudo_split_stats(
                    data_loader.client_datasets,
                    data_loader.num_clients,
                    num_old_classes=k1_eval,
                    num_classes=data_loader.num_classes,
                )
                test_block = {k: v for k, v in clm.items() if k != "test_acc_all"}
                self.metrics['continual_eval'].append({
                    'k1': k1_eval,
                    'k2': self.num_classes,
                    'test': test_block,
                    'pseudo_unlabeled_agg': pss,
                })
            else:
                self.metrics['continual_eval'].append(None)
            self.metrics['pseudo_mean_confidence'].append(
                round_train_pl_sum_conf / round_train_pl_n if round_train_pl_n else 0.0
            )
            self.metrics['pseudo_selected'].append(int(round_train_pl_n))
            self.metrics['pseudo_train_conf_mean_old'].append(
                round_sum_conf_true_old / round_n_true_old if round_n_true_old else None
            )
            self.metrics['pseudo_train_conf_mean_new'].append(
                round_sum_conf_true_new / round_n_true_new if round_n_true_new else None
            )
            self.metrics['pseudo_train_precision_old'].append(
                round_n_correct_true_old / round_n_true_old if round_n_true_old else None
            )
            self.metrics['pseudo_train_precision_new'].append(
                round_n_correct_true_new / round_n_true_new if round_n_true_new else None
            )
            self.metrics['pseudo_train_precision_all'].append(
                round_train_pl_correct / round_train_pl_n if round_train_pl_n else None
            )
            self.metrics['pseudo_train_pred_old_accuracy'].append(
                round_n_correct_if_pred_old / round_n_pred_old if round_n_pred_old else None
            )
            self.metrics['pseudo_train_pred_new_accuracy'].append(
                round_n_correct_if_pred_new / round_n_pred_new if round_n_pred_new else None
            )
            self.metrics['pseudo_train_pred_old_count_total'].append(int(round_n_pred_old))
            self.metrics['pseudo_train_pred_new_count_total'].append(int(round_n_pred_new))
            self.metrics['pseudo_scheduled_misclass_new_to_old'].append(
                p_sched.get("pseudo_misclass_new_to_old_rate")
            )
            self.metrics['round'].append(round_offset + round_num)
            self.metrics['selected_clients'].append(selected_clients)
            self.metrics['pseudo_per_class_per_client'].append(round_pseudo_per_class)
            self.metrics['pseudo_train_pred_new_count_per_client'].append(pred_new_count_pc)
            self.metrics['pseudo_train_pred_old_count_per_client'].append(pred_old_count_pc)
            self.metrics['pseudo_precision_old_scheduled'].append(
                p_sched.get("pseudo_precision_old")
            )
            self.metrics['pseudo_precision_new_scheduled'].append(
                p_sched.get("pseudo_precision_new")
            )
            self.metrics['pseudo_precision_total_scheduled'].append(
                p_sched.get("pseudo_precision_total")
            )
            if supervision_mode == "full_true":
                self.metrics["pseudo_masked_wrong_total"].append(0)
                self.metrics["pseudo_masked_wrong_runner_hit_total"].append(0)
                self.metrics["pseudo_masked_wrong_runnerup_rate"].append(None)
            else:
                self.metrics["pseudo_masked_wrong_total"].append(int(round_masked_wrong))
                self.metrics["pseudo_masked_wrong_runner_hit_total"].append(
                    int(round_runner_hit)
                )
                rate_ru = (
                    float(round_runner_hit) / float(round_masked_wrong)
                    if round_masked_wrong > 0
                    else None
                )
                self.metrics["pseudo_masked_wrong_runnerup_rate"].append(rate_ru)

            if round_num % 10 == 0:
                if supervision_mode == "full_true":
                    sup = "监督=全样本真标签"
                elif supervision_mode == "masked_true":
                    sup = "监督=同mask真类别"
                else:
                    sup = "监督=伪类别"
                gr = round_offset + round_num
                unsup_mean = (
                    float(np.mean(round_unlabeled_mean_losses))
                    if round_unlabeled_mean_losses
                    else 0.0
                )
                _pta = self.metrics["pseudo_train_precision_all"][-1]
                _pta_s = f"{_pta:.3f}" if _pta is not None else "n/a"
                print(f"轮次 {gr:3d} | 测试准确率: {test_acc:.2f}% | {sup} | "
                      f"测试损失(CE): {test_loss:.4f} | "
                      f"无监督损失(均值): {unsup_mean:.4f} | "
                      f"伪标签置信度: {self.metrics['pseudo_mean_confidence'][-1]:.3f} | "
                      f"伪标签精度: {_pta_s} | "
                      f"选中: {selected_clients}")
                ce = self.metrics['continual_eval'][-1]
                if ce is not None:
                    t = ce['test']
                    p = ce['pseudo_unlabeled_agg']
                    def _fmt(x):
                        return f"{x:.3f}" if x is not None else "n/a"
                    print(
                        f"    [CL] 测试: "
                        f"Acc_old={_fmt(t.get('test_acc_old'))} "
                        f"Acc_new={_fmt(t.get('test_acc_new'))} "
                        f"新→旧误判={_fmt(t.get('misclass_new_to_old_rate'))} "
                        f"旧→新误判={_fmt(t.get('misclass_old_to_new_rate'))} | "
                        f"伪标签(无标签汇总): "
                        f"P_old={_fmt(p.get('pseudo_precision_old'))} "
                        f"P_new={_fmt(p.get('pseudo_precision_new'))} "
                        f"P_all={_fmt(p.get('pseudo_precision_total'))} "
                        f"新真→伪旧={_fmt(p.get('pseudo_misclass_new_to_old_rate'))}"
                    )
                for cid in selected_clients:
                    client_dataset = data_loader.client_datasets[cid]
                    total_samples = len(client_dataset)
                    if supervision_mode == "full_true":
                        print(f"    客户端{cid} | 全样本监督: {total_samples}")
                        continue
                    pc = np.array(round_pseudo_per_class.get(cid, [0] * self.num_classes))
                    total_selected = pc.sum()
                    pct_overall = 100.0 * total_selected / total_samples if total_samples > 0 else 0
                    pct_per_class = (100.0 * pc / total_selected) if total_selected > 0 else np.zeros(self.num_classes)
                    cls_str = ','.join(f'c{i}:{pct_per_class[i]:.1f}%' for i in range(self.num_classes))
                    print(f"    客户端{cid} | 通过阈值: {total_selected}/{total_samples} ({pct_overall:.1f}%) | "
                          f"按类别占比: [{cls_str}]")

            if ckpt_enabled and checkpoint_root:
                next_round = round_num + 1
                if next_round % ckpt_every == 0 or next_round >= num_rounds:
                    self._save_federated_checkpoint(
                        checkpoint_root, experiment_type, data_loader, next_round, num_rounds
                    )

    def _run_federated_frame1(
        self,
        experiment_type: str,
        dl2,
        base_exp: str,
        supervision_mode: str,
        k1: int,
        k2: int,
        n2: int,
    ) -> Dict:
        cfg = self.config or {}
        checkpoint_root = _resolve_checkpoint_root(cfg)
        start_round = 0

        base = cfg.get("frame1_pre_warmup_model_path")
        load_path = get_frame1_pre_k2_warmup_model_path(base, cfg, for_save=False)

        ckpt = None
        if checkpoint_root and self._federated_checkpoint_enabled(cfg) and cfg.get("federated_auto_resume", True):
            ckpt_path = _find_latest_checkpoint_path(checkpoint_root, experiment_type)
            if ckpt_path is not None:
                if not os.path.isfile(load_path):
                    raise RuntimeError(
                        f"续训需要 frame1_pre 模型路径存在（用于校验）: {load_path}"
                    )
                self.num_classes = k2
                self.global_model = build_model(self.model_name, num_classes=k2).to(self.device)
                ckpt = self._try_load_federated_checkpoint(checkpoint_root, experiment_type, dl2, k1, k2)
                if ckpt is not None:
                    start_round = int(ckpt["next_round"])

        if ckpt is None:
            self.metrics = self._init_metrics()
            self.warmup_metrics = {}
            if not os.path.isfile(load_path):
                raise RuntimeError(
                    f"联邦 frame1 需要已存在的 frame1_pre（K2 全类预热）模型: {load_path}\n"
                    "请先跑 continual_run_stage=frame1_pre。"
                )
            self.num_classes = k2
            self.global_model = build_model(self.model_name, num_classes=k2).to(self.device)
            print(f"\n📂 加载 frame1_pre（K2 全类预热）模型: {load_path}")
            load_warmup_model(
                self.global_model,
                load_path,
                current_config=cfg,
                strict_match=False,
                skip_config_check=True,
                device=self.device,
            )
            _set_continual_class_state(self.global_model, k1, k2)
            freeze_old = bool(cfg.get("frame1_freeze_old_logits", True))
            self.global_model.freeze_old_classes(freeze_old)
            if not freeze_old:
                print("ℹ️ frame1_freeze_old_logits=False：旧类 logits 可训练，以缓解骨干漂移与旧类头冻结的错配")

            self.server.model_state_dict = _clone_model_state_dict(self.global_model)
        else:
            self.warmup_metrics = {}

        if start_round >= n2:
            print(f"ℹ️ 检查点显示训练已完成（{start_round}/{n2} 轮），跳过训练循环")
            return self.metrics

        _mbits, _msize = calculate_model_size_bits(self.global_model)
        print(
            f"[模型 联邦 frame1] 可训练参数: {_msize['total_params']:,} | "
            f"体积: {_msize['total_kb']:.2f} KB ({_msize['total_mb']:.4f} MB) | "
            f"{_msize['total_bits']:,} bits | num_classes={k2}"
        )

        wireless_ch, use_wireless = self._setup_wireless_channel(dl2, base_exp)
        self._print_training_header(
            experiment_type, n2, supervision_mode, wireless_ch, use_wireless, base_exp,
            continual_tag=f" | 持续学习 联邦 frame1 (K2 全类, 类数={k2})",
        )

        self._run_training_rounds(
            dl2, experiment_type, n2, 0, wireless_ch,
            base_exp, supervision_mode, continual_run_stage_tag="frame1",
            start_round=start_round,
        )
        return self.metrics

    def run_experiment(self, experiment_type: str, data_loader) -> Dict:
        """
        联邦持续学习：continual_run_stage=frame1 时运行（K2 全类、partition=full_k 的数据加载器）。

        continual_run_stage:
          - frame1_pre / frame2_pre：仅预热（应在 main 中路由，不经本方法）
          - frame1：联邦训练；从 frame1_pre 检查点加载
          - frame2：CIFAR-100 预留（未实现）
        """
        print(f"\n{'='*70}")
        print(f"开始实验: {experiment_type}")
        print(f"{'='*70}")

        base_exp, supervision_mode = parse_client_supervision_suffix(experiment_type)
        cfg = self.config or {}

        _require_federated_training_stage(cfg)

        total_start = time.perf_counter()

        n2 = int(cfg.get('frame1_federated_rounds', 1000))
        k1 = int(cfg.get('frame1_initial_num_classes', 6))
        k2 = int(get_frame1_total_num_classes(cfg))

        print(
            f"ℹ️ continual_run_stage=frame1：联邦阶段，数据加载器 partition=full_k（K={k2} 全类），"
            f"旧/新分界 K1={k1}"
        )
        self._run_federated_frame1(
            experiment_type, data_loader, base_exp, supervision_mode, k1, k2, n2
        )
        session_elapsed = time.perf_counter() - total_start
        total_elapsed = session_elapsed + float(getattr(self, "_training_elapsed_before", 0.0))
        self.metrics['total_time_seconds'] = total_elapsed
        self.metrics['total_time_formatted'] = _format_duration(total_elapsed)
        print(f"\n⏱️ 总训练时间: {self.metrics['total_time_formatted']} "
              f"（本次会话 {_format_duration(session_elapsed)}）")
        return self.metrics
    