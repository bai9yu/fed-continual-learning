"""
================================================================================
数据处理和分配 - 数据增强、数据集类、联邦数据加载器
================================================================================
"""

import torch
import os
from typing import Dict, Optional, Mapping
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random


def _to_numpy(x):
    """torch.Tensor 或 ndarray → ndarray（伪标签 mask 可能来自 numpy 比较或 torch）。"""
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# 归一化统计信息
DATASET_STATS = {
    'cifar10': {
        'mean': (0.4914, 0.4822, 0.4465),
        'std': (0.2023, 0.1994, 0.2010),
    },
    'cifar100': {
        'mean': (0.5071, 0.4867, 0.4408),
        'std': (0.2675, 0.2565, 0.2761),
    },
    'svhn': {
        'mean': (0.4377, 0.4438, 0.4728),
        'std': (0.1980, 0.2010, 0.1970),
    },
}


def _server_labeled_class_counts(all_labels: np.ndarray, labeled_indices: list) -> Dict[int, int]:
    """根据 labeled_indices（指向 all_labels 的下标）统计服务器有标签样本的类别分布。"""
    if not labeled_indices:
        return {}
    srv = all_labels[np.asarray(labeled_indices, dtype=np.int64)]
    uniq, cnt = np.unique(srv, return_counts=True)
    return {int(u): int(c) for u, c in zip(uniq, cnt)}


def svhn_label_map(t):
    """SVHN标签映射：官方标签10表示数字0"""
    return int(t) % 10
# ================================================================================
# RandAugment 数据增强
# ================================================================================
def _rand_rotate(img, m):
    return transforms.functional.rotate(img, angle=m * 3)


def _rand_translate_x(img, m):
    return transforms.functional.affine(img, angle=0, translate=(m / 30, 0), scale=1, shear=0)


def _rand_translate_y(img, m):
    return transforms.functional.affine(img, angle=0, translate=(0, m / 30), scale=1, shear=0)


def _rand_brightness(img, m):
    return transforms.functional.adjust_brightness(img, 1 + m / 10)


def _rand_contrast(img, m):
    return transforms.functional.adjust_contrast(img, 1 + m / 10)


def _rand_saturation(img, m):
    return transforms.functional.adjust_saturation(img, 1 + m / 10)


def _rand_shear(img, m):
    return transforms.functional.affine(img, angle=0, translate=(0, 0), scale=1, shear=m)


class RandAugment:
    """FixMatch中使用的强增强（使用可pickle的函数，兼容Windows多进程）"""

    def __init__(self, n=2, m=10):
        self.n = n
        self.m = m

        self.augment_pool = [
            _rand_rotate,
            _rand_translate_x,
            _rand_translate_y,
            _rand_brightness,
            _rand_contrast,
            _rand_saturation,
            _rand_shear,
        ]

    def __call__(self, img):
        ops = random.choices(self.augment_pool, k=self.n)
        for op in ops:
            img = op(img, self.m)
        return img


# ================================================================================
# FixMatch 数据增强类
# ================================================================================
class FixMatchAugmentation:
    """实现FixMatch的弱增强和强增强"""
    
    def __init__(self, normalize_mean, normalize_std, dataset_name: str = 'cifar10'):
        dataset_key = dataset_name.lower()

        if dataset_key == 'svhn':
            self.weak_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            ])
        else:
            self.weak_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            ])
        
        self.strong_transform_pil = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            RandAugment(n=2, m=10),
        ])
        
        # 仅在CIFAR系列上保留随机水平翻转，SVHN不翻转以避免数字混淆
        if dataset_key != 'svhn':
            self.strong_transform_pil.transforms.insert(0, transforms.RandomHorizontalFlip(p=0.5))

        self.strong_transform_tensor = transforms.Compose([
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
        ])
        
        self.normalize = transforms.Normalize(mean=normalize_mean, std=normalize_std)
    
    def weak_augment(self, image):
        """弱增强：随机翻转 + 随机裁剪"""
        if isinstance(image, np.ndarray):
            if image.max() > 1.0:
                image = image / 255.0
            if len(image.shape) == 3 and image.shape[0] != 3:
                image = np.transpose(image, (2, 0, 1))
            image = torch.from_numpy(image).float()
        
        image = transforms.ToPILImage()(image)
        image = self.weak_transform(image)
        image = transforms.ToTensor()(image)
        image = self.normalize(image)
        return image
    
    def strong_augment(self, image):
        """强增强：弱增强 + RandAugment + RandomErasing"""
        if isinstance(image, np.ndarray):
            if image.max() > 1.0:
                image = image / 255.0
            if len(image.shape) == 3 and image.shape[0] != 3:
                image = np.transpose(image, (2, 0, 1))
            image = torch.from_numpy(image).float()
        
        image = transforms.ToPILImage()(image)
        image = self.strong_transform_pil(image)
        image = transforms.ToTensor()(image)
        image = self.strong_transform_tensor(image)
        image = self.normalize(image)
        return image


# ================================================================================
# 客户端数据集类
# ================================================================================
class ClientDataset(Dataset):
    """客户端数据集 - 支持伪标签统计"""
    
    def __init__(self, base_dataset, indices, is_labeled=True,
                 normalize_mean=None, normalize_std=None, dataset_name: str = 'cifar10'):
        self.base_dataset = base_dataset
        self.indices = indices
        self.is_labeled = is_labeled
        
        # 存储真实标签
        self.labels = torch.tensor([base_dataset[i][1] for i in indices], dtype=torch.long)
        
        # 伪标签相关
        self.pseudo_labels = None
        self.pseudo_mask = None
        self.pseudo_confidence = None
        # sticky_new 策略：样本曾被高置信判为新类后，不再赋予旧类伪标签（与 len 一致）
        self.ever_predicted_new = np.zeros(len(self), dtype=bool)
        # 无标签客户端本地监督：pseudo | masked_true（同 mask、真类别）| full_true（全样本、真类别）
        self.supervision_mode = "pseudo"
        
        # 归一化统计默认使用CIFAR-10，除非外部提供
        if normalize_mean is None or normalize_std is None:
            stats = DATASET_STATS['cifar10']
            normalize_mean = stats['mean']
            normalize_std = stats['std']

        # 增强器
        self.augmentor = FixMatchAugmentation(normalize_mean, normalize_std, dataset_name)
        self.use_strong_aug = False
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        image, _ = self.base_dataset[real_idx]
        
        if self.is_labeled:
            image = self.augmentor.weak_augment(image)
            label = self.labels[idx]

            return image, label, idx

        weak_image = self.augmentor.weak_augment(image)

        if not self.use_strong_aug:
            return weak_image, self.labels[idx], idx

        strong_image = self.augmentor.strong_augment(image)
        if self.supervision_mode == "full_true":
            return (
                weak_image,
                strong_image,
                int(self.labels[idx].item()),
                True,
                idx,
            )

        selected = bool(self.pseudo_mask[idx]) if self.pseudo_mask is not None else False
        if self.supervision_mode == "masked_true":
            target = int(self.labels[idx].item()) if selected else -1
            return weak_image, strong_image, target, selected, idx

        pseudo_label = self.pseudo_labels[idx] if self.pseudo_labels is not None and selected else -1

        return weak_image, strong_image, pseudo_label, selected, idx
    
    def set_pseudo_labels(self, pseudo_labels, mask, confidence=None):
        """设置伪标签和置信度"""
        self.pseudo_labels = pseudo_labels
        self.pseudo_mask = mask
        self.pseudo_confidence = confidence
    
    def set_supervision_mode(self, mode: str):
        """无标签客户端监督模式：pseudo | masked_true | full_true"""
        if mode not in ("pseudo", "masked_true", "full_true"):
            raise ValueError(f"supervision_mode 必须是 pseudo/masked_true/full_true，收到: {mode}")
        self.supervision_mode = mode
    
    def set_strong_augmentation(self, use_strong):
        """设置是否使用强增强"""
        self.use_strong_aug = use_strong
    
    def get_class_distribution(self):
        """获取类别分布（用于调度）"""
        labels = self.labels.numpy()
        unique, counts = np.unique(labels, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))
    
    def get_pseudo_stats(self):
        """获取伪标签统计信息（用于调度）"""
        if self.pseudo_mask is None:
            return 0, 0, 0
        
        num_selected = self.pseudo_mask.sum()
        
        # 计算平均置信度
        if self.pseudo_confidence is not None and num_selected > 0:
            mean_confidence = self.pseudo_confidence[self.pseudo_mask].mean()
        else:
            mean_confidence = 0
        
        # 计算伪标签精度
        true_labels = self.labels.numpy()
        if num_selected > 0:
            correct = (self.pseudo_labels[self.pseudo_mask] == true_labels[self.pseudo_mask]).sum()
            precision = correct / num_selected
        else:
            precision = 0
        
        return num_selected, mean_confidence, precision
    
    def get_pseudo_per_class_counts(self, num_classes: int):
        """获取通过置信度阈值的样本数量（按伪标签类别统计）"""
        if self.pseudo_mask is None or not self.pseudo_mask.any():
            return np.zeros(num_classes, dtype=np.int64)
        selected_preds = np.asarray(self.pseudo_labels[self.pseudo_mask])
        return np.bincount(selected_preds, minlength=num_classes)

    def get_pseudo_stats_split(self, num_old_classes: int, num_classes: int) -> dict:
        """
        按真实标签区分旧类 / 新类，仅统计通过伪标签阈值的样本：
        - precision_old / precision_new：该子集上伪标签与真标签一致的比例
        - n_true_new_selected 与 n_pred_old_among_true_new：真为新类且被选中时，伪预测落在旧类区间的数量（用于新→旧误判）
        """
        k1 = int(num_old_classes)
        k = int(num_classes)
        empty = {
            "n_selected_old": 0,
            "n_selected_new": 0,
            "precision_old": 0.0,
            "precision_new": 0.0,
            "n_correct_old": 0,
            "n_correct_new": 0,
            "n_true_new_selected": 0,
            "n_pred_old_among_true_new": 0,
        }
        if self.pseudo_mask is None or not self.pseudo_mask.any():
            return empty
        true_labels = self.labels.numpy()
        pl = np.asarray(self.pseudo_labels)
        mask = _to_numpy(self.pseudo_mask)
        y = true_labels[mask]
        pred = pl[mask]
        correct = pred == y

        is_old_true = (y < k1) & (y < k)
        is_new_true = (y >= k1) & (y < k)

        n_old = int(is_old_true.sum())
        n_new = int(is_new_true.sum())
        out = {
            "n_selected_old": n_old,
            "n_selected_new": n_new,
            "precision_old": float(correct[is_old_true].sum() / n_old) if n_old else 0.0,
            "precision_new": float(correct[is_new_true].sum() / n_new) if n_new else 0.0,
            "n_correct_old": int(correct[is_old_true].sum()),
            "n_correct_new": int(correct[is_new_true].sum()),
            "n_true_new_selected": n_new,
            "n_pred_old_among_true_new": int((pred[is_new_true] < k1).sum()) if n_new else 0,
        }
        return out

    def get_pseudo_pred_as_new_ratio(self, num_old_classes: int, num_classes: int) -> float:
        """
        在通过阈值的样本中，伪标签预测为「新类」(k1..K-1) 的占比；无选中样本时为 nan。
        """
        k1 = int(num_old_classes)
        k = int(num_classes)
        if self.pseudo_mask is None or not self.pseudo_mask.any():
            return float("nan")
        pl = np.asarray(self.pseudo_labels)
        mask = _to_numpy(self.pseudo_mask)
        pred = pl[mask]
        if pred.size == 0:
            return float("nan")
        n_new = int(((pred >= k1) & (pred < k)).sum())
        return float(n_new) / float(pred.size)


def aggregate_pseudo_split_stats(
    client_datasets: list,
    num_clients: int,
    num_old_classes: int,
    num_classes: int,
) -> dict:
    """
    汇总所有无标签客户端上 get_pseudo_stats_split，得到全局伪标签旧/新精度与误判计数。
    """
    k1 = int(num_old_classes)
    k = int(num_classes)
    tot = {
        "n_selected_old": 0,
        "n_selected_new": 0,
        "n_correct_old": 0,
        "n_correct_new": 0,
        "n_true_new_selected": 0,
        "n_pred_old_among_true_new": 0,
    }
    for cid in range(num_clients):
        st = client_datasets[cid].get_pseudo_stats_split(k1, k)
        tot["n_selected_old"] += st["n_selected_old"]
        tot["n_selected_new"] += st["n_selected_new"]
        tot["n_correct_old"] += st["n_correct_old"]
        tot["n_correct_new"] += st["n_correct_new"]
        tot["n_true_new_selected"] += st["n_true_new_selected"]
        tot["n_pred_old_among_true_new"] += st["n_pred_old_among_true_new"]

    prec_old = (
        tot["n_correct_old"] / tot["n_selected_old"] if tot["n_selected_old"] else None
    )
    prec_new = (
        tot["n_correct_new"] / tot["n_selected_new"] if tot["n_selected_new"] else None
    )
    n_sel = tot["n_selected_old"] + tot["n_selected_new"]
    prec_total = (
        (tot["n_correct_old"] + tot["n_correct_new"]) / n_sel if n_sel else None
    )
    rate_new_to_old = (
        tot["n_pred_old_among_true_new"] / tot["n_true_new_selected"]
        if tot["n_true_new_selected"]
        else None
    )
    return {
        "pseudo_precision_old": float(prec_old) if prec_old is not None else None,
        "pseudo_precision_new": float(prec_new) if prec_new is not None else None,
        "pseudo_precision_total": float(prec_total) if prec_total is not None else None,
        "pseudo_n_selected_old": int(tot["n_selected_old"]),
        "pseudo_n_selected_new": int(tot["n_selected_new"]),
        "pseudo_misclass_new_to_old_rate": float(rate_new_to_old)
        if rate_new_to_old is not None
        else None,
    }


def pseudo_precision_total_from_unlabeled_agg(pss: Optional[dict]) -> Optional[float]:
    """
    无标签汇总块 pseudo_unlabeled_agg 上的总体伪标签精度（过阈旧+新合并）。
    新结果含 pseudo_precision_total；旧 JSON 仅有 old/new 分项与计数时，按加权还原。
    """
    if not pss:
        return None
    v = pss.get("pseudo_precision_total")
    if v is not None:
        return float(v)
    no = int(pss.get("pseudo_n_selected_old") or 0)
    nn = int(pss.get("pseudo_n_selected_new") or 0)
    if no + nn == 0:
        return None
    po, pn = pss.get("pseudo_precision_old"), pss.get("pseudo_precision_new")
    if no > 0 and po is None:
        return None
    if nn > 0 and pn is None:
        return None
    if no == 0:
        return float(pn) if pn is not None else None
    if nn == 0:
        return float(po) if po is not None else None
    return (float(po) * no + float(pn) * nn) / float(no + nn)


def aggregate_pseudo_split_stats_for_clients(
    client_datasets: list,
    client_ids: list,
    num_old_classes: int,
    num_classes: int,
) -> dict:
    """
    仅汇总 ``client_ids`` 中客户端的 get_pseudo_stats_split，用于「当轮被调度客户端」上的
    旧/新/总体伪标签准确率及过阈值样本总数。
    """
    k1 = int(num_old_classes)
    k = int(num_classes)
    tot = {
        "n_selected_old": 0,
        "n_selected_new": 0,
        "n_correct_old": 0,
        "n_correct_new": 0,
        "n_pred_old_among_true_new": 0,
        "n_true_new_selected": 0,
    }
    for cid in client_ids:
        st = client_datasets[cid].get_pseudo_stats_split(k1, k)
        tot["n_selected_old"] += st["n_selected_old"]
        tot["n_selected_new"] += st["n_selected_new"]
        tot["n_correct_old"] += st["n_correct_old"]
        tot["n_correct_new"] += st["n_correct_new"]
        tot["n_pred_old_among_true_new"] += st["n_pred_old_among_true_new"]
        tot["n_true_new_selected"] += st["n_true_new_selected"]

    n_sel = tot["n_selected_old"] + tot["n_selected_new"]
    prec_old = (
        tot["n_correct_old"] / tot["n_selected_old"] if tot["n_selected_old"] else None
    )
    prec_new = (
        tot["n_correct_new"] / tot["n_selected_new"] if tot["n_selected_new"] else None
    )
    prec_total = (tot["n_correct_old"] + tot["n_correct_new"]) / n_sel if n_sel else None
    rate_new_to_old = (
        tot["n_pred_old_among_true_new"] / tot["n_true_new_selected"]
        if tot["n_true_new_selected"]
        else None
    )
    return {
        "pseudo_precision_old": float(prec_old) if prec_old is not None else None,
        "pseudo_precision_new": float(prec_new) if prec_new is not None else None,
        "pseudo_precision_total": float(prec_total) if prec_total is not None else None,
        "pseudo_n_above_threshold": int(n_sel),
        "pseudo_misclass_new_to_old_rate": float(rate_new_to_old)
        if rate_new_to_old is not None
        else None,
    }


# ================================================================================
# 归一化子数据集
# ================================================================================
class NormalizedSubset(Dataset):
    """带归一化的子数据集"""
    
    def __init__(self, base_dataset, indices, transform):
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        data, label = self.base_dataset[real_idx]
        return self.transform(data), label


# ================================================================================
# 联邦数据加载器
# ================================================================================
class FederatedDataLoader:
    """管理联邦学习的数据分布"""
    
    def __init__(self, num_clients, num_classes: int = 10,
                 alpha: float = 0.5, batch_size: int = 64, seed: int = 42, dataset_name='cifar10',
                 labeled_per_class: int = None,
                 server_batch_size: int = 10,
                 num_workers: int = None, pin_memory: bool = None,
                 server_labeled_mode: str = 'balanced',
                 server_labeled_per_class: Optional[Mapping[int, int]] = None,
                 manual_allocation: Optional[Dict[int, Dict[int, int]]] = None,
                 server_only: bool = False):
        self.num_clients = num_clients
        self.num_classes = num_classes
        self.alpha = alpha
        self.batch_size = batch_size
        self.seed = seed
        self.dataset_name = dataset_name.lower()
        # 服务器有标签：balanced 时每类 labeled_per_class；imbalanced 时按 server_labeled_per_class[class_id]
        self.server_labeled_mode = (server_labeled_mode or 'balanced').lower()
        if self.server_labeled_mode not in ('balanced', 'imbalanced'):
            self.server_labeled_mode = 'balanced'
        self.server_labeled_per_class: Optional[Dict[int, int]] = None
        if self.server_labeled_mode == 'imbalanced':
            if not server_labeled_per_class:
                raise ValueError(
                    "server_labeled_mode='imbalanced' 时必须提供 server_labeled_per_class（非空），"
                    "且在本帧类别范围 0..K-1 内至少有一个正数；例如总表 {0:100,...,9:100}，"
                    "第一帧 K1=6 时自动取 0..5 的条目"
                )
            self.server_labeled_per_class = {}
            for k, v in server_labeled_per_class.items():
                kk, vv = int(k), int(v)
                if vv <= 0:
                    continue
                if kk < 0 or kk >= num_classes:
                    raise ValueError(
                        f"server_labeled_per_class 中类别 {kk} 超出本帧范围 0..{num_classes - 1}"
                    )
                self.server_labeled_per_class[kk] = vv
            if not self.server_labeled_per_class:
                raise ValueError("server_labeled_per_class 中各类数量需为正整数")
        self.normalize_mean = DATASET_STATS.get(self.dataset_name, DATASET_STATS['cifar10'])['mean']
        self.normalize_std = DATASET_STATS.get(self.dataset_name, DATASET_STATS['cifar10'])['std']
        # DataLoader settings：如果未指定则自动选择合理值
        cpu_count = os.cpu_count() or 1
        default_workers = max(2, min(12, cpu_count // 2))
        self.num_workers = num_workers if num_workers is not None else default_workers
        self.pin_memory = pin_memory if pin_memory is not None else torch.cuda.is_available()
        
        # 每类有标签样本数（默认值随数据集）
        default_labeled = 25
        if self.dataset_name == 'cifar10':
            default_labeled = 25
        elif self.dataset_name == 'svhn':
            default_labeled = 25
        elif self.dataset_name == 'cifar100':
            default_labeled = 25
        self.labeled_per_class = labeled_per_class if labeled_per_class is not None else default_labeled

        self.server_dataset = None
        self.server_loader = None
        self.server_batch_size = server_batch_size
        # 若指定，则按客户端手工分配各类样本数（非 Dirichlet），用于持续学习等场景
        self.manual_allocation = manual_allocation
        # True：仅构建服务器有标签集 + 测试集，不划分客户端（如 server_init_pre）
        self.server_only = bool(server_only)
        
        self._load_data()
        self._distribute_data()

    def _load_data(self):
        """根据配置加载数据集"""
        dataset_key = self.dataset_name.lower()

        if dataset_key == 'svhn':
            self.trainset = torchvision.datasets.SVHN(
                root='./data', split='train', download=True, transform=transforms.ToTensor(),
                target_transform=svhn_label_map
            )
            testset = torchvision.datasets.SVHN(
                root='./data', split='test', download=True, transform=transforms.ToTensor(),
                target_transform=svhn_label_map
            )
        elif dataset_key == 'cifar10':
            self.trainset = torchvision.datasets.CIFAR10(
                root='./data', train=True, download=True, transform=transforms.ToTensor()
            )
            
            testset = torchvision.datasets.CIFAR10(
                root='./data', train=False, download=True, transform=transforms.ToTensor()
            )
        elif dataset_key == 'cifar100':
            self.trainset = torchvision.datasets.CIFAR100(
                root='./data', train=True, download=True, transform=transforms.ToTensor()
            )
            
            testset = torchvision.datasets.CIFAR100(
                root='./data', train=False, download=True, transform=transforms.ToTensor()
            )
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
        
        # 筛选指定类别
        self.train_indices = [i for i, (_, label) in enumerate(self.trainset) if label < self.num_classes]
        test_indices = [i for i, (_, label) in enumerate(testset) if label < self.num_classes]
        
        # 测试集归一化
        test_transform = transforms.Normalize(self.normalize_mean, self.normalize_std)
        normalized_test = NormalizedSubset(testset, test_indices, test_transform)
        self.test_loader = DataLoader(normalized_test, batch_size=32, shuffle=False)
        k_hi = max(0, self.num_classes - 1)
        print(
            f"数据加载完成[{self.dataset_name}]: 本帧有效类别 0..{k_hi} "
            f"（训练池 / 测试评估 / 服务器有标签子集与此一致，与客户端本帧标签空间对齐）| "
            f"训练样本={len(self.train_indices)}, 测试样本={len(test_indices)}"
        )
    
    def _server_labeled_take_for_class(self, class_id: int, available: int) -> int:
        """每类从池中划给服务器的有标签样本数上限（再与 available 取 min）。"""
        if self.server_labeled_mode == 'imbalanced':
            want = int(self.server_labeled_per_class.get(class_id, 0))
            return min(want, available)
        return min(self.labeled_per_class, available)
        
    def _distribute_data_manual(self, manual_allocation: Dict[int, Dict[int, int]]):
        """按 manual_allocation[client_id][class_id] = count 分配样本（服务器先取有标签）。"""
        np.random.seed(self.seed)
        all_labels = np.array([self.trainset[i][1] for i in self.train_indices])
        class_indices = {c: np.where(all_labels == c)[0].tolist() for c in range(self.num_classes)}
        for c in range(self.num_classes):
            np.random.shuffle(class_indices[c])

        labeled_indices = []
        for class_id in range(self.num_classes):
            take = self._server_labeled_take_for_class(class_id, len(class_indices[class_id]))
            take = min(take, len(class_indices[class_id]))
            labeled_indices.extend(class_indices[class_id][:take])
            class_indices[class_id] = class_indices[class_id][take:]

        server_real_indices = [self.train_indices[idx] for idx in labeled_indices]
        server_class_dist = _server_labeled_class_counts(all_labels, labeled_indices)
        self.server_dataset = ClientDataset(
            self.trainset,
            server_real_indices,
            is_labeled=True,
            normalize_mean=self.normalize_mean,
            normalize_std=self.normalize_std,
            dataset_name=self.dataset_name,
        )
        self.server_loader = DataLoader(
            self.server_dataset,
            batch_size=self.server_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

        ptr = {c: 0 for c in range(self.num_classes)}

        print(f"\n{'='*60}")
        print("数据分配 - 手工 manual_allocation（非 Dirichlet）")
        k_hi = max(0, self.num_classes - 1)
        print(
            f"本帧类别空间 0..{k_hi}：服务器有标签仅从此范围抽样，"
            f"与客户端 manual_allocation 一致；全 K 类划分请用 partition=\"full_k\"。"
        )
        print(f"服务器有标签类别分布: {server_class_dist}")
        print(f"{'='*60}")

        srv_desc = (
            f"每类={self.labeled_per_class}"
            if self.server_labeled_mode == 'balanced'
            else f"按类指定={self.server_labeled_per_class}"
        )
        print(
            f"服务器端有标签样本: {len(server_real_indices)} "
            f"(模式={self.server_labeled_mode}, {srv_desc})"
        )

        if self.server_only:
            self.client_datasets = []
            self.num_clients = 0
            print(
                "server_only=True：已跳过客户端样本划分，仅保留服务器有标签集与 test_loader。"
            )
            print(f"{'='*60}\n")
            return

        client_indices_list = [[] for _ in range(self.num_clients)]

        for cid in range(self.num_clients):
            alloc = manual_allocation.get(cid, {})
            for cls_id, cnt in sorted(alloc.items()):
                if cnt <= 0:
                    continue
                if cls_id < 0 or cls_id >= self.num_classes:
                    raise ValueError(
                        f"manual_allocation 中类别 {cls_id} 超出当前 num_classes={self.num_classes}"
                    )
                start = ptr[cls_id]
                avail = len(class_indices[cls_id]) - start
                if cnt > avail:
                    raise ValueError(
                        f"客户端 {cid} 需要类 {cls_id} 样本 {cnt}，但池中仅剩 {avail}（"
                        f"请检查总样本量或 manual_allocation）"
                    )
                chunk = class_indices[cls_id][start:start + cnt]
                ptr[cls_id] = start + cnt
                client_indices_list[cid].extend(chunk)

            class_dist = {}
            for real_idx in client_indices_list[cid]:
                lab = int(all_labels[real_idx])
                class_dist[lab] = class_dist.get(lab, 0) + 1
            total = len(client_indices_list[cid])
            print(f"客户端 {cid:2d} (无标签): 总数={total:4d}, 类别分布={class_dist}")

        self.client_datasets = []
        for i in range(self.num_clients):
            real_indices = [self.train_indices[idx] for idx in client_indices_list[i]] if client_indices_list[i] else []
            dataset = ClientDataset(
                self.trainset,
                real_indices,
                False,
                normalize_mean=self.normalize_mean,
                normalize_std=self.normalize_std,
                dataset_name=self.dataset_name,
            )
            self.client_datasets.append(dataset)

        print(f"\n{'='*60}")
        print("分配统计（手工）")
        print(f"{'='*60}")
        total_samples = sum(len(ds) for ds in self.client_datasets)
        print(f"总分配样本: {total_samples}")
        print(f"每客户端平均: {total_samples / len(self.client_datasets):.1f}")
        print(f"{'='*60}\n")

    def _distribute_data(self):
        """使用Dirichlet分布分配Non-IID数据，或为手工指定 manual_allocation"""
        if self.manual_allocation is not None:
            self._distribute_data_manual(self.manual_allocation)
            return

        np.random.seed(self.seed)
        
        # 按类别组织数据
        all_labels = np.array([self.trainset[i][1] for i in self.train_indices])
        class_indices = {c: np.where(all_labels == c)[0].tolist() for c in range(self.num_classes)}
        
        # 打乱每个类别的数据
        for c in range(self.num_classes):
            np.random.shuffle(class_indices[c])

        # 为服务器端抽取有标签样本
        labeled_indices = []
        for class_id in range(self.num_classes):
            take = self._server_labeled_take_for_class(class_id, len(class_indices[class_id]))
            take = min(take, len(class_indices[class_id]))
            labeled_indices.extend(class_indices[class_id][:take])
            class_indices[class_id] = class_indices[class_id][take:]

        # 服务器端数据集（仅用于聚合后微调，不参与客户端选择）
        server_real_indices = [self.train_indices[idx] for idx in labeled_indices]
        server_class_dist = _server_labeled_class_counts(all_labels, labeled_indices)
        self.server_dataset = ClientDataset(
            self.trainset,
            server_real_indices,
            is_labeled=True,
            normalize_mean=self.normalize_mean,
            normalize_std=self.normalize_std,
            dataset_name=self.dataset_name,
        )
        self.server_loader = DataLoader(
            self.server_dataset,
            batch_size=self.server_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

        # 剩余数据用于无标签客户端分配
        unlabeled_clients = self.num_clients
        client_proportions = []
        for _ in range(unlabeled_clients):
            proportions = np.random.dirichlet([self.alpha] * self.num_classes)
            client_proportions.append(proportions)
        
        remaining_total = sum(len(v) for v in class_indices.values())
        samples_per_client = remaining_total // max(unlabeled_clients, 1)
        client_class_allocation = np.zeros((unlabeled_clients, self.num_classes), dtype=int)
        
        for class_id in range(self.num_classes):
            class_total = len(class_indices[class_id])
            demands = np.array([client_proportions[c][class_id] * samples_per_client 
                            for c in range(unlabeled_clients)])
            
            total_demand = demands.sum()
            if total_demand > class_total:
                demands = demands * (class_total / max(total_demand, 1))
            
            int_demands = demands.astype(int)
            remainder = class_total - int_demands.sum()
            if remainder > 0:
                sorted_clients = np.argsort(demands)[::-1]
                for i in range(min(remainder, unlabeled_clients)):
                    int_demands[sorted_clients[i]] += 1
            
            client_class_allocation[:, class_id] = int_demands
        
        # 分配数据到客户端（全部为无标签客户端）
        class_pointers = [0] * self.num_classes
        client_indices_list = [[] for _ in range(self.num_clients)]
        
        print(f"\n{'='*60}")
        srv_tag = f"服务器有标签={self.server_labeled_mode}"
        if self.server_labeled_mode == 'imbalanced':
            srv_tag += f" {self.server_labeled_per_class}"
        else:
            srv_tag += f" 每类={self.labeled_per_class}"
        print(f"数据分配 - Non-IID (Dirichlet α={self.alpha}) | 全局训练池各类均衡 | {srv_tag}")
        k_hi = max(0, self.num_classes - 1)
        print(
            f"本帧类别空间 0..{k_hi}：服务器与无标签客户端均仅使用此范围；"
            f"服务器有标签类别分布: {server_class_dist}"
        )
        print(f"{'='*60}")
        
        for idx, client_offset in enumerate(range(self.num_clients)):
            client_class_counts = []
            
            for class_id in range(self.num_classes):
                n_samples = client_class_allocation[idx, class_id]
                start_idx = class_pointers[class_id]
                end_idx = start_idx + n_samples
                
                selected = class_indices[class_id][start_idx:end_idx]
                client_indices_list[client_offset].extend(selected)
                class_pointers[class_id] = end_idx
                
                client_class_counts.append(len(selected))
            
            class_dist = dict(zip(range(self.num_classes), client_class_counts))
            total = len(client_indices_list[client_offset])
            print(f"客户端 {client_offset:2d} (无标签): 总数={total:4d}, 类别分布={class_dist}")

        srv_desc = (
            f"每类={self.labeled_per_class}"
            if self.server_labeled_mode == 'balanced'
            else f"按类指定={self.server_labeled_per_class}"
        )
        print(f"服务器端有标签样本: {len(server_real_indices)} (模式={self.server_labeled_mode}, {srv_desc})")
        
        # 创建客户端数据集
        self.client_datasets = []
        for i in range(self.num_clients):
            real_indices = [self.train_indices[idx] for idx in client_indices_list[i]] if client_indices_list[i] else []
            dataset = ClientDataset(
                self.trainset,
                real_indices,
                False,
                normalize_mean=self.normalize_mean,
                normalize_std=self.normalize_std,
                dataset_name=self.dataset_name,
            )
            self.client_datasets.append(dataset)
        
        # 打印统计信息
        print(f"\n{'='*60}")
        print("分配统计")
        print(f"{'='*60}")
        total_samples = sum(len(ds) for ds in self.client_datasets)
        print(f"总分配样本: {total_samples}")
        print(f"每客户端平均: {total_samples / len(self.client_datasets):.1f}")
        print(f"{'='*60}\n")
    
    def get_client_loader(self, client_id, include_pseudo=True, use_strong_aug=False):
        """获取指定客户端的数据加载器"""
        dataset = self.client_datasets[client_id]
        
        # 设置增强方式
        if not dataset.is_labeled:
            dataset.set_strong_augmentation(use_strong_aug)
        
        # 无标签客户端：通常需先完成伪标签生成；full_true 模式不依赖伪标签
        if not dataset.is_labeled and include_pseudo:
            if dataset.supervision_mode != "full_true":
                if dataset.pseudo_mask is None or dataset.pseudo_labels is None:
                    return None

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


def _manual_allocation_k1(cfg: dict) -> Dict[int, Dict[int, int]]:
    """按 frame1 总类数 K 从完整 MANUAL 表截取「类号 < K」的客户端划分（partition=k1 时使用）。"""
    from config import MANUAL_ALLOCATION_FRAME1, get_frame1_total_num_classes

    k1 = int(get_frame1_total_num_classes(cfg))
    full = cfg.get("manual_allocation_frame1") or MANUAL_ALLOCATION_FRAME1
    out: Dict[int, Dict[int, int]] = {}
    for cid, alloc in full.items():
        out[cid] = {c: n for c, n in alloc.items() if int(c) < k1}
    return out


def _resolve_server_labeled_per_class(cfg: dict, num_classes: int) -> Optional[Dict[int, int]]:
    """
    仅从 server_labeled_per_class 中取本帧使用的条目：标签 k 满足 0 <= k < num_classes。
    可配置一张「总表」（含所有可能出现的类）；按本帧 num_classes 截取 0..K-1。
    """
    raw = cfg.get('server_labeled_per_class')
    if not raw:
        return None
    out: Dict[int, int] = {}
    for k, v in raw.items():
        kk, vv = int(k), int(v)
        if vv <= 0:
            continue
        if kk < 0 or kk >= num_classes:
            continue
        out[kk] = vv
    return out if out else None


def build_federated_loader_from_config(
    cfg: dict,
    partition: str,
    seed: int,
    *,
    server_only: bool = False,
) -> FederatedDataLoader:
    """
    按配置构建 FederatedDataLoader（命名与 continual_run_stage 一致，不用「数据 frame1/frame2」混指）。

    partition:
      ``k1`` — 类别空间为 frame1 总类数 K（get_frame1_total_num_classes）；MANUAL 表按类号 < K 截取；
      ``full_k`` — 当前联邦阶段全类（同 frame1 总类数），manual_allocation_frame1 全表
      （联邦 continual_run_stage=frame1、frame1_pre 预热）。

    server_only：仅构建服务器有标签集与测试集，不划分客户端（server_init_pre）。

    本加载器 ``num_classes`` 即当前划分类别数 K。
    """
    common = dict(
        num_clients=0 if server_only else cfg['num_clients'],
        alpha=cfg['alpha'],
        batch_size=cfg['batch_size'],
        seed=seed,
        dataset_name=cfg.get('dataset_name', 'cifar10'),
        labeled_per_class=cfg.get('labeled_per_class'),
        server_batch_size=cfg.get('server_batch_size', 10),
        num_workers=cfg.get('num_workers'),
        server_only=server_only,
    )
    from config import get_frame1_total_num_classes

    f1k = int(get_frame1_total_num_classes(cfg))
    if partition == "k1":
        common['num_classes'] = f1k
        common['manual_allocation'] = {} if server_only else _manual_allocation_k1(cfg)
    elif partition == "full_k":
        common['num_classes'] = f1k
        common['manual_allocation'] = cfg.get('manual_allocation_frame1')
    else:
        raise ValueError(
            f'build_federated_loader_from_config: partition 须为 "k1" 或 "full_k"；收到: {partition!r}'
        )
    resolved = _resolve_server_labeled_per_class(cfg, common['num_classes'])
    common['server_labeled_per_class'] = resolved
    common['server_labeled_mode'] = 'imbalanced' if resolved else 'balanced'
    return FederatedDataLoader(**common)
