"""
================================================================================
模型定义 - 轻量级 CNN 与可选的 ResNet18
================================================================================
"""

import math
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


def _group_norm_num_groups(num_channels: int) -> int:
    """选取能整除 num_channels 的组数（优先不超过 32）。"""
    for g in (32, 16, 8, 4, 2, 1):
        if num_channels % g == 0:
            return g
    return 1


def _bn2d_to_gn(bn: nn.BatchNorm2d) -> nn.GroupNorm:
    c = bn.num_features
    g = _group_norm_num_groups(c)
    gn = nn.GroupNorm(num_groups=g, num_channels=c, affine=True)
    with torch.no_grad():
        if bn.weight is not None:
            gn.weight.copy_(bn.weight)
        if bn.bias is not None:
            gn.bias.copy_(bn.bias)
    return gn.to(device=next(bn.parameters()).device, dtype=next(bn.parameters()).dtype)


def replace_bn2d_with_groupnorm(module: nn.Module) -> None:
    """递归将子模块中的 BatchNorm2d 替换为 GroupNorm（用于 ResNet18CIFAR）。"""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, _bn2d_to_gn(child))
        else:
            replace_bn2d_with_groupnorm(child)


def _register_freeze_old_logits_hooks(linear: nn.Linear, num_old_classes: int) -> List[Any]:
    """对分类头中「旧类」对应的权重行与偏置分量屏蔽梯度，仅训练新类。返回 hook 句柄便于移除。"""

    def _hook_weight(grad: torch.Tensor) -> torch.Tensor:
        if grad is None:
            return None
        out = grad.clone()
        out[:num_old_classes] = 0
        return out

    def _hook_bias(grad: torch.Tensor) -> torch.Tensor:
        if grad is None:
            return None
        out = grad.clone()
        out[:num_old_classes] = 0
        return out

    handles: List[Any] = []
    handles.append(linear.weight.register_hook(_hook_weight))
    if linear.bias is not None:
        handles.append(linear.bias.register_hook(_hook_bias))
    return handles


def _ensure_old_logits_hook_list(model: nn.Module) -> List:
    if not hasattr(model, "_old_logits_hook_handles"):
        model._old_logits_hook_handles = []  # type: ignore[attr-defined]
    return model._old_logits_hook_handles  # type: ignore[return-value]


def _remove_old_logits_hooks(model: nn.Module) -> None:
    for h in _ensure_old_logits_hook_list(model):
        try:
            h.remove()
        except Exception:
            pass
    _ensure_old_logits_hook_list(model).clear()


def _set_continual_class_state(model: nn.Module, num_old_classes: int, new_num_classes: int) -> None:
    """扩类后：当前总类别数、本次扩类前的旧类数（用于 freeze_old_classes）。"""
    model.current_num_classes = new_num_classes  # type: ignore[attr-defined]
    model._last_expand_old_classes = num_old_classes  # type: ignore[attr-defined]


def _is_classifier_head_param_name(name: str) -> bool:
    """扩类时仅替换的最后一层线性头：CNN 为 fc2，ResNet 为 model.fc。"""
    if name.startswith("fc2.") or name == "fc2.weight" or name == "fc2.bias":
        return True
    if name.startswith("model.fc.") or name in ("model.fc.weight", "model.fc.bias"):
        return True
    return False


def freeze_backbone_except_classifier_head(model: nn.Module) -> None:
    """
    持续学习常用：冻结特征提取与 fc1（若有），仅保留最后一层分类头可训练。
    与 docs/continual_learning_experiment_design.md 中「冻结 backbone，仅初始化新类输出头」一致。
    """
    for name, p in model.named_parameters():
        p.requires_grad = _is_classifier_head_param_name(name)
    print("ℹ️ continual: 已冻结 backbone（除最后一层分类头外 requires_grad=False）")


def unfreeze_all_parameters(model: nn.Module) -> None:
    """恢复全部参数可训练。"""
    for p in model.parameters():
        p.requires_grad = True


class ContinualClassifierMixin:
    """
    持续学习内置接口（与常见 CNNCifar/ResNet 写法一致）：
    - ``current_num_classes``：当前分类头输出类别数
    - ``expand_classifier(new_num_classes, ...)``：扩类并拷贝旧权重、新行初始化
    - ``freeze_old_classes(freeze=True)``：仅屏蔽「扩类前旧类」对应 logits 行/偏置的梯度（非整层 requires_grad=False）
    - 冻结骨干：调用模块函数 ``freeze_backbone_except_classifier_head(model)``
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._old_logits_hook_handles = []  # type: ignore[misc]

    def _continual_last_linear(self) -> nn.Linear:
        if hasattr(self, "fc2") and isinstance(self.fc2, nn.Linear):
            return self.fc2
        if hasattr(self, "model") and hasattr(self.model, "fc") and isinstance(self.model.fc, nn.Linear):
            return self.model.fc
        raise AttributeError("当前模型无 fc2 / model.fc 线性分类头")

    def expand_classifier(
        self,
        new_num_classes: int,
        *,
        new_head_init: Optional[str] = None,
        freeze_old_logits: bool = False,
        freeze_backbone: bool = False,
    ) -> None:
        """扩展最后一层分类器；新类行 ResNet 默认 Xavier、CNN 默认 Kaiming（与 ``expand_classifier_num_classes`` 一致）。"""
        expand_classifier_num_classes(
            self,
            new_num_classes,
            new_head_init=new_head_init,
            freeze_old_logits=freeze_old_logits,
        )
        if freeze_backbone:
            freeze_backbone_except_classifier_head(self)

    def freeze_old_classes(self, freeze: bool = True) -> None:
        """
        对扩类前已存在的类别（``_last_expand_old_classes`` 行）屏蔽 logits 梯度；新类行仍可训练。
        ``freeze=False`` 时移除上述 hook。若需冻结整段骨干，请用 ``freeze_backbone_except_classifier_head(self)``。
        """
        n_old = getattr(self, "_last_expand_old_classes", None)
        if n_old is None or n_old <= 0:
            return
        _remove_old_logits_hooks(self)
        if not freeze:
            return
        lin = self._continual_last_linear()
        self._old_logits_hook_handles.extend(_register_freeze_old_logits_hooks(lin, n_old))
        print(f"ℹ️ freeze_old_classes: 已屏蔽前 {n_old} 类 logits 的梯度（新类行仍更新）")

    def unfreeze_all_logits_hooks(self) -> None:
        """移除旧类 logits 梯度屏蔽 hook。"""
        _remove_old_logits_hooks(self)


class CNN(ContinualClassifierMixin, nn.Module):
    """CNN：两层 32 通道卷积块 + 两层 64 通道卷积块 + FC120。"""

    def __init__(self, num_classes: int = 10):
        super(CNN, self).__init__()
        self.current_num_classes = num_classes
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(64 * 5 * 5, 120),
            nn.ReLU()
        )

        self.fc2 = nn.Linear(120, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x



class ResNet18CIFAR(ContinualClassifierMixin, nn.Module):
    """ResNet18 适配 CIFAR/SVHN 尺寸：3x32x32，移除初始 7x7+maxpool；骨干用 GroupNorm 替代 BatchNorm。"""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.current_num_classes = num_classes
        self.model = tv_models.resnet18(weights=None)
        # 替换首层卷积以适配小尺寸图像
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()
        replace_bn2d_with_groupnorm(self.model)
        # 替换分类头
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


def build_model(model_name: str, num_classes: int):
    """根据名称构建模型：'resnet18' → ResNet18CIFAR；其余 → CNN。"""
    name = (model_name or "cnn").lower()
    if name == "resnet18":
        return ResNet18CIFAR(num_classes=num_classes)
    return CNN(num_classes=num_classes)


def expand_classifier_num_classes(
    model: nn.Module,
    new_num_classes: int,
    *,
    new_head_init: Optional[str] = None,
    freeze_old_logits: bool = False,
) -> None:
    """
    将分类头从当前输出维度扩展到 new_num_classes（原地修改）。
    旧类权重/偏置从旧层拷贝；新类输出维度按 new_head_init 初始化。
    成功后会写入 ``current_num_classes``、``_last_expand_old_classes``（供 ``freeze_old_classes``）。

    new_head_init:
      - None：ResNet18CIFAR 用 xavier_uniform_，CNN 的 fc2 用 kaiming_uniform_（与旧行为一致）
      - 'xavier' / 'kaiming'：显式指定

    freeze_old_logits:
      - False（默认）：旧类与新类 logits 均可训练。
      - True：对旧类对应的权重行与偏置分量注册梯度 hook，反向传播中仅更新新类（特征骨干仍全程训练）。
    """
    _remove_old_logits_hooks(model)

    is_resnet = hasattr(model, "model") and hasattr(model.model, "fc")
    if new_head_init is None:
        new_head_init = "xavier" if is_resnet else "kaiming"
    else:
        new_head_init = new_head_init.lower()
        if new_head_init not in ("xavier", "kaiming"):
            raise ValueError(f"new_head_init 须为 'xavier' 或 'kaiming'，收到: {new_head_init!r}")

    def _expand_linear(layer: nn.Linear, new_out: int) -> Tuple[nn.Linear, int]:
        old_out = layer.out_features
        if new_out <= old_out:
            raise ValueError(
                f"expand_classifier_num_classes: new_out={new_out} 必须大于当前 out_features={old_out}"
            )
        device = layer.weight.device
        dtype = layer.weight.dtype
        new_layer = nn.Linear(layer.in_features, new_out, bias=layer.bias is not None).to(device=device, dtype=dtype)
        with torch.no_grad():
            new_layer.weight[:old_out] = layer.weight
            if layer.bias is not None:
                new_layer.bias[:old_out] = layer.bias
        if new_head_init == "xavier":
            nn.init.xavier_uniform_(new_layer.weight[old_out:])
        else:
            nn.init.kaiming_uniform_(new_layer.weight[old_out:], a=math.sqrt(5))
        if new_layer.bias is not None:
            nn.init.zeros_(new_layer.bias[old_out:])
        return new_layer, old_out

    if hasattr(model, "fc2") and isinstance(model.fc2, nn.Linear):
        new_layer, old_out = _expand_linear(model.fc2, new_num_classes)
        model.fc2 = new_layer
        _set_continual_class_state(model, old_out, new_num_classes)
        if freeze_old_logits:
            _ensure_old_logits_hook_list(model).extend(
                _register_freeze_old_logits_hooks(model.fc2, old_out)
            )
        return
    if hasattr(model, "model") and hasattr(model.model, "fc"):
        fc = model.model.fc
        if isinstance(fc, nn.Linear):
            new_layer, old_out = _expand_linear(fc, new_num_classes)
            model.model.fc = new_layer
            _set_continual_class_state(model, old_out, new_num_classes)
            if freeze_old_logits:
                _ensure_old_logits_hook_list(model).extend(
                    _register_freeze_old_logits_hooks(model.model.fc, old_out)
                )
            return
    raise ValueError(f"expand_classifier_num_classes 不支持该模型结构: {type(model)}")
    