"""
无线信道和带宽管理模块

信道增益、传输速率、基于 Lambert W 的最优带宽、计算延时（负指数随机项）等。
可与联邦实验配置字典对接：使用 ``wireless_args_from_config`` 生成与 argparse Namespace 兼容的参数对象。

示例::

    from modules.wireless_channel import WirelessChannel, wireless_args_from_config
    cfg = apply_dataset_overrides({**DEFAULT_CONFIG, **user_cfg})
    channel = WirelessChannel(wireless_args_from_config(cfg, seed=42), model=global_model)
    gains = channel.calculate_channel_gains(round_num=0, seed=42)
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from scipy import special


def calculate_model_size_bits(model: torch.nn.Module):
    """
    动态计算模型的实际大小（以 bits 为单位）。

    Args:
        model: PyTorch 模型

    Returns:
        total_bits: 模型总大小（bits）
        size_info: 详细的大小信息字典
    """
    total_params = 0
    total_bytes = 0
    param_details = {}

    for name, param in model.named_parameters():
        if param.requires_grad:
            param_count = param.numel()
            param_bytes = param_count * 4

            total_params += param_count
            total_bytes += param_bytes

            param_details[name] = {
                'shape': list(param.shape),
                'params': param_count,
                'bytes': param_bytes,
            }

    total_bits = total_bytes * 8

    size_info = {
        'total_params': total_params,
        'total_bytes': total_bytes,
        'total_bits': total_bits,
        'total_kb': total_bytes / 1024,
        'total_mb': total_bytes / (1024 * 1024),
        'param_details': param_details,
    }

    return total_bits, size_info


def wireless_args_from_config(
    config: Dict[str, Any],
    seed: int,
    num_users: Optional[int] = None,
) -> SimpleNamespace:
    """
    将联邦实验 ``config`` 转为 ``WirelessChannel`` 所需的 Namespace（字段名与 argparse 一致）。

    - ``num_users`` 默认取 ``config['num_clients']``
    - ``local_ep`` / ``local_bs`` 分别对应 ``local_epochs_unlabeled`` / ``batch_size``
    """
    nu = int(num_users if num_users is not None else config.get('num_clients', 20))
    return SimpleNamespace(
        total_bandwidth_mhz=float(config.get('total_bandwidth_mhz', 20.0)),
        tx_power_dbm=float(config.get('tx_power_dbm', 23.0)),
        upload_deadline=float(config.get('upload_deadline', 1.0)),
        cell_radius=float(config.get('cell_radius', 250.0)),
        num_users=nu,
        seed=int(seed),
        comp_delay_a_i=float(config.get('comp_delay_a_i', 0.5e-3)),
        comp_delay_mu_i=float(config.get('comp_delay_mu_i', 2.0e3)),
        local_ep=int(config.get('local_epochs_unlabeled', 5)),
        local_bs=int(config.get('batch_size', 64)),
    )


class WirelessChannel:
    """无线信道管理器：路径损耗、阴影衰落、香农速率、计算延时与最优带宽（Lambert W）。"""

    def __init__(self, args: Union[SimpleNamespace, Any], model: torch.nn.Module):
        """
        model: 必须传入，按 ``requires_grad`` 参数以 float32 计上传比特数（不再使用配置中的 KB 估算）。
        """
        self.total_bandwidth = args.total_bandwidth_mhz * 1e6
        self.tx_power_dbm = args.tx_power_dbm
        self.tx_power = 10 ** (self.tx_power_dbm / 10) * 1e-3
        self.total_deadline = args.upload_deadline
        self.cell_radius = args.cell_radius

        self.carrier_freq = 3.5e9
        self.noise_psd_dbmhz = -174

        self.model_size_bits, self.model_size_info = calculate_model_size_bits(model)

        self.noise_psd = 10 ** (self.noise_psd_dbmhz / 10) * 1e-3

        self.device_positions = self._generate_device_positions(args.num_users, args.seed)

        self.base_seed = args.seed

        np.random.seed(args.seed)
        self.device_comp_params = {}
        for i in range(args.num_users):
            self.device_comp_params[i] = {
                'a_i': args.comp_delay_a_i,
                'mu_i': args.comp_delay_mu_i,
                'tau': args.local_ep,
                'd_i': args.local_bs,
            }

        print("无线信道管理器初始化完成:")
        print(f"    总带宽: {self.total_bandwidth / 1e6:.1f} MHz (配置: {args.total_bandwidth_mhz})")
        print(f"    发射功率: {self.tx_power_dbm} dBm")
        print(f"    总延时约束: {self.total_deadline} s")
        print(f"    蜂窝半径: {self.cell_radius} m")
        # 上传比特数由真实模型统计，体积见联邦阶段 [模型] 打印
        print(f"    噪声功率谱密度: {self.noise_psd_dbmhz} dBm/Hz")
        print(
            f"    计算延时: a_i={args.comp_delay_a_i * 1e3:.2f} ms/sample, "
            f"mu_i={args.comp_delay_mu_i / 1e3:.1f} k sample/s"
        )
        print(f"    设备数量: {args.num_users}")
        print("    路径损耗: 128.1 + 37.6*log10(d) dB (d 单位 km); 阴影: 0 均值、8 dB 标准差正态 (dB)")

    def update_model_size(self, model: torch.nn.Module):
        """模型结构变化时更新比特数。"""
        old_size_kb = self.model_size_bits / (8 * 1024)
        self.model_size_bits, self.model_size_info = calculate_model_size_bits(model)
        new_size_kb = self.model_size_bits / (8 * 1024)
        print(f"模型大小更新: {old_size_kb:.1f} KB -> {new_size_kb:.1f} KB")

    def get_configuration_summary(self):
        """当前配置摘要（便于写入 JSON）。"""
        first = next(iter(self.device_comp_params.values()), None)
        return {
            'total_bandwidth_mhz': self.total_bandwidth / 1e6,
            'tx_power_dbm': self.tx_power_dbm,
            'upload_deadline_s': self.total_deadline,
            'cell_radius_m': self.cell_radius,
            'model_size_kb': self.model_size_bits / (8 * 1024),
            'num_devices': len(self.device_positions),
            'comp_delay_a_i_ms': first['a_i'] * 1e3 if first else None,
            'comp_delay_mu_i_ksps': first['mu_i'] / 1e3 if first else None,
            'path_loss_model': '128.1 + 37.6*log10(d) dB',
            'shadow_fading_model': 'sigma=8 dB (normal in dB domain)',
        }

    def _generate_device_positions(self, num_users, seed):
        np.random.seed(seed)
        positions = []
        for _ in range(num_users):
            r = self.cell_radius * np.sqrt(np.random.random())
            theta = 2 * np.pi * np.random.random()
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            positions.append((x, y))
        return positions

    def _calculate_distance(self, device_id):
        x, y = self.device_positions[device_id]
        return float(np.sqrt(x ** 2 + y ** 2))

    def _calculate_path_loss_db(self, d_distance):
        d_km = d_distance / 1000.0
        if d_km <= 0:
            d_km = 0.001
        return 128.1 + 37.6 * np.log10(d_km)

    def _generate_shadow_fading_db(self):
        return float(np.random.normal(0, 8.0))

    def generate_computation_delays(self, round_num, device_ids, seed):
        """
        计算延时: t_comp = a_i * tau * d_i + Exp( rate = mu_i/(tau*d_i) )。
        """
        np.random.seed(seed + round_num * 10000)
        comp_delays = {}

        for device_id in device_ids:
            if device_id in self.device_comp_params:
                params = self.device_comp_params[device_id]
                a_i = params['a_i']
                mu_i = params['mu_i']
                tau = params['tau']
                d_i = params['d_i']

                min_delay = a_i * tau * d_i
                exponential_rate = mu_i / (tau * d_i)
                additional_delay = np.random.exponential(scale=1.0 / exponential_rate)
                comp_delays[device_id] = float(min_delay + additional_delay)
            else:
                comp_delays[device_id] = float(np.random.uniform(0.2, 1.0))

        return comp_delays

    def calculate_channel_gains(self, round_num, seed):
        np.random.seed(seed + round_num * 1000)
        channel_gains = {}

        for device_id in range(len(self.device_positions)):
            d = self._calculate_distance(device_id)
            path_loss_db = self._calculate_path_loss_db(d)
            shadow_fading_db = self._generate_shadow_fading_db()
            total_loss_db = path_loss_db + shadow_fading_db
            channel_gain = 10 ** (-total_loss_db / 10)
            channel_gains[device_id] = channel_gain

        return channel_gains

    def calculate_transmission_rate(self, device_id, allocated_bandwidth, channel_gain):
        if allocated_bandwidth <= 0:
            return 0.0
        snr = self.tx_power * channel_gain / (allocated_bandwidth * self.noise_psd)
        rate = allocated_bandwidth * np.log2(1 + snr)
        return float(rate)

    def calculate_transmission_time(self, model_size_bits, transmission_rate):
        if transmission_rate <= 0:
            return float('inf')
        return float(model_size_bits / transmission_rate)

    def calculate_required_bandwidth(self, device_id, channel_gain, computation_delay):
        """
        在给定计算延时与信道下，使传输在 deadline 内完成所需的最小带宽（解析式，含 Lambert W_{-1}）。
        """
        if channel_gain <= 0:
            return float('inf')

        available_transmission_time = self.total_deadline - computation_delay
        if available_transmission_time <= 0:
            return float('inf')

        numerator_gamma = self.noise_psd * self.model_size_bits * np.log(2)
        denominator_gamma = available_transmission_time * self.tx_power * channel_gain
        if denominator_gamma <= 0:
            return float('inf')

        gamma_n = numerator_gamma / denominator_gamma  # Γ_n
        
        lambert_arg = -gamma_n * np.exp(-gamma_n)
        if lambert_arg < -1 / np.e or lambert_arg >= 0:
            return float('inf')

        lambert_w = special.lambertw(lambert_arg, k=-1)
        if not np.isfinite(lambert_w) or np.isnan(lambert_w):
            return float('inf')

        denominator_lambert = lambert_w + gamma_n
        real_den = float(np.real(denominator_lambert))
        if real_den == 0 or not np.isfinite(real_den):
            return float('inf')

        numerator_bandwidth = -self.model_size_bits * np.log(2)
        optimal_bandwidth = float(numerator_bandwidth / (available_transmission_time * real_den))
        optimal_bandwidth = abs(optimal_bandwidth)

        if optimal_bandwidth <= 0 or not np.isfinite(optimal_bandwidth):
            return float('inf')
        if optimal_bandwidth > 10 * self.total_bandwidth:
            print(
                f"警告: 计算带宽过大 {optimal_bandwidth / 1e6:.2f} MHz > "
                f"{10 * self.total_bandwidth / 1e6:.2f} MHz"
            )

        return optimal_bandwidth


def greedy_select_clients_by_bandwidth(
    ordered_client_ids: List[int],
    channel: "WirelessChannel",
    channel_gains: Dict[int, float],
    comp_delays: Dict[int, float],
) -> Tuple[List[int], float]:
    """
    按给定顺序做 best-effort：依次为客户端预留解析带宽 B*，累加直至剩余上行带宽不足以下一个客户端。

    信道无效或 B* 非有限（含 +∞）时**跳过该客户端**并继续尝试后续用户；仅当当前用户所需有限带宽
    仍大于剩余带宽时**停止**（与 FedTeddi 侧实现一致）。

    返回 (选中的客户端 ID 列表, 剩余带宽 Hz)。
    """
    remaining = float(channel.total_bandwidth)
    selected: List[int] = []
    for cid in ordered_client_ids:
        g = float(channel_gains.get(cid, 0.0))
        if g <= 0 or not np.isfinite(g):
            continue
        cd = float(comp_delays.get(cid, 0.0))
        req = channel.calculate_required_bandwidth(cid, g, cd)
        if req == float('inf') or not np.isfinite(req) or req <= 0:
            continue
        if req > remaining:
            break
        selected.append(int(cid))
        remaining -= req
    return selected, remaining


__all__ = [
    'calculate_model_size_bits',
    'wireless_args_from_config',
    'WirelessChannel',
    'greedy_select_clients_by_bandwidth',
]
