"""
================================================================================
联邦半监督学习 - FixMatch + 智能客户端调度策略
================================================================================
"""

from ..config import DEFAULT_CONFIG, EXPERIMENT_TYPES, set_seed, get_device
from .models import CNN, ResNet18CIFAR
from .fixmatch import FederatedDataLoader, ClientDataset, FixMatchAugmentation
from ..scheduler import select_clients_by_experiment
from ..local_update import (
    generate_pseudo_labels_fixmatch,
    local_training_fixmatch,
)
from ..warmup import (
    warmup_training,
    save_warmup_model,
    load_warmup_model,
)
from .aggregation import federated_averaging, evaluate_model
from ..federated import FederatedLearning
from ..visualization import (
    plot_all_results,
    plot_single_experiment,
    plot_comparison,
    plot_summary_bar,
    plot_client_selection,
)
from ..main import run_all_experiments, run_single_experiment

__all__ = [
    'DEFAULT_CONFIG',
    'EXPERIMENT_TYPES',
    'set_seed',
    'get_device',
    'CNN',
    'ResNet18CIFAR',
    'FederatedDataLoader',
    'ClientDataset',
    'FixMatchAugmentation',
    'select_clients_by_experiment',
    'generate_pseudo_labels_fixmatch',
    'local_training_fixmatch',
    'warmup_training',
    'save_warmup_model',
    'load_warmup_model',
    'federated_averaging',
    'evaluate_model',
    'FederatedLearning',
    'plot_all_results',
    'plot_single_experiment',
    'plot_comparison',
    'plot_summary_bar',
    'plot_client_selection',
    'run_all_experiments',
    'run_single_experiment',
]
