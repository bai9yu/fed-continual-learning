# 联邦持续半监督学习（FixMatch + 客户端调度）

> Public portfolio version. This repository keeps the runnable source code,
> dependency notes, selected result figures, and the s42 experiment outputs.
> Large datasets, checkpoints, virtual environments, and local paper/reference
> files are intentionally excluded from Git history.

基于 FixMatch 的联邦半监督实验框架，支持多种客户端调度策略与持续学习式阶段（预热 → 联邦）。

## 项目亮点

- 面向联邦半监督学习场景，整合 FixMatch、本地伪标签学习、FedAvg 聚合与多种客户端调度策略。
- 支持持续学习式实验流程：服务器预热、全类预热、联邦主实验与结果重绘。
- 输出 JSON 指标和对比图，便于分析不同调度策略在准确率、收敛轮次和客户端参与量上的表现。

## 主要文件

| 文件 | 说明 |
|------|------|
| `main.py` | 入口：单实验 / 多实验、`apply_dataset_overrides`、实验目录与 JSON |
| `federated.py` | `FederatedLearning`：训练轮次、调度、聚合、持续学习联邦 |
| `config.py` | `DEFAULT_CONFIG`、手工数据分配表、持续学习阶段默认值 |
| `warmup.py` | `server_init_pre` / `frame1_pre` 预热与检查点路径（文件名含超参，见 `_warmup_param_tag`） |
| `local_update.py` | 客户端 FixMatch 与伪标签 |
| `modules/fixmatch.py` | `FederatedDataLoader`、数据增强与按配置建 loader |
| `modules/models.py` | `cnn` / `resnet18` 与持续学习分类头 |
| `modules/aggregation.py` | FedAvg 与评估 |
| `scheduler.py` | 客户端选择与调度 |
| `visualization.py` / `plot_from_json.py` | 结果绘图 |
| `results/` | 精简后的代表性结果图，供仓库首页和作品集引用 |
| `runs/exp_r8000_ud0.9_s42/` | `user_drop=0.9`、seed 42 的完整实验结果 |
| `runs/exp_r8000_ud1.0_s42/` | `user_drop=1.0`、seed 42 的完整实验结果 |

## 代表性结果

| 图表 | 说明 |
|------|------|
| `results/comparison_frame1_r8000_t0.8.png` | 多策略整体性能对比 |
| `results/comparison_rounds_to_target_frame1_r8000_t0.8.png` | 达到目标精度所需轮次对比 |

## 依赖

```bash
python -m pip install -r requirements.txt
```

## 快速开始

1. **预热**（按 `continual_run_stage` 二选一或都跑）  
   - 默认 `python warmup.py` → `server_init_pre`（仅服务器、K1 类）  
   - `python warmup.py frame1_pre` → `frame1_pre`（**K2**=frame1 总类数；需已存在 `server_init_pre_warmup_model_path` 解析后的检查点）

2. **联邦主实验**（需将 `config.py` 中 `continual_run_stage` 设为 `frame1`，且已存在 `frame1_pre` 生成的权重路径解析结果）

```bash
python main.py
```

3. **编程调用**

```python
from main import run_single_experiment, run_all_experiments, DEFAULT_CONFIG

metrics, cfg, json_path, exp_dir, plot_files = run_single_experiment("Random")

cfg = {**DEFAULT_CONFIG, "dataset_name": "svhn", "continual_run_stage": "frame1"}
all_results, cfg, json_files, exp_dir, plot_files = run_all_experiments(config=cfg, num_runs=1)
```

## 配置要点（`DEFAULT_CONFIG`）

- **`continual_run_stage`**（必填，由 `normalize_continual_settings` 校验）：  
  `server_init_pre` | `frame1_pre` | `frame1` | `frame2_pre` | `frame2`（后两者仅 CIFAR-100 预留，未实现会报错）
- **联邦 `frame1`**：数据为 **K2**（frame1 总类数）；**仅从 `frame1_pre` 全类预热检查点加载**，不再支持通过配置在 `server_init` 与 `frame1_pre` 之间切换。
- **轮次**：由阶段决定，例如 `server_init_pre_warmup_rounds`、`frame1_pre_warmup_rounds`、联邦 `frame1_federated_rounds`（`continual_run_stage=frame1` 时）；见 `modules/utils.get_num_rounds`。
- **服务器有标签**：`server_labeled_per_class` 默认由 `build_default_server_labeled_per_class` 生成（旧类 0–5 各 1000，新类按 `labeled_per_class`）；`FederatedDataLoader` 根据是否有表自动 `balanced` / `imbalanced`。
- **预热路径**：`server_init_pre_warmup_model_path` / `frame1_pre_warmup_model_path` 仅用于**扩展名**；实际文件名由 `get_server_init_pre_model_path`（含 **`kold`**）/ `get_frame1_pre_k2_warmup_model_path`（含 **`k2`**=总类数，无 k1）按超参生成，落在 `pre/serverinit/`、`pre/frame1pre/`。
- **持续学习（frame1 / frame2）**：`frame1_initial_num_classes`、`frame1_new_num_classes`、`frame2_new_num_classes`（CIFAR-100 示例：30+20=frame1，再+20 为 frame2）、`frame1_federated_rounds`、`frame2_federated_rounds`、`manual_allocation_frame1`、`continual_new_class_client_ids` 等见 `config.continual_frame_defaults()`；总类数可用 `get_frame1_total_num_classes` / `get_frame2_total_num_classes`。

## 输出目录

```
runs/
└── exp_<label>_<dataset>_lpc<lpc>_<stage>_le<le>_r<rounds>_lr..._pt..._ud..._<MMDD_HHMMSS>/
    ├── config.json
    ├── <ExpName>_result.json
    ├── comparison_*.png
    └── ...
```

（目录名以 `main.create_exp_dir` 为准。）

> 仓库已纳入 `runs/exp_r8000_ud0.9_s42/` 与 `runs/exp_r8000_ud1.0_s42/` 两组 seed 42 实验结果。其它 `runs/` 结果、`data/`、`ckpt/`、`pre/` 目录体积较大，未纳入公开仓库。复现实验时请按代码配置自行下载数据集并生成预训练/实验输出。

## 从已有结果重绘

```bash
python plot_from_json.py <runs/exp_... 目录>
```
