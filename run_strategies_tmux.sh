#!/usr/bin/env bash
# Start detached tmux sessions (one strategy each):
#   结果 JSON/图表 -> runs/exp_<abbrev>_svhn_..._<timestamp>/
#   断点/模型       -> ckpt/<abbrev>_ud<upload_deadline>_s<seed>/
#
# Usage:
#   ./run_strategies_tmux.sh
#
# Optional env:
#   RESUME=1               # 1=从 ckpt 自动续训（默认）；0=从头开始（--no-resume）
#   TAG=mytag              # tmux 会话名后缀（默认当前时间）
#   CKPT_BASE=ckpt         # 策略输出根目录（默认 $ROOT/ckpt）
#   CONDA_ENV=fl           # conda 环境名（默认 fl）
#   CONDA_BASE=...         # miniconda 根目录（默认 /opt/data/private/zcy/miniconda3）
#   PYTHON=/path/to/python # 可选，覆盖自动解析的 Python
# upload_deadline / seed 从 config.py DEFAULT_CONFIG 读取，无需传参

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

TAG="${TAG:-$(date +%m%d_%H%M%S)}"
CKPT_BASE="${CKPT_BASE:-$ROOT/ckpt}"
CONDA_ENV="${CONDA_ENV:-fl}"
CONDA_BASE="${CONDA_BASE:-/opt/data/private/zcy/miniconda3}"
RESUME="${RESUME:-1}"

resolve_python() {
  local candidate
  if [[ -n "${PYTHON:-}" ]]; then
    if command -v "$PYTHON" &>/dev/null; then
      command -v "$PYTHON"
      return 0
    fi
    if [[ -x "$PYTHON" ]]; then
      echo "$PYTHON"
      return 0
    fi
  fi
  if [[ -n "${CONDA_PREFIX:-}" && "$(basename "$CONDA_PREFIX")" == "$CONDA_ENV" ]]; then
    candidate="${CONDA_PREFIX}/bin/python"
    if [[ -x "$candidate" ]]; then
      echo "$candidate"
      return 0
    fi
  fi
  candidate="${CONDA_BASE}/envs/${CONDA_ENV}/bin/python"
  if [[ -x "$candidate" ]]; then
    echo "$candidate"
    return 0
  fi
  if command -v conda &>/dev/null; then
    candidate="$(conda run -n "$CONDA_ENV" which python 2>/dev/null || true)"
    if [[ -n "$candidate" && -x "$candidate" ]]; then
      echo "$candidate"
      return 0
    fi
  fi
  return 1
}

PY="$(resolve_python || true)"
if [[ -z "$PY" ]]; then
  echo "错误：未找到 conda 环境 ${CONDA_ENV} 的 Python。" >&2
  echo "可设置 CONDA_BASE、CONDA_ENV 或 PYTHON，例如：" >&2
  echo "  CONDA_BASE=/opt/data/private/zcy/miniconda3 CONDA_ENV=fl ./run_strategies_tmux.sh" >&2
  exit 127
fi

want_resume() {
  case "${RESUME,,}" in
    0|false|no|off) return 1 ;;
    *) return 0 ;;
  esac
}

has_strategy_checkpoint() {
  local strategy="$1"
  local ckpt_dir="$2"
  local ckpt_subdir="${ckpt_dir}/checkpoints/${strategy}"
  compgen -G "${ckpt_subdir}/*.pth" >/dev/null
}

load_config_meta() {
  local out
  out="$("$PY" -c 'from config import DEFAULT_CONFIG; c=DEFAULT_CONFIG; print(c["upload_deadline"], c["seed"])')" || {
    echo "错误：无法从 config.py 读取 upload_deadline / seed（python: $PY）。" >&2
    exit 1
  }
  read -r UPLOAD_DEADLINE SEED <<< "$out"
}

load_config_meta

if ! command -v tmux &>/dev/null; then
  echo "错误：未找到 tmux（不在 PATH）。" >&2
  echo "在 Debian/Ubuntu 上可先安装：" >&2
  echo "  sudo apt-get update && sudo apt-get install -y tmux" >&2
  exit 127
fi

mkdir -p "$CKPT_BASE"

# 策略简称（与 config.ABBREVIATIONS 一致）；未列出的策略回退为原名
strategy_abbrev() {
  case "$1" in
    PO_FPM) echo "PO_FPM" ;;
    PO_FPLN) echo "PO_FPLN" ;;
    PO_MFLN) echo "PO_MFLN" ;;
    Random_PriorityOldNew) echo "R_PO" ;;
    Random_StickyNew) echo "R_SN" ;;
    Random_FLFL) echo "R_FLFL" ;;
    Random_FedLGMatch) echo "R_FedLG" ;;
    Random_FSSL_UC) echo "R_FSSL_UC" ;;
    NoClientTrain) echo "NCT" ;;
    AllClientsTrain) echo "ACT" ;;
    Random) echo "R" ;;
    BestChannel) echo "BC" ;;
    NewClassClientsOnly) echo "NCC" ;;
    *) echo "$1" ;;
  esac
}

strategy_ckpt_dir() {
  local strategy="$1"
  local abbrev
  abbrev="$(strategy_abbrev "$strategy")"
  echo "${CKPT_BASE}/${abbrev}_ud${UPLOAD_DEADLINE}_s${SEED}"
}

# 按需增删；与 main.py 中 EXPERIMENT_TYPES 名称一致
STRATEGIES=(
  # "PO_FPM"
  # "PO_FPLN"
  "PO_MFLN"
  # "Random_PriorityOldNew"
  # "Random_FLFL"
  "Random_FedLGMatch"
  # "Random_FSSL_UC"
  # "NoClientTrain"
  # "AllClientsTrain"
  # "Random"
  # "BestChannel"
  # "NewClassClientsOnly"
)

echo "Starting ${#STRATEGIES[@]} detached tmux sessions (tag=$TAG) in: $ROOT"
if want_resume; then
  echo "RESUME=1  （自动从 ckpt/<策略>_ud${UPLOAD_DEADLINE}_s${SEED}/ 续训；无断点则从 frame1_pre 开始）"
else
  echo "RESUME=0  （忽略已有断点，全部从头开始）"
fi
echo "CKPT_BASE=$CKPT_BASE  CONDA_ENV=$CONDA_ENV  PYTHON=$PY  UPLOAD_DEADLINE=$UPLOAD_DEADLINE  SEED=$SEED"
echo ""

for s in "${STRATEGIES[@]}"; do
  name="fed_${s}_${TAG}"
  ckpt_dir="$(strategy_ckpt_dir "$s")"
  mkdir -p "$ckpt_dir"

  resume_flag=""
  if want_resume; then
    if has_strategy_checkpoint "$s" "$ckpt_dir"; then
      echo "  $s -> runs/ (auto)  ckpt=$ckpt_dir  [续训]"
    else
      echo "  $s -> runs/ (auto)  ckpt=$ckpt_dir  [续训: 无断点，从头开始]"
    fi
  else
    resume_flag=" --no-resume"
    echo "  $s -> runs/ (auto)  ckpt=$ckpt_dir  [从头开始]"
  fi

  # 若 python 立刻退出，会话仍保留交互 shell，便于 attach 看报错；否则 tmux 无 session 后 server 也会退出。
  tmux new-session -d -s "$name" -c "$ROOT" \
    bash -lc "$(printf 'set -euo pipefail; %q main.py %q --checkpoint-dir %q%s || exec bash -i' \
      "$PY" "$s" "$ckpt_dir" "$resume_flag")"
done

echo ""
echo "List: tmux ls"
echo "Attach one (example):"
echo "  tmux attach -t fed_${STRATEGIES[0]}_${TAG}"
echo "Kill these sessions:"
for s in "${STRATEGIES[@]}"; do
  echo "  tmux kill-session -t fed_${s}_${TAG}"
done
