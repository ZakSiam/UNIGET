#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export MKL_THREADING_LAYER=GNU
RUNTIME_ROOT="${GRAPH_GPT_RUNTIME_DIR:-${SLURM_TMPDIR:-/tmp}/${USER:-graph_gpt}/graph-gpt/${SLURM_JOB_ID:-manual}}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${RUNTIME_ROOT}/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${RUNTIME_ROOT}/torchinductor}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/network/rit/lab/aistudents22948/zs933749/cache/xdg}"
export TMPDIR="${TMPDIR:-${RUNTIME_ROOT}/tmp}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${RUNTIME_ROOT}/cuda}"

mkdir -p \
  logs checkpoints/molhiv/pretrain_get_mini \
  "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" \
  "$XDG_CACHE_HOME" "$TMPDIR" "$CUDA_CACHE_PATH"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints/molhiv/pretrain_get_mini}"
TOTAL_TOKENS="${TOTAL_TOKENS:-2e8}"
WARMUP_TOKENS="${WARMUP_TOKENS:-2e7}"

args=(
  "--config-name=experiments/molhiv_get_mini_pretrain"
  "training.output_dir=${OUTPUT_DIR}"
  "training.schedule.total_tokens=${TOTAL_TOKENS}"
  "training.schedule.warmup_tokens=${WARMUP_TOKENS}"
)

if [[ -n "${DEEPSPEED_CONFIG:-}" ]]; then
  args+=("training.deepspeed_conf_file=${DEEPSPEED_CONFIG}")
fi

"${PYTHON_BIN}" ./examples/train_pretrain.py "${args[@]}"
