#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

SEED="${1:-${SEED:-0}}"
PRETRAIN_CPT="${PRETRAIN_CPT:-./checkpoints/molhiv/pretrain_get_mini}"
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints/molhiv/finetune_get_mini_seed${SEED}}"

if [[ -f "${OUTPUT_DIR}/test_metrics.json" && -f "${OUTPUT_DIR}/test_predictions.csv" ]]; then
  echo "Final outputs already exist for seed ${SEED}; skipping ${OUTPUT_DIR}."
  exit 0
fi

export MKL_THREADING_LAYER=GNU
RUNTIME_ROOT="${GRAPH_GPT_RUNTIME_DIR:-${SLURM_TMPDIR:-/tmp}/${USER:-graph_gpt}/graph-gpt/${SLURM_JOB_ID:-manual}_${SEED}}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${RUNTIME_ROOT}/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${RUNTIME_ROOT}/torchinductor}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/network/rit/lab/aistudents22948/zs933749/cache/xdg}"
export TMPDIR="${TMPDIR:-${RUNTIME_ROOT}/tmp}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${RUNTIME_ROOT}/cuda}"

mkdir -p \
  logs "${OUTPUT_DIR}" \
  "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" \
  "$XDG_CACHE_HOME" "$TMPDIR" "$CUDA_CACHE_PATH"

PYTHON_BIN="${PYTHON_BIN:-python}"

args=(
  "--config-name=experiments/molhiv_get_mini_finetune"
  "training.finetune.seed=${SEED}"
  "training.pretrain_cpt=${PRETRAIN_CPT}"
  "training.output_dir=${OUTPUT_DIR}"
)

if [[ -n "${DEEPSPEED_CONFIG:-}" ]]; then
  args+=("training.deepspeed_conf_file=${DEEPSPEED_CONFIG}")
fi

"${PYTHON_BIN}" ./examples/train_supervised.py "${args[@]}"
