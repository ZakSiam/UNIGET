#!/bin/bash
#SBATCH --job-name=molhiv_cgt_mini_pretrain
#SBATCH --output=logs/molhiv_cgt_mini_pretrain_%A.out
#SBATCH --error=logs/molhiv_cgt_mini_pretrain_%A.err
#SBATCH --partition=dgx
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gpus=1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=zsiam@albany.edu

set -euo pipefail

module load slurm
eval "$(/network/rit/lab/aistudents22948/zs933749/miniforge3/bin/conda shell.bash hook)"
conda activate /network/rit/lab/aistudents22948/zs933749/conda_envs/graph_gpt
cd "${SLURM_SUBMIT_DIR:-/network/rit/lab/aistudents22948/zs933749/graph-gpt}"

export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export MKL_THREADING_LAYER=GNU

RUNTIME_ROOT="${SLURM_TMPDIR:-/tmp}/${USER:-graph_gpt}/graph-gpt/${SLURM_JOB_ID:-manual}"
export PYG_HOME=/network/rit/lab/aistudents22948/zs933749/cache/pyg
export XDG_CACHE_HOME=/network/rit/lab/aistudents22948/zs933749/cache/xdg
export TMPDIR="${RUNTIME_ROOT}/tmp"
export TRITON_CACHE_DIR="${RUNTIME_ROOT}/triton"
export TORCHINDUCTOR_CACHE_DIR="${RUNTIME_ROOT}/torchinductor"
export CUDA_CACHE_PATH="${RUNTIME_ROOT}/cuda"
mkdir -p \
  logs checkpoints/molhiv/pretrain_cgt_mini results \
  "$PYG_HOME" "$XDG_CACHE_HOME" "$TMPDIR" \
  "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH"

echo "===== NODE/GPU INFO ====="
hostname
nvidia-smi
which python

bash examples/graph_lvl/molhiv_cgt_pretrain.sh
