# UNIGET Course Project Code Package

This repository contains the executable code package for the course report
**UNIGET: Canonical Graph Tokenization for Universal Graph-to-Sequence Transformers**.
The released experiment scope matches the final report: `ogbg-molhiv` only.

The package contains:

- a reproduced GraphGPT-Mini GET baseline,
- a runnable molhiv CGT implementation derived from the report design,
- Slurm scripts for the UAlbany DGX A100 cluster,
- result aggregation, plotting, and sequence-length scripts.

The final report compares:

| Model | Pretraining | Test ROC-AUC | Mean train tokens |
| --- | --- | ---: | ---: |
| GraphGPT-Mini GET | SMTP | `0.7305 +/- 0.0157` | `42.2` |
| GraphGPT-Mini CGT | NTP | `0.7461 +/- 0.0140` | `24.8` |

The dual-stream UNIGET architecture and SC-LoRA + REPLAY-SI continual-learning
extensions remain future work; they are not part of this course submission.

## Dataset

The project uses the official Open Graph Benchmark dataset
[`ogbg-molhiv`](https://ogb.stanford.edu/docs/graphprop/#ogbg-molhiv).
The code downloads it through OGB on first use; no dataset files are stored in
GitHub. See [`dataset/README.md`](dataset/README.md).

## Repository Layout

```text
UNIGET/
├── README.md
├── dataset/
│   └── README.md
└── code/
    ├── configs/
    ├── examples/
    ├── scripts/
    ├── slurm/
    ├── src/
    └── tests/
```

## Cluster Environment Used

The reported runs used the UAlbany DGX cluster with NVIDIA A100 GPUs:

- login host: `dgx-head01.its.albany.edu`
- Slurm partition: `dgx`
- conda environment:
  `/network/rit/lab/aistudents22948/zs933749/conda_envs/graph_gpt`
- Python activation:

```bash
module load slurm
eval "$(/network/rit/lab/aistudents22948/zs933749/miniforge3/bin/conda shell.bash hook)"
conda activate /network/rit/lab/aistudents22948/zs933749/conda_envs/graph_gpt
```

## Reproduce The Paper Results

### 1. Clone And Enter The Code Directory

```bash
git clone https://github.com/ZakSiam/UNIGET.git
cd UNIGET/code
```

### 2. Prepare Runtime Folders

```bash
mkdir -p logs checkpoints/molhiv results
```

### 3. Verify The Official Split

```bash
python - <<'PY'
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

dataset = PygGraphPropPredDataset(name="ogbg-molhiv", root="./data/OGB")
split = dataset.get_idx_split()
print({key: len(val) for key, val in split.items()})
print("metric:", Evaluator(name="ogbg-molhiv").eval_metric)
assert set(split["train"].tolist()).isdisjoint(set(split["valid"].tolist()))
assert set(split["train"].tolist()).isdisjoint(set(split["test"].tolist()))
print("OGB scaffold split sanity check passed.")
PY
```

Pretraining uses only `dataset.get_idx_split()["train"]`. Fine-tuning uses the
official OGB train/valid/test scaffold splits. Validation selects the best
checkpoint; test is evaluated once per seed with the official OGB evaluator.

### 4. Submit GET Jobs

```bash
GET_PRETRAIN_JOB_ID=$(sbatch --parsable slurm/molhiv_get_mini_pretrain.slurm)
echo "GET pretrain job: ${GET_PRETRAIN_JOB_ID}"

GET_FINETUNE_JOB_ID=$(sbatch --parsable --dependency=afterok:${GET_PRETRAIN_JOB_ID} slurm/molhiv_get_mini_finetune.slurm)
echo "GET finetune array job: ${GET_FINETUNE_JOB_ID}"
```

### 5. Submit CGT Jobs

```bash
CGT_PRETRAIN_JOB_ID=$(sbatch --parsable slurm/molhiv_cgt_mini_pretrain.slurm)
echo "CGT pretrain job: ${CGT_PRETRAIN_JOB_ID}"

CGT_FINETUNE_JOB_ID=$(sbatch --parsable --dependency=afterok:${CGT_PRETRAIN_JOB_ID} slurm/molhiv_cgt_mini_finetune.slurm)
echo "CGT finetune array job: ${CGT_FINETUNE_JOB_ID}"
```

### 6. Monitor Jobs

```bash
squeue -u "$USER" -j "${GET_PRETRAIN_JOB_ID},${GET_FINETUNE_JOB_ID},${CGT_PRETRAIN_JOB_ID},${CGT_FINETUNE_JOB_ID}"
watch -n 60 "squeue -u $USER"
```

Requested wall times:

- pretraining: up to `18:00:00` each,
- fine-tuning array: up to `08:00:00` per seed.

### 7. Aggregate And Plot GET

```bash
python scripts/aggregate_molhiv_results.py
python scripts/plot_molhiv_figures.py
python scripts/compute_molhiv_get_sequence_lengths.py
```

### 8. Aggregate And Plot CGT

```bash
python scripts/aggregate_molhiv_results.py \
  --run-prefix finetune_cgt_mini \
  --model-label CGT \
  --output results/molhiv_cgt_mini.json

python scripts/plot_molhiv_figures.py \
  --run-prefix finetune_cgt_mini \
  --experiment-label CGT \
  --results-json results/molhiv_cgt_mini.json \
  --output-dir results/figures/molhiv_cgt_mini

python scripts/compute_molhiv_cgt_sequence_lengths.py
```

## Expected Outputs

GET:

- `checkpoints/molhiv/pretrain_get_mini/`
- `checkpoints/molhiv/finetune_get_mini_seed*/test_metrics.json`
- `results/molhiv_get_mini.json`
- `results/figures/molhiv_get_mini/`
- `results/molhiv_get_sequence_lengths.json`

CGT:

- `checkpoints/molhiv/pretrain_cgt_mini/`
- `checkpoints/molhiv/finetune_cgt_mini_seed*/test_metrics.json`
- `results/molhiv_cgt_mini.json`
- `results/figures/molhiv_cgt_mini/`
- `results/molhiv_cgt_sequence_lengths.json`

## Implementation Notes

- GET uses the original Eulerian-path tokenizer and SMTP setup.
- CGT uses deterministic attributed-graph canonicalization followed by a
  canonical edge-once serialization compatible with the GraphGPT stacked input
  interface.
- The released CGT code is a faithful runnable implementation of the method in
  the course report. The collaborator implementation used to produce the final
  report figures was not available for direct vendoring into this repository.
- This project builds on the MIT-licensed GraphGPT codebase from Alibaba; see
  [`LICENSE`](LICENSE) and [`code/LICENSE`](code/LICENSE).
