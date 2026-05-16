# Code Package

This folder contains the executable source used for the `ogbg-molhiv` study.

## Main Entry Points

GET:

- `examples/graph_lvl/molhiv_pretrain.sh`
- `examples/graph_lvl/molhiv_supervised.sh`
- `slurm/molhiv_get_mini_pretrain.slurm`
- `slurm/molhiv_get_mini_finetune.slurm`

CGT:

- `examples/graph_lvl/molhiv_cgt_pretrain.sh`
- `examples/graph_lvl/molhiv_cgt_supervised.sh`
- `slurm/molhiv_cgt_mini_pretrain.slurm`
- `slurm/molhiv_cgt_mini_finetune.slurm`

Post-processing:

- `scripts/aggregate_molhiv_results.py`
- `scripts/plot_molhiv_figures.py`
- `scripts/compute_molhiv_get_sequence_lengths.py`
- `scripts/compute_molhiv_cgt_sequence_lengths.py`

## CGT Files

- `src/data/tokenizer/canonical.py`: WL-style refinement plus deterministic
  individualization search and edge-once linearization metadata.
- `src/data/tokenizer/core.py`: `CanonicalStackedGSTTokenizer`.
- `src/data/tokenizer/strategies/task_prep/pretrain.py`: `PretrainNTPStrategy`.
- `configs/tokenization/graph_lvl/ogbg_molhiv_cgt.yaml`
- `configs/experiments/molhiv_cgt_mini_pretrain.yaml`
- `configs/experiments/molhiv_cgt_mini_finetune.yaml`

## Quick Local Checks

```bash
python -m py_compile \
  src/data/tokenizer/canonical.py \
  src/data/tokenizer/core.py \
  src/data/tokenizer/strategies/task_prep/pretrain.py \
  scripts/aggregate_molhiv_results.py \
  scripts/plot_molhiv_figures.py \
  scripts/compute_molhiv_cgt_sequence_lengths.py

bash -n \
  examples/graph_lvl/molhiv_pretrain.sh \
  examples/graph_lvl/molhiv_supervised.sh \
  examples/graph_lvl/molhiv_cgt_pretrain.sh \
  examples/graph_lvl/molhiv_cgt_supervised.sh \
  slurm/molhiv_get_mini_pretrain.slurm \
  slurm/molhiv_get_mini_finetune.slurm \
  slurm/molhiv_cgt_mini_pretrain.slurm \
  slurm/molhiv_cgt_mini_finetune.slurm
```
