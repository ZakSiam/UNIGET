# Dataset Package

## Dataset Link

Official dataset page:
[`ogbg-molhiv` on Open Graph Benchmark](https://ogb.stanford.edu/docs/graphprop/#ogbg-molhiv)

## What The Dataset Is

`ogbg-molhiv` is a molecular graph classification benchmark. Each graph is a
molecule, the target is a binary HIV inhibition label, the official split is a
scaffold split, and the official metric is ROC-AUC.

## How The Code Gets The Data

The project does not store raw dataset files in GitHub. OGB downloads and
processes the dataset automatically the first time this command is run from
`code/`:

```bash
python - <<'PY'
from ogb.graphproppred import PygGraphPropPredDataset
PygGraphPropPredDataset(name="ogbg-molhiv", root="./data/OGB")
print("ogbg-molhiv is ready.")
PY
```

## Split Policy Used In The Project

- pretraining: official OGB `train` scaffold split only,
- fine-tuning: official OGB `train` split,
- model selection: official OGB `valid` split,
- final reporting: official OGB `test` split once per seed.

The code never mixes valid or test molecules into pretraining.
