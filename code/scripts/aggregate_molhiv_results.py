#!/usr/bin/env python
"""Aggregate ogbg-molhiv GraphGPT-Mini test ROC-AUC over seeds."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-root",
        default="checkpoints/molhiv",
        help="Directory containing finetune_get_mini_seed{seed} folders.",
    )
    parser.add_argument(
        "--output",
        default="results/molhiv_get_mini.json",
        help="JSON file to write aggregated results.",
    )
    parser.add_argument(
        "--run-prefix",
        default="finetune_get_mini",
        help="Checkpoint-folder prefix before _seed{seed}.",
    )
    parser.add_argument(
        "--model-label",
        default="GET",
        help="Human-readable tokenizer/model label stored in the JSON.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2],
        help="Fine-tuning seeds to aggregate.",
    )
    return parser.parse_args()


def read_seed_metrics(checkpoint_root: Path, run_prefix: str, seed: int) -> dict:
    metrics_file = checkpoint_root / f"{run_prefix}_seed{seed}" / "test_metrics.json"
    if not metrics_file.is_file():
        raise FileNotFoundError(f"Missing metrics file for seed {seed}: {metrics_file}")
    with metrics_file.open() as fp:
        payload = json.load(fp)
    test_rocauc = float(payload["test"]["rocauc"])
    best_valid = payload.get("best_valid") or {}
    return {
        "seed": seed,
        "test_rocauc": test_rocauc,
        "best_valid_rocauc": best_valid.get("rocauc"),
        "best_epoch": payload.get("best_epoch"),
        "metrics_file": str(metrics_file),
    }


def population_std(values: list[float]) -> float:
    mean = sum(values) / len(values)
    return math.sqrt(sum((val - mean) ** 2 for val in values) / len(values))


def main() -> None:
    args = parse_args()
    checkpoint_root = Path(args.checkpoint_root)
    rows = [
        read_seed_metrics(checkpoint_root, args.run_prefix, seed)
        for seed in args.seeds
    ]
    values = [row["test_rocauc"] for row in rows]
    mean = sum(values) / len(values)
    std = population_std(values)

    result = {
        "metric": "rocauc",
        "dataset": "ogbg-molhiv",
        "model_label": args.model_label,
        "std_type": "population",
        "per_seed": rows,
        "mean": mean,
        "std": std,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as fp:
        json.dump(result, fp, indent=2, sort_keys=True)

    print("seed  best_valid_rocauc  best_epoch  test_rocauc")
    for row in rows:
        best_valid = row["best_valid_rocauc"]
        best_valid_str = "NA" if best_valid is None else f"{best_valid:.6f}"
        print(
            f"{row['seed']:>4}  {best_valid_str:>17}  "
            f"{str(row['best_epoch']):>10}  {row['test_rocauc']:.6f}"
        )
    print(f"\nmean ± std ({result['std_type']}): {mean:.6f} ± {std:.6f}")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
