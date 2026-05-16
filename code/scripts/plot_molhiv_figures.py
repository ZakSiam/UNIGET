#!/usr/bin/env python
"""Generate presentation figures for ogbg-molhiv GET/CGT Mini experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Iterable

mpl_config_dir = Path(tempfile.gettempdir()) / "graph_gpt_matplotlib"
mpl_config_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-root",
        default="checkpoints/molhiv",
        help="Directory containing finetune_get_mini_seed{seed} folders.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/figures/molhiv_get_mini",
        help="Directory for generated figures and the manifest.",
    )
    parser.add_argument(
        "--results-json",
        default="results/molhiv_get_mini.json",
        help="Aggregated scalar result JSON, if available.",
    )
    parser.add_argument(
        "--run-prefix",
        default="finetune_get_mini",
        help="Checkpoint-folder prefix before _seed{seed}.",
    )
    parser.add_argument(
        "--experiment-label",
        default="GET",
        help="Short label used in titles and the manifest.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2],
        help="Fine-tuning seeds to plot.",
    )
    parser.add_argument(
        "--ogb-root",
        default="./data/OGB",
        help="OGB root used only for the split class-balance figure.",
    )
    parser.add_argument(
        "--skip-split-balance",
        action="store_true",
        help="Skip loading OGB to draw the split class-balance figure.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "pdf"],
        help="Figure formats to write.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    with path.open() as fp:
        return json.load(fp)


def read_predictions(path: Path) -> dict[str, np.ndarray]:
    rows = []
    with path.open(newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            rows.append(row)
    if not rows:
        raise ValueError(f"No prediction rows found in {path}")

    idx = np.array([int(float(row["idx"])) for row in rows], dtype=np.int64)
    y_true = np.array([float(row["y_true"]) for row in rows], dtype=np.float64)
    y_logit = np.array([float(row["y_score_logit"]) for row in rows], dtype=np.float64)
    if "y_score_prob" in rows[0]:
        y_prob = np.array(
            [float(row["y_score_prob"]) for row in rows], dtype=np.float64
        )
    else:
        y_prob = 1.0 / (1.0 + np.exp(-y_logit))

    order = np.argsort(idx)
    labeled = ~np.isnan(y_true[order])
    return {
        "idx": idx[order][labeled],
        "y_true": y_true[order][labeled].astype(np.int64),
        "y_logit": y_logit[order][labeled],
        "y_prob": y_prob[order][labeled],
    }


def load_seed_artifacts(
    checkpoint_root: Path, seeds: Iterable[int], run_prefix: str
) -> list[dict]:
    artifacts = []
    for seed in seeds:
        seed_dir = checkpoint_root / f"{run_prefix}_seed{seed}"
        pred_path = seed_dir / "test_predictions.csv"
        metrics_path = seed_dir / "test_metrics.json"
        if not pred_path.is_file():
            raise FileNotFoundError(f"Missing predictions for seed {seed}: {pred_path}")
        if not metrics_path.is_file():
            raise FileNotFoundError(f"Missing metrics for seed {seed}: {metrics_path}")

        pred = read_predictions(pred_path)
        metrics = read_json(metrics_path)
        calc_auc = roc_auc_score(pred["y_true"], pred["y_prob"])
        artifacts.append(
            {
                "seed": seed,
                "seed_dir": seed_dir,
                "predictions_file": pred_path,
                "metrics_file": metrics_path,
                "pred": pred,
                "metrics": metrics,
                "calculated_rocauc": float(calc_auc),
            }
        )
    return artifacts


def setup_style() -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def save_figure(fig, output_dir: Path, stem: str, formats: list[str]) -> list[str]:
    paths = []
    for fmt in formats:
        path = output_dir / f"{stem}.{fmt}"
        fig.savefig(path, bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)
    return paths


def plot_roc_curves(
    artifacts: list[dict], output_dir: Path, formats: list[str], experiment_label: str
):
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for item in artifacts:
        pred = item["pred"]
        fpr, tpr, _ = roc_curve(pred["y_true"], pred["y_prob"])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f"seed {item['seed']} AUC={roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], color="0.55", lw=1.2, linestyle="--", label="chance")
    ax.set_title(f"ogbg-molhiv {experiment_label} Test ROC Curves")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")
    return save_figure(fig, output_dir, "roc_curves_by_seed", formats)


def plot_mean_roc(
    artifacts: list[dict], output_dir: Path, formats: list[str], experiment_label: str
):
    common_fpr = np.linspace(0, 1, 201)
    interp_tprs = []
    aucs = []
    for item in artifacts:
        pred = item["pred"]
        fpr, tpr, _ = roc_curve(pred["y_true"], pred["y_prob"])
        interp = np.interp(common_fpr, fpr, tpr)
        interp[0] = 0.0
        interp_tprs.append(interp)
        aucs.append(auc(fpr, tpr))
    tprs = np.vstack(interp_tprs)
    mean_tpr = tprs.mean(axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = tprs.std(axis=0)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.plot(
        common_fpr,
        mean_tpr,
        color="#1f77b4",
        lw=2.4,
        label=f"mean AUC={np.mean(aucs):.3f} +/- {np.std(aucs):.3f}",
    )
    ax.fill_between(
        common_fpr,
        np.maximum(mean_tpr - std_tpr, 0),
        np.minimum(mean_tpr + std_tpr, 1),
        color="#1f77b4",
        alpha=0.18,
        label="seed variability",
    )
    ax.plot([0, 1], [0, 1], color="0.55", lw=1.2, linestyle="--")
    ax.set_title(f"{experiment_label} Mean Test ROC Curve Across Seeds")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")
    return save_figure(fig, output_dir, "roc_mean_with_band", formats)


def plot_pr_curves(artifacts: list[dict], output_dir: Path, formats: list[str]):
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for item in artifacts:
        pred = item["pred"]
        precision, recall, _ = precision_recall_curve(pred["y_true"], pred["y_prob"])
        ap = average_precision_score(pred["y_true"], pred["y_prob"])
        ax.plot(recall, precision, lw=2, label=f"seed {item['seed']} AP={ap:.3f}")
    prevalence = np.concatenate([item["pred"]["y_true"] for item in artifacts]).mean()
    ax.axhline(prevalence, color="0.55", lw=1.2, linestyle="--", label="prevalence")
    ax.set_title("ogbg-molhiv Test Precision-Recall Curves")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="best")
    return save_figure(fig, output_dir, "precision_recall_curves", formats)


def plot_score_distribution(
    artifacts: list[dict], output_dir: Path, formats: list[str]
):
    y_true = np.concatenate([item["pred"]["y_true"] for item in artifacts])
    y_prob = np.concatenate([item["pred"]["y_prob"] for item in artifacts])

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    bins = np.linspace(0, 1, 31)
    ax.hist(
        y_prob[y_true == 0],
        bins=bins,
        density=True,
        alpha=0.62,
        label="inactive / label 0",
        color="#4c78a8",
    )
    ax.hist(
        y_prob[y_true == 1],
        bins=bins,
        density=True,
        alpha=0.62,
        label="active / label 1",
        color="#f58518",
    )
    ax.set_title("Predicted Probability Distribution")
    ax.set_xlabel("Predicted HIV activity probability")
    ax.set_ylabel("Density")
    ax.legend(loc="best")
    return save_figure(fig, output_dir, "score_distribution", formats)


def plot_calibration(artifacts: list[dict], output_dir: Path, formats: list[str]):
    y_true = np.concatenate([item["pred"]["y_true"] for item in artifacts])
    y_prob = np.concatenate([item["pred"]["y_prob"] for item in artifacts])
    bins = np.linspace(0, 1, 11)
    bin_ids = np.digitize(y_prob, bins, right=True) - 1
    bin_ids = np.clip(bin_ids, 0, len(bins) - 2)

    mean_prob = []
    frac_pos = []
    counts = []
    for bin_id in range(len(bins) - 1):
        mask = bin_ids == bin_id
        if not np.any(mask):
            continue
        mean_prob.append(y_prob[mask].mean())
        frac_pos.append(y_true[mask].mean())
        counts.append(int(mask.sum()))

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.plot([0, 1], [0, 1], color="0.55", lw=1.2, linestyle="--", label="perfect")
    ax.plot(mean_prob, frac_pos, marker="o", lw=2, label="model")
    ax.set_title("Calibration-Style Reliability Plot")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive fraction")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")

    ax2 = ax.twinx()
    ax2.bar(mean_prob, counts, width=0.06, color="0.3", alpha=0.15, label="count")
    ax2.set_ylabel("Samples per bin")
    return save_figure(fig, output_dir, "calibration_reliability", formats)


def plot_auc_summary(
    artifacts: list[dict], output_dir: Path, formats: list[str], aggregate: dict
):
    seeds = [item["seed"] for item in artifacts]
    aucs = [
        float(item["metrics"].get("test", {}).get("rocauc", item["calculated_rocauc"]))
        for item in artifacts
    ]
    mean = float(aggregate.get("mean", np.mean(aucs)))
    std = float(aggregate.get("std", np.std(aucs)))

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    x = np.arange(len(seeds))
    ax.bar(x, aucs, color="#54a24b", alpha=0.82)
    ax.set_xticks(x, [str(seed) for seed in seeds])
    ax.axhline(mean, color="#d62728", lw=2, label=f"mean={mean:.3f}")
    ax.fill_between(
        [-0.5, len(seeds) - 0.5],
        [mean - std, mean - std],
        [mean + std, mean + std],
        color="#d62728",
        alpha=0.12,
        label=f"std={std:.3f}",
    )
    ax.set_title("Final Test ROC-AUC By Seed")
    ax.set_xlabel("Fine-tuning seed")
    ax.set_ylabel("Test ROC-AUC")
    ax.set_ylim(max(0.0, min(aucs) - 0.05), min(1.0, max(aucs) + 0.05))
    ax.legend(loc="best")
    return save_figure(fig, output_dir, "test_rocauc_summary", formats)


def parse_result_history(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    with path.open() as fp:
        next(fp, None)
        for line in fp:
            parts = [part.strip() for part in line.strip().split(",")]
            if len(parts) < 2 or not parts[0]:
                continue
            roc_positions = [i for i, part in enumerate(parts) if part == "rocauc"]
            values = []
            for pos in roc_positions:
                if pos + 1 < len(parts):
                    try:
                        values.append(float(parts[pos + 1]))
                    except ValueError:
                        values.append(math.nan)
            rows.append(
                {
                    "epoch": int(float(parts[0])),
                    "global_step": int(float(parts[1])),
                    "train_rocauc": values[0] if len(values) > 0 else math.nan,
                    "valid_rocauc": values[1] if len(values) > 1 else math.nan,
                    "test_rocauc": values[2] if len(values) > 2 else math.nan,
                }
            )
    return rows


def parse_loss_history(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    with path.open(newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            try:
                rows.append(
                    {
                        "epoch": int(float(row["epoch"])),
                        "global_step": int(float(row["global_step"])),
                        "task_loss": float(row["task_loss"]),
                        "train_loss": float(row["train_loss"]),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def plot_validation_history(
    artifacts: list[dict], output_dir: Path, formats: list[str]
) -> list[str]:
    histories = []
    for item in artifacts:
        history = parse_result_history(item["seed_dir"] / "result.csv")
        history = [row for row in history if not math.isnan(row["valid_rocauc"])]
        if history:
            histories.append((item, history))
    if not histories:
        return []

    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    for item, history in histories:
        epochs = [row["epoch"] for row in history]
        valid = [row["valid_rocauc"] for row in history]
        ax.plot(epochs, valid, marker="o", ms=3, lw=1.8, label=f"seed {item['seed']}")
        best_epoch = item["metrics"].get("best_epoch")
        if best_epoch is not None:
            best_rows = [row for row in history if row["epoch"] == best_epoch]
            if best_rows:
                ax.scatter(
                    [best_epoch],
                    [best_rows[0]["valid_rocauc"]],
                    s=70,
                    marker="*",
                    zorder=4,
                )
    ax.set_title("Validation ROC-AUC History")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Valid ROC-AUC")
    ax.legend(loc="best")
    return save_figure(fig, output_dir, "validation_rocauc_history", formats)


def plot_training_loss_history(
    artifacts: list[dict], output_dir: Path, formats: list[str]
) -> list[str]:
    histories = []
    for item in artifacts:
        history = parse_loss_history(item["seed_dir"] / "loss.csv")
        if history:
            histories.append((item, history))
    if not histories:
        return []

    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    for item, history in histories:
        steps = [row["global_step"] for row in history]
        loss = [row["train_loss"] for row in history]
        ax.plot(steps, loss, lw=1.6, label=f"seed {item['seed']}")
    ax.set_title("Fine-Tuning Training Loss")
    ax.set_xlabel("Global step")
    ax.set_ylabel("Training loss")
    ax.legend(loc="best")
    return save_figure(fig, output_dir, "training_loss_history", formats)


def load_split_balance(ogb_root: str) -> list[dict]:
    from ogb.graphproppred import PygGraphPropPredDataset

    dataset = PygGraphPropPredDataset(name="ogbg-molhiv", root=ogb_root)
    split = dataset.get_idx_split()
    rows = []
    for split_name in ["train", "valid", "test"]:
        labels = []
        for idx in split[split_name].tolist():
            y_val = float(dataset[int(idx)].y.view(-1)[0].item())
            if not math.isnan(y_val):
                labels.append(int(y_val))
        labels_np = np.array(labels, dtype=np.int64)
        positives = int((labels_np == 1).sum())
        negatives = int((labels_np == 0).sum())
        rows.append(
            {
                "split": split_name,
                "positive": positives,
                "negative": negatives,
                "total": positives + negatives,
                "positive_rate": positives / max(positives + negatives, 1),
            }
        )
    return rows


def plot_split_balance(
    split_rows: list[dict], output_dir: Path, formats: list[str]
) -> list[str]:
    names = [row["split"] for row in split_rows]
    negatives = np.array([row["negative"] for row in split_rows])
    positives = np.array([row["positive"] for row in split_rows])
    rates = np.array([row["positive_rate"] for row in split_rows])
    x = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.bar(x, negatives, label="label 0", color="#4c78a8", alpha=0.84)
    ax.bar(x, positives, bottom=negatives, label="label 1", color="#f58518", alpha=0.84)
    ax.set_title("OGB Scaffold Split Class Balance")
    ax.set_xlabel("Split")
    ax.set_ylabel("Molecules")
    ax.set_xticks(x, names)
    ax.legend(loc="upper left")

    ax2 = ax.twinx()
    ax2.plot(x, rates, color="#d62728", marker="o", lw=2, label="positive rate")
    ax2.set_ylabel("Positive rate")
    ax2.set_ylim(0, max(0.1, rates.max() * 1.35))
    return save_figure(fig, output_dir, "split_class_balance", formats)


def main() -> None:
    args = parse_args()
    checkpoint_root = Path(args.checkpoint_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    formats = [fmt.lstrip(".") for fmt in args.formats]
    setup_style()

    aggregate = read_json(Path(args.results_json))
    artifacts = load_seed_artifacts(checkpoint_root, args.seeds, args.run_prefix)

    manifest = {
        "dataset": "ogbg-molhiv",
        "experiment_label": args.experiment_label,
        "run_prefix": args.run_prefix,
        "figure_dir": str(output_dir),
        "figures": {},
        "skipped": [],
        "per_seed": [],
    }
    for item in artifacts:
        manifest["per_seed"].append(
            {
                "seed": item["seed"],
                "test_metrics_file": str(item["metrics_file"]),
                "test_predictions_file": str(item["predictions_file"]),
                "official_test_rocauc": item["metrics"].get("test", {}).get("rocauc"),
                "calculated_test_rocauc": item["calculated_rocauc"],
            }
        )

    manifest["figures"]["roc_curves_by_seed"] = plot_roc_curves(
        artifacts, output_dir, formats, args.experiment_label
    )
    manifest["figures"]["roc_mean_with_band"] = plot_mean_roc(
        artifacts, output_dir, formats, args.experiment_label
    )
    figure_fns = [
        ("precision_recall_curves", plot_pr_curves),
        ("score_distribution", plot_score_distribution),
        ("calibration_reliability", plot_calibration),
    ]
    for name, fn in figure_fns:
        manifest["figures"][name] = fn(artifacts, output_dir, formats)
    manifest["figures"]["test_rocauc_summary"] = plot_auc_summary(
        artifacts, output_dir, formats, aggregate
    )

    history_paths = plot_validation_history(artifacts, output_dir, formats)
    if history_paths:
        manifest["figures"]["validation_rocauc_history"] = history_paths
    else:
        manifest["skipped"].append("validation_rocauc_history: no result.csv history")

    loss_paths = plot_training_loss_history(artifacts, output_dir, formats)
    if loss_paths:
        manifest["figures"]["training_loss_history"] = loss_paths
    else:
        manifest["skipped"].append("training_loss_history: no loss.csv history")

    if not args.skip_split_balance:
        try:
            split_rows = load_split_balance(args.ogb_root)
            manifest["split_balance"] = split_rows
            manifest["figures"]["split_class_balance"] = plot_split_balance(
                split_rows, output_dir, formats
            )
        except Exception as exc:  # pragma: no cover - depends on local OGB install/data.
            manifest["skipped"].append(f"split_class_balance: {exc}")
    else:
        manifest["skipped"].append("split_class_balance: skipped by CLI flag")

    manifest_path = output_dir / "figure_manifest.json"
    with manifest_path.open("w") as fp:
        json.dump(manifest, fp, indent=2, sort_keys=True)

    print("Generated molhiv figure bundle:")
    for name, paths in manifest["figures"].items():
        print(f"  {name}: {', '.join(paths)}")
    if manifest["skipped"]:
        print("Skipped:")
        for item in manifest["skipped"]:
            print(f"  {item}")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
