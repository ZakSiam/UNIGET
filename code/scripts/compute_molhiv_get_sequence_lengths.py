#!/usr/bin/env python
"""Measure GET sequence lengths on the ogbg-molhiv train split."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import tempfile
from pathlib import Path

mpl_config_dir = Path(tempfile.gettempdir()) / "graph_gpt_matplotlib"
mpl_config_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.conf import base_configs
from src.data import read_dataset, tokenizer
from src.utils import conf_utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-name",
        default="experiments/molhiv_get_mini_pretrain",
        help="Hydra config to load from configs/.",
    )
    parser.add_argument(
        "--output-csv",
        default="results/molhiv_get_sequence_lengths.csv",
        help="Per-graph length table to write.",
    )
    parser.add_argument(
        "--output-json",
        default="results/molhiv_get_sequence_lengths.json",
        help="Summary JSON to write.",
    )
    parser.add_argument(
        "--figure-dir",
        default="results/figures/molhiv_get_mini",
        help="Directory for the sequence-length histogram.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "pdf"],
        help="Figure formats to write.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=60,
        help="Number of histogram bins.",
    )
    parser.add_argument(
        "--max-graphs",
        type=int,
        default=None,
        help="Optional cap for quick smoke tests.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed used for deterministic tokenization diagnostics.",
    )
    return parser.parse_args()


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(config_name: str):
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=str(REPO_ROOT / "configs"),
        version_base=None,
    ):
        cfg = compose(config_name=config_name)

    cfg.training.pretrain_mode = True
    cfg.training.do_valid = False
    cfg.training.do_test = False
    cfg.tokenization.data.return_valid_test = False
    base_configs.update_odps_cfg_from_token_cfg(cfg, "train")
    base_configs.init_stacked_feat(cfg)
    base_configs.init_embed_dim(cfg)
    base_configs.sync_config(cfg)
    return cfg


def build_tokenizer(cfg):
    tokenizer_config = conf_utils.convert_to_legacy_tokenization_config(cfg)
    vocab_path = (
        Path(tokenizer_config["name_or_path"])
        / tokenizer_config.get("vocab_file", "vocab")
    )
    if not vocab_path.is_file():
        raise FileNotFoundError(
            f"Missing GET vocabulary at {vocab_path}. Run pretraining setup first."
        )

    tokenizer_cls = getattr(tokenizer, tokenizer_config["tokenizer_class"])
    return tokenizer_cls(
        tokenizer_config,
        add_eos=cfg.tokenization.add_eos,
        stack_method=cfg.model.graph_input.stack_method,
        train_cfg=cfg.training,
    )


def iter_progress(items, total: int):
    try:
        from tqdm import tqdm

        return tqdm(items, total=total, desc="tokenizing molhiv train graphs")
    except Exception:
        return items


def summarize(values: list[int]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    percentiles = np.percentile(arr, [5, 25, 50, 75, 95, 99])
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std_population": float(arr.std(ddof=0)),
        "std_sample": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": int(arr.min()),
        "p05": float(percentiles[0]),
        "p25": float(percentiles[1]),
        "median": float(percentiles[2]),
        "p75": float(percentiles[3]),
        "p95": float(percentiles[4]),
        "p99": float(percentiles[5]),
        "max": int(arr.max()),
        "total": int(arr.sum()),
    }


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "graph_idx",
        "num_nodes",
        "num_edges",
        "raw_get_tokens",
        "pretrain_input_tokens",
    ]
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_histogram(
    rows: list[dict],
    summary: dict,
    figure_dir: Path,
    formats: list[str],
    bins: int,
):
    figure_dir.mkdir(parents=True, exist_ok=True)
    lengths = np.asarray([row["raw_get_tokens"] for row in rows], dtype=np.float64)
    mean = summary["raw_get_tokens"]["mean"]
    median = summary["raw_get_tokens"]["median"]

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

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.hist(lengths, bins=bins, color="#4c78a8", alpha=0.82, edgecolor="white")
    ax.axvline(mean, color="#f58518", lw=2, label=f"mean={mean:.1f}")
    ax.axvline(
        median,
        color="#54a24b",
        lw=2,
        linestyle="--",
        label=f"median={median:.1f}",
    )
    ax.set_title("GET Sequence Lengths on ogbg-molhiv Train Split")
    ax.set_xlabel("GET tokens per graph (unpadded, unpacked)")
    ax.set_ylabel("Number of training graphs")
    ax.legend(loc="upper right")

    raw = summary["raw_get_tokens"]
    pretrain = summary["pretrain_input_tokens"]
    text = (
        f"n={raw['count']:,}\n"
        f"raw mean={raw['mean']:.1f}\n"
        f"raw p95={raw['p95']:.1f}\n"
        f"raw max={raw['max']:,}\n"
        f"SMTP mean={pretrain['mean']:.1f}"
    )
    ax.text(
        0.98,
        0.72,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.88},
    )

    paths = []
    for fmt in formats:
        path = figure_dir / f"sequence_length_histogram.{fmt}"
        fig.savefig(path, bbox_inches="tight")
        paths.append(
            str(path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path)
        )
    plt.close(fig)
    return paths


def main() -> None:
    args = parse_args()
    os.chdir(REPO_ROOT)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = load_config(args.config_name)
    gtokenizer = build_tokenizer(cfg)
    train_dataset, _ = read_dataset(
        name=cfg.tokenization.data.dataset,
        data_cfg=cfg.tokenization.data,
        train_cfg=cfg.training,
    )

    train_indices = sorted(train_dataset.sample_idx.tolist())
    if args.max_graphs is not None:
        train_indices = train_indices[: args.max_graphs]

    rows = []
    for graph_idx in iter_progress(train_indices, len(train_indices)):
        idx, graph = train_dataset[int(graph_idx)]
        token_res = gtokenizer.tokenize(graph)
        raw_get_tokens = len(token_res.ls_tokens)
        rows.append(
            {
                "graph_idx": int(idx),
                "num_nodes": int(graph.num_nodes),
                "num_edges": int(graph.edge_index.shape[1]),
                "raw_get_tokens": int(raw_get_tokens),
                "pretrain_input_tokens": int(raw_get_tokens + 1),
            }
        )

    output_csv = repo_path(args.output_csv)
    output_json = repo_path(args.output_json)
    figure_dir = repo_path(args.figure_dir)

    raw_values = [row["raw_get_tokens"] for row in rows]
    pretrain_values = [row["pretrain_input_tokens"] for row in rows]
    summary = {
        "dataset": cfg.tokenization.data.dataset,
        "split": "train",
        "config_name": args.config_name,
        "tokenizer_class": cfg.tokenization.tokenizer_class,
        "stack_method": cfg.model.graph_input.stack_method,
        "count_is_full_train_split": args.max_graphs is None,
        "pretrain_appends_eos": True,
        "length_definitions": {
            "raw_get_tokens": (
                "len(StackedGSTTokenizer.tokenize(graph).ls_tokens), before padding, "
                "packing, masking, or task preparation"
            ),
            "pretrain_input_tokens": (
                "raw_get_tokens + 1 for the EOS token appended by SMTP pretraining"
            ),
        },
        "raw_get_tokens": summarize(raw_values),
        "pretrain_input_tokens": summarize(pretrain_values),
    }

    write_csv(rows, output_csv)
    figure_paths = write_histogram(rows, summary, figure_dir, args.formats, args.bins)
    summary["outputs"] = {
        "csv": str(
            output_csv.relative_to(REPO_ROOT)
            if output_csv.is_relative_to(REPO_ROOT)
            else output_csv
        ),
        "figures": figure_paths,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w") as fp:
        json.dump(summary, fp, indent=2, sort_keys=True)

    print("GET sequence length summary for ogbg-molhiv train split")
    print(f"graphs: {summary['raw_get_tokens']['count']}")
    print(
        "raw GET tokens mean +/- std: "
        f"{summary['raw_get_tokens']['mean']:.3f} +/- "
        f"{summary['raw_get_tokens']['std_population']:.3f}"
    )
    print(
        "SMTP input tokens mean +/- std: "
        f"{summary['pretrain_input_tokens']['mean']:.3f} +/- "
        f"{summary['pretrain_input_tokens']['std_population']:.3f}"
    )
    print(f"wrote {summary['outputs']['csv']}")
    print(
        f"wrote "
        f"{output_json.relative_to(REPO_ROOT) if output_json.is_relative_to(REPO_ROOT) else output_json}"
    )
    for path in figure_paths:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
