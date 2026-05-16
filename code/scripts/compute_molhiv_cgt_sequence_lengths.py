#!/usr/bin/env python
"""Measure CGT sequence lengths on the ogbg-molhiv train split."""

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
        default="experiments/molhiv_cgt_mini_pretrain",
        help="Hydra config to load from configs/.",
    )
    parser.add_argument(
        "--output-csv",
        default="results/molhiv_cgt_sequence_lengths.csv",
        help="Per-graph length table to write.",
    )
    parser.add_argument(
        "--output-json",
        default="results/molhiv_cgt_sequence_lengths.json",
        help="Summary JSON to write.",
    )
    parser.add_argument(
        "--figure-dir",
        default="results/figures/molhiv_cgt_mini",
        help="Directory for the sequence-length histogram.",
    )
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    parser.add_argument("--bins", type=int, default=60)
    parser.add_argument("--max-graphs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(config_name: str):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base=None):
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
            f"Missing CGT vocabulary at {vocab_path}. Run pretraining setup first."
        )
    tokenizer_cls = getattr(tokenizer, tokenizer_config["tokenizer_class"])
    return tokenizer_cls(
        tokenizer_config,
        add_eos=cfg.tokenization.add_eos,
        stack_method=cfg.model.graph_input.stack_method,
        train_cfg=cfg.training,
    )


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


def write_histogram(rows, summary, figure_dir: Path, formats: list[str], bins: int):
    figure_dir.mkdir(parents=True, exist_ok=True)
    lengths = np.asarray([row["raw_cgt_tokens"] for row in rows], dtype=np.float64)
    mean = summary["raw_cgt_tokens"]["mean"]
    median = summary["raw_cgt_tokens"]["median"]
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("default")

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.hist(lengths, bins=bins, color="#59a14f", alpha=0.82, edgecolor="white")
    ax.axvline(mean, color="#f28e2b", lw=2, label=f"mean={mean:.1f}")
    ax.axvline(median, color="#4e79a7", lw=2, linestyle="--", label=f"median={median:.1f}")
    ax.set_title("CGT Sequence Lengths on ogbg-molhiv Train Split")
    ax.set_xlabel("CGT tokens per graph (unpadded, unpacked)")
    ax.set_ylabel("Number of training graphs")
    ax.legend(loc="upper right")
    paths = []
    for fmt in formats:
        path = figure_dir / f"sequence_length_histogram.{fmt}"
        fig.savefig(path, bbox_inches="tight")
        paths.append(str(path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path))
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
    for graph_idx in train_indices:
        idx, graph = train_dataset[int(graph_idx)]
        token_res = gtokenizer.tokenize(graph)
        raw_tokens = len(token_res.ls_tokens)
        rows.append(
            {
                "graph_idx": int(idx),
                "num_nodes": int(graph.num_nodes),
                "num_edges": int(graph.edge_index.shape[1]),
                "raw_cgt_tokens": int(raw_tokens),
                "pretrain_input_tokens": int(raw_tokens + 1),
            }
        )

    output_csv = repo_path(args.output_csv)
    output_json = repo_path(args.output_json)
    figure_dir = repo_path(args.figure_dir)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "dataset": cfg.tokenization.data.dataset,
        "split": "train",
        "config_name": args.config_name,
        "tokenizer_class": cfg.tokenization.tokenizer_class,
        "stack_method": cfg.model.graph_input.stack_method,
        "count_is_full_train_split": args.max_graphs is None,
        "pretrain_appends_eos": True,
        "length_definitions": {
            "raw_cgt_tokens": (
                "len(CanonicalStackedGSTTokenizer.tokenize(graph).ls_tokens), before "
                "padding, packing, or task preparation"
            ),
            "pretrain_input_tokens": (
                "raw_cgt_tokens + 1 for the terminal EOS input appended by NTP setup"
            ),
        },
        "raw_cgt_tokens": summarize([row["raw_cgt_tokens"] for row in rows]),
        "pretrain_input_tokens": summarize([row["pretrain_input_tokens"] for row in rows]),
    }
    summary["outputs"] = {
        "csv": str(output_csv.relative_to(REPO_ROOT) if output_csv.is_relative_to(REPO_ROOT) else output_csv),
        "figures": write_histogram(rows, summary, figure_dir, args.formats, args.bins),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w") as fp:
        json.dump(summary, fp, indent=2, sort_keys=True)

    print("CGT sequence length summary for ogbg-molhiv train split")
    print(f"graphs: {summary['raw_cgt_tokens']['count']}")
    print(
        "raw CGT tokens mean +/- std: "
        f"{summary['raw_cgt_tokens']['mean']:.3f} +/- "
        f"{summary['raw_cgt_tokens']['std_population']:.3f}"
    )
    print(f"wrote {summary['outputs']['csv']}")
    print(f"wrote {output_json}")
    for path in summary["outputs"]["figures"]:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
