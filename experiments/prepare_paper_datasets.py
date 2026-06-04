from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd

from common import REPO_ROOT, load_config, processed_dataset_dir, slugify


def read_raw_dataset(dataset: str) -> tuple[pd.DataFrame, pd.Series]:
    raw_dir = REPO_ROOT / "data" / "raw"
    slug = slugify(dataset)
    candidates = list(raw_dir.glob(f"{slug}.*"))
    if not candidates and dataset in {"Energy cooling", "Energy heating"}:
        candidates = list(raw_dir.glob("energy.*"))
    if not candidates:
        raise FileNotFoundError(f"No raw file found for {dataset}. Expected data/raw/{slug}.*")

    path = candidates[0]
    if dataset == "Boston housing":
        text = path.read_text(encoding="utf-8")
        values = [float(v) for v in re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", "\n".join(text.splitlines()[22:]))]
        if len(values) % 14 != 0:
            raise ValueError(f"Could not parse Boston housing data from {path}")
        frame = pd.DataFrame(np.asarray(values, dtype=float).reshape(-1, 14))
    elif dataset == "Dow chemical":
        sheets = pd.read_excel(path, sheet_name=None)
        frame = pd.concat(sheets.values(), ignore_index=True)
    elif path.suffix in {".xls", ".xlsx"}:
        frame = pd.read_excel(path)
    elif dataset == "Tower":
        frame = pd.read_csv(path, header=None)
    elif "winequality" in path.name or path.suffix == ".csv":
        frame = pd.read_csv(path, sep=";" if "wine" in path.name else None, engine="python")
    else:
        frame = pd.read_csv(path, sep=r"\s+", header=None)

    frame = frame.dropna(axis=0)
    if dataset == "Energy heating":
        target = frame.iloc[:, -2]
        X = frame.iloc[:, :-2]
    elif dataset == "Energy cooling":
        target = frame.iloc[:, -1]
        X = frame.iloc[:, :-2]
    else:
        target = frame.iloc[:, -1]
        X = frame.iloc[:, :-1]
    return X.astype(float), target.astype(float)


def write_split(dataset: str, seed: int, split_cfg: dict) -> None:
    X_frame, y_series = read_raw_dataset(dataset)
    X = X_frame.to_numpy(dtype=float)
    y = y_series.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    X = X[order]
    y = y[order]

    n = len(y)
    n_train = int(round(n * float(split_cfg["train"])))
    n_val = int(round(n * float(split_cfg["validation"])))
    idx = {
        "train": slice(0, n_train),
        "val": slice(n_train, n_train + n_val),
        "test": slice(n_train + n_val, n),
    }

    out_dir = processed_dataset_dir(dataset)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, sl in idx.items():
        pd.DataFrame(X[sl]).to_csv(out_dir / f"X_{split}.csv", index=False)
        pd.DataFrame({"y": y[sl]}).to_csv(out_dir / f"y_{split}.csv", index=False)
    print(f"wrote {dataset} -> {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare deterministic paper-style train/validation/test splits.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "experiments" / "paper_reproduction_config.yaml")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    datasets = args.datasets or cfg["datasets"]
    seed = args.seed if args.seed is not None else int(cfg.get("data_split_seed", 0))
    for dataset in datasets:
        write_split(dataset, seed, cfg["train_validation_test_split"])


if __name__ == "__main__":
    main()
