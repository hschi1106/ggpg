from __future__ import annotations

import argparse
import csv
import math
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import sympy as sp
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "experiments" / "paper_reproduction_config.yaml"
FUNCTION_SET_SYMBOLS = {
    "+": "+",
    "-": "-",
    "¬": "¬",
    "*": "*",
    "/": "/",
    "aq": "aq",
    "AQ": "aq",
    "analytic_quotient": "aq",
    "analytic-quotient": "aq",
    "1/": "1/",
    "**2": "**2",
    "sqrt": "sqrt",
    "**3": "**3",
    "sin": "sin",
    "cos": "cos",
    "log": "log",
}
FINAL_COLUMNS = [
    "run_id",
    "dataset",
    "backend",
    "seed",
    "tree_height",
    "max_nodes",
    "population_size",
    "ims_g",
    "gpu_batch_size",
    "elapsed_sec",
    "train_nmse",
    "validation_nmse",
    "test_nmse",
    "num_evaluations",
    "num_gpu_evaluations",
    "evals_per_sec",
    "best_expression",
]
GENERATION_COLUMNS = [
    "run_id",
    "dataset",
    "backend",
    "seed",
    "generation",
    "elapsed_sec",
    "tree_height",
    "max_nodes",
    "population_size",
    "ims_g",
    "gpu_batch_size",
    "best_train_fitness",
    "best_validation_fitness",
    "best_train_nmse",
    "best_validation_nmse",
    "num_evaluations",
    "num_gpu_evaluations",
    "num_cpu_evaluations",
    "num_accepted_moves",
    "num_rejected_moves",
    "num_meaningful_candidates",
    "num_nonmeaningful_candidates",
    "avg_actual_gpu_batch_size",
]


def load_config(path: Path = CONFIG_PATH) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def cli_function_set(config: dict | None = None) -> str:
    config = config if config is not None else load_config()
    configured = config.get("function_set", ["+", "-", "*", "aq"])
    if isinstance(configured, str):
        return configured
    symbols = []
    for name in configured:
        key = str(name).strip()
        try:
            symbols.append(FUNCTION_SET_SYMBOLS[key])
        except KeyError as exc:
            raise ValueError(f"Unsupported upstream gpg function_set symbol in paper config: {name}") from exc
    return ",".join(symbols)


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def find_paper(config: dict) -> Path | None:
    configured = config.get("paper_path")
    candidates = []
    if configured:
        candidates.append(REPO_ROOT / configured)
    candidates.extend((REPO_ROOT / "paper").glob("*.pdf"))
    candidates.extend((REPO_ROOT / "docs").glob("*.pdf"))
    for path in candidates:
        if path.exists():
            return path
    return None


def ensure_csv_header(path: Path, columns: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(list(columns))


def append_csv_row(path: Path, row: dict, columns: Iterable[str]) -> None:
    ensure_csv_header(path, columns)
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns))
        writer.writerow({col: row.get(col, "") for col in columns})


def processed_dataset_dir(dataset: str) -> Path:
    return REPO_ROOT / "data" / "processed" / slugify(dataset)


def load_processed_split(dataset: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    root = processed_dataset_dir(dataset)
    x_path = root / f"X_{split}.csv"
    y_path = root / f"y_{split}.csv"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"Missing processed {split} split for {dataset}: {root}")
    X = pd.read_csv(x_path).to_numpy(dtype=float)
    y = pd.read_csv(y_path).iloc[:, 0].to_numpy(dtype=float)
    return X, y


def write_gpg_training_csv(dataset: str, seed: int) -> Path:
    X, y = load_processed_split(dataset, "train")
    out_dir = REPO_ROOT / "results" / "tmp"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{slugify(dataset)}_seed{seed}_train.csv"
    frame = pd.DataFrame(X)
    frame["target"] = y
    frame.to_csv(out, index=False, header=False)
    return out


def nmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.mean((y_true - np.mean(y_true)) ** 2)
    if denom <= 0 or not math.isfinite(denom):
        return float("nan")
    return float(np.mean((y_true - y_pred) ** 2) / denom)


def gpg_binary() -> Path:
    env_binary = os.environ.get("GPG_BINARY")
    if env_binary:
        path = Path(env_binary)
        if path.exists():
            return path
    candidates = [
        REPO_ROOT / "build" / "release" / "gpg",
        REPO_ROOT / "build" / "debug" / "gpg",
        REPO_ROOT / "build" / "gpg",
        REPO_ROOT / "build" / "cuda" / "gpg",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find built gpg executable under build/. Run cmake/make first.")


def gpg_binary_for_backend(backend: str) -> Path:
    env_binary = os.environ.get("GPG_BINARY")
    if env_binary:
        path = Path(env_binary)
        if path.exists():
            return path
    if backend.startswith("gpu"):
        candidate = REPO_ROOT / "build" / "cuda" / "gpg"
        if candidate.exists():
            return candidate
    return gpg_binary()


def extract_best_expression(stdout: str) -> str:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    for i, line in enumerate(lines):
        if line.startswith("Best w.r.t. complexity") and i + 1 < len(lines):
            return lines[i + 1]
    return lines[-1] if lines else ""


def parse_counter_line(stdout: str) -> dict:
    pattern = (
        r"Evaluation counters: total=(?P<num_evaluations>[-0-9]+), "
        r"gpu=(?P<num_gpu_evaluations>[-0-9]+), "
        r"cpu=(?P<num_cpu_evaluations>[-0-9]+), "
        r"accepted_moves=(?P<num_accepted_moves>[-0-9]+), "
        r"rejected_moves=(?P<num_rejected_moves>[-0-9]+), "
        r"meaningful_candidates=(?P<num_meaningful_candidates>[-0-9]+), "
        r"nonmeaningful_candidates=(?P<num_nonmeaningful_candidates>[-0-9]+)"
    )
    match = re.search(pattern, stdout)
    if not match:
        return {}
    return {k: int(v) for k, v in match.groupdict().items()}


def parse_generation_rows(stdout: str) -> list[dict]:
    rows = []
    for line in stdout.splitlines():
        match = re.search(r"~ macro generation: (?P<generation>[0-9]+), curr\. best fit: (?P<fitness>[-+0-9.eEinf]+)", line)
        if not match:
            continue
        fitness = match.group("fitness")
        rows.append({
            "generation": int(match.group("generation")),
            "best_train_fitness": fitness,
            "best_train_nmse": fitness,
        })
    return rows


def expression_to_callable(expr: str, n_features: int):
    if not expr:
        return None
    symbols = {f"x_{i}": sp.Symbol(f"x_{i}") for i in range(n_features)}
    safe = {
        **symbols,
        "sin": sp.sin,
        "cos": sp.cos,
        "log": sp.log,
        "sqrt": sp.sqrt,
        "max": sp.Max,
    }
    parsed = sp.sympify(expr, locals=safe)
    fn = sp.lambdify([symbols[f"x_{i}"] for i in range(n_features)], parsed, modules="numpy")
    return fn


def predict_expression(expr: str, X: np.ndarray) -> np.ndarray:
    try:
        fn = expression_to_callable(expr, X.shape[1])
        if fn is None:
            return np.full(X.shape[0], np.nan)
        pred = fn(*[X[:, i] for i in range(X.shape[1])])
        if np.isscalar(pred):
            pred = np.full(X.shape[0], float(pred))
        pred = np.asarray(pred, dtype=float)
        if pred.shape == ():
            pred = np.full(X.shape[0], float(pred))
        if pred.shape[0] != X.shape[0]:
            pred = np.full(X.shape[0], float(pred.flat[0]) if pred.size else np.nan)
        pred[~np.isfinite(pred)] = np.nan
        return pred
    except Exception:
        return np.full(X.shape[0], np.nan)


def evaluate_expression_on_splits(dataset: str, expr: str) -> dict:
    metrics = {}
    for split, key in [("train", "train_nmse"), ("val", "validation_nmse"), ("test", "test_nmse")]:
        X, y = load_processed_split(dataset, split)
        pred = predict_expression(expr, X)
        metrics[key] = nmse(y, pred) if np.isfinite(pred).any() else float("nan")
    return metrics


def run_gpg_cli(
    *,
    dataset: str,
    backend: str,
    seed: int,
    tree_height: int,
    population_size: int,
    time_limit_seconds: int,
    gpu_batch_size: int,
    disable_ims: bool = True,
    ims_g: int | None = None,
    function_set: str | None = None,
) -> dict:
    train_csv = write_gpg_training_csv(dataset, seed)
    function_set = function_set or cli_function_set()
    cmd = [
        str(gpg_binary_for_backend(backend)),
        "-train", str(train_csv),
        "-backend", backend,
        "-ff", "mse",
        "-fset", function_set,
        "-d", str(tree_height),
        "-pop", str(population_size),
        "-t", str(time_limit_seconds),
        "-g", "-1",
        "-random_state", str(seed),
        "-feat_sel", "-1",
        "-gpu_batch_size", str(gpu_batch_size),
        "-verbose",
    ]
    if disable_ims:
        cmd.append("-disable_ims")
    elif ims_g is not None:
        cmd.extend(["-ims_g", str(ims_g)])

    started = time.time()
    proc = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    elapsed = time.time() - started
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or proc.stdout)
    counters = parse_counter_line(proc.stdout)
    best_expression = extract_best_expression(proc.stdout)
    split_metrics = evaluate_expression_on_splits(dataset, best_expression)
    num_evaluations = counters.get("num_evaluations", 0)
    return {
        "elapsed_sec": elapsed,
        "best_expression": best_expression,
        "evals_per_sec": num_evaluations / elapsed if elapsed > 0 else float("nan"),
        **counters,
        **split_metrics,
        "generation_rows": parse_generation_rows(proc.stdout),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def write_run_outputs(
    *,
    result: dict,
    run_id: str,
    dataset: str,
    backend: str,
    seed: int,
    tree_height: int,
    max_nodes: int | str,
    population_size: int,
    ims_g: int | str,
    gpu_batch_size: int,
    backend_csv: Path,
) -> None:
    row = {
        "run_id": run_id,
        "dataset": dataset,
        "backend": backend,
        "seed": seed,
        "tree_height": tree_height,
        "max_nodes": max_nodes,
        "population_size": population_size,
        "ims_g": ims_g,
        "gpu_batch_size": gpu_batch_size,
        "elapsed_sec": result.get("elapsed_sec", ""),
        "train_nmse": result.get("train_nmse", ""),
        "validation_nmse": result.get("validation_nmse", ""),
        "test_nmse": result.get("test_nmse", ""),
        "num_evaluations": result.get("num_evaluations", ""),
        "num_gpu_evaluations": result.get("num_gpu_evaluations", ""),
        "evals_per_sec": result.get("evals_per_sec", ""),
        "best_expression": result.get("best_expression", ""),
    }
    append_csv_row(backend_csv, row, FINAL_COLUMNS)
    append_csv_row(REPO_ROOT / "results" / "final_results.csv", row, FINAL_COLUMNS)

    gen_path = REPO_ROOT / "results" / "logs" / f"{run_id}_generation.csv"
    counters = {
        "num_evaluations": result.get("num_evaluations", ""),
        "num_gpu_evaluations": result.get("num_gpu_evaluations", ""),
        "num_cpu_evaluations": result.get("num_cpu_evaluations", ""),
        "num_accepted_moves": result.get("num_accepted_moves", ""),
        "num_rejected_moves": result.get("num_rejected_moves", ""),
        "num_meaningful_candidates": result.get("num_meaningful_candidates", ""),
        "num_nonmeaningful_candidates": result.get("num_nonmeaningful_candidates", ""),
        "avg_actual_gpu_batch_size": gpu_batch_size if backend == "gpu_batch_gom" else "",
    }
    for gen_row in result.get("generation_rows", []) or [{"generation": ""}]:
        append_csv_row(gen_path, {
            "run_id": run_id,
            "dataset": dataset,
            "backend": backend,
            "seed": seed,
            "elapsed_sec": result.get("elapsed_sec", ""),
            "tree_height": tree_height,
            "max_nodes": max_nodes,
            "population_size": population_size,
            "ims_g": ims_g,
            "gpu_batch_size": gpu_batch_size,
            "best_validation_nmse": result.get("validation_nmse", ""),
            **gen_row,
            **counters,
        }, GENERATION_COLUMNS)


def base_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=[0])
    parser.add_argument("--tree-heights", nargs="*", type=int, default=None)
    parser.add_argument("--time-limit", type=int, default=None)
    parser.add_argument("--population-size", type=int, default=None)
    return parser
