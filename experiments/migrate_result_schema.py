from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pandas as pd

from common import FINAL_COLUMNS, GENERATION_COLUMNS, REPO_ROOT, cli_function_set, load_config
from run_paper_full_suite import suite_defaults


def fill_metadata(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    function_set = cli_function_set(config)
    if "split_seed" not in frame:
        frame["split_seed"] = config.get("data_split_seed", 0)
    else:
        missing_split_seed = frame["split_seed"].isna() | (frame["split_seed"].astype(str) == "")
        frame.loc[missing_split_seed, "split_seed"] = config.get("data_split_seed", 0)
    for column in [
        "fitness_function",
        "function_set",
        "time_limit_seconds",
        "generations",
        "coefficient_mutation_probability",
    ]:
        if column not in frame:
            frame[column] = ""

    if "suite" not in frame:
        return frame
    if {"backend", "avg_actual_gpu_batch_size"}.issubset(frame.columns):
        frame.loc[frame["backend"] == "cpu_original", "avg_actual_gpu_batch_size"] = float("nan")
        frame.loc[frame["backend"] == "gpu_exact_gom", "avg_actual_gpu_batch_size"] = 1.0
    for suite in frame["suite"].dropna().astype(str).unique():
        try:
            defaults = suite_defaults(suite, config)
        except KeyError:
            continue
        mask = frame["suite"].astype(str) == suite
        values = {
            "fitness_function": defaults.get("fitness_function", config.get("fitness_function", "mse")),
            "function_set": function_set,
            "time_limit_seconds": defaults.get("time_limit_seconds", config.get("time_limit_seconds", "")),
            "generations": defaults.get("generations", config.get("generations", "")),
            "coefficient_mutation_probability": defaults.get(
                "coefficient_mutation_probability", config.get("coefficient_mutation_probability", "")
            ),
        }
        for column, value in values.items():
            missing = frame[column].isna() | (frame[column].astype(str) == "")
            frame.loc[mask & missing, column] = value
    return frame


def migrate(path: Path, config: dict) -> bool:
    try:
        frame = pd.read_csv(path)
    except (pd.errors.EmptyDataError, UnicodeDecodeError):
        return False
    if "run_id" not in frame:
        return False

    is_generation = "generation" in frame.columns or path.name.endswith("_generation.csv")
    columns = GENERATION_COLUMNS if is_generation else FINAL_COLUMNS
    frame = fill_metadata(frame, config)
    for column in columns:
        if column not in frame:
            frame[column] = ""
    frame = frame[columns]
    dedupe_keys = ["run_id", "generation"] if is_generation else ["run_id"]
    frame = frame.drop_duplicates(dedupe_keys, keep="last")

    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate generated benchmark CSV files to the current result schema.")
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    args = parser.parse_args()
    config = load_config()
    migrated = 0
    for path in sorted(args.results_dir.rglob("*.csv")):
        if migrate(path, config):
            migrated += 1
            print(f"[migrated] {path}")
    print(f"Migrated {migrated} result CSV files.")


if __name__ == "__main__":
    main()
