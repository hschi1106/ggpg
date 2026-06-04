from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from common import REPO_ROOT


def grouping_keys(df: pd.DataFrame) -> list[str]:
    candidates = ["suite", "dataset", "row_scale", "tree_height", "population_size"]
    return [key for key in candidates if key in df.columns]


def write_table_cpu_vs_gpu_exact(df: pd.DataFrame, out: Path) -> None:
    subset = df[df["backend"].isin(["cpu_original", "gpu_exact_gom"])]
    table = subset.groupby([*grouping_keys(df), "backend"], dropna=False).agg(
        median_test_nmse=("test_nmse", "median"),
        median_elapsed_sec=("elapsed_sec", "median"),
        median_evals_per_sec=("evals_per_sec", "median"),
    ).reset_index()
    table.to_csv(out, index=False)


def write_table_batch_ablation(df: pd.DataFrame, out: Path) -> None:
    subset = df[df["backend"] == "gpu_batch_gom"]
    table = subset.groupby([*grouping_keys(df), "gpu_batch_size"], dropna=False).agg(
        median_test_nmse=("test_nmse", "median"),
        median_elapsed_sec=("elapsed_sec", "median"),
        median_evals_per_sec=("evals_per_sec", "median"),
    ).reset_index()
    table.to_csv(out, index=False)


def write_paper_style_nmse(df: pd.DataFrame, out: Path) -> None:
    table = df.groupby([*grouping_keys(df), "backend"], dropna=False).agg(
        train_nmse=("train_nmse", "median"),
        validation_nmse=("validation_nmse", "median"),
        test_nmse=("test_nmse", "median"),
    ).reset_index()
    table.to_csv(out, index=False)


def plot_if_columns(df: pd.DataFrame, x: str, y: str, hue: str, out: Path) -> None:
    if x not in df or y not in df or hue not in df:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    for key, group in df.dropna(subset=[x, y]).groupby(hue):
        group = group.sort_values(x)
        ax.plot(group[x], group[y], marker="o", label=str(key))
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate comparison tables and figures from final_results.csv.")
    parser.add_argument("--input", type=Path, default=REPO_ROOT / "results" / "final_results.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    tables = REPO_ROOT / "results" / "tables"
    figs = REPO_ROOT / "results" / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    write_table_cpu_vs_gpu_exact(df, tables / "table_cpu_vs_gpu_exact.csv")
    write_table_batch_ablation(df, tables / "table_batch_size_ablation.csv")
    write_paper_style_nmse(df, tables / "table_paper_style_nmse.csv")

    plot_if_columns(df, "elapsed_sec", "validation_nmse", "backend", figs / "validation_nmse_vs_time.png")
    plot_if_columns(df[df["backend"] == "gpu_batch_gom"], "gpu_batch_size", "test_nmse", "dataset", figs / "test_nmse_vs_batch_size.png")
    plot_if_columns(df[df["backend"] == "gpu_batch_gom"], "gpu_batch_size", "evals_per_sec", "dataset", figs / "evals_per_sec_vs_batch_size.png")

    if {"backend", "gpu_batch_size", "elapsed_sec"}.issubset(df.columns):
        keys = grouping_keys(df)
        cpu = df[df["backend"] == "cpu_original"].groupby(keys, dropna=False)["elapsed_sec"].median().reset_index()
        cpu = cpu.rename(columns={"elapsed_sec": "cpu_elapsed_sec"})
        batch = df[df["backend"] == "gpu_batch_gom"].copy()
        if not batch.empty:
            batch = batch.merge(cpu, on=keys, how="left")
            batch["speedup"] = batch["cpu_elapsed_sec"] / batch["elapsed_sec"]
            plot_if_columns(batch, "gpu_batch_size", "speedup", "dataset", figs / "speedup_vs_batch_size.png")
            if "suite" in batch:
                plot_if_columns(batch[batch["suite"] == "acceleration"], "population_size", "speedup", "dataset", figs / "acceleration_speedup_vs_population.png")
                plot_if_columns(batch[batch["suite"] == "exact-row-scale"], "row_scale", "speedup", "dataset", figs / "exact_row_scale_speedup.png")


if __name__ == "__main__":
    main()
