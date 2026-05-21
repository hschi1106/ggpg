from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from common import REPO_ROOT


KEYS = ["dataset", "seed", "tree_height", "population_size", "time_limit_seconds"]


def median_or_nan(series: pd.Series) -> float:
    return float(series.median()) if len(series.dropna()) else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze batch-GOM fitness distortion relative to gpu_exact_gom.")
    parser.add_argument("--input", type=Path, default=REPO_ROOT / "results" / "report" / "batch_distortion_runs.csv")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "report")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    exact = df[df["backend"] == "gpu_exact_gom"].copy()
    batch = df[df["backend"] == "gpu_batch_gom"].copy()
    exact_cols = KEYS + ["train_nmse", "validation_nmse", "test_nmse", "elapsed_sec", "evals_per_sec"]
    exact = exact[exact_cols].rename(columns={
        "train_nmse": "exact_train_nmse",
        "validation_nmse": "exact_validation_nmse",
        "test_nmse": "exact_test_nmse",
        "elapsed_sec": "exact_elapsed_sec",
        "evals_per_sec": "exact_evals_per_sec",
    })
    paired = batch.merge(exact, on=KEYS, how="inner")

    for metric in ["train_nmse", "validation_nmse", "test_nmse"]:
        paired[f"delta_{metric}"] = paired[metric] - paired[f"exact_{metric}"]
        paired[f"ratio_{metric}"] = paired[metric] / paired[f"exact_{metric}"]
        paired[f"abs_delta_{metric}"] = paired[f"delta_{metric}"].abs()
        paired[f"{metric}_paper_scaled"] = 100.0 * paired[metric]
        paired[f"exact_{metric}_paper_scaled"] = 100.0 * paired[f"exact_{metric}"]
        paired[f"delta_{metric}_paper_scaled"] = 100.0 * paired[f"delta_{metric}"]
        paired[f"abs_delta_{metric}_paper_scaled"] = 100.0 * paired[f"abs_delta_{metric}"]
    paired["speedup_vs_exact"] = paired["exact_elapsed_sec"] / paired["elapsed_sec"]
    paired["eval_rate_ratio_vs_exact"] = paired["evals_per_sec"] / paired["exact_evals_per_sec"]

    pairwise_path = args.output_dir / "batch_distortion_pairwise.csv"
    paired.to_csv(pairwise_path, index=False)

    summary = paired.groupby(["dataset", "gpu_batch_size"], dropna=False).agg(
        runs=("run_id", "count"),
        median_test_nmse=("test_nmse", "median"),
        median_test_nmse_paper_scaled=("test_nmse_paper_scaled", "median"),
        median_exact_test_nmse=("exact_test_nmse", "median"),
        median_exact_test_nmse_paper_scaled=("exact_test_nmse_paper_scaled", "median"),
        median_delta_test_nmse=("delta_test_nmse", "median"),
        median_delta_test_nmse_paper_scaled=("delta_test_nmse_paper_scaled", "median"),
        median_abs_delta_test_nmse=("abs_delta_test_nmse", "median"),
        median_abs_delta_test_nmse_paper_scaled=("abs_delta_test_nmse_paper_scaled", "median"),
        median_ratio_test_nmse=("ratio_test_nmse", "median"),
        median_validation_nmse=("validation_nmse", "median"),
        median_validation_nmse_paper_scaled=("validation_nmse_paper_scaled", "median"),
        median_delta_validation_nmse=("delta_validation_nmse", "median"),
        median_delta_validation_nmse_paper_scaled=("delta_validation_nmse_paper_scaled", "median"),
        median_train_nmse=("train_nmse", "median"),
        median_train_nmse_paper_scaled=("train_nmse_paper_scaled", "median"),
        median_delta_train_nmse=("delta_train_nmse", "median"),
        median_delta_train_nmse_paper_scaled=("delta_train_nmse_paper_scaled", "median"),
        median_elapsed_sec=("elapsed_sec", "median"),
        median_speedup_vs_exact=("speedup_vs_exact", "median"),
        median_eval_rate_ratio_vs_exact=("eval_rate_ratio_vs_exact", "median"),
    ).reset_index()
    summary_path = args.output_dir / "batch_distortion_summary.csv"
    summary.to_csv(summary_path, index=False)

    overall = paired.groupby("gpu_batch_size", dropna=False).agg(
        runs=("run_id", "count"),
        median_delta_test_nmse=("delta_test_nmse", "median"),
        median_delta_test_nmse_paper_scaled=("delta_test_nmse_paper_scaled", "median"),
        median_abs_delta_test_nmse=("abs_delta_test_nmse", "median"),
        median_abs_delta_test_nmse_paper_scaled=("abs_delta_test_nmse_paper_scaled", "median"),
        median_ratio_test_nmse=("ratio_test_nmse", "median"),
        median_delta_validation_nmse=("delta_validation_nmse", "median"),
        median_delta_validation_nmse_paper_scaled=("delta_validation_nmse_paper_scaled", "median"),
        median_speedup_vs_exact=("speedup_vs_exact", "median"),
        median_eval_rate_ratio_vs_exact=("eval_rate_ratio_vs_exact", "median"),
    ).reset_index()
    overall_path = args.output_dir / "batch_distortion_overall.csv"
    overall.to_csv(overall_path, index=False)

    acceleration = paired.groupby("gpu_batch_size", dropna=False).agg(
        runs=("run_id", "count"),
        median_batch_evals_per_sec=("evals_per_sec", "median"),
        median_exact_evals_per_sec=("exact_evals_per_sec", "median"),
        median_eval_rate_ratio_vs_exact=("eval_rate_ratio_vs_exact", "median"),
        min_eval_rate_ratio_vs_exact=("eval_rate_ratio_vs_exact", "min"),
        max_eval_rate_ratio_vs_exact=("eval_rate_ratio_vs_exact", "max"),
        q25_eval_rate_ratio_vs_exact=("eval_rate_ratio_vs_exact", lambda s: s.quantile(0.25)),
        q75_eval_rate_ratio_vs_exact=("eval_rate_ratio_vs_exact", lambda s: s.quantile(0.75)),
        median_speedup_vs_exact_wall_clock=("speedup_vs_exact", "median"),
    ).reset_index().sort_values("gpu_batch_size")
    acceleration["incremental_ratio_vs_previous_batch"] = (
        acceleration["median_eval_rate_ratio_vs_exact"]
        / acceleration["median_eval_rate_ratio_vs_exact"].shift(1)
    )
    acceleration_path = args.output_dir / "batch_acceleration_overall.csv"
    acceleration.to_csv(acceleration_path, index=False)

    dataset_acceleration = paired.groupby(["dataset", "gpu_batch_size"], dropna=False).agg(
        runs=("run_id", "count"),
        median_batch_evals_per_sec=("evals_per_sec", "median"),
        median_exact_evals_per_sec=("exact_evals_per_sec", "median"),
        median_eval_rate_ratio_vs_exact=("eval_rate_ratio_vs_exact", "median"),
        median_speedup_vs_exact_wall_clock=("speedup_vs_exact", "median"),
    ).reset_index().sort_values(["dataset", "gpu_batch_size"])
    dataset_acceleration_path = args.output_dir / "batch_acceleration_by_dataset.csv"
    dataset_acceleration.to_csv(dataset_acceleration_path, index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    for dataset, group in summary.groupby("dataset"):
        group = group.sort_values("gpu_batch_size")
        ax.plot(group["gpu_batch_size"], group["median_delta_test_nmse"], marker="o", label=dataset)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("GPU batch size")
    ax.set_ylabel("Median test NMSE delta vs exact")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output_dir / "batch_distortion_test_delta.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for dataset, group in summary.groupby("dataset"):
        group = group.sort_values("gpu_batch_size")
        ax.plot(group["gpu_batch_size"], group["median_eval_rate_ratio_vs_exact"], marker="o", label=dataset)
    ax.axhline(1, color="black", linewidth=0.8)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("GPU batch size")
    ax.set_ylabel("Median eval/s ratio vs exact")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output_dir / "batch_distortion_eval_rate.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    acceleration = acceleration.sort_values("gpu_batch_size")
    ax.plot(
        acceleration["gpu_batch_size"],
        acceleration["median_eval_rate_ratio_vs_exact"],
        marker="o",
        color="#1f77b4",
        label="median",
    )
    ax.fill_between(
        acceleration["gpu_batch_size"],
        acceleration["q25_eval_rate_ratio_vs_exact"],
        acceleration["q75_eval_rate_ratio_vs_exact"],
        color="#1f77b4",
        alpha=0.18,
        label="IQR",
    )
    ax.axhline(1, color="black", linewidth=0.8)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("GPU batch size")
    ax.set_ylabel("Eval/s ratio vs exact GPU GOM")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output_dir / "batch_acceleration_ratio.png", dpi=160)
    plt.close(fig)

    print(f"wrote {pairwise_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {overall_path}")
    print(f"wrote {acceleration_path}")
    print(f"wrote {dataset_acceleration_path}")


if __name__ == "__main__":
    main()
