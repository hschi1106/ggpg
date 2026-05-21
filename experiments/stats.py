from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

from common import REPO_ROOT


METRICS = ["train_nmse", "validation_nmse", "test_nmse", "wall_clock_time", "evals_per_sec"]
STATS_COLUMNS = [
    "dataset",
    "metric",
    "method_a",
    "method_b",
    "median_a",
    "median_b",
    "p_value",
    "significant",
    "winner",
]


def paired_wilcoxon(df: pd.DataFrame, metric: str, method_a: str, method_b: str) -> list[dict]:
    rows = []
    for dataset, dset_df in df.groupby("dataset"):
        a = dset_df[dset_df["backend"] == method_a].sort_values("seed")
        b = dset_df[dset_df["backend"] == method_b].sort_values("seed")
        paired = pd.merge(a[["seed", metric]], b[["seed", metric]], on="seed", suffixes=("_a", "_b")).dropna()
        if len(paired) < 2:
            continue
        try:
            stat = wilcoxon(paired[f"{metric}_a"], paired[f"{metric}_b"])
        except ValueError:
            continue
        median_a = paired[f"{metric}_a"].median()
        median_b = paired[f"{metric}_b"].median()
        rows.append({
            "dataset": dataset,
            "metric": metric,
            "method_a": method_a,
            "method_b": method_b,
            "median_a": median_a,
            "median_b": median_b,
            "p_value": stat.pvalue,
            "significant": False,
            "winner": method_a if median_a < median_b else method_b,
        })
    return rows


def apply_bonferroni(rows: list[dict]) -> list[dict]:
    if not rows:
        return rows
    alpha = 0.05 / len(rows)
    for row in rows:
        row["significant"] = bool(row["p_value"] <= alpha)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute paper-style paired significance summaries.")
    parser.add_argument("--input", type=Path, default=REPO_ROOT / "results" / "final_results.csv")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "results" / "stats" / "significance_summary.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if "wall_clock_time" not in df and "elapsed_sec" in df:
        df["wall_clock_time"] = df["elapsed_sec"]

    methods = [m for m in ["cpu_original", "gpu_exact_gom", "gpu_batch_gom"] if m in set(df["backend"].dropna())]
    rows = []
    for metric in METRICS:
        if metric not in df:
            continue
        for i, method_a in enumerate(methods):
            for method_b in methods[i + 1:]:
                rows.extend(paired_wilcoxon(df, metric, method_a, method_b))

    rows = apply_bonferroni(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=STATS_COLUMNS).to_csv(args.output, index=False)

    if len(methods) >= 3 and "test_nmse" in df:
        pivot = df.pivot_table(index=["dataset", "seed"], columns="backend", values="test_nmse")
        available = [m for m in methods if m in pivot]
        complete = pivot[available].dropna()
        if len(complete) >= 2:
            stat = friedmanchisquare(*[complete[m] for m in available])
            print(f"Friedman test on test_nmse over {available}: statistic={stat.statistic}, p={stat.pvalue}")


if __name__ == "__main__":
    main()
