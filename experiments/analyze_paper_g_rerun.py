from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from common import REPO_ROOT, slugify


DATASETS = ["Airfoil", "Dow chemical", "Wine white", "Yacht hydrodynamics"]
PAPER_G_L31_NPOP1000 = {
    "Airfoil": {"validation": 26.40, "test": 26.40},
    "Dow chemical": {"validation": 21.30, "test": 20.30},
    "Wine white": {"validation": 70.20, "test": 69.00},
    "Yacht hydrodynamics": {"validation": 0.46, "test": 0.52},
}


def expected_run_ids(datasets: list[str], seeds: list[int], tree_height: int) -> set[str]:
    return {
        f"cpu_original_{slugify(dataset)}_seed{seed}_h{tree_height}"
        for dataset in datasets
        for seed in seeds
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize selected local Paper-G reruns for the report.")
    parser.add_argument("--input", type=Path, default=REPO_ROOT / "results" / "final_results.csv")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "report")
    parser.add_argument("--datasets", nargs="*", default=DATASETS)
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--tree-height", type=int, default=4)
    parser.add_argument("--population-size", type=int, default=1000)
    parser.add_argument("--time-limit", type=int, default=60)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    run_ids = expected_run_ids(args.datasets, args.seeds, args.tree_height)
    rerun = df[df["run_id"].isin(run_ids)].copy()
    if rerun.empty:
        raise SystemExit("No selected Paper-G rerun rows found in final_results.csv")

    # final_results.csv is append-only; if a selected run was repeated, use the most recent row.
    rerun = rerun.drop_duplicates("run_id", keep="last")
    rerun = rerun[
        (rerun["backend"] == "cpu_original")
        & (rerun["tree_height"] == args.tree_height)
        & (rerun["population_size"] == args.population_size)
    ].copy()
    rerun["time_limit_seconds"] = args.time_limit
    for metric in ["train_nmse", "validation_nmse", "test_nmse"]:
        rerun[f"{metric}_paper_scaled"] = 100.0 * rerun[metric]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs_path = args.output_dir / "paper_g_rerun_runs.csv"
    rerun.to_csv(runs_path, index=False)

    summary = rerun.groupby("dataset", dropna=False).agg(
        local_runs=("run_id", "count"),
        local_median_elapsed_sec=("elapsed_sec", "median"),
        local_median_evals_per_sec=("evals_per_sec", "median"),
        local_train_nmse=("train_nmse_paper_scaled", "median"),
        local_validation_nmse=("validation_nmse_paper_scaled", "median"),
        local_test_nmse=("test_nmse_paper_scaled", "median"),
    ).reset_index()
    summary.insert(1, "tree_height", args.tree_height)
    summary.insert(2, "max_nodes", 2 ** (args.tree_height + 1) - 1)
    summary.insert(3, "population_size", args.population_size)
    summary.insert(4, "time_limit_seconds", args.time_limit)
    summary["paper_g_validation_nmse"] = summary["dataset"].map(
        lambda name: PAPER_G_L31_NPOP1000[name]["validation"]
    )
    summary["paper_g_test_nmse"] = summary["dataset"].map(
        lambda name: PAPER_G_L31_NPOP1000[name]["test"]
    )
    summary["local_minus_paper_test_nmse"] = summary["local_test_nmse"] - summary["paper_g_test_nmse"]
    summary["local_over_paper_test_nmse"] = summary["local_test_nmse"] / summary["paper_g_test_nmse"]
    summary = summary[
        [
            "dataset",
            "tree_height",
            "max_nodes",
            "population_size",
            "time_limit_seconds",
            "local_runs",
            "paper_g_validation_nmse",
            "paper_g_test_nmse",
            "local_validation_nmse",
            "local_test_nmse",
            "local_minus_paper_test_nmse",
            "local_over_paper_test_nmse",
            "local_median_evals_per_sec",
            "local_median_elapsed_sec",
        ]
    ]
    summary_path = args.output_dir / "paper_g_rerun_comparison.csv"
    summary.to_csv(summary_path, index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(summary))
    width = 0.36
    ax.bar([i - width / 2 for i in x], summary["paper_g_test_nmse"], width, label="paper G")
    ax.bar([i + width / 2 for i in x], summary["local_test_nmse"], width, label="local rerun")
    ax.set_xticks(list(x))
    ax.set_xticklabels(summary["dataset"], rotation=20, ha="right")
    ax.set_ylabel("Test NMSE, paper scale")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output_dir / "paper_g_rerun_test_nmse.png", dpi=160)
    plt.close(fig)

    print(f"wrote {runs_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
