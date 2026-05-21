from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd

from common import REPO_ROOT, load_config, run_gpg_cli, slugify


REPORT_COLUMNS = [
    "run_id",
    "dataset",
    "backend",
    "seed",
    "tree_height",
    "max_nodes",
    "population_size",
    "time_limit_seconds",
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


def read_completed(path: Path) -> set[str]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    return set(pd.read_csv(path, usecols=["run_id"])["run_id"].astype(str))


def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow({col: row.get(col, "") for col in REPORT_COLUMNS})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a bounded exact-vs-batch GPU distortion study for the report.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "experiments" / "paper_reproduction_config.yaml")
    parser.add_argument("--datasets", nargs="*", default=["Yacht hydrodynamics", "Airfoil", "Wine white", "Dow chemical"])
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--tree-height", type=int, default=4)
    parser.add_argument("--population-size", type=int, default=128)
    parser.add_argument("--time-limit", type=int, default=5)
    parser.add_argument("--gpu-batch-sizes", nargs="*", type=int, default=[1, 8, 16, 32, 64, 128])
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "results" / "report" / "batch_distortion_runs.csv")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    completed = read_completed(args.output) if args.resume else set()
    max_nodes = cfg["max_nodes"].get(f"h{args.tree_height}", "")

    jobs = []
    for dataset in args.datasets:
        for seed in args.seeds:
            jobs.append(("gpu_exact_gom", dataset, seed, 1))
            for batch_size in args.gpu_batch_sizes:
                jobs.append(("gpu_batch_gom", dataset, seed, batch_size))

    print(f"report distortion jobs: {len(jobs)}")
    for backend, dataset, seed, batch_size in jobs:
        run_id = f"report_{backend}_{slugify(dataset)}_seed{seed}_h{args.tree_height}"
        if backend == "gpu_batch_gom":
            run_id += f"_b{batch_size}"
        if run_id in completed:
            print(f"[skip] {run_id}")
            continue

        print(f"[run] {run_id}")
        result = run_gpg_cli(
            dataset=dataset,
            backend=backend,
            seed=seed,
            tree_height=args.tree_height,
            population_size=args.population_size,
            time_limit_seconds=args.time_limit,
            gpu_batch_size=batch_size,
        )
        append_row(args.output, {
            "run_id": run_id,
            "dataset": dataset,
            "backend": backend,
            "seed": seed,
            "tree_height": args.tree_height,
            "max_nodes": max_nodes,
            "population_size": args.population_size,
            "time_limit_seconds": args.time_limit,
            "gpu_batch_size": batch_size,
            **result,
        })


if __name__ == "__main__":
    main()
