from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path

import pandas as pd

from common import (
    REPO_ROOT,
    load_config,
    run_gpg_cli,
    slugify,
    write_run_outputs,
)


def read_completed_run_ids(path: Path) -> set[str]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    frame = pd.read_csv(path, usecols=["run_id"])
    return set(frame["run_id"].dropna().astype(str))


def paper_suite_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the full paper-style benchmark matrix.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "experiments" / "paper_reproduction_config.yaml")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--tree-heights", nargs="*", type=int, default=None)
    parser.add_argument("--backends", nargs="*", default=["cpu_original", "gpu_exact_gom", "gpu_batch_gom"])
    parser.add_argument("--gpu-batch-sizes", nargs="*", type=int, default=[32])
    parser.add_argument("--time-limit", type=int, default=None)
    parser.add_argument("--population-size", type=int, default=None)
    parser.add_argument("--include-ims", action="store_true", help="Run IMS settings from ims_g_values instead of fixed population only.")
    parser.add_argument("--ims-g-values", nargs="*", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-long-run", action="store_true")
    return parser


def build_jobs(args: argparse.Namespace, cfg: dict) -> list[dict]:
    datasets = args.datasets or cfg["datasets"]
    seeds = args.seeds if args.seeds is not None else list(range(int(cfg["repetitions"])))
    tree_heights = args.tree_heights or cfg["tree_heights"]
    time_limit = args.time_limit or cfg["time_limit_seconds"]
    population_size = args.population_size or cfg["fixed_population_size"]
    ims_g_values = args.ims_g_values or cfg["ims_g_values"]

    jobs = []
    for dataset in datasets:
        for seed in seeds:
            for height in tree_heights:
                for backend in args.backends:
                    batch_sizes = args.gpu_batch_sizes if backend == "gpu_batch_gom" else [1]
                    for batch_size in batch_sizes:
                        ims_values = ims_g_values if args.include_ims else [None]
                        for ims_g in ims_values:
                            run_id_parts = [
                                backend,
                                slugify(dataset),
                                f"seed{seed}",
                                f"h{height}",
                            ]
                            if backend == "gpu_batch_gom":
                                run_id_parts.append(f"b{batch_size}")
                            if ims_g is not None:
                                run_id_parts.append(f"ims{ims_g}")
                            jobs.append({
                                "run_id": "_".join(run_id_parts),
                                "dataset": dataset,
                                "backend": backend,
                                "seed": seed,
                                "tree_height": height,
                                "max_nodes": cfg["max_nodes"].get(f"h{height}", ""),
                                "population_size": population_size,
                                "time_limit_seconds": time_limit,
                                "gpu_batch_size": batch_size,
                                "ims_g": ims_g,
                                "disable_ims": ims_g is None,
                            })
    return jobs


def backend_csv_for(job: dict) -> Path:
    backend = job["backend"]
    if backend == "gpu_batch_gom":
        return REPO_ROOT / "results" / "paper_reproduction" / backend / "full_suite.csv"
    return REPO_ROOT / "results" / "paper_reproduction" / backend / "runs.csv"


def main() -> None:
    parser = paper_suite_parser()
    args = parser.parse_args()
    cfg = load_config(args.config)
    jobs = build_jobs(args, cfg)
    total_budget = sum(job["time_limit_seconds"] for job in jobs)
    print(f"paper-suite jobs: {len(jobs)}")
    print(f"configured max wall-clock budget: {timedelta(seconds=total_budget)}")
    if jobs:
        print("first job:", jobs[0])
        print("last job:", jobs[-1])

    if args.dry_run:
        return
    if total_budget > 3600 and not args.confirm_long_run:
        raise SystemExit("Refusing to launch a long benchmark matrix without --confirm-long-run.")

    completed = read_completed_run_ids(REPO_ROOT / "results" / "final_results.csv") if args.resume else set()
    for job in jobs:
        if job["run_id"] in completed:
            print(f"[skip] completed {job['run_id']}")
            continue
        print(f"[run] {job['run_id']}")
        result = run_gpg_cli(
            dataset=job["dataset"],
            backend=job["backend"],
            seed=job["seed"],
            tree_height=job["tree_height"],
            population_size=job["population_size"],
            time_limit_seconds=job["time_limit_seconds"],
            gpu_batch_size=job["gpu_batch_size"],
            disable_ims=job["disable_ims"],
            ims_g=job["ims_g"],
        )
        write_run_outputs(
            result=result,
            run_id=job["run_id"],
            dataset=job["dataset"],
            backend=job["backend"],
            seed=job["seed"],
            tree_height=job["tree_height"],
            max_nodes=job["max_nodes"],
            population_size=job["population_size"],
            ims_g=job["ims_g"] if job["ims_g"] is not None else "",
            gpu_batch_size=job["gpu_batch_size"],
            backend_csv=backend_csv_for(job),
        )


if __name__ == "__main__":
    main()
