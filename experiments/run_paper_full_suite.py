from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait
from datetime import timedelta
from pathlib import Path

import pandas as pd

from common import (
    REPO_ROOT,
    cli_function_set,
    load_config,
    run_gpg_cli,
    slugify,
    write_run_outputs,
)

SUITES = {
    "paper",
    "acceleration",
    "batch-scaling",
    "exact-row-scale",
    "fitness-evolution",
    "paper-g-reference",
}


def read_completed_run_ids(path: Path) -> set[str]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    frame = pd.read_csv(path, usecols=["run_id"])
    return set(frame["run_id"].dropna().astype(str))


def paper_suite_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run paper-style and acceleration-focused benchmark matrices.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "experiments" / "paper_reproduction_config.yaml")
    parser.add_argument("--suite", choices=sorted(SUITES), default="paper")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--row-scales", nargs="*", type=int, default=None)
    parser.add_argument("--tree-heights", nargs="*", type=int, default=None)
    parser.add_argument("--backends", nargs="*", default=None)
    parser.add_argument("--gpu-batch-sizes", nargs="*", type=int, default=None)
    parser.add_argument("--time-limit", type=int, default=None)
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--coefficient-mutation-probability", type=float, default=None)
    parser.add_argument("--fitness-function", default=None)
    parser.add_argument("--population-size", type=int, default=None)
    parser.add_argument("--population-sizes", nargs="*", type=int, default=None)
    parser.add_argument("--include-ims", action="store_true", help="Run IMS settings from ims_g_values instead of fixed population only.")
    parser.add_argument("--ims-g-values", nargs="*", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=1, help="Number of runs to execute concurrently. Use with care for GPU backends.")
    parser.add_argument(
        "--gpu-devices",
        nargs="*",
        default=None,
        help="CUDA_VISIBLE_DEVICES values assigned round-robin to GPU jobs, for example: --gpu-devices 0 1",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-long-run", action="store_true")
    return parser


def suite_defaults(suite: str, cfg: dict) -> dict:
    if suite == "paper":
        return {
            "datasets": cfg["datasets"],
            "row_scales": [1],
            "tree_heights": cfg["tree_heights"],
            "population_sizes": cfg.get("population_sizes", [cfg["fixed_population_size"]]),
            "gpu_batch_sizes": cfg.get("gpu_batch_sizes", [64, 128, 256, 512, 1024]),
            "generations": cfg.get("generations", -1),
            "fitness_function": cfg.get("fitness_function", "mse"),
            "time_limit_seconds": cfg["time_limit_seconds"],
            "coefficient_mutation_probability": cfg.get("coefficient_mutation_probability"),
            "seeds": cfg.get("repetitions", 5),
            "backends": ["cpu_original", "gpu_exact_gom", "gpu_batch_gom"],
        }
    key = {
        "acceleration": "acceleration_suite",
        "batch-scaling": "batch_size_scaling_suite",
        "exact-row-scale": "exact_row_scale_suite",
        "fitness-evolution": "fitness_evolution_suite",
        "paper-g-reference": "paper_g_reference_suite",
    }[suite]
    defaults = dict(cfg.get(key, {}))
    defaults.setdefault("datasets", cfg["datasets"])
    defaults.setdefault("row_scales", [1])
    defaults.setdefault("tree_heights", cfg["tree_heights"])
    defaults.setdefault("population_sizes", cfg.get("population_sizes", [cfg["fixed_population_size"]]))
    defaults.setdefault("gpu_batch_sizes", cfg.get("gpu_batch_sizes", [64, 128, 256, 512, 1024]))
    defaults.setdefault("generations", cfg.get("generations", -1))
    defaults.setdefault("fitness_function", cfg.get("fitness_function", "mse"))
    defaults.setdefault("time_limit_seconds", cfg["time_limit_seconds"])
    defaults.setdefault("coefficient_mutation_probability", cfg.get("coefficient_mutation_probability"))
    defaults.setdefault("seeds", cfg.get("repetitions", 5))
    defaults.setdefault("backends", ["cpu_original", "gpu_exact_gom", "gpu_batch_gom"])
    return defaults


def expand_seeds(value: int | list[int]) -> list[int]:
    if isinstance(value, int):
        return list(range(value))
    return list(value)


def build_jobs(args: argparse.Namespace, cfg: dict) -> list[dict]:
    defaults = suite_defaults(args.suite, cfg)
    datasets = args.datasets or defaults["datasets"]
    seeds = args.seeds if args.seeds is not None else expand_seeds(defaults["seeds"])
    row_scales = args.row_scales or defaults["row_scales"]
    tree_heights = args.tree_heights or defaults["tree_heights"]
    time_limit = args.time_limit if args.time_limit is not None else defaults["time_limit_seconds"]
    generations = args.generations if args.generations is not None else defaults["generations"]
    coefficient_mutation_probability = (
        args.coefficient_mutation_probability
        if args.coefficient_mutation_probability is not None
        else defaults["coefficient_mutation_probability"]
    )
    fitness_function = args.fitness_function or defaults["fitness_function"]
    function_set = cli_function_set(cfg)
    if args.population_sizes is not None:
        population_sizes = args.population_sizes
    elif args.population_size is not None:
        population_sizes = [args.population_size]
    else:
        population_sizes = defaults["population_sizes"]
    backends = args.backends or defaults["backends"]
    gpu_batch_sizes = args.gpu_batch_sizes or defaults["gpu_batch_sizes"]
    ims_g_values = args.ims_g_values or cfg["ims_g_values"]

    jobs = []
    for dataset in datasets:
        for seed in seeds:
            for row_scale in row_scales:
                for height in tree_heights:
                    for population_size in population_sizes:
                        for backend in backends:
                            batch_sizes = gpu_batch_sizes if backend == "gpu_batch_gom" else [1]
                            for batch_size in batch_sizes:
                                ims_values = ims_g_values if args.include_ims else [None]
                                for ims_g in ims_values:
                                    run_id_parts = [
                                        args.suite,
                                        backend,
                                        slugify(dataset),
                                        f"seed{seed}",
                                        f"r{row_scale}",
                                        f"h{height}",
                                        f"p{population_size}",
                                        f"g{generations}",
                                    ]
                                    if backend == "gpu_batch_gom":
                                        run_id_parts.append(f"b{batch_size}")
                                    if ims_g is not None:
                                        run_id_parts.append(f"ims{ims_g}")
                                    jobs.append({
                                        "run_id": "_".join(run_id_parts),
                                        "suite": args.suite,
                                        "dataset": dataset,
                                        "backend": backend,
                                        "seed": seed,
                                        "split_seed": cfg.get("data_split_seed", 0),
                                        "row_scale": row_scale,
                                        "tree_height": height,
                                        "max_nodes": cfg["max_nodes"].get(f"h{height}", ""),
                                        "population_size": population_size,
                                        "time_limit_seconds": time_limit,
                                        "generations": generations,
                                        "coefficient_mutation_probability": coefficient_mutation_probability,
                                        "fitness_function": fitness_function,
                                        "function_set": function_set,
                                        "gpu_batch_size": batch_size,
                                        "ims_g": ims_g,
                                        "disable_ims": ims_g is None,
                                    })
    return jobs


def backend_csv_for(job: dict) -> Path:
    backend = job["backend"]
    root = REPO_ROOT / "results" / job["suite"] / backend
    if backend == "gpu_batch_gom":
        return root / "full_suite.csv"
    return root / "runs.csv"


def execute_job(job: dict) -> tuple[dict, dict]:
    result = run_gpg_cli(
        dataset=job["dataset"],
        backend=job["backend"],
        seed=job["seed"],
        row_scale=job["row_scale"],
        tree_height=job["tree_height"],
        population_size=job["population_size"],
        time_limit_seconds=job["time_limit_seconds"],
        gpu_batch_size=job["gpu_batch_size"],
        generations=job["generations"],
        coefficient_mutation_probability=job["coefficient_mutation_probability"],
        fitness_function=job["fitness_function"],
        function_set=job["function_set"],
        disable_ims=job["disable_ims"],
        ims_g=job["ims_g"],
        cuda_visible_device=job.get("cuda_visible_device"),
    )
    result.pop("stdout", None)
    result.pop("stderr", None)
    return job, result


def persist_job_result(job: dict, result: dict) -> None:
    write_run_outputs(
        result=result,
        run_id=job["run_id"],
        suite=job["suite"],
        dataset=job["dataset"],
        backend=job["backend"],
        seed=job["seed"],
        split_seed=job["split_seed"],
        row_scale=job["row_scale"],
        tree_height=job["tree_height"],
        max_nodes=job["max_nodes"],
        population_size=job["population_size"],
        ims_g=job["ims_g"] if job["ims_g"] is not None else "",
        gpu_batch_size=job["gpu_batch_size"],
        backend_csv=backend_csv_for(job),
        fitness_function=job["fitness_function"],
        function_set=job["function_set"],
        time_limit_seconds=job["time_limit_seconds"],
        generations=job["generations"],
        coefficient_mutation_probability=job["coefficient_mutation_probability"],
    )


def run_gpu_device_queues(pending_jobs: list[dict], gpu_devices: list[str], jobs: int) -> None:
    devices = gpu_devices[: min(jobs, len(gpu_devices))]
    if not devices:
        raise ValueError("At least one GPU device is required.")
    pending = iter(pending_jobs)
    with ProcessPoolExecutor(max_workers=len(devices)) as executor:
        futures = {}
        for device in devices:
            try:
                job = next(pending)
            except StopIteration:
                break
            job = dict(job)
            job["cuda_visible_device"] = device
            print(f"[run gpu={device}] {job['run_id']}")
            futures[executor.submit(execute_job, job)] = (job, device)

        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                job, device = futures.pop(future)
                try:
                    finished_job, result = future.result()
                except Exception as exc:
                    raise RuntimeError(f"{job['run_id']} failed") from exc
                persist_job_result(finished_job, result)
                print(f"[done gpu={device}] {finished_job['run_id']}")
                try:
                    next_job = dict(next(pending))
                except StopIteration:
                    continue
                next_job["cuda_visible_device"] = device
                print(f"[run gpu={device}] {next_job['run_id']}")
                futures[executor.submit(execute_job, next_job)] = (next_job, device)


def main() -> None:
    parser = paper_suite_parser()
    args = parser.parse_args()
    if args.jobs < 1:
        raise SystemExit("--jobs must be at least 1.")
    cfg = load_config(args.config)
    jobs = build_jobs(args, cfg)
    finite_budgets = [job["time_limit_seconds"] for job in jobs if job["time_limit_seconds"] > 0]
    total_budget = sum(finite_budgets)
    print(f"{args.suite} jobs: {len(jobs)}")
    if finite_budgets:
        print(f"configured max wall-clock budget: {timedelta(seconds=total_budget)}")
    else:
        print("configured max wall-clock budget: unbounded by -t; generation budget controls each run")
    if jobs:
        print("first job:", jobs[0])
        print("last job:", jobs[-1])

    if args.dry_run:
        return
    if (not finite_budgets or total_budget > 3600) and not args.confirm_long_run:
        raise SystemExit("Refusing to launch a long benchmark matrix without --confirm-long-run.")
    if args.jobs > 1 and any(job["backend"].startswith("gpu") for job in jobs) and not args.gpu_devices:
        print("[warn] Running multiple GPU jobs concurrently can reduce per-run throughput through device contention.")

    completed = read_completed_run_ids(REPO_ROOT / "results" / "final_results.csv") if args.resume else set()
    pending_jobs = []
    for job in jobs:
        if job["run_id"] in completed:
            print(f"[skip] completed {job['run_id']}")
            continue
        pending_jobs.append(job)

    if args.jobs == 1:
        for job in pending_jobs:
            print(f"[run] {job['run_id']}")
            _, result = execute_job(job)
            persist_job_result(job, result)
            print(f"[done] {job['run_id']}")
        return

    gpu_devices = args.gpu_devices or []
    if gpu_devices and pending_jobs and all(job["backend"].startswith("gpu") for job in pending_jobs):
        run_gpu_device_queues(pending_jobs, gpu_devices, args.jobs)
        return

    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = {}
        for job in pending_jobs:
            print(f"[run] {job['run_id']}")
            futures[executor.submit(execute_job, job)] = job
        for future in as_completed(futures):
            job = futures[future]
            try:
                finished_job, result = future.result()
            except Exception as exc:
                raise RuntimeError(f"{job['run_id']} failed") from exc
            persist_job_result(finished_job, result)
            print(f"[done] {finished_job['run_id']}")


if __name__ == "__main__":
    main()
