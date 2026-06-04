from __future__ import annotations

import argparse
import csv
import io
import subprocess
from pathlib import Path

import pandas as pd

from common import REPO_ROOT


PROFILE_SETTINGS = {
    "dataset": "Wine white",
    "row_scale": 32,
    "tree_height": 3,
    "population_size": 4096,
    "seed": 0,
}


def number(value: object) -> float:
    return float(str(value).replace(",", "").strip())


def nsys_report(profile: Path, report: str) -> list[dict[str, str]]:
    process = subprocess.run(
        [
            "nsys",
            "stats",
            "--report",
            report,
            "--format",
            "csv",
            "--timeunit",
            "usec",
            str(profile),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(process.stderr or process.stdout)
    lines = process.stdout.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("Time (%),"):
            return list(csv.DictReader(io.StringIO("\n".join(lines[index:]))))
    raise RuntimeError(f"Could not find CSV header in nsys {report} output for {profile}")


def total_time_us(rows: list[dict[str, str]]) -> float:
    return sum(number(row["Total Time (us)"]) for row in rows)


def api_call_count(rows: list[dict[str, str]], prefix: str) -> int:
    return sum(int(number(row["Num Calls"])) for row in rows if str(row.get("Name", "")).startswith(prefix))


def summarize_profile(profile: Path, backend: str, final_results: pd.DataFrame) -> dict:
    profile = profile.resolve()
    kernels = nsys_report(profile, "cuda_gpu_kern_sum")
    apis = nsys_report(profile, "cuda_api_sum")
    memops = nsys_report(profile, "cuda_gpu_mem_time_sum")
    kernel_instances = sum(int(number(row["Instances"])) for row in kernels)
    kernel_time_us = total_time_us(kernels)
    graph_launches = api_call_count(apis, "cudaGraphLaunch")
    direct_kernel_launches = api_call_count(apis, "cudaLaunchKernel")
    effective_launches = graph_launches if graph_launches else direct_kernel_launches

    mask = (
        (final_results["suite"] == "acceleration")
        & (final_results["backend"] == backend)
        & (final_results["dataset"] == PROFILE_SETTINGS["dataset"])
        & (final_results["row_scale"] == PROFILE_SETTINGS["row_scale"])
        & (final_results["tree_height"] == PROFILE_SETTINGS["tree_height"])
        & (final_results["population_size"] == PROFILE_SETTINGS["population_size"])
        & (final_results["seed"] == PROFILE_SETTINGS["seed"])
    )
    if backend == "gpu_batch_gom":
        mask &= final_results["gpu_batch_size"] == 512
    matching = final_results[mask].drop_duplicates("run_id", keep="last")
    if len(matching) != 1:
        raise RuntimeError(f"Expected one matching result for {backend}, found {len(matching)}.")
    result = matching.iloc[0]
    gpu_evaluations = int(result["num_gpu_evaluations"])

    return {
        "backend": backend,
        "profile": str(profile.relative_to(REPO_ROOT)),
        "num_gpu_evaluations": gpu_evaluations,
        "kernel_instances": kernel_instances,
        "effective_launches": effective_launches,
        "programs_per_launch": gpu_evaluations / effective_launches if effective_launches else float("nan"),
        "nsys_kernel_instances": kernel_instances,
        "total_kernel_ms": kernel_time_us / 1000.0,
        "avg_kernel_us": kernel_time_us / kernel_instances if kernel_instances else float("nan"),
        "total_cuda_api_ms": total_time_us(apis) / 1000.0,
        "total_memop_ms": total_time_us(memops) / 1000.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize representative Nsight Systems GPU profiles.")
    parser.add_argument(
        "--exact-profile",
        type=Path,
        default=REPO_ROOT / "results" / "profiling" / "exact.nsys-rep",
    )
    parser.add_argument(
        "--batch-profile",
        type=Path,
        default=REPO_ROOT / "results" / "profiling" / "batch.nsys-rep",
    )
    parser.add_argument(
        "--final-results",
        type=Path,
        default=REPO_ROOT / "results" / "final_results.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results" / "profiling" / "summary.csv",
    )
    args = parser.parse_args()

    for profile in [args.exact_profile, args.batch_profile]:
        if not profile.exists():
            raise SystemExit(f"Missing profile: {profile}")
    final_results = pd.read_csv(args.final_results)
    rows = [
        summarize_profile(args.exact_profile, "gpu_exact_gom", final_results),
        summarize_profile(args.batch_profile, "gpu_batch_gom", final_results),
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(args.output, index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
