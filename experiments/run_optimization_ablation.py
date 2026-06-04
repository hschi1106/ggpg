from __future__ import annotations

import argparse
import csv
import subprocess
import time
from pathlib import Path

from common import REPO_ROOT, cli_function_set, parse_counter_line, write_gpg_training_csv


COLUMNS = [
    "implementation",
    "backend",
    "seed",
    "dataset",
    "row_scale",
    "tree_height",
    "population_size",
    "gpu_batch_size",
    "elapsed_sec",
    "num_evaluations",
    "evals_per_sec",
]


def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow({column: row.get(column, "") for column in COLUMNS})


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare the pre-optimization and current CUDA implementations.")
    parser.add_argument("--baseline-binary", type=Path, required=True)
    parser.add_argument("--optimized-binary", type=Path, default=REPO_ROOT / "build" / "cuda-release" / "gpg")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "results" / "optimization_ablation.csv")
    parser.add_argument("--dataset", default="Wine white")
    parser.add_argument("--row-scale", type=int, default=32)
    parser.add_argument("--tree-height", type=int, default=3)
    parser.add_argument("--population-size", type=int, default=4096)
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--gpu-batch-size", type=int, default=512)
    args = parser.parse_args()

    implementations = [
        ("pre_optimization", args.baseline_binary),
        ("optimized", args.optimized_binary),
    ]
    backends = [("gpu_exact_gom", 1), ("gpu_batch_gom", args.gpu_batch_size)]
    for _, binary in implementations:
        if not binary.exists():
            raise SystemExit(f"Missing binary: {binary}")
    for backend, batch_size in backends:
        for seed in args.seeds:
            ordered_implementations = implementations if seed % 2 == 0 else list(reversed(implementations))
            for implementation, binary in ordered_implementations:
                train_csv = write_gpg_training_csv(args.dataset, seed, row_scale=args.row_scale)
                cmd = [
                    str(binary),
                    "-train",
                    str(train_csv),
                    "-backend",
                    backend,
                    "-ff",
                    "mse",
                    "-fset",
                    cli_function_set(),
                    "-d",
                    str(args.tree_height),
                    "-pop",
                    str(args.population_size),
                    "-t",
                    "-1",
                    "-g",
                    "1",
                    "-random_state",
                    str(seed),
                    "-feat_sel",
                    "-1",
                    "-gpu_batch_size",
                    str(batch_size),
                    "-cmp",
                    "0.1",
                    "-disable_ims",
                    "-verbose",
                ]
                print(f"[run] {implementation} {backend} seed={seed}")
                started = time.time()
                process = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
                elapsed = time.time() - started
                if process.returncode != 0:
                    raise RuntimeError(process.stderr or process.stdout)
                counters = parse_counter_line(process.stdout)
                evaluations = counters.get("num_evaluations", 0)
                append_row(
                    args.output,
                    {
                        "implementation": implementation,
                        "backend": backend,
                        "seed": seed,
                        "dataset": args.dataset,
                        "row_scale": args.row_scale,
                        "tree_height": args.tree_height,
                        "population_size": args.population_size,
                        "gpu_batch_size": batch_size,
                        "elapsed_sec": elapsed,
                        "num_evaluations": evaluations,
                        "evals_per_sec": evaluations / elapsed if elapsed > 0 else "",
                    },
                )


if __name__ == "__main__":
    main()
