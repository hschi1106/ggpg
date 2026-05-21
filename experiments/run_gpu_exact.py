from __future__ import annotations

from common import base_arg_parser, load_config, run_gpg_cli, write_run_outputs, REPO_ROOT


def main() -> None:
    parser = base_arg_parser("Run the gpu_exact_gom backend.")
    args = parser.parse_args()
    cfg = load_config(args.config)

    datasets = args.datasets or cfg["fast_development_benchmark"]["datasets"]
    tree_heights = args.tree_heights or [cfg["fast_development_benchmark"]["tree_height"]]
    population_size = args.population_size or cfg["fast_development_benchmark"]["population_size"]
    time_limit = args.time_limit or cfg["fast_development_benchmark"]["time_limit_seconds"]
    out = REPO_ROOT / "results" / "paper_reproduction" / "gpu_exact_gom" / "runs.csv"

    for dataset in datasets:
        for seed in args.seeds:
            for height in tree_heights:
                result = run_gpg_cli(
                    dataset=dataset,
                    backend="gpu_exact_gom",
                    seed=seed,
                    tree_height=height,
                    population_size=population_size,
                    time_limit_seconds=time_limit,
                    gpu_batch_size=1,
                )
                write_run_outputs(
                    result=result,
                    run_id=f"gpu_exact_gom_{dataset}_{seed}_h{height}",
                    dataset=dataset,
                    backend="gpu_exact_gom",
                    seed=seed,
                    tree_height=height,
                    max_nodes=cfg["max_nodes"].get(f"h{height}", ""),
                    population_size=population_size,
                    ims_g="",
                    gpu_batch_size=1,
                    backend_csv=out,
                )


if __name__ == "__main__":
    main()
