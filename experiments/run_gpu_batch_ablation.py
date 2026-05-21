from __future__ import annotations

from common import base_arg_parser, load_config, run_gpg_cli, write_run_outputs, REPO_ROOT


def main() -> None:
    parser = base_arg_parser("Run gpu_batch_gom batch-size ablations.")
    parser.add_argument("--gpu-batch-sizes", nargs="*", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    fast = cfg["fast_development_benchmark"]

    datasets = args.datasets or fast["datasets"]
    tree_heights = args.tree_heights or [fast["tree_height"]]
    population_size = args.population_size or fast["population_size"]
    time_limit = args.time_limit or fast["time_limit_seconds"]
    batch_sizes = args.gpu_batch_sizes or [1, 8, 16, 32, 64, 128]
    out = REPO_ROOT / "results" / "paper_reproduction" / "gpu_batch_gom" / "batch_ablation.csv"

    for dataset in datasets:
        for seed in args.seeds:
            for height in tree_heights:
                for batch_size in batch_sizes:
                    result = run_gpg_cli(
                        dataset=dataset,
                        backend="gpu_batch_gom",
                        seed=seed,
                        tree_height=height,
                        population_size=population_size,
                        time_limit_seconds=time_limit,
                        gpu_batch_size=batch_size,
                    )
                    write_run_outputs(
                        result=result,
                        run_id=f"gpu_batch_gom_{dataset}_{seed}_h{height}_b{batch_size}",
                        dataset=dataset,
                        backend="gpu_batch_gom",
                        seed=seed,
                        tree_height=height,
                        max_nodes=cfg["max_nodes"].get(f"h{height}", ""),
                        population_size=population_size,
                        ims_g="",
                        gpu_batch_size=batch_size,
                        backend_csv=out,
                    )


if __name__ == "__main__":
    main()
