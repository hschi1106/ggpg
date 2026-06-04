from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from common import REPO_ROOT, derive_gom_batch_size, load_config, load_processed_split


REPORT_ROOT = REPO_ROOT / "report"
FIGURE_ROOT = REPORT_ROOT / "figures"
GENERATED_ROOT = REPORT_ROOT / "generated"

PAIR_KEYS = ["dataset", "seed", "split_seed", "row_scale", "tree_height", "population_size"]
PAPER_G_REFERENCE = {
    "Airfoil": {"validation": 26.40, "test": 26.40},
    "Dow chemical": {"validation": 21.30, "test": 20.30},
    "Wine white": {"validation": 70.20, "test": 69.00},
    "Yacht hydrodynamics": {"validation": 0.46, "test": 0.52},
}
BACKEND_LABELS = {
    "cpu_original": "CPU",
    "gpu_exact_gom": "Exact GPU",
    "gpu_batch_gom": "Batch GPU",
}
COLORS = {
    "CPU": "#333333",
    "Exact GPU": "#1f77b4",
    "Batch 64": "#2ca02c",
    "Batch 128": "#ff7f0e",
    "Batch 256": "#d62728",
    "Batch 512": "#9467bd",
    "Batch 1024": "#8c564b",
}


def latex_escape(value: object) -> str:
    text = str(value)
    for source, target in [
        ("\\", r"\textbackslash{}"),
        ("_", r"\_"),
        ("%", r"\%"),
        ("&", r"\&"),
        ("#", r"\#"),
    ]:
        text = text.replace(source, target)
    return text


def fmt(value: object, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "--"
    return f"{float(value):.{digits}f}"


def pct(value: object, digits: int = 0) -> str:
    if value is None or pd.isna(value):
        return "--"
    return f"{100.0 * float(value):.{digits}f}\\%"


def bootstrap_median_ci(values: pd.Series | np.ndarray, seed: int = 17) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(4000, len(array)), replace=True)
    medians = np.median(samples, axis=1)
    return float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))


def holm_adjust(p_values: pd.Series) -> pd.Series:
    values = p_values.to_numpy(dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return pd.Series(adjusted, index=p_values.index)


def backend_variant(frame: pd.DataFrame) -> pd.Series:
    labels = frame["backend"].map(BACKEND_LABELS)
    batch_mask = frame["backend"] == "gpu_batch_gom"
    labels.loc[batch_mask] = "Batch " + frame.loc[batch_mask, "gpu_batch_size"].astype(int).astype(str)
    return labels


def latest_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop_duplicates("run_id", keep="last").copy()


def normalize_batch_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"backend", "avg_actual_gpu_batch_size", "num_gpu_evaluations", "population_size"}
    if not required.issubset(frame.columns):
        return frame
    frame = frame.copy()
    frame["avg_actual_gpu_batch_size"] = frame.apply(
        lambda row: derive_gom_batch_size(
            row["backend"],
            row["avg_actual_gpu_batch_size"],
            row["num_gpu_evaluations"],
            row["population_size"],
        ),
        axis=1,
    )
    return frame


def expected_suite_runs(config: dict, section_name: str) -> int:
    section = config[section_name]
    seeds = section["seeds"] if isinstance(section["seeds"], int) else len(section["seeds"])
    backend_variants = 0
    for backend in section["backends"]:
        backend_variants += len(section["gpu_batch_sizes"]) if backend == "gpu_batch_gom" else 1
    return (
        len(section["datasets"])
        * seeds
        * len(section["row_scales"])
        * len(section["tree_heights"])
        * len(section["population_sizes"])
        * backend_variants
    )


def validate_completeness(frame: pd.DataFrame, generations: pd.DataFrame) -> None:
    config = load_config()
    required = {
        "acceleration": expected_suite_runs(config, "acceleration_suite"),
        "batch-scaling": expected_suite_runs(config, "batch_size_scaling_suite"),
        "exact-row-scale": expected_suite_runs(config, "exact_row_scale_suite"),
        "fitness-evolution": expected_suite_runs(config, "fitness_evolution_suite"),
        "paper-g-reference": expected_suite_runs(config, "paper_g_reference_suite"),
    }
    for suite, expected in required.items():
        actual = frame.loc[frame["suite"] == suite, "run_id"].nunique()
        if actual != expected:
            raise SystemExit(f"{suite} suite is incomplete: expected {expected} runs, found {actual}.")
    expected_generation_rows = required["fitness-evolution"] * int(config["fitness_evolution_suite"]["generations"])
    actual_generation_rows = len(generations)
    if actual_generation_rows != expected_generation_rows:
        raise SystemExit(
            "fitness-evolution generation logs are incomplete: "
            f"expected {expected_generation_rows} rows, found {actual_generation_rows}."
        )


def pair_acceleration(frame: pd.DataFrame, suite: str = "acceleration") -> tuple[pd.DataFrame, pd.DataFrame]:
    data = latest_rows(frame[frame["suite"] == suite])
    cpu = data[data["backend"] == "cpu_original"][PAIR_KEYS + ["elapsed_sec", "evals_per_sec"]].rename(
        columns={"elapsed_sec": "cpu_elapsed_sec", "evals_per_sec": "cpu_evals_per_sec"}
    )
    exact = data[data["backend"] == "gpu_exact_gom"].merge(cpu, on=PAIR_KEYS, how="inner")
    batch = data[data["backend"] == "gpu_batch_gom"].merge(cpu, on=PAIR_KEYS, how="inner")
    for paired in [exact, batch]:
        paired["wall_speedup_vs_cpu"] = paired["cpu_elapsed_sec"] / paired["elapsed_sec"]
        paired["throughput_ratio_vs_cpu"] = paired["evals_per_sec"] / paired["cpu_evals_per_sec"]
    return exact, batch


def acceleration_summary(exact: pd.DataFrame, batch: pd.DataFrame) -> pd.DataFrame:
    rows = []
    variants = [("Exact GPU", exact)]
    variants.extend((f"Batch {int(batch_size)}", group) for batch_size, group in batch.groupby("gpu_batch_size"))
    for label, group in variants:
        lo, hi = bootstrap_median_ci(group["wall_speedup_vs_cpu"])
        rows.append(
            {
                "variant": label,
                "runs": len(group),
                "median_wall_speedup": group["wall_speedup_vs_cpu"].median(),
                "q25_wall_speedup": group["wall_speedup_vs_cpu"].quantile(0.25),
                "q75_wall_speedup": group["wall_speedup_vs_cpu"].quantile(0.75),
                "ci_low": lo,
                "ci_high": hi,
                "median_throughput_ratio": group["throughput_ratio_vs_cpu"].median(),
                "fraction_faster_than_cpu": (group["wall_speedup_vs_cpu"] > 1.0).mean(),
                "max_wall_speedup": group["wall_speedup_vs_cpu"].max(),
            }
        )
    return pd.DataFrame(rows)


def fitness_pairs(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = latest_rows(frame[frame["suite"] == "fitness-evolution"])
    cpu = data[data["backend"] == "cpu_original"].copy()
    exact = data[data["backend"] == "gpu_exact_gom"].copy()
    batch = data[data["backend"] == "gpu_batch_gom"].copy()

    exact_reference = exact[
        PAIR_KEYS + ["elapsed_sec", "train_nmse", "validation_nmse", "test_nmse", "evals_per_sec"]
    ].rename(
        columns={
            "elapsed_sec": "exact_elapsed_sec",
            "train_nmse": "exact_train_nmse",
            "validation_nmse": "exact_validation_nmse",
            "test_nmse": "exact_test_nmse",
            "evals_per_sec": "exact_evals_per_sec",
        }
    )
    cpu_reference = cpu[PAIR_KEYS + ["elapsed_sec", "test_nmse", "evals_per_sec"]].rename(
        columns={
            "elapsed_sec": "cpu_elapsed_sec",
            "test_nmse": "cpu_test_nmse",
            "evals_per_sec": "cpu_evals_per_sec",
        }
    )
    batch_pairs = batch.merge(exact_reference, on=PAIR_KEYS, how="inner").merge(cpu_reference, on=PAIR_KEYS, how="left")
    for metric in ["train_nmse", "validation_nmse", "test_nmse"]:
        batch_pairs[f"delta_{metric}"] = batch_pairs[metric] - batch_pairs[f"exact_{metric}"]
        batch_pairs[f"ratio_{metric}"] = batch_pairs[metric] / batch_pairs[f"exact_{metric}"]
    batch_pairs["wall_speedup_vs_exact"] = batch_pairs["exact_elapsed_sec"] / batch_pairs["elapsed_sec"]
    batch_pairs["wall_speedup_vs_cpu"] = batch_pairs["cpu_elapsed_sec"] / batch_pairs["elapsed_sec"]
    batch_pairs["throughput_ratio_vs_exact"] = batch_pairs["evals_per_sec"] / batch_pairs["exact_evals_per_sec"]

    cpu_exact = exact.merge(cpu_reference, on=PAIR_KEYS, how="inner", suffixes=("", "_cpu_ref"))
    cpu_exact["test_nmse_ratio_vs_cpu"] = cpu_exact["test_nmse"] / cpu_exact["cpu_test_nmse"]
    cpu_exact["wall_speedup_vs_cpu"] = cpu_exact["cpu_elapsed_sec"] / cpu_exact["elapsed_sec"]
    return cpu_exact, batch_pairs


def fitness_distortion_summary(batch_pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    tolerance = 1e-12
    for batch_size, group in batch_pairs.groupby("gpu_batch_size"):
        lo, hi = bootstrap_median_ci(group["ratio_test_nmse"])
        delta = group["delta_test_nmse"]
        try:
            p_value = float(wilcoxon(delta, alternative="two-sided").pvalue)
        except ValueError:
            p_value = 1.0
        rows.append(
            {
                "gpu_batch_size": int(batch_size),
                "pairs": len(group),
                "median_test_nmse_ratio": group["ratio_test_nmse"].median(),
                "q25_test_nmse_ratio": group["ratio_test_nmse"].quantile(0.25),
                "q75_test_nmse_ratio": group["ratio_test_nmse"].quantile(0.75),
                "ratio_ci_low": lo,
                "ratio_ci_high": hi,
                "median_test_nmse_delta_paper_scale": 100.0 * delta.median(),
                "median_abs_test_nmse_delta_paper_scale": 100.0 * delta.abs().median(),
                "wins": int((delta < -tolerance).sum()),
                "ties": int((delta.abs() <= tolerance).sum()),
                "losses": int((delta > tolerance).sum()),
                "median_wall_speedup_vs_exact": group["wall_speedup_vs_exact"].median(),
                "median_wall_speedup_vs_cpu": group["wall_speedup_vs_cpu"].median(),
                "median_throughput_ratio_vs_exact": group["throughput_ratio_vs_exact"].median(),
                "wilcoxon_p": p_value,
            }
        )
    summary = pd.DataFrame(rows).sort_values("gpu_batch_size")
    summary["wilcoxon_holm_p"] = holm_adjust(summary["wilcoxon_p"])
    return summary


def read_generation_logs() -> pd.DataFrame:
    paths = sorted((REPO_ROOT / "results" / "logs").glob("fitness-evolution_*_generation.csv"))
    frames = []
    for path in paths:
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if "best_test_nmse" not in frame or frame["best_test_nmse"].isna().all():
            continue
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    data = pd.concat(frames, ignore_index=True)
    data = data.drop_duplicates(["run_id", "generation"], keep="last")
    data = normalize_batch_metrics(data)
    data["variant"] = backend_variant(data)
    return data


def paper_g_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    data = latest_rows(frame[frame["suite"] == "paper-g-reference"])
    rows = []
    for dataset, reference in PAPER_G_REFERENCE.items():
        group = data[data["dataset"] == dataset]
        rows.append(
            {
                "dataset": dataset,
                "local_runs": len(group),
                "paper_validation_nmse": reference["validation"],
                "paper_test_nmse": reference["test"],
                "local_validation_nmse": 100.0 * group["validation_nmse"].median() if len(group) else float("nan"),
                "local_test_nmse": 100.0 * group["test_nmse"].median() if len(group) else float("nan"),
                "local_over_paper_test_nmse": (
                    100.0 * group["test_nmse"].median() / reference["test"] if len(group) else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def gom_behavior_summary(generations: pd.DataFrame) -> pd.DataFrame:
    if generations.empty:
        return pd.DataFrame()
    final = generations.sort_values("generation").groupby("run_id", as_index=False).tail(1).copy()
    decisions = final["num_accepted_moves"] + final["num_rejected_moves"]
    candidates = final["num_meaningful_candidates"] + final["num_nonmeaningful_candidates"]
    final["acceptance_rate"] = final["num_accepted_moves"] / decisions.replace(0, np.nan)
    final["meaningful_fraction"] = final["num_meaningful_candidates"] / candidates.replace(0, np.nan)
    final.loc[final["variant"] == "CPU", "avg_actual_gpu_batch_size"] = np.nan
    final.loc[final["variant"] == "Exact GPU", "avg_actual_gpu_batch_size"] = 1.0
    return (
        final.groupby("variant")
        .agg(
            runs=("run_id", "count"),
            median_acceptance_rate=("acceptance_rate", "median"),
            median_meaningful_fraction=("meaningful_fraction", "median"),
            median_evaluations=("num_evaluations", "median"),
            median_actual_batch_size=("avg_actual_gpu_batch_size", "median"),
        )
        .reset_index()
    )


def optimization_ablation() -> pd.DataFrame:
    path = REPO_ROOT / "results" / "optimization_ablation.csv"
    if not path.exists():
        return pd.DataFrame()
    data = pd.read_csv(path).drop_duplicates(["implementation", "backend", "seed"], keep="last")
    baseline = data[data["implementation"] == "pre_optimization"][
        ["backend", "seed", "elapsed_sec", "evals_per_sec"]
    ].rename(columns={"elapsed_sec": "baseline_elapsed_sec", "evals_per_sec": "baseline_evals_per_sec"})
    optimized = data[data["implementation"] == "optimized"].merge(baseline, on=["backend", "seed"], how="inner")
    optimized["wall_speedup"] = optimized["baseline_elapsed_sec"] / optimized["elapsed_sec"]
    optimized["throughput_ratio"] = optimized["evals_per_sec"] / optimized["baseline_evals_per_sec"]
    return (
        optimized.groupby("backend")
        .agg(
            runs=("seed", "count"),
            baseline_elapsed_sec=("baseline_elapsed_sec", "median"),
            optimized_elapsed_sec=("elapsed_sec", "median"),
            median_wall_speedup=("wall_speedup", "median"),
            median_throughput_ratio=("throughput_ratio", "median"),
        )
        .reset_index()
    )


def profiling_summary() -> pd.DataFrame:
    path = REPO_ROOT / "results" / "profiling" / "summary.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def dataset_summary(frame: pd.DataFrame) -> pd.DataFrame:
    datasets = sorted(set(frame["dataset"].dropna()) | set(PAPER_G_REFERENCE))
    rows = []
    for dataset in datasets:
        train_x, _ = load_processed_split(dataset, "train")
        val_x, _ = load_processed_split(dataset, "val")
        test_x, _ = load_processed_split(dataset, "test")
        rows.append(
            {
                "dataset": dataset,
                "features": train_x.shape[1],
                "train_rows": train_x.shape[0],
                "validation_rows": val_x.shape[0],
                "test_rows": test_x.shape[0],
            }
        )
    return pd.DataFrame(rows)


def write_latex_table(path: Path, header: list[str], rows: list[list[str]]) -> None:
    column_spec = "l" + "r" * (len(header) - 1)
    lines = [fr"\begin{{tabular}}{{{column_spec}}}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    lines.extend(" & ".join(row) + r" \\" for row in rows)
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_tables(
    datasets: pd.DataFrame,
    acceleration: pd.DataFrame,
    exact: pd.DataFrame,
    batch: pd.DataFrame,
    scaling_exact: pd.DataFrame,
    scaling_batch: pd.DataFrame,
    distortion: pd.DataFrame,
    paper_g: pd.DataFrame,
    ablation: pd.DataFrame,
    gom_behavior: pd.DataFrame,
    profiling: pd.DataFrame,
) -> None:
    write_latex_table(
        GENERATED_ROOT / "table_datasets.tex",
        ["Dataset", "Features", "Train", "Validation", "Test"],
        [
            [
                latex_escape(row.dataset),
                str(int(row.features)),
                str(int(row.train_rows)),
                str(int(row.validation_rows)),
                str(int(row.test_rows)),
            ]
            for row in datasets.itertuples()
        ],
    )
    write_latex_table(
        GENERATED_ROOT / "table_acceleration_summary.tex",
        ["Backend", "Runs", "Median", "IQR", "Eval/s ratio", "Faster"],
        [
            [
                latex_escape(row.variant),
                str(int(row.runs)),
                fmt(row.median_wall_speedup),
                f"[{fmt(row.q25_wall_speedup)}, {fmt(row.q75_wall_speedup)}]",
                fmt(row.median_throughput_ratio),
                pct(row.fraction_faster_than_cpu),
            ]
            for row in acceleration.itertuples()
        ],
    )

    headline_exact = exact[(exact["row_scale"] == 32) & (exact["population_size"] == 8192)].copy()
    headline_batch = batch[(batch["row_scale"] == 32) & (batch["population_size"] == 8192)].copy()
    rows = []
    for dataset in sorted(headline_exact["dataset"].unique()):
        exact_group = headline_exact[headline_exact["dataset"] == dataset]
        batch_group = headline_batch[headline_batch["dataset"] == dataset]
        best_batch = (
            batch_group.groupby("gpu_batch_size")["wall_speedup_vs_cpu"].median().sort_values(ascending=False)
        )
        best_size = int(best_batch.index[0]) if len(best_batch) else 0
        best_group = batch_group[batch_group["gpu_batch_size"] == best_size]
        rows.append(
            [
                latex_escape(dataset),
                fmt(exact_group["wall_speedup_vs_cpu"].median()),
                str(best_size),
                fmt(best_group["wall_speedup_vs_cpu"].median()),
                fmt(best_group["throughput_ratio_vs_cpu"].median()),
            ]
        )
    write_latex_table(
        GENERATED_ROOT / "table_headline_acceleration.tex",
        ["Dataset", "Exact speedup", "Best batch", "Batch speedup", "Batch eval/s ratio"],
        rows,
    )
    scaling_rows = []
    if not scaling_exact.empty:
        scaling_rows.append(
            [
                "Exact GPU",
                "1",
                "1.0",
                fmt(scaling_exact["wall_speedup_vs_cpu"].median()),
                fmt(scaling_exact["throughput_ratio_vs_cpu"].median()),
            ]
        )
    for batch_size, group in scaling_batch.groupby("gpu_batch_size"):
        scaling_rows.append(
            [
                "Batch GPU",
                str(int(batch_size)),
                fmt(group["avg_actual_gpu_batch_size"].median(), 1),
                fmt(group["wall_speedup_vs_cpu"].median()),
                fmt(group["throughput_ratio_vs_cpu"].median()),
            ]
        )
    write_latex_table(
        GENERATED_ROOT / "table_batch_scaling.tex",
        ["Backend", "Requested batch", "Actual GOM batch", "Wall speedup", "Eval/s ratio"],
        scaling_rows,
    )

    write_latex_table(
        GENERATED_ROOT / "table_fitness_distortion.tex",
        ["Batch", "Pairs", "Test ratio", r"$100\Delta$ NMSE", "W/T/L", "Speedup"],
        [
            [
                str(int(row.gpu_batch_size)),
                str(int(row.pairs)),
                fmt(row.median_test_nmse_ratio, 3),
                fmt(row.median_test_nmse_delta_paper_scale, 2),
                f"{int(row.wins)}/{int(row.ties)}/{int(row.losses)}",
                fmt(row.median_wall_speedup_vs_cpu),
            ]
            for row in distortion.itertuples()
        ],
    )
    write_latex_table(
        GENERATED_ROOT / "table_paper_g.tex",
        ["Dataset", "Paper val.", "Paper test", "Local val.", "Local test", "Local/Paper"],
        [
            [
                latex_escape(row.dataset),
                fmt(row.paper_validation_nmse),
                fmt(row.paper_test_nmse),
                fmt(row.local_validation_nmse),
                fmt(row.local_test_nmse),
                fmt(row.local_over_paper_test_nmse),
            ]
            for row in paper_g.itertuples()
        ],
    )
    if not ablation.empty:
        write_latex_table(
            GENERATED_ROOT / "table_optimization_ablation.tex",
            ["Backend", "Pre-opt. (s)", "Optimized (s)", "Wall speedup", "Eval/s ratio"],
            [
                [
                    latex_escape(BACKEND_LABELS.get(row.backend, row.backend)),
                    fmt(row.baseline_elapsed_sec),
                    fmt(row.optimized_elapsed_sec),
                    fmt(row.median_wall_speedup),
                    fmt(row.median_throughput_ratio),
                ]
                for row in ablation.itertuples()
            ],
        )
    if not gom_behavior.empty:
        order = {label: index for index, label in enumerate(COLORS)}
        gom_behavior = gom_behavior.assign(sort_key=gom_behavior["variant"].map(order).fillna(999)).sort_values("sort_key")
        write_latex_table(
            GENERATED_ROOT / "table_gom_behavior.tex",
            ["Variant", "Runs", "Acceptance", "Meaningful", "Actual GOM batch", "Evaluations"],
            [
                [
                    latex_escape(row.variant),
                    str(int(row.runs)),
                    pct(row.median_acceptance_rate, 1),
                    pct(row.median_meaningful_fraction, 1),
                    fmt(row.median_actual_batch_size, 1),
                    fmt(row.median_evaluations, 0),
                ]
                for row in gom_behavior.itertuples()
            ],
        )
    if not profiling.empty:
        order = {"gpu_exact_gom": 0, "gpu_batch_gom": 1}
        profiling = profiling.assign(sort_key=profiling["backend"].map(order).fillna(999)).sort_values("sort_key")
        write_latex_table(
            GENERATED_ROOT / "table_profiling.tex",
            ["Backend", "GPU evals", "Launches", "Programs/launch", "Kernel time (ms)", "CUDA API (ms)"],
            [
                [
                    latex_escape(BACKEND_LABELS.get(row.backend, row.backend)),
                    fmt(row.num_gpu_evaluations, 0),
                    fmt(row.effective_launches, 0),
                    fmt(row.programs_per_launch, 1),
                    fmt(row.total_kernel_ms, 1),
                    fmt(row.total_cuda_api_ms, 1),
                ]
                for row in profiling.itertuples()
            ],
        )


def heatmap(
    values: pd.DataFrame,
    title: str,
    output: Path,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    image = ax.imshow(values.to_numpy(), aspect="auto", cmap="YlGnBu", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(values.columns)), [str(int(value)) for value in values.columns])
    ax.set_yticks(range(len(values.index)), [f"x{int(value)}" for value in values.index])
    ax.set_xlabel("Population size")
    ax.set_ylabel("Training-row scale")
    ax.set_title(title)
    for y in range(len(values.index)):
        for x in range(len(values.columns)):
            value = values.iloc[y, x]
            ax.text(x, y, fmt(value, 1), ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="Wall-clock speedup vs CPU")
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_acceleration(
    exact: pd.DataFrame,
    batch: pd.DataFrame,
    scaling_exact: pd.DataFrame,
    scaling_batch: pd.DataFrame,
    row_exact: pd.DataFrame,
    row_batch: pd.DataFrame,
) -> None:
    exact_heat = exact.pivot_table(
        index="row_scale", columns="population_size", values="wall_speedup_vs_cpu", aggfunc="median"
    ).sort_index()
    batch_512 = batch[batch["gpu_batch_size"] == 512]
    batch_heat = batch_512.pivot_table(
        index="row_scale", columns="population_size", values="wall_speedup_vs_cpu", aggfunc="median"
    ).sort_index()
    vmax = max(float(exact_heat.max().max()), float(batch_heat.max().max()))
    heatmap(exact_heat, "Exact GPU", FIGURE_ROOT / "exact_speedup_heatmap.png", vmin=0, vmax=vmax)
    heatmap(batch_heat, "Batch GPU, batch size 512", FIGURE_ROOT / "batch_speedup_heatmap.png", vmin=0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    exact_x32 = exact[exact["row_scale"] == 32].groupby("population_size")["wall_speedup_vs_cpu"].median()
    ax.plot(exact_x32.index, exact_x32.values, marker="o", label="Exact GPU", color=COLORS["Exact GPU"])
    for batch_size, group in batch[batch["row_scale"] == 32].groupby("gpu_batch_size"):
        med = group.groupby("population_size")["wall_speedup_vs_cpu"].median()
        label = f"Batch {int(batch_size)}"
        ax.plot(med.index, med.values, marker="o", label=label, color=COLORS.get(label))
    ax.axhline(1.0, color="#777777", linewidth=0.8, linestyle="--")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Population size")
    ax.set_ylabel("Wall-clock speedup vs CPU")
    ax.legend(ncol=2, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURE_ROOT / "speedup_vs_population_x32.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    row_exact_source = row_exact if not row_exact.empty else exact
    row_batch_source = row_batch if not row_batch.empty else batch_512
    if not row_batch.empty:
        row_batch_source = row_batch[row_batch["gpu_batch_size"] == 512]
    exact_rows = row_exact_source.groupby("row_scale")["wall_speedup_vs_cpu"].median()
    batch_rows = row_batch_source.groupby("row_scale")["wall_speedup_vs_cpu"].median()
    ax.plot(exact_rows.index, exact_rows.values, marker="o", label="Exact GPU", color=COLORS["Exact GPU"])
    ax.plot(batch_rows.index, batch_rows.values, marker="o", label="Batch 512", color=COLORS["Batch 512"])
    ax.axhline(1.0, color="#777777", linewidth=0.8, linestyle="--")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Training-row scale")
    ax.set_ylabel("Wall-clock speedup vs CPU")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_ROOT / "speedup_vs_row_scale.png", dpi=220)
    plt.close(fig)

    if not scaling_batch.empty and not scaling_exact.empty:
        fixed = scaling_batch.copy()
        exact_fixed = scaling_exact.copy()
    else:
        fixed = batch[
            (batch["dataset"] == "Wine white")
            & (batch["row_scale"] == 32)
            & (batch["population_size"] == 4096)
        ].copy()
        exact_fixed = exact[
            (exact["dataset"] == "Wine white")
            & (exact["row_scale"] == 32)
            & (exact["population_size"] == 4096)
        ]
    exact_ratio = exact_fixed["throughput_ratio_vs_cpu"].median()
    med = fixed.groupby("gpu_batch_size").agg(
        speedup=("wall_speedup_vs_cpu", "median"),
        throughput=("throughput_ratio_vs_cpu", "median"),
    )
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(med.index, med["speedup"], marker="o", label="Wall-clock speedup")
    ax.plot(med.index, med["throughput"], marker="s", label="Eval/s ratio")
    ax.axhline(exact_ratio, color=COLORS["Exact GPU"], linestyle="--", label="Exact GPU eval/s ratio")
    ax.axhline(1.0, color="#777777", linewidth=0.8, linestyle=":")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("GPU batch size")
    ax.set_ylabel("Ratio vs CPU")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_ROOT / "batch_size_scaling.png", dpi=220)
    plt.close(fig)


def plot_fitness(generations: pd.DataFrame, batch_pairs: pd.DataFrame) -> None:
    if generations.empty:
        return
    variants = ["CPU", "Exact GPU", "Batch 64", "Batch 128", "Batch 256", "Batch 512", "Batch 1024"]
    config = load_config()
    configured_datasets = config.get("fitness_evolution_suite", {}).get("datasets", [])
    available_datasets = set(generations["dataset"].dropna().astype(str).unique())
    datasets = [dataset for dataset in configured_datasets if dataset in available_datasets]
    datasets.extend(sorted(available_datasets - set(datasets)))
    subset = generations[
        (generations["tree_height"] == 4)
        & (generations["dataset"].isin(datasets))
        & (generations["variant"].isin(variants))
    ].copy()
    if subset.empty:
        return
    datasets = [dataset for dataset in datasets if not subset[subset["dataset"] == dataset].empty]
    cols = 2
    rows = int(math.ceil(len(datasets) / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(8.0, 2.9 * rows), sharex=True, squeeze=False)
    for ax, dataset in zip(axes.flat, datasets):
        data = subset[subset["dataset"] == dataset]
        for variant in variants:
            group = data[data["variant"] == variant]
            if group.empty:
                continue
            med = group.groupby("generation")["best_test_nmse"].median()
            ax.plot(med.index, med.values, label=variant, color=COLORS.get(variant), linewidth=1.4)
        ax.set_title(dataset)
        ax.set_yscale("log")
        ax.grid(alpha=0.2)
    for ax in axes.flat[len(datasets) :]:
        ax.set_visible(False)
    for ax in axes[-1, :]:
        ax.set_xlabel("Generation")
    for ax in axes[:, 0]:
        ax.set_ylabel("Median best test NMSE")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="lower center", fontsize=7)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIGURE_ROOT / "fitness_evolution_by_generation.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(rows, cols, figsize=(8.0, 2.9 * rows), squeeze=False)
    for ax, dataset in zip(axes.flat, datasets):
        data = subset[subset["dataset"] == dataset]
        for variant in variants:
            group = data[data["variant"] == variant]
            if group.empty:
                continue
            med = group.groupby("generation").agg(
                elapsed_sec=("elapsed_sec", "median"),
                best_test_nmse=("best_test_nmse", "median"),
            )
            ax.plot(
                med["elapsed_sec"],
                med["best_test_nmse"],
                label=variant,
                color=COLORS.get(variant),
                linewidth=1.4,
            )
        ax.set_title(dataset)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(alpha=0.2)
    for ax in axes.flat[len(datasets) :]:
        ax.set_visible(False)
    for ax in axes[-1, :]:
        ax.set_xlabel("Median elapsed time (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Median best test NMSE")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="lower center", fontsize=7)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIGURE_ROOT / "fitness_evolution_by_time.png", dpi=220)
    plt.close(fig)

    summary = batch_pairs.groupby("gpu_batch_size")["ratio_test_nmse"].agg(
        median="median",
        q25=lambda series: series.quantile(0.25),
        q75=lambda series: series.quantile(0.75),
    )
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.plot(summary.index, summary["median"], marker="o", color="#d62728")
    ax.fill_between(summary.index, summary["q25"], summary["q75"], color="#d62728", alpha=0.18)
    ax.axhline(1.0, color="#777777", linewidth=0.8, linestyle="--")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("GPU batch size")
    ax.set_ylabel("Final test NMSE ratio vs exact GPU")
    fig.tight_layout()
    fig.savefig(FIGURE_ROOT / "batch_fitness_distortion.png", dpi=220)
    plt.close(fig)

    pareto = batch_pairs.groupby("gpu_batch_size").agg(
        speedup=("wall_speedup_vs_cpu", "median"),
        test_ratio=("ratio_test_nmse", "median"),
    )
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    ax.scatter(pareto["speedup"], pareto["test_ratio"], color="#d62728")
    for batch_size, row in pareto.iterrows():
        ax.annotate(str(int(batch_size)), (row["speedup"], row["test_ratio"]), xytext=(4, 3), textcoords="offset points")
    ax.axhline(1.0, color="#777777", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Median wall-clock speedup vs CPU")
    ax.set_ylabel("Median final test NMSE ratio vs exact GPU")
    fig.tight_layout()
    fig.savefig(FIGURE_ROOT / "quality_speed_tradeoff.png", dpi=220)
    plt.close(fig)


def write_metrics(
    frame: pd.DataFrame,
    acceleration: pd.DataFrame,
    exact: pd.DataFrame,
    batch: pd.DataFrame,
    cpu_exact: pd.DataFrame,
    distortion: pd.DataFrame,
    generations: pd.DataFrame,
    paper_g: pd.DataFrame,
    ablation: pd.DataFrame,
) -> None:
    exact_summary = acceleration[acceleration["variant"] == "Exact GPU"].iloc[0]
    batch_summary = acceleration[acceleration["variant"].str.startswith("Batch")].copy()
    best_batch = batch_summary.sort_values("median_wall_speedup", ascending=False).iloc[0]
    x32_exact = exact[exact["row_scale"] == 32]["wall_speedup_vs_cpu"]
    x32_batch = batch[batch["row_scale"] == 32]["wall_speedup_vs_cpu"]
    best_distortion = distortion.iloc[(distortion["median_test_nmse_ratio"] - 1.0).abs().argmin()]
    worst_distortion = distortion.iloc[(distortion["median_test_nmse_ratio"] - 1.0).abs().argmax()]
    metrics = {
        "AccelerationRuns": len(frame[frame["suite"] == "acceleration"]),
        "FitnessRuns": len(frame[frame["suite"] == "fitness-evolution"]),
        "GenerationRecords": len(generations),
        "ExactMedianSpeedup": fmt(exact_summary["median_wall_speedup"]),
        "ExactSpeedupCI": f"{fmt(exact_summary['ci_low'])}--{fmt(exact_summary['ci_high'])}",
        "ExactMaxSpeedup": fmt(exact_summary["max_wall_speedup"]),
        "ExactFasterFraction": pct(exact_summary["fraction_faster_than_cpu"]),
        "BestBatchVariant": latex_escape(best_batch["variant"]),
        "BestBatchMedianSpeedup": fmt(best_batch["median_wall_speedup"]),
        "BestBatchMaxSpeedup": fmt(batch["wall_speedup_vs_cpu"].max()),
        "BestBatchFasterFraction": pct(best_batch["fraction_faster_than_cpu"]),
        "ExactXThirtyTwoMedianSpeedup": fmt(x32_exact.median()),
        "BatchXThirtyTwoMedianSpeedup": fmt(x32_batch.median()),
        "ExactFitnessMedianRatioVsCPU": fmt(cpu_exact["test_nmse_ratio_vs_cpu"].median(), 3),
        "BestFidelityBatch": str(int(best_distortion["gpu_batch_size"])),
        "BestFidelityRatio": fmt(best_distortion["median_test_nmse_ratio"], 3),
        "WorstFidelityBatch": str(int(worst_distortion["gpu_batch_size"])),
        "WorstFidelityRatio": fmt(worst_distortion["median_test_nmse_ratio"], 3),
        "SignificantDistortionBatchCount": int((distortion["wilcoxon_holm_p"] < 0.05).sum()),
        "PaperGLocalRuns": int(paper_g["local_runs"].sum()),
        "ExactOptimizationSpeedup": "--",
        "BatchOptimizationSpeedup": "--",
    }
    if not ablation.empty:
        exact_ablation = ablation[ablation["backend"] == "gpu_exact_gom"]
        batch_ablation = ablation[ablation["backend"] == "gpu_batch_gom"]
        if len(exact_ablation):
            metrics["ExactOptimizationSpeedup"] = fmt(exact_ablation.iloc[0]["median_wall_speedup"])
        if len(batch_ablation):
            metrics["BatchOptimizationSpeedup"] = fmt(batch_ablation.iloc[0]["median_wall_speedup"])
    lines = []
    for name, value in metrics.items():
        lines.append(rf"\newcommand{{\{name}}}{{{value}}}")
    (GENERATED_ROOT / "metrics.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate tables, figures, and LaTeX macros for the conference report.")
    parser.add_argument("--input", type=Path, default=REPO_ROOT / "results" / "final_results.csv")
    args = parser.parse_args()

    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    GENERATED_ROOT.mkdir(parents=True, exist_ok=True)
    frame = normalize_batch_metrics(latest_rows(pd.read_csv(args.input)))

    exact, batch = pair_acceleration(frame)
    if exact.empty or batch.empty:
        raise SystemExit("The acceleration suite is incomplete.")
    acceleration = acceleration_summary(exact, batch)
    scaling_exact, scaling_batch = pair_acceleration(frame, suite="batch-scaling")
    row_exact, row_batch = pair_acceleration(frame, suite="exact-row-scale")
    cpu_exact, batch_pairs = fitness_pairs(frame)
    if cpu_exact.empty or batch_pairs.empty:
        raise SystemExit("The fitness-evolution suite is incomplete.")
    distortion = fitness_distortion_summary(batch_pairs)
    generations = read_generation_logs()
    if generations.empty:
        raise SystemExit("No valid fitness-evolution generation logs were found.")
    validate_completeness(frame, generations)
    paper_g = paper_g_comparison(frame)
    ablation = optimization_ablation()
    gom_behavior = gom_behavior_summary(generations)
    profiling = profiling_summary()
    if ablation.empty:
        raise SystemExit("The optimization ablation is missing.")
    if profiling.empty:
        raise SystemExit("The Nsight Systems profiling summary is missing.")
    datasets = dataset_summary(frame)

    acceleration.to_csv(GENERATED_ROOT / "acceleration_summary.csv", index=False)
    exact.to_csv(GENERATED_ROOT / "acceleration_exact_pairs.csv", index=False)
    batch.to_csv(GENERATED_ROOT / "acceleration_batch_pairs.csv", index=False)
    if not scaling_exact.empty:
        scaling_exact.to_csv(GENERATED_ROOT / "batch_scaling_exact_pairs.csv", index=False)
    if not scaling_batch.empty:
        scaling_batch.to_csv(GENERATED_ROOT / "batch_scaling_batch_pairs.csv", index=False)
    if not row_exact.empty:
        row_exact.to_csv(GENERATED_ROOT / "row_scale_exact_pairs.csv", index=False)
    if not row_batch.empty:
        row_batch.to_csv(GENERATED_ROOT / "row_scale_batch_pairs.csv", index=False)
    cpu_exact.to_csv(GENERATED_ROOT / "fitness_cpu_exact_pairs.csv", index=False)
    batch_pairs.to_csv(GENERATED_ROOT / "fitness_batch_pairs.csv", index=False)
    distortion.to_csv(GENERATED_ROOT / "fitness_distortion_summary.csv", index=False)
    paper_g.to_csv(GENERATED_ROOT / "paper_g_comparison.csv", index=False)
    if not ablation.empty:
        ablation.to_csv(GENERATED_ROOT / "optimization_ablation_summary.csv", index=False)
    if not gom_behavior.empty:
        gom_behavior.to_csv(GENERATED_ROOT / "gom_behavior_summary.csv", index=False)
    if not profiling.empty:
        profiling.to_csv(GENERATED_ROOT / "profiling_summary.csv", index=False)
    datasets.to_csv(GENERATED_ROOT / "datasets.csv", index=False)

    write_tables(
        datasets,
        acceleration,
        exact,
        batch,
        scaling_exact,
        scaling_batch,
        distortion,
        paper_g,
        ablation,
        gom_behavior,
        profiling,
    )
    plot_acceleration(exact, batch, scaling_exact, scaling_batch, row_exact, row_batch)
    plot_fitness(generations, batch_pairs)
    write_metrics(frame, acceleration, exact, batch, cpu_exact, distortion, generations, paper_g, ablation)
    print(f"Wrote report assets under {REPORT_ROOT}")


if __name__ == "__main__":
    main()
