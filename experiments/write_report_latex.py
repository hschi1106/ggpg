from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import REPO_ROOT


def fmt(value: float, digits: int = 2) -> str:
    if pd.isna(value):
        return "--"
    return f"{float(value):.{digits}f}"


def batch_acceleration_rows(accel: pd.DataFrame) -> str:
    rows = []
    for _, row in accel.sort_values("gpu_batch_size").iterrows():
        rows.append(
            " & ".join(
                [
                    str(int(row["gpu_batch_size"])),
                    str(int(row["runs"])),
                    fmt(row["median_eval_rate_ratio_vs_exact"], 2),
                    fmt(row["q25_eval_rate_ratio_vs_exact"], 2),
                    fmt(row["q75_eval_rate_ratio_vs_exact"], 2),
                    fmt(row["incremental_ratio_vs_previous_batch"], 2)
                    if not pd.isna(row["incremental_ratio_vs_previous_batch"])
                    else "--",
                ]
            )
            + r" \\"
        )
    return "\n".join(rows)


def batch_distortion_rows(distortion: pd.DataFrame) -> str:
    rows = []
    for _, row in distortion.sort_values("gpu_batch_size").iterrows():
        rows.append(
            " & ".join(
                [
                    str(int(row["gpu_batch_size"])),
                    str(int(row["runs"])),
                    fmt(row["median_delta_test_nmse_paper_scaled"], 2),
                    fmt(row["median_abs_delta_test_nmse_paper_scaled"], 2),
                    fmt(row["median_ratio_test_nmse"], 2),
                    fmt(row["median_eval_rate_ratio_vs_exact"], 2),
                ]
            )
            + r" \\"
        )
    return "\n".join(rows)


def paper_g_rows(comparison: pd.DataFrame, short_reference: pd.DataFrame) -> str:
    merged = comparison.merge(
        short_reference[
            [
                "dataset",
                "report_exact_test_nmse",
                "best_batch_size",
                "best_batch_test_nmse",
                "best_batch_eval_rate_ratio_vs_exact",
            ]
        ],
        on="dataset",
        how="left",
    )
    rows = []
    for _, row in merged.sort_values("dataset").iterrows():
        rows.append(
            " & ".join(
                [
                    row["dataset"],
                    fmt(row["paper_g_test_nmse"], 2),
                    fmt(row["local_test_nmse"], 2),
                    fmt(row["local_over_paper_test_nmse"], 2),
                    fmt(row["report_exact_test_nmse"], 2),
                    str(int(row["best_batch_size"])),
                    fmt(row["best_batch_test_nmse"], 2),
                    fmt(row["best_batch_eval_rate_ratio_vs_exact"], 2),
                ]
            )
            + r" \\"
        )
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the conference-style LaTeX report.")
    parser.add_argument("--report-dir", type=Path, default=REPO_ROOT / "results" / "report")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "results" / "report" / "gpu_gomea_paper_style_report.tex")
    args = parser.parse_args()

    report_dir = args.report_dir
    accel = pd.read_csv(report_dir / "batch_acceleration_overall.csv")
    distortion = pd.read_csv(report_dir / "batch_distortion_overall.csv")
    paper_g = pd.read_csv(report_dir / "paper_g_rerun_comparison.csv")
    short_reference = pd.read_csv(report_dir / "paper_reference_comparison.csv")

    selected_time_limit = int(paper_g["time_limit_seconds"].iloc[0])
    selected_runs = int(paper_g["local_runs"].sum())
    selected_seeds = int(paper_g["local_runs"].max())
    max_speedup = fmt(accel["median_eval_rate_ratio_vs_exact"].max(), 2)
    batch_8_jump = fmt(
        accel.loc[accel["gpu_batch_size"] == 8, "incremental_ratio_vs_previous_batch"].iloc[0],
        2,
    )
    batch_128_gain = fmt(
        accel.loc[accel["gpu_batch_size"] == 128, "incremental_ratio_vs_previous_batch"].iloc[0],
        2,
    )

    tex = rf"""\documentclass[10pt,twocolumn]{{article}}
\usepackage[margin=0.72in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{hyperref}}
\usepackage{{amsmath}}
\usepackage{{caption}}
\usepackage{{microtype}}
\graphicspath{{{{results/report/}}{{./}}}}

\title{{Batched GPU Fitness Evaluation for Model-Based Genetic Programming in Symbolic Regression}}
\author{{Anonymous Authors}}
\date{{}}

\begin{{document}}
\maketitle

\begin{{abstract}}
Model-based genetic programming is attractive for symbolic regression because linkage learning can exploit regularities in expression structure, but the sequential accept--reject behavior of Gene-pool Optimal Mixing (GOM) creates a poor match for massively parallel hardware.
We study a CUDA implementation of GP-GOMEA that separates two acceleration strategies: an exact GPU fitness backend that preserves GOM timing, and a batched GPU variant that evaluates multiple candidate expressions per launch while deliberately relaxing the immediacy of population updates.
On a bounded four-dataset study, batching increases candidate-evaluation throughput up to {max_speedup}\(\times\) relative to exact GPU GOM.
However, the final fitness changes are non-monotonic across batch sizes, confirming that batched GOM is an approximate search algorithm rather than a semantics-preserving implementation optimization.
\end{{abstract}}

\section{{Introduction}}

Symbolic regression searches for compact mathematical expressions that fit numerical data.
The GP-GOMEA approach studied by Virgolin et al.~\cite{{virgolin}} improves standard genetic programming by using a model of gene dependencies to guide variation over expression trees.
This model-based variation is effective for small expressions, but it is also difficult to accelerate on GPUs: GOM evaluates candidate edits sequentially, and each accepted edit changes the state seen by subsequent edits.

This work asks whether GPU evaluation can still be useful when the evolutionary operator itself is sequential.
We investigate two designs.
The first preserves the original GOM control flow and moves only fitness evaluation to the GPU.
The second batches candidate evaluations across individuals and FOS steps, reducing kernel-launch overhead at the cost of delayed update visibility.
The central question is therefore not only how much speed is gained, but how much the batched approximation changes search quality.

\section{{Research Questions}}

We organize the study around three questions.

\textbf{{RQ1.}} How does candidate-evaluation throughput scale as GPU batch size increases?

\textbf{{RQ2.}} How does batched GOM affect final validation and test fitness relative to exact GPU GOM under the same short budget?

\textbf{{RQ3.}} How do selected local reruns of the reference paper's fixed-population GP-GOMEA setting compare with the paper's reported scale?

\section{{Method}}

\subsection{{Execution Backends}}

The implementation exposes three GP-GOMEA execution modes.
\texttt{{cpu\_original}} preserves the upstream CPU behavior.
\texttt{{gpu\_exact\_gom}} preserves the original GOM accept--reject order and evaluates each meaningful candidate immediately on GPU.
\texttt{{gpu\_batch\_gom}} forms batches of candidate expressions, evaluates them in one CUDA launch, and then applies local accept--reject decisions.

\subsection{{GPU Program Representation}}

Expression trees remain host-side objects.
Before GPU evaluation, each active tree is serialized into a postfix token sequence containing opcodes, feature indices, and constants.
The device kernel evaluates these token programs with a stack machine over the training samples.
For batched evaluation, token programs are stored in a fixed-stride padded layout so a CUDA grid can parallelize over both program index and sample slice.
This representation avoids sending pointer-rich tree structures, virtual operators, or recursive traversal to device code.

\subsection{{Approximate Batched GOM}}

Exact GOM commits an accepted edit immediately.
The batched backend changes that dependency structure: several candidates are generated from a temporarily shared state, evaluated together, and committed after the batch-local decision step.
The approximation is intentional.
It exposes enough independent work to use the GPU efficiently, but it also changes donor visibility and random-search trajectory.
We therefore evaluate batching as a distinct algorithmic variant.

\section{{Experimental Design}}

The bounded GPU study uses four datasets from the reference-paper benchmark suite: Airfoil, Dow chemical, Wine white, and Yacht hydrodynamics.
All runs use height 4 expressions, corresponding to the \(l=31\) setting in the paper.
For the batch-size study, we run seeds 0--2, population 128, and batch sizes \(1, 8, 16, 32, 64, 128\).
Errors are reported as normalized MSE on the paper scale \(100 \times \mathrm{{MSE}}/\mathrm{{var}}(y)\).

We also rerun the selected Paper-G rows shown in the earlier comparison table.
Those local reruns use \texttt{{cpu\_original}}, \(l=31\), \(\mathrm{{npop}}=1000\), seeds 0--{selected_seeds - 1}, and a {selected_time_limit}s budget, yielding {selected_runs} completed runs.
The original paper values remain a reference scale rather than a claim of protocol-equivalent reproduction: we did not rerun the full 30-repetition paper matrix, and the current scripted function set uses \texttt{{+,-,*,/}} rather than the paper's analytic quotient configuration.

\section{{Results}}

\subsection{{Acceleration by Batch Size}}

Let \(R_b\) denote the ratio between batched and exact-GPU candidate evaluations per second at batch size \(b\).
Table~\ref{{tab:batch-acceleration}} shows that throughput improves rapidly once multiple candidates are evaluated per launch.
The largest incremental gain is from batch 1 to batch 8, where median throughput increases by {batch_8_jump}\(\times\).
The curve then saturates: batch 128 reaches the best median ratio, {max_speedup}\(\times\), but improves over batch 64 by only {batch_128_gain}\(\times\).

\begin{{table}}[t]
\centering
\caption{{Throughput ratio by GPU batch size. \(R_b\) is eval/s relative to exact GPU GOM; Q1 and Q3 summarize per-run ratios.}}
\label{{tab:batch-acceleration}}
\begin{{tabular}}{{rrrrrr}}
\toprule
Batch & Runs & Median \(R_b\) & Q1 & Q3 & Incr. \\
\midrule
{batch_acceleration_rows(accel)}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{figure}}[t]
\centering
\includegraphics[width=\linewidth]{{batch_acceleration_ratio.png}}
\caption{{Evaluation-throughput ratio relative to exact GPU GOM. The shaded band is the interquartile range across datasets and seeds.}}
\label{{fig:batch-acceleration}}
\end{{figure}}

\subsection{{Fitness Distortion}}

We define test-set distortion for batch size \(b\) as
\begin{{align*}}
  \Delta_b &=
  \mathrm{{NMSE}}_{{test}}(\texttt{{batch}}, b) \\
  &\quad - \mathrm{{NMSE}}_{{test}}(\texttt{{exact}}).
\end{{align*}}
Negative values mean that the batched variant found a lower test error within the same short-budget setting.
Table~\ref{{tab:batch-distortion}} shows that the relationship between batch size and final fitness is not monotonic.
Some larger batches improve median test NMSE because they evaluate many more candidates, but the nonzero median absolute distortions show that batching changes the search path.

\begin{{table}}[t]
\centering
\caption{{Batch-induced test-fitness distortion. NMSE deltas use the paper scale.}}
\label{{tab:batch-distortion}}
\begin{{tabular}}{{rrrrrr}}
\toprule
Batch & Runs & Med. \(\Delta_b\) & Med. \(|\Delta_b|\) & Ratio & \(R_b\) \\
\midrule
{batch_distortion_rows(distortion)}
\bottomrule
\end{{tabular}}
\end{{table}}

\subsection{{Selected Paper-G Rerun}}

Table~\ref{{tab:paper-g-rerun}} compares the original paper's fixed-population \(G\), \(l=31\), \(\mathrm{{npop}}=1000\) test results with our selected local rerun and the best short-budget GPU-batch result from the bounded ablation.
The local rerun is weaker than the paper reference on Airfoil, Dow chemical, and Yacht hydrodynamics, and closer on Wine white.
This is consistent with the reduced budget, fewer repetitions, and remaining function-set mismatch.

\begin{{table*}}[t]
\centering
\caption{{Original Paper-G reference, selected local Paper-G rerun, and bounded GPU-ablation reference. Lower test NMSE is better.}}
\label{{tab:paper-g-rerun}}
\begin{{tabular}}{{lrrrrrrr}}
\toprule
Dataset & Paper G & Local G & Local/Paper & Exact GPU & Best batch & Batch test & Batch \(R_b\) \\
\midrule
{paper_g_rows(paper_g, short_reference)}
\bottomrule
\end{{tabular}}
\end{{table*}}

\section{{Discussion}}

The results support a throughput--fidelity tradeoff.
Exact GPU GOM is useful as a semantic baseline, but the workload is too fine-grained to fully amortize GPU launch overhead.
Batched GOM exposes more parallelism and reaches substantially higher candidate-evaluation throughput, especially for batch sizes 32 and above.
At the same time, the final-fitness response is not determined by throughput alone.
Batching changes when successful edits become visible, so it can either help by expanding the number of evaluated candidates or hurt by moving away from exact GOM's trajectory.

For practical use, batch size should therefore be treated as a search hyperparameter.
The bounded study suggests that batch sizes 64 and 128 are attractive for throughput, while intermediate sizes may sometimes provide a better quality--speed compromise.
Publication-quality conclusions about symbolic-regression accuracy require the full reference matrix and exact alignment with the paper's operator semantics.

\section{{Limitations}}

The full reference-paper experiment matrix was not rerun.
The bounded GPU study uses four datasets, three seeds, population 128, and short runs.
The selected Paper-G rerun uses only four datasets and three seeds, with a shorter budget than the reference paper's 30-repetition, 1000s protocol.
The current scripted function set also differs from the paper's analytic quotient setup.
Accordingly, this report supports claims about implementation feasibility, throughput scaling, and batch-induced search distortion, but not final benchmark superiority over the published GP-GOMEA results.

\section{{Conclusion}}

We studied GPU acceleration for GP-GOMEA symbolic regression under the sequential constraints of Gene-pool Optimal Mixing.
The exact GPU backend preserves GOM semantics but remains limited by many small evaluations.
The batched backend substantially improves evaluation throughput, reaching {max_speedup}\(\times\) exact-GPU eval/s in the bounded study, but it also changes the search process.
The main research conclusion is that batched GPU GOM should be presented as an approximate model-based GP variant with a tunable batch-size parameter, not as a transparent replacement for exact GP-GOMEA.

\begin{{thebibliography}}{{1}}
\bibitem{{virgolin}}
Virgolin et al.,
\emph{{Improving Model-based Genetic Programming for Symbolic Regression of Small Expressions}},
Evolutionary Computation, DOI: 10.1162/evco\_a\_00278.
\end{{thebibliography}}

\end{{document}}
"""

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(tex, encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
