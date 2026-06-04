# Conference Report

`main.tex` is the conference-style report for the exact and batched GPU GP-GOMEA work.

The report reads benchmark outputs from `results/` through
`experiments/generate_conference_report_assets.py`. Raw benchmark data remains outside
the report directory; generated tables, figures, and LaTeX macros are written under
`report/generated/` and `report/figures/`.

The report also requires the cumulative optimization ablation and representative
Nsight Systems profiles. Summarize the profiles, generate the report assets, and
build the PDF after all experiments have completed:

```bash
python3 experiments/summarize_nsys_profiles.py
python3 experiments/generate_conference_report_assets.py
make -C report
```

The selected Paper-G comparison is an external-reference comparison only. The report
explicitly states that the full 30-repetition, 1000-second paper matrix was not rerun.
