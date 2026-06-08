# Progress Reports

This page lists the staged progress reports for the fundus-constrained retinal vasculature modeling project.

## Reports

| Date | File | Focus |
|---|---|---|
| 2026-04-16 | [`progress_2026-04-16.md`](progress_2026-04-16.md) | Repository setup, baseline model, initial constraint extraction. |
| 2026-04-21 | [`4_21_progress_report.pdf`](4_21_progress_report.pdf) | Early image-constrained pipeline and initial results. |
| 2026-04-29 | [`progress_2026-04-29_density_update.pdf`](progress_2026-04-29_density_update.pdf) | Density-aware direction and survival update. |
| 2026-05-02 | [`5_2_progress_report.pdf`](5_2_progress_report.pdf) | Fundus-constrained density-aware model over 15 images. |
| 2026-05-20 | [`5_20_progress_report.pdf`](5_20_progress_report.pdf) / [`progress_2026-05-20_density_calibration.md`](progress_2026-05-20_density_calibration.md) | Density-aware calibration, ablation study, and reduced over-pruning. |
| 2026-06-05 | [`6_05_progress_report.pdf`](6_05_progress_report.pdf) / [`progress_2026-06-05_parameter_search.md`](progress_2026-06-05_parameter_search.md) | Parameter search, fair density metrics, and terminal-over-density visualization. |

## Current Project Scope

OCTA images are not yet available, so the current implementation remains fundus-only. The latest work focuses on recovering spatial reach in the density-aware model while preserving image-specific structure and improving density evaluation.

For the latest executable notebook, see [`notebooks/03_parameter_search_density_evaluation.ipynb`](../notebooks/03_parameter_search_density_evaluation.ipynb).
