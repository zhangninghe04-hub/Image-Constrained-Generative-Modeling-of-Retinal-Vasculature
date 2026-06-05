# Current Stage Summary

This repository is currently focused on the fundus-only stage of the project.
The original proposal includes a future fundus + OCTA framework, but OCTA data
are not yet available. The present implementation therefore evaluates the
fundus-constrained generative model and the density-aware branching update.

## Current Modeling Status

The working pipeline has three model levels:

1. Baseline: fixed manual branching parameters.
2. Image-constrained: optic disc, macula, and major vessel orientation are
   extracted from each fundus image and used to parameterize generation.
3. Density-aware: the fundus-derived vessel-density map affects branch depth,
   branch direction, and branch survival.

The density-aware model is now implemented as separable mechanisms so the
result pipeline can run ablation comparisons:

- Density Depth Only
- Density Direction Only
- Density Survival Only
- Full Density-Aware

The latest selected full density-aware setting comes from a focused parameter
search and prioritizes recovering spatial reach:

- alpha: `0.76`
- max depth: `7`
- initial length: `0.23`
- density depth weight: `0.75`
- density direction weight: `0.70`
- density survival weight: `0.00`
- global density weight: `0.60`

## Evaluation Priorities

The next runs should focus on whether the full density-aware model preserves
the coverage improvement reported in the May 2026 progress report without
over-pruning the generated tree.

The most important quantities are:

- terminal count
- total vessel length
- occupied grid coverage
- coverage dispersion
- density correlation
- terminal density score

Coverage dispersion is the nearest-neighbor distance standard deviation among
terminal nodes, where lower values indicate more uniform terminal spacing.
Occupied grid coverage is a positive coverage metric, where higher values mean
terminal nodes reach more of the retinal field.

## How to Regenerate Results

Place the 15 fundus images in `data/raw/healthy/` and run:

```bash
python run_results.py
```

The script writes:

- `results/evaluation_results.csv`
- `results/evaluation_summary.csv`
- `results/ablation_summary.csv`
- `results/latest_run_summary.md`
- `results/parameter_search.csv`
- `results/best_parameter_summary.md`
- `figures/fig1_constraint_overlays.png`
- `figures/fig2_model_comparison.png`
- `figures/fig3_density_terminals.png`
- `figures/fig4_evaluation_summary.png`
- `figures/fig5_terminal_density_overlay.png`
