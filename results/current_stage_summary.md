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

The latest density-aware update refines branch direction selection. Candidate
directions are now evaluated using endpoint density and short-horizon density
sampled slightly beyond the endpoint. This helps the model prefer directions
with better forward density support rather than only reacting to one local cell.

The current selected full density-aware setting is:

- alpha: `0.76`
- max depth: `6`
- initial length: `0.26`
- density depth weight: `0.50`
- density direction weight: `0.50`
- density survival weight: `0.05`
- density angle span: `1.50`
- density horizon weight: `0.50`
- global density weight: `0.60`

## Evaluation Priorities

The current model should be evaluated as a fundus-only constrained generator.
The most important quantities are:

- terminal count
- total vessel length
- occupied grid coverage
- coverage dispersion
- terminal density score
- matched terminal density score
- density lift over random

The latest full density-aware model averages `24.533` terminals and occupied
grid coverage of `0.055` across 15 fundus images. Density lift over random is
`-0.013`, which is close to the random-point baseline but still leaves room for
better absolute density matching.

## How to Regenerate Results

Place the 15 fundus images in `data/raw/healthy/` and run:

```bash
python run_results.py
```

The script writes:

- `results/evaluation_results.csv`
- `results/evaluation_summary.csv`
- `results/ablation_summary.csv`
- `figures/fig1_constraint_overlays.png`
- `figures/fig2_model_comparison.png`
- `figures/fig3_density_terminals.png`
- `figures/fig4_evaluation_summary.png`
- `figures/fig5_terminal_density_overlay.png`

To rerun the parameter search:

```bash
python run_parameter_search.py
```
