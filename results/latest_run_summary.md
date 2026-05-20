# Latest Run Summary

Generated on 2026-05-20 from the 15 fundus images in `data/raw/healthy/`.

## Main Update

The density-aware generator was calibrated to reduce over-pruning. The previous
full density-aware setting made survival too strong and produced an overly sparse
tree. The current run uses a lighter setting:

- density depth weight: `0.75`
- density direction weight: `0.50`
- density survival weight: `0.10`
- global density weight: `0.60`

This keeps the full density-aware model close to the image-constrained model in
terminal count and total length while retaining a stronger terminal density score
than the constrained-only model.

## 15-Image Average Metrics

| model | terminals | length | occupied grid coverage | coverage dispersion | density corr | terminal density score |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 29.000 | 4.234 | 0.057 | 0.010 | -0.030 | 0.277 |
| Constrained | 15.533 | 2.692 | 0.028 | 0.072 | -0.069 | 0.169 |
| Density Depth Only | 14.733 | 2.579 | 0.026 | 0.085 | -0.059 | 0.210 |
| Density Direction Only | 15.333 | 2.711 | 0.027 | 0.093 | -0.058 | 0.214 |
| Density Survival Only | 14.200 | 2.551 | 0.026 | 0.088 | -0.059 | 0.209 |
| Density-Aware | 14.867 | 2.661 | 0.027 | 0.088 | -0.056 | 0.217 |

## Interpretation

The updated density-aware model no longer collapses into a very small terminal
set. It remains slightly sparser than the image-constrained model, but the
terminal density score improves from `0.169` to `0.217`, suggesting that terminal
placement is more responsive to the fundus-derived density map.

The baseline still has the highest occupied grid coverage because it is a fixed,
larger tree. This should not be interpreted as anatomical superiority; it means
the baseline is more diffuse and less image-specific. The next modeling priority
is to recover more spatial reach in the constrained models without losing their
image-specific root placement and vessel orientation.

## Outputs

- `results/evaluation_results.csv`
- `results/evaluation_summary.csv`
- `results/ablation_summary.csv`
- `figures/fig1_constraint_overlays.png`
- `figures/fig2_model_comparison.png`
- `figures/fig3_density_terminals.png`
- `figures/fig4_evaluation_summary.png`
