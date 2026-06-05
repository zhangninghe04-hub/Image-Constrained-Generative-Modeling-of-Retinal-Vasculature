# Latest Run Summary

Generated on 2026-06-05 from the 15 fundus images in `data/raw/healthy/`.

## Main Update

This run focuses on recovering spatial reach in the fundus-constrained and
density-aware models while keeping density-aware terminal placement active.
OCTA data are still unavailable, so the project remains fundus-only.

The update adds:

- `run_parameter_search.py`
- `results/parameter_search.csv`
- `results/best_parameter_summary.md`
- matched terminal-count density metrics
- `figures/fig5_terminal_density_overlay.png`

## Selected Parameters

The focused parameter search tested 96 combinations. The selected full
density-aware setting is:

- alpha: `0.76`
- max_depth: `7`
- initial_length: `0.23`
- global density weight: `0.60`
- density depth weight: `0.75`
- density direction weight: `0.70`
- density survival weight: `0.00`

This setting prioritizes recovering terminal count and occupied grid coverage.

## 15-Image Average Metrics

| model | terminals | length | occupied grid coverage | coverage dispersion | terminal density score | matched density score | density lift vs random |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 29.000 | 4.234 | 0.057 | 0.010 | 0.277 | 0.284 | -0.015 |
| Constrained | 22.200 | 3.645 | 0.039 | 0.077 | 0.157 | 0.158 | -0.148 |
| Density Depth Only | 18.867 | 3.186 | 0.033 | 0.047 | 0.176 | 0.177 | -0.136 |
| Density Direction Only | 20.733 | 3.551 | 0.037 | 0.051 | 0.187 | 0.188 | -0.126 |
| Density Survival Only | 16.267 | 2.899 | 0.030 | 0.058 | 0.176 | 0.176 | -0.137 |
| Density-Aware | 20.733 | 3.551 | 0.037 | 0.051 | 0.187 | 0.188 | -0.126 |

## Interpretation

The previous calibrated density-aware model averaged `14.867` terminals and
occupied grid coverage of `0.027`. The new selected setting increases the full
density-aware model to `20.733` terminals and occupied grid coverage of `0.037`.
This directly addresses the prior weakness that constrained models were too
spatially limited.

The tradeoff is that terminal density score decreases from the previous
full density-aware value of `0.217` to `0.187`. This means the model now reaches
more of the retinal field, but the terminal distribution is less concentrated in
the highest-density regions. This is a useful tradeoff to document because the
project is now balancing two goals: spatial reach and density matching.

The matched terminal-count density score was added to make density comparison
fairer across models with different terminal counts. Density lift over random is
still negative for all models, which shows that the current density map and
terminal placement metric need further refinement.

## Next Priority

The next modeling step should improve density matching under the recovered
terminal count. In practical terms, the model should keep approximately
`20-25` terminals while increasing matched terminal density score and density
lift over random.

The most promising next direction is to improve the direction-selection rule
and terminal placement objective, rather than increasing branch survival
pruning again.
