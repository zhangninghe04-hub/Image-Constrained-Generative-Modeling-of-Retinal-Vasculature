# Latest Run Summary

Generated on 2026-06-08 from the 15 fundus images in `data/raw/healthy/`.

## Main Update

This run continues the fundus-only modeling stage because OCTA images are not
yet available. The update refines the density-aware direction rule by sampling
fundus-derived vessel density both at the candidate branch endpoint and along a
short forward horizon. The parameter search was also adjusted to preserve the
target terminal-count range instead of selecting overly dense trees.

The update adds:

- short-horizon density-guided direction selection
- `density_angle_span` and `density_horizon_weight` configuration parameters
- a focused 288-combination parameter search
- updated evaluation CSV files and figures
- tests for the density horizon behavior

## Selected Parameters

The focused parameter search tested 288 combinations. The selected full
density-aware setting is:

- alpha: `0.76`
- max_depth: `6`
- initial_length: `0.26`
- global density weight: `0.60`
- density depth weight: `0.50`
- density direction weight: `0.50`
- density survival weight: `0.05`
- density angle span: `1.50`
- density horizon weight: `0.50`

This setting keeps the full density-aware model near the intended terminal-count
range while improving the density-lift metric relative to the previous run.

## 15-Image Average Metrics

| model | terminals | length | occupied grid coverage | coverage dispersion | terminal density score | matched density score | density lift vs random |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 29.000 | 4.234 | 0.057 | 0.010 | 0.327 | 0.324 | 0.157 |
| Constrained | 24.400 | 5.466 | 0.055 | 0.075 | 0.134 | 0.130 | -0.032 |
| Density Depth Only | 25.400 | 5.687 | 0.056 | 0.049 | 0.127 | 0.122 | -0.042 |
| Density Direction Only | 26.200 | 5.814 | 0.057 | 0.079 | 0.152 | 0.150 | -0.012 |
| Density Survival Only | 22.667 | 5.268 | 0.050 | 0.052 | 0.121 | 0.119 | -0.046 |
| Density-Aware | 24.533 | 5.589 | 0.055 | 0.076 | 0.152 | 0.150 | -0.013 |

## Interpretation

The previous full density-aware run averaged `20.733` terminals and occupied
grid coverage of `0.037`. The updated model averages `24.533` terminals and
occupied grid coverage of `0.055`, which gives broader retinal reach while
remaining near the target terminal-count range.

The matched density score is `0.150`, and density lift over random is `-0.013`.
This is close to the random-point baseline and improves the relative density
placement behavior compared with the previous clearly negative lift value. The
absolute terminal density score still remains below the baseline model, so the
fundus-only density objective is not fully solved.

## Next Priority

The next modeling step should improve absolute terminal placement in
fundus-derived high-density regions while preserving the current terminal count
and spatial coverage. A useful next direction is to improve the density map or
terminal placement objective before adding OCTA-based constraints.
