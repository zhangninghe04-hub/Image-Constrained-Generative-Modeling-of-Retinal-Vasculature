# Soft Connected Terminal Refinement

This update continues from the previous HRF small-vessel analysis. The earlier four-method comparison showed that `connected_recovery` had the highest vessel sensitivity and terminal-vessel sensitivity, but it also produced the most false positives. The latest structure-aware filtering reduced false positives more strongly, but it over-pruned terminal small vessels.

The current step keeps `connected_recovery` as the single base method and applies a softer structural constraint. The goal is not to repeat all four methods, but to refine the most useful candidate for terminal small-vessel recovery.

## Method

The refinement starts from `connected_recovery_raw`. Candidate vessel components are preserved when they are structurally close to the baseline vessel tree or supported near baseline endpoints. Weak vessel-like components are removed when they are not sufficiently supported by area or vesselness strength.

This is a softer version of the previous structure-aware filtering:

- previous filtering: stronger FP reduction, but terminal sensitivity dropped too much;
- current filtering: smaller FP reduction, but terminal sensitivity is better preserved.

## Quantitative Result

| Method | Dice | Sensitivity | Precision | Terminal Sensitivity | FP / GT Area | FN / GT Area |
|---|---:|---:|---:|---:|---:|---:|
| baseline_raw | 0.550 | 0.879 | 0.406 | 0.668 | 1.391 | 0.121 |
| connected_recovery_raw | 0.532 | 0.902 | 0.382 | 0.684 | 1.578 | 0.098 |
| soft_structure_refined | 0.545 | 0.886 | 0.399 | 0.673 | 1.449 | 0.114 |

Compared with `connected_recovery_raw`, the soft refinement improves Dice and precision and reduces FP / GT area, while keeping terminal sensitivity close to the raw connected-recovery result.

## Interpretation

The result confirms that terminal small-vessel recovery and false-positive control need to be balanced. A hard structural filter removes too many terminal vessels, while the softer constraint keeps more terminal vessels but only moderately reduces false positives.

This makes the current result a better next-stage baseline than the previous hard filter. The next improvement should focus on separating true terminal branches from background noise more selectively, instead of increasing or decreasing sensitivity globally.
