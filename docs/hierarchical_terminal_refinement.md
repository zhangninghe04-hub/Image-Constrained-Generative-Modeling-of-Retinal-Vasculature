# Hierarchical Terminal-Vessel Refinement

This update extends the previous soft structural refinement by explicitly adding a hierarchical vessel strategy and skeleton-distance-based evaluation.

## Motivation

The previous soft refinement improved the balance between terminal-vessel preservation and false-positive control, but it still treated candidate vessels mainly through local support rules. Based on the latest discussion, the next step focused on two supported directions:

- process vessels hierarchically by separating a stable main-vessel layer from terminal small-vessel candidates;
- evaluate terminal structures with skeleton-distance metrics in addition to pixel-level overlap.

The goal is not to replace ground-truth evaluation, but to make the analysis more suitable for small terminal vessels, where small spatial shifts can strongly affect pixel overlap.

## Method

The method keeps `baseline_raw` as a conservative main-vessel layer and uses `connected_recovery_raw` as the source of additional small-vessel candidates.

The refinement follows these steps:

1. Extract the stable main-vessel mask from `baseline_raw`.
2. Extract baseline skeleton and endpoint regions.
3. Identify additional candidates from `connected_recovery_raw - baseline_raw`.
4. Recover candidates near endpoint regions using softer terminal-vessel criteria.
5. Apply branch-level skeleton checks, including skeleton length and endpoint-direction support.
6. Evaluate results with both pixel metrics and skeleton-distance metrics.

## Quantitative Summary

| Method | Dice | Sensitivity | Precision | Terminal Sensitivity | FP / GT Area | FN / GT Area | Skeleton F1 r=5 | Terminal Skeleton F1 r=5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_raw | 0.550 | 0.879 | 0.406 | 0.668 | 1.391 | 0.121 | 0.721 | 0.896 |
| connected_recovery_raw | 0.532 | 0.902 | 0.382 | 0.684 | 1.578 | 0.098 | 0.723 | 0.899 |
| soft_structure_refined | 0.545 | 0.886 | 0.399 | 0.673 | 1.449 | 0.114 | 0.720 | 0.897 |
| hierarchical_terminal_refined | 0.545 | 0.881 | 0.400 | 0.670 | 1.433 | 0.119 | 0.721 | 0.897 |

## Interpretation

The hierarchical refinement produced a small but useful shift toward better false-positive control compared with the previous soft refinement. FP / GT area decreased from 1.449 to 1.433, and precision increased slightly from 0.399 to 0.400. Dice remained similar at 0.545.

Terminal sensitivity decreased slightly from 0.673 to 0.670, which indicates that the current hierarchical rules are still conservative around some terminal vessels. However, the drop is much smaller than the earlier hard filtering trial, where terminal sensitivity dropped to 0.545.

This result supports the direction of hierarchical refinement, but also shows that the terminal-recovery rules need further tuning. The next step should refine endpoint-region recovery and branch-level skeleton validation without making the terminal branch criteria too strict.

## Next Step

The next refinement should focus on:

- adjusting terminal-region parameters separately from main-vessel parameters;
- improving branch-level direction continuity;
- using skeleton-distance metrics as a more prominent tuning target;
- reducing false positives without causing a hard-filter-like drop in terminal sensitivity.
