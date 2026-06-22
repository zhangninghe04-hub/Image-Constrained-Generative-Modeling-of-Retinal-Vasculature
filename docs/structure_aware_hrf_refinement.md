# Structure-Aware HRF Refinement

This update adds a structure-aware post-processing step for HRF vessel segmentation after the previous FP/FN analysis showed that small-vessel recovery was too sensitive and produced many false-positive vessel candidates.

## Motivation

The previous methods intentionally increased sensitivity to recover missed terminal small vessels. This helped reduce false negatives, but it also introduced many isolated and short vessel-like structures in the background. The next refinement therefore focuses on adding structural constraints instead of only increasing small-vessel sensitivity.

## Method

The same four HRF segmentation outputs were used as input:

- `baseline`
- `multiscale_line`
- `clean_recovery`
- `connected_recovery`

For each method, the raw prediction was compared with a structure-filtered version. The filtering step includes:

- removal of small connected components,
- removal of short skeleton branches,
- suppression of vessel candidates that are not connected to the main vessel tree,
- skeleton-distance evaluation in addition to pixel-overlap metrics.

This step is designed as a post-processing constraint. It does not replace the existing segmentation pipeline.

## Quantitative Summary

The structure-aware filtering improved overall overlap and precision for all four methods. It reduced false positives substantially, but terminal sensitivity decreased because some true small terminal vessels were also filtered out.

| Method | Refinement | Dice | Sensitivity | Precision | Terminal Sensitivity | FP / GT Area | FN / GT Area | Skeleton F1 r=5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | raw | 0.550 | 0.879 | 0.406 | 0.668 | 1.391 | 0.121 | 0.721 |
| baseline | structure_filtered | 0.673 | 0.807 | 0.586 | 0.475 | 0.607 | 0.193 | 0.803 |
| multiscale_line | raw | 0.549 | 0.887 | 0.403 | 0.673 | 1.422 | 0.113 | 0.721 |
| multiscale_line | structure_filtered | 0.669 | 0.837 | 0.564 | 0.506 | 0.694 | 0.163 | 0.814 |
| clean_recovery | raw | 0.545 | 0.893 | 0.398 | 0.676 | 1.464 | 0.107 | 0.721 |
| clean_recovery | structure_filtered | 0.661 | 0.842 | 0.550 | 0.510 | 0.737 | 0.158 | 0.815 |
| connected_recovery | raw | 0.532 | 0.902 | 0.382 | 0.684 | 1.578 | 0.098 | 0.723 |
| connected_recovery | structure_filtered | 0.634 | 0.865 | 0.507 | 0.545 | 0.903 | 0.135 | 0.819 |

## Interpretation

The results show a clear trade-off. Structure-aware filtering is effective for removing isolated false positives and improving skeleton-level agreement, but it is too conservative for terminal vessels. This confirms that the next step should not be simple hard filtering. A better strategy is to separate large-vessel structure preservation from small-vessel recovery.

## Next Step

The next refinement should use a hierarchical vessel strategy:

- extract and stabilize the main vessel tree first,
- recover small terminal vessels around the main tree,
- prune short isolated branches only when they are not structurally supported,
- evaluate results using both pixel metrics and skeleton-distance metrics.
