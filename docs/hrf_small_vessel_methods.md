# HRF Small-Vessel Segmentation Method Comparison

This document records the HRF-focused benchmark requested after the project
discussion. The goal is not to turn the project into a standalone segmentation
or classification task. The goal is to improve the reliability of image-derived
constraints for retinal vascular modeling, especially vessel masks, terminal
branches, and skeleton connectivity.

## Motivation

The current retinal vascular model depends on vessel segmentation, vessel
density maps, terminal vessel structure, and skeleton extraction. If small
terminal vessels are missed, the downstream density-aware model and future
OCTA network analysis can be affected. The HRF dataset is used here because it
provides ground-truth vessel masks.

## Data

The benchmark uses the HRF dataset:

- 45 fundus images
- 45 corresponding ground-truth vessel masks

The raw HRF data are not committed to this repository. The script expects a
local structure:

```text
HRF/
  images/
  masks/
```

## Added Script

The new comparison script is:

```bash
python scripts/compare_hrf_small_vessel_methods.py \
  --hrf-root /path/to/HRF \
  --output-dir results/hrf_small_vessel_methods \
  --figure-dir figures \
  --max-image-side 1600
```

The `--max-image-side 1600` option keeps the HRF ground-truth comparison
reproducible and fast enough for iterative method testing. The script can also
run without resizing for selected full-resolution validation cases.

## Compared Methods

Four lightweight image-processing methods are compared:

1. `baseline`: green-channel enhancement, local contrast enhancement,
   background subtraction, thresholding, and post-processing.
2. `multiscale_line`: baseline plus multiscale line-filter enhancement for
   thin vessel-like structures.
3. `connected_recovery`: baseline plus a more permissive recovery of
   vesselness responses connected to the existing vessel mask.
4. `clean_recovery`: a more conservative recovery rule that only keeps
   vesselness components connected to existing vessels and supported by a
   stronger line response.

These methods are used as interpretable baselines before introducing a more
complex learning-based segmentation model.

## Metrics

Whole-mask metrics:

- Dice
- IoU
- sensitivity
- specificity
- precision

Small-vessel and skeleton-related metrics:

- terminal-region sensitivity
- endpoint recall
- missed endpoint count
- skeleton overlap
- pruned skeleton overlap
- predicted skeleton component count

These metrics are more relevant to the current modeling problem than whole-mask
Dice alone, because the downstream model depends on terminal branches and
skeleton connectivity.

## Preliminary HRF Results

The preliminary HRF benchmark gives the following method-level averages:

| Method | Dice | Sensitivity | Precision | Terminal Sensitivity | Endpoint Recall | Missed Endpoints | Skeleton Overlap | Pruned Skeleton Overlap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.529 | 0.867 | 0.385 | 0.650 | 0.821 | 96.489 | 0.361 | 0.342 |
| multiscale_line | 0.530 | 0.878 | 0.384 | 0.656 | 0.817 | 98.444 | 0.365 | 0.350 |
| clean_recovery | 0.529 | 0.886 | 0.381 | 0.660 | 0.816 | 99.333 | 0.363 | 0.348 |
| connected_recovery | 0.519 | 0.898 | 0.369 | 0.673 | 0.808 | 103.556 | 0.361 | 0.350 |

## Interpretation

The `multiscale_line` method slightly improves Dice, terminal sensitivity, and
skeleton overlap compared with the baseline. This suggests that multiscale
thin-vessel enhancement is a useful direction for recovering small vessels.

The `clean_recovery` method improves terminal sensitivity and pruned skeleton
overlap while keeping the precision drop smaller than the more aggressive
`connected_recovery` method. This makes it a more useful next-step direction for
retinal vascular modeling, where cleaner skeleton connectivity matters.

The `connected_recovery` method gives the highest terminal sensitivity, but it
reduces precision and endpoint recall. This indicates that simply adding more
thin vessel-like responses can introduce false positives and noisy skeleton
branches. For vascular modeling, this tradeoff is important because a noisier
skeleton may harm connectivity analysis even when terminal-region sensitivity
increases.

The next improvement should therefore focus on cleaner recovery of true small
terminal vessels, not only higher pixel-level sensitivity.

## Next Step

The next implementation step should refine the small-vessel recovery rule by:

- preserving thin branches connected to plausible vessel structures
- suppressing isolated false-positive line responses
- pruning skeleton spurs after segmentation
- evaluating terminal-vessel errors and skeleton connectivity
- checking whether improved masks produce more reliable skeleton connectivity

This keeps the work aligned with retinal vascular modeling rather than a
standalone segmentation benchmark.
