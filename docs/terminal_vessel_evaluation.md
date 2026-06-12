# Terminal Small Vessel Evaluation

This document describes the new fundus-image evaluation step added after the
project direction shifted toward improving the identification of small terminal
vessels in fundus vessel segmentation.

## Motivation

Current fundus vessel segmentation methods often identify major vessels more
reliably than small terminal vessels. Since terminal branches affect downstream
tree extraction and artery/vein tree matching, the next step is to quantify and
improve terminal small vessel detection before further extending the generative
model.

## Data

The local test data folder contains six fundus datasets with image-mask pairs:

| Dataset | Paired Images |
|---|---:|
| AFIO | 99 |
| CHASEDB1 | 28 |
| DualModal2019 | 30 |
| HRF | 45 |
| LES | 22 |
| STARE | 20 |
| Total | 244 |

The data are not committed to the repository. The evaluation script expects the
same structure locally:

```text
test_data/
  DATASET_NAME/
    images/
    masks/
```

## Added Script

The new script is:

```bash
python scripts/evaluate_terminal_vessels.py \
  --data-root /path/to/test_data \
  --output-dir results/terminal_vessel_evaluation \
  --figure-path figures/terminal_vessel_examples.png
```

The script runs a lightweight fundus vessel segmentation baseline and compares
the result with the provided masks.

## Metrics

Whole-mask segmentation metrics:

- Dice
- IoU
- sensitivity
- specificity
- precision

Terminal-vessel metrics:

- terminal-region Dice
- terminal-region sensitivity
- terminal-region precision
- skeleton endpoint recall
- skeleton endpoint precision
- missed endpoint count

Terminal regions are defined from ground-truth skeleton endpoints and a local
neighborhood around each endpoint. This makes the evaluation focus on the
small terminal branches that are most likely to be missed by generic vessel
segmentation.

## Preliminary Baseline Results

The baseline was evaluated on 244 image-mask pairs.

| Dataset | Dice | IoU | Sensitivity | Terminal Sensitivity | Endpoint Recall | Missed Endpoints |
|---|---:|---:|---:|---:|---:|---:|
| AFIO | 0.479 | 0.317 | 0.588 | 0.280 | 0.713 | 28.949 |
| CHASEDB1 | 0.470 | 0.307 | 0.781 | 0.339 | 0.764 | 13.964 |
| DualModal2019 | 0.448 | 0.290 | 0.871 | 0.543 | 0.803 | 15.933 |
| HRF | 0.435 | 0.279 | 0.782 | 0.566 | 0.858 | 42.422 |
| LES | 0.475 | 0.314 | 0.726 | 0.388 | 0.635 | 24.455 |
| STARE | 0.453 | 0.296 | 0.845 | 0.523 | 0.730 | 22.100 |
| Overall | 0.463 | 0.303 | 0.714 | 0.402 | 0.751 | 27.148 |

The overall Dice score is `0.463`, while terminal-vessel sensitivity is only
`0.402`. This confirms that terminal small vessels are a weaker part of the
current baseline and should be the next improvement target.

## Next Step

The next implementation step should improve the segmentation pipeline with
special attention to missed small terminal branches. Useful directions include:

- stronger green-channel and contrast enhancement
- multiscale vessel filtering
- adaptive threshold refinement
- post-processing that preserves thin branches
- endpoint-aware recovery of missed terminal segments

The improved method should be compared with the current baseline using both
overall segmentation metrics and terminal-specific metrics.
