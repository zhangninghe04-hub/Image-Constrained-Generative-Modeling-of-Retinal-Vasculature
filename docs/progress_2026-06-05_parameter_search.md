# Progress Report: Fundus-Constrained Density-Aware Model Refinement

**Project:** Image-Constrained Generative Modeling of Retinal Vasculature  
**Author:** Ninghe Zhang  
**Date:** June 5, 2026

## 1. Current Stage

The project remains in the fundus-only stage because OCTA images are not yet available. The current work therefore focuses on improving the fundus-constrained generative model before extending the framework to additional imaging modalities.

Since the previous progress report, the main objective has been to recover spatial coverage in the image-constrained and density-aware models while preserving density-aware terminal placement. The previous density-aware model reduced over-pruning, but the generated trees still remained smaller than the baseline. The current update addresses this limitation through parameter search, fairer density evaluation, and improved visualization.

## 2. Completed Updates

A focused parameter-search pipeline was added through `run_parameter_search.py`. The search tested 96 parameter combinations involving length decay, maximum depth, initial branch length, and density-aware depth, direction, and survival weights.

The selected full density-aware setting is:

- alpha: 0.76
- max depth: 7
- initial length: 0.23
- global density weight: 0.60
- density depth weight: 0.75
- density direction weight: 0.70
- density survival weight: 0.00

This setting prioritizes recovering terminal count and occupied grid coverage. It also avoids returning to strong survival-based pruning, which previously produced overly sparse trees.

Two additional density-evaluation metrics were added. Matched terminal density score compares models after controlling for terminal count. Density lift over random compares generated terminal placement with uniformly random retinal points. These metrics reduce the bias caused by models that generate more terminal nodes.

A new visualization, `fig5_terminal_density_overlay.png`, was added. This figure overlays generated terminal nodes directly on the fundus-derived density map, making density response easier to interpret qualitatively.

## 3. Main Results

The parameter search improved the spatial reach of the full density-aware model. In the previous run, the full density-aware model averaged 14.867 terminal nodes and occupied grid coverage of 0.027. The updated setting increases these values to 20.733 terminal nodes and occupied grid coverage of 0.037.

The 15-image average metrics are:

| Model | Terminals | Length | Occupied Grid Coverage | Coverage Dispersion | Terminal Density Score | Matched Density Score | Density Lift vs Random |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 29.000 | 4.234 | 0.057 | 0.010 | 0.277 | 0.284 | -0.015 |
| Image-Constrained | 22.200 | 3.645 | 0.039 | 0.077 | 0.157 | 0.158 | -0.148 |
| Density Depth Only | 18.867 | 3.186 | 0.033 | 0.047 | 0.176 | 0.177 | -0.136 |
| Density Direction Only | 20.733 | 3.551 | 0.037 | 0.051 | 0.187 | 0.188 | -0.126 |
| Density Survival Only | 16.267 | 2.899 | 0.030 | 0.058 | 0.176 | 0.176 | -0.137 |
| Density-Aware | 20.733 | 3.551 | 0.037 | 0.051 | 0.187 | 0.188 | -0.126 |

Figure 1 compares the baseline, image-constrained, and density-aware models on representative fundus images. The baseline remains generic, while the constrained and density-aware models respond to image-derived optic disc location and major vessel orientation.

Figure 2 summarizes the updated quantitative metrics over all 15 fundus images. The density-aware model now has substantially better terminal count and spatial coverage than the previous calibrated version.

Figure 3 overlays terminal nodes on the fundus-derived density map. This visualization shows that the generated structures are image-specific, but density matching remains incomplete.

## 4. Interpretation

The main improvement is recovery of spatial reach. The full density-aware model now generates an average of 20.733 terminal nodes, closer to the intended range of 20 to 25 terminals. This directly addresses the previous limitation that constrained models were too spatially restricted.

The update also reveals a clear tradeoff. Terminal density score decreases from the previous full density-aware value of 0.217 to 0.187. This indicates that the model now reaches more of the retinal field, but terminal nodes are less concentrated in the highest-density regions. The current stage is therefore balancing two objectives: spatial coverage and density matching.

The ablation results suggest that direction guidance is more useful than survival pruning. The density direction-only model and full density-aware model have the same average terminal count, length, occupied grid coverage, and density scores under the selected setting. This indicates that the direction-selection rule currently drives most of the density-aware behavior, while survival pruning is not useful in the current configuration.

The density lift over random remains negative for all models. This result suggests that the current density map and terminal placement metric require further refinement. The terminal density score is useful, but the model still does not outperform a random retinal placement baseline under the matched-count comparison.

## 5. Remaining Issues

The constrained and density-aware models still do not match the baseline in occupied grid coverage. The baseline remains larger and more spatially diffuse, but it is not image-specific. The key challenge is to increase spatial reach without losing anatomical anchoring.

Density matching remains weaker than expected. The improved terminal count makes the model more structurally reasonable, but density-guided placement still needs stronger local response.

The current density map is derived from fundus segmentation only. Since OCTA data are unavailable, microvascular density remains approximated rather than directly observed.

## 6. Next Steps

The next step is to improve density matching while preserving the recovered terminal count. The target range should remain approximately 20 to 25 terminal nodes.

The most promising direction is to refine the branch direction-selection rule. Instead of only choosing among nearby candidate angles, the generator should evaluate whether a candidate branch moves toward a region of increasing density over a short spatial horizon.

The density metric should also be refined. A stronger metric should compare generated terminal placement with density maps while controlling for terminal count, retinal region, and random-placement baselines.

The final fundus-only model should then be documented as a complete stage of the project, with OCTA extension described as future work once data become available.
