# Progress Report: Density-Aware Fundus Model Calibration

**Project:** Image-Constrained Generative Modeling of Retinal Vasculature  
**Author:** Ninghe Zhang  
**Date:** May 20, 2026

## 1. Current Stage of the Project

This project is currently in the fundus-only stage. The original proposal describes a larger fundus + OCTA image-constrained modeling framework, but OCTA data are not yet available. Therefore, the present work focuses on strengthening the fundus-constrained generative model and making the density-aware branching rule more stable and interpretable.

Compared with the previous progress report, the main update is calibration and evaluation rather than adding a new modeling component. The earlier density-aware version made density influence branch depth, direction, and survival, but the survival rule was too aggressive in some runs. The current update adjusts the density-aware parameters to reduce over-pruning while keeping the model responsive to fundus-derived vessel density.

The current full density-aware setting uses:

- global density weight: 0.60
- density depth weight: 0.75
- density direction weight: 0.50
- density survival weight: 0.10

This keeps density guidance active while preventing the generated tree from collapsing into a very small number of terminal branches.

## 2. What Has Been Completed

The current update includes four main improvements.

First, the density-aware generator has been reorganized so that density affects branching through three separable mechanisms: effective branching depth, branch direction selection, and branch survival. This makes the model easier to evaluate because each density mechanism can be tested independently.

Second, an ablation evaluation has been added. The pipeline now compares six models:

- Baseline
- Image-Constrained
- Density Depth Only
- Density Direction Only
- Density Survival Only
- Full Density-Aware

This ablation structure makes it clearer which density mechanism contributes to the observed behavior.

Third, the evaluation metrics were clarified. The old nearest-neighbor coverage score is now treated as coverage dispersion, where lower values mean terminal nodes are more uniformly spaced. A new occupied grid coverage metric was added as a positive coverage measure, where higher values mean terminals occupy more of the retinal field. A terminal density score was also added to measure whether terminal nodes fall in higher-density regions of the fundus-derived density map.

Fourth, the result-generation pipeline was made more reproducible. A lightweight Pillow/numpy fallback was added so that figures and CSV files can still be generated even when OpenCV or Matplotlib are not available. The preferred image extraction method remains the OpenCV-based implementation, but the fallback makes the current progress run easier to reproduce on a minimal environment.

## 3. Main Results

Figure 1 shows the extracted fundus constraints for representative images, including the retinal boundary, optic disc, macula estimate, and major vessel orientation. The extracted optic disc locations and initial branch orientations vary across images, confirming that the constrained model remains image-specific rather than using a single default geometry.

Figure 1: Extracted structural constraints from representative fundus images.

Figure 2 compares the baseline, image-constrained, and density-aware models on three representative images. The baseline model still produces the same generic tree across images. The image-constrained model changes root placement and primary orientation according to each fundus image. The calibrated density-aware model remains close to the constrained model in overall size, but its terminal placement is more influenced by the fundus-derived density map.

Figure 2: Model comparison among baseline, image-constrained, and density-aware models.

Figure 3 shows vessel segmentation, density maps, and terminal-node distributions for representative images. This figure is useful for interpreting whether generated terminals respond to the spatial density structure extracted from the fundus images.

Figure 3: Vessel segmentation, density map, and terminal-node distribution.

Figure 4 summarizes the quantitative comparison averaged over all 15 fundus images.

Figure 4: Quantitative evaluation metrics averaged over 15 fundus images.

The full 15-image average metrics are:

| Model | Terminals | Total Length | Occupied Grid Coverage | Coverage Dispersion | Density Corr. | Terminal Density Score |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 29.000 | 4.234 | 0.057 | 0.010 | -0.030 | 0.277 |
| Image-Constrained | 15.533 | 2.692 | 0.028 | 0.072 | -0.069 | 0.169 |
| Density Depth Only | 14.733 | 2.579 | 0.026 | 0.085 | -0.059 | 0.210 |
| Density Direction Only | 15.333 | 2.711 | 0.027 | 0.093 | -0.058 | 0.214 |
| Density Survival Only | 14.200 | 2.551 | 0.026 | 0.088 | -0.059 | 0.209 |
| Density-Aware | 14.867 | 2.661 | 0.027 | 0.088 | -0.056 | 0.217 |

The main result is that the full density-aware model no longer over-prunes as severely as the previous setting. Its average number of terminals is 14.867, close to the image-constrained model's 15.533, and its total length is 2.661, close to the image-constrained value of 2.692. This means the calibrated density-aware rule preserves a similar level of structural complexity.

At the same time, the terminal density score improves from 0.169 in the image-constrained model to 0.217 in the full density-aware model. This suggests that terminal nodes are more likely to fall in regions that the fundus-derived density map identifies as vessel-dense.

The density correlation remains weak and negative for all models. This confirms that correlation alone is not yet a reliable summary of local density matching in this pipeline. The terminal density score appears to be more interpretable for the current stage.

## 4. Comparison with the Previous Stage

The previous progress report identified a tradeoff: the density-aware rule had become directionally effective, but branch survival could be too aggressive, producing sparse generated trees in some images. The current update directly addresses that issue.

The survival component was reduced, and the full density-aware model now stays close to the image-constrained model in terminal count and total length. This is an improvement over the earlier over-pruned behavior.

The current update also improves interpretability. Instead of treating density-aware generation as one combined mechanism, the pipeline now separates depth-only, direction-only, survival-only, and full density-aware variants. This makes it easier to explain how density is influencing the model.

The results show that all three density mechanisms raise the terminal density score relative to the image-constrained model. Direction-only gives 0.214, depth-only gives 0.210, survival-only gives 0.209, and the full model gives 0.217. This suggests that density information is being used by the generator, but the improvement is still modest.

## 5. Interpretation

First, the project has now moved from simply adding density-aware behavior to calibrating it. The latest model is less likely to prune away too many branches, which makes it more suitable for comparison with the image-constrained model.

Second, the ablation results suggest that density direction guidance is especially useful. The direction-only model keeps a terminal count close to the constrained model while improving terminal density score. This indicates that guiding where branches grow may be more stable than using density primarily to decide whether branches survive.

Third, the baseline model still has the highest occupied grid coverage and terminal density score. This should not be interpreted as the baseline being anatomically better. The baseline is a larger fixed tree with more terminals, so it naturally covers more grid cells. However, it is not image-specific and does not adapt to optic disc placement or major vessel orientation.

Fourth, the main weakness of the current constrained models is that they are still less spatially extensive than the baseline. The current generator is anatomically anchored, but it may be too conservative in some images. Future tuning should recover more spatial reach without losing image-specific structure.

## 6. Next Steps

The first priority is to improve spatial reach in the constrained and density-aware models. The model should preserve image-specific root placement and major orientation while generating trees that occupy more of the retinal field.

The second priority is to continue refining density metrics. Density correlation is still difficult to interpret, while terminal density score is more directly meaningful at the current stage. A future metric should compare generated terminal distributions against density maps while controlling for the number of terminals.

The third priority is to improve the fundus constraint extraction quality. The current lightweight fallback is useful for reproducibility, but the OpenCV-based segmentation should remain the preferred extraction path for final results. The next version should ensure that the final reported figures are generated with the strongest available extraction method.

The fourth priority is to prepare a clearer final narrative: the project is currently a fundus-constrained generative branching model, with OCTA extension deferred until data become available.

## 7. Summary

In summary, this update calibrates the density-aware fundus model and adds ablation-based evaluation. The density-aware generator now uses separable depth, direction, and survival mechanisms, and the survival rule has been reduced to avoid excessive pruning. The full density-aware model remains close to the image-constrained model in terminal count and total length while improving terminal density score from 0.169 to 0.217.

The main remaining challenge is to increase spatial reach and coverage in the constrained models without returning to the generic behavior of the baseline. The project is now in a stronger position because the density-aware mechanism is more stable, the evaluation is more interpretable, and the current fundus-only scope is clearly separated from the future OCTA extension.
