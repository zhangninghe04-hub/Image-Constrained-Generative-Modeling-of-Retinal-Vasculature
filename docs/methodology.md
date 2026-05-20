# Methodology

This document describes the intended computational pipeline for the project. The pipeline has three main stages: constraint extraction, generative branching, and quantitative evaluation.

## Stage 1: Image Constraint Extraction

**Goal:** Extract structural features from retinal images to serve as constraints for the generative model. The current implementation is fundus-only because OCTA data are not yet available.

### Planned constraint types:

- **Optic disc location** — Detected from fundus images using intensity-based or morphological methods. Defines the root position for the vascular tree.
- **Major vessel orientation** — The initial branching directions (superior and inferior temporal arcades) extracted from vessel segmentation or orientation analysis.
- **Regional vessel density** — Currently derived from fundus vessel segmentation. OCTA-derived density remains future work.
- **Macula avascular zone (FAZ)** — The foveal region where vessels are absent. Currently modeled as a circular exclusion zone; future work may extract its actual shape if OCTA data become available.

### Current status:

- Retina boundary, optic disc, macula estimate, vessel segmentation, fundus density map, and coarse vessel orientation are implemented for fundus images.
- The current model should be interpreted as the fundus-constrained component of the original multimodal proposal.
- OCTA-based constraints are deferred until OCTA data become available.
- A lightweight Pillow/numpy fallback is available for result generation when OpenCV is not installed. This fallback is intended for reproducible batch runs and quick progress reporting; OpenCV-based extraction remains the preferred path when available.

## Stage 2: Generative Branching Model

**Goal:** Generate synthetic retinal vascular trees using recursive branching rules parameterized by image-derived constraints.

### Current baseline model:

The implemented model (`RetinalTreeGenerator`) works as follows:

1. Place a root node at the optic disc location
2. Initialize two main branches (superior and inferior) at configurable base angles
3. At each branching node, spawn two child branches with:
   - Length scaled by decay factor `alpha` (e.g., 0.72)
   - Angles drawn from a normal distribution around a mean branch angle
4. Terminate branches that:
   - Exceed maximum generation depth
   - Fall below minimum segment length
   - Exit the retina boundary
   - Enter the macula avascular zone
   - Cross an existing vessel segment

### Configurable parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `retina_radius` | Radius of the retinal domain | 1.0 |
| `root` | Root node position (optic disc) | (0.18, 0.0) |
| `initial_length` | Length of the first-generation segments | 0.23 |
| `alpha` | Length decay factor per generation | 0.72 |
| `max_depth` | Maximum branching depth | 6 |
| `branch_angle_mean` | Mean branching angle (radians) | 28° |
| `branch_angle_std` | Standard deviation of branching angle | 7° |
| `macula_center` | Center of the macula avascular zone | (-0.25, 0.0) |
| `macula_radius` | Radius of the macula avascular zone | 0.16 |

### Planned extensions:

- **Density-guided branching:** Implemented as separable depth, direction, and survival mechanisms so ablation studies can isolate which part improves density response and which part causes over-pruning
- **Murray's law integration:** Constrain branch radii and angles according to Murray's law of minimum work
- **Optimization-based rules:** Use energy minimization to determine branch placement

## Stage 3: Quantitative Evaluation

**Goal:** Assess generated networks for spatial coverage, structural efficiency, and consistency with image-derived constraints.

### Implemented metrics:

- **Coverage dispersion** — Standard deviation of nearest-neighbor distances among terminal nodes. Lower values indicate more uniform spatial distribution.
- **Occupied grid coverage** — Fraction of circular retinal grid cells containing at least one terminal node. Higher values indicate broader retinal reach.
- **Total vessel length** — Sum of all segment lengths. Measures structural cost.
- **Terminal node count** — Number of leaf nodes in the generated tree.
- **Density correlation** — Correlation between generated terminal density and image-derived vessel-density map.
- **Terminal density score** — Mean fundus-derived vessel-density value sampled at generated terminal locations.

### Planned metrics:

- **Fractal dimension** — Box-counting dimension to quantify the space-filling properties of the generated network
- **Branch angle distribution** — Statistical comparison of generated branch angles with empirically observed distributions
- **Topological comparison** — Graph-based comparison with segmented vascular structures from real images

## Parameter Experiment Framework

The current codebase includes an experiment runner that:

1. Sweeps across combinations of `alpha`, `branch_angle_mean`, and random seeds
2. Generates a tree for each parameter combination
3. Records network statistics (edges, nodes, terminals, total length, coverage score)
4. Produces a grouped summary table for comparison

This framework can be extended to include new parameters and metrics as the model develops.
