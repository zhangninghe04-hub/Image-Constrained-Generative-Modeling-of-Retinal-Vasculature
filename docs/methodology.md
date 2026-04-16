# Methodology

This document describes the intended computational pipeline for the project. The pipeline has three main stages: constraint extraction, generative branching, and quantitative evaluation.

## Stage 1: Image Constraint Extraction

**Goal:** Extract structural features from fundus and OCTA images to serve as constraints for the generative model.

### Planned constraint types:

- **Optic disc location** — Detected from fundus images using intensity-based or morphological methods. Defines the root position for the vascular tree.
- **Major vessel orientation** — The initial branching directions (superior and inferior temporal arcades) extracted from vessel segmentation or orientation analysis.
- **Regional vessel density** — Derived from OCTA images. Provides a spatial map indicating where branching should be denser or sparser.
- **Macula avascular zone (FAZ)** — The foveal region where vessels are absent. Currently modeled as a circular exclusion zone; future work may extract its actual shape from OCTA.

### Current status:

- Optic disc position is currently set manually as a model parameter
- Macula avascular zone is modeled as a fixed circular region
- Automated extraction from images is planned but not yet implemented

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

- **Density-guided branching:** Adjust branching probability or depth based on regional vessel density maps
- **Murray's law integration:** Constrain branch radii and angles according to Murray's law of minimum work
- **Optimization-based rules:** Use energy minimization to determine branch placement

## Stage 3: Quantitative Evaluation

**Goal:** Assess generated networks for spatial coverage, structural efficiency, and consistency with image-derived constraints.

### Implemented metrics:

- **Coverage score** — Standard deviation of nearest-neighbor distances among terminal nodes. Lower values indicate more uniform spatial distribution.
- **Total vessel length** — Sum of all segment lengths. Measures structural cost.
- **Terminal node count** — Number of leaf nodes in the generated tree.

### Planned metrics:

- **Fractal dimension** — Box-counting dimension to quantify the space-filling properties of the generated network
- **Branch angle distribution** — Statistical comparison of generated branch angles with empirically observed distributions
- **Vessel density correlation** — Correlation between generated vessel density and image-derived density maps
- **Topological comparison** — Graph-based comparison with segmented vascular structures from real images

## Parameter Experiment Framework

The current codebase includes an experiment runner that:

1. Sweeps across combinations of `alpha`, `branch_angle_mean`, and random seeds
2. Generates a tree for each parameter combination
3. Records network statistics (edges, nodes, terminals, total length, coverage score)
4. Produces a grouped summary table for comparison

This framework can be extended to include new parameters and metrics as the model develops.
