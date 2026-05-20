# Image-Constrained Generative Modeling of Retinal Vasculature

**Author:** Ninghe Zhang
**Date:** March 2026
**Status:** Fundus-only image-constrained and density-aware model in progress; OCTA extension deferred until data are available

---

## Summary

This project investigates generative branching models for retinal vascular networks under structural constraints derived from retinal images. The goal is to identify which branching rules produce synthetic vascular trees that achieve efficient spatial coverage while remaining consistent with anatomical features observable in fundus and OCTA imaging.

## Research Question

> **Which branching rules generate retinal vascular networks that achieve efficient spatial coverage while remaining consistent with structural constraints derived from retinal images?**

## Motivation

Retinal vasculature forms a hierarchical branching network that distributes blood across the retinal surface. From a mathematical perspective, this system can be represented as a spatial tree embedded in a two-dimensional domain. A realistic vascular network must:

1. **Distribute vessels** across the retinal surface (spatial coverage)
2. **Maintain structural efficiency** in total vessel length and branching complexity
3. **Remain consistent** with structural information observable in retinal imaging

Existing approaches to vascular modeling typically fall into two categories:
- **Reconstruction-based methods** that extract vessel structures directly from images
- **Data-driven generative models** (e.g., GANs) that synthesize vascular patterns from training data

This project takes a different approach: rather than reconstructing vessels from images or learning implicit representations, we use **explicit branching rules** to generate vascular networks, with **image-derived features as structural constraints**. This allows direct investigation of how branching parameters influence network geometry and coverage.

## How This Differs from Reconstruction

| Aspect | Reconstruction approaches | This project |
|--------|--------------------------|-------------|
| Input | Fundus/OCTA image | Image-derived constraints (optic disc location, vessel orientation, density) |
| Method | Segmentation, tracing, surface fitting | Rule-based generative branching model |
| Output | Extracted vessel map | Synthetic full vascular network |
| Goal | Recover existing structure | Generate plausible structure under constraints |

## Current Project Status

- [x] Research proposal complete
- [x] Literature review (Murray's law, West-Brown-Enquist, Guedri et al., Yeh et al., Brown et al.)
- [x] Baseline recursive branching model implemented
- [x] Macula avascular zone constraint implemented
- [x] Retina boundary constraint implemented
- [x] Segment crossing detection implemented
- [x] Coverage score metric (nearest-neighbor distance std)
- [x] Parameter sweep experiment (alpha x branch angle x seed)
- [x] Visualization (network plot, terminal node distribution)
- [x] Fundus image constraint extraction
- [x] Optic disc detection and orientation extraction
- [x] Fundus-derived vessel density map constraint
- [x] Density-aware depth, direction, and survival rules
- [x] Extended evaluation metrics (fractal dimension, branch angle statistics, density metrics)
- [x] Ablation setup for density depth/direction/survival rules
- [ ] Comparison with real vascular topology
- [ ] OCTA-derived constraints when OCTA data become available

## Repository Structure

```
.
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── data/
│   ├── raw/                          # Raw retinal images (not tracked by git)
│   │   └── healthy/                  # 15 healthy fundus/OCTA images (01_h.jpg–15_h.jpg)
│   ├── processed/                    # Processed constraint maps
│   └── external/                     # External datasets or references
├── notebooks/
│   └── 01_baseline_branching_model.ipynb   # Baseline model with parameter experiments
├── src/
│   ├── __init__.py
│   ├── constraint_extraction/        # Extract structural constraints from images
│   │   ├── __init__.py
│   │   └── extract_constraints.py
│   ├── generative_models/            # Branching rule implementations
│   │   ├── __init__.py
│   │   └── branching_model.py
│   ├── evaluation/                   # Quantitative evaluation metrics
│   │   ├── __init__.py
│   │   └── metrics.py
│   ├── visualization/                # Plotting and rendering
│   │   ├── __init__.py
│   │   └── plot_network.py
│   └── utils/                        # Shared utilities
│       ├── __init__.py
│       └── io.py
├── docs/
│   ├── project_overview.md
│   ├── methodology.md
│   ├── roadmap.md
│   └── references.md
├── papers/                           # Reference papers (PDFs not tracked by git)
├── figures/                          # Generated figures and diagrams
├── results/                          # Experiment outputs
├── tests/                            # Unit tests
└── archive/                          # Older drafts or unused materials
```

## Planned Workflow

```
Fundus + OCTA Images
        │
        ▼
┌─────────────────────┐
│ Constraint Extraction│  ← optic disc location, vessel orientation, density map
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Generative Model   │  ← recursive branching with configurable rules
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│    Evaluation       │  ← spatial coverage, structural efficiency, constraint consistency
└─────────────────────┘
```

## Setup

### Requirements

- Python 3.9+
- See `requirements.txt` for package dependencies

### Installation

```bash
git clone https://github.com/zhangninghe04-hub/Image-Constrained-Generative-Modeling-of-Retinal-Vasculature.git
cd Image-Constrained-Generative-Modeling-of-Retinal-Vasculature
pip install -r requirements.txt
```

### Running the Baseline Model

The baseline branching model is in the notebook:

```bash
jupyter notebook notebooks/01_baseline_branching_model.ipynb
```

This notebook generates synthetic retinal vascular trees using recursive branching rules with:
- Configurable length decay factor (`alpha`)
- Stochastic branch angles (normal distribution)
- Macula avascular zone avoidance
- Retina boundary enforcement
- Segment crossing prevention

It also runs a parameter sweep across `alpha`, `branch_angle_mean`, and random seeds, producing a summary table of network statistics and coverage scores.

## Data

The `data/raw/healthy/` directory contains 15 healthy retinal images (`01_h.jpg` through `15_h.jpg`). These images serve as references for extracting structural constraints — they are **not** reconstruction targets.

> **Note:** Raw image data is not tracked by git due to file size. Place images in `data/raw/healthy/` after cloning.

## Next Steps

1. **Image constraint extraction** — Detect optic disc location and major vessel orientations from the healthy retinal images
2. **Density-aware branching** — Incorporate regional vessel density maps as soft constraints on branching probability
3. **Extended evaluation** — Add fractal dimension, branch angle distribution analysis, and comparison with real vascular topology
4. **Model variants** — Explore alternative branching rules (e.g., optimization-based, Murray's law-constrained)
5. **Quantitative comparison** — Compare generated networks against image-derived ground truth

## References

- **Guedri, H., Bajahzar, A., & Belmabrouk, H.** (2021). Three-Dimensional Modeling of the Retinal Vascular Tree via Fractal Interpolation. *Computer Modeling in Engineering & Sciences*, 127(1). DOI: 10.32604/cmes.2021.013632
- **Murray, C. D.** (1926). The physiological principle of minimum work: I. The vascular system and the cost of blood volume. *Proceedings of the National Academy of Sciences*, 12(3), 207–214.
- **West, G. B., Brown, J. H., & Enquist, B. J.** (1997). A general model for the origin of allometric scaling laws in biology. *Science*, 276(5309), 122–126.
- **Brown, A. et al.** (2024). Physics-informed generative adversarial network for synthesizing retinal vascular networks. (Referenced in proposal)
- **Yeh, F. et al.** Shape-grammar approach for generating artificial retinal vascular networks. (Referenced in proposal)

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
