# Project Roadmap

## Milestone 1: Literature and Foundation (Complete)

- [x] Write research proposal
- [x] Review core literature: Murray's law, West-Brown-Enquist scaling, Guedri et al. fractal modeling, Yeh et al. shape grammars, Brown et al. physics-informed GAN
- [x] Define research question and scope
- [x] Identify the distinction between reconstruction and generative approaches

## Milestone 2: Baseline Branching Model (Complete)

- [x] Implement recursive branching tree generator
- [x] Define data structures: `Node`, `Edge`, `TreeGeneratorConfig`
- [x] Implement retina boundary constraint
- [x] Implement macula avascular zone constraint
- [x] Implement segment crossing detection
- [x] Generate and visualize baseline trees
- [x] Implement terminal node distribution visualization

## Milestone 3: Evaluation Framework (In Progress)

- [x] Implement coverage score (nearest-neighbor distance std)
- [x] Implement total vessel length metric
- [x] Build parameter sweep experiment runner
- [x] Generate summary tables across parameter combinations
- [ ] Add fractal dimension computation
- [ ] Add branch angle distribution analysis
- [ ] Add statistical comparison tooling

## Milestone 4: Data Organization and Image Preprocessing (Planned)

- [ ] Organize healthy retinal image dataset (15 images available)
- [ ] Implement fundus image preprocessing pipeline
- [ ] Implement optic disc detection
- [ ] Implement major vessel orientation extraction
- [ ] Generate vessel density maps from OCTA images (if OCTA data available)
- [ ] Define a standard constraint data format

## Milestone 5: Image-Constrained Generation (Planned)

- [ ] Integrate extracted optic disc location as root position
- [ ] Use extracted vessel orientations for initial branch angles
- [ ] Incorporate density maps as branching probability modifiers
- [ ] Implement per-image constrained generation pipeline
- [ ] Compare constrained vs. unconstrained generation results

## Milestone 6: Advanced Branching Rules (Planned)

- [ ] Implement Murray's law-constrained branching (branch radii and angles)
- [ ] Explore optimization-based branch placement
- [ ] Explore asymmetric branching rules
- [ ] Compare branching rule variants quantitatively

## Milestone 7: Validation and Comparison (Planned)

- [ ] Segment vessels from healthy retinal images
- [ ] Extract topological features from real vascular networks
- [ ] Compare generated networks with real networks (density, branching angles, fractal dimension)
- [ ] Produce final evaluation report and figures

## Milestone 8: Documentation and Publication (Future)

- [ ] Clean up codebase for reproducibility
- [ ] Write complete methodology section
- [ ] Prepare publication-quality figures
- [ ] Draft manuscript
