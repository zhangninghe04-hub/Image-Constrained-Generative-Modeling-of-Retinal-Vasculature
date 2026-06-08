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
- [x] Add fractal dimension computation
- [x] Add branch angle distribution analysis
- [x] Add density correlation and terminal density score
- [x] Add occupied grid coverage as a positive coverage metric
- [x] Add matched terminal-count density score
- [x] Add density lift over random baseline
- [x] Run focused parameter search for coverage-density tradeoff
- [ ] Further calibrate density metrics against qualitative results

## Milestone 4: Data Organization and Image Preprocessing (In Progress)

- [ ] Organize healthy retinal image dataset (15 images available)
- [x] Implement fundus image preprocessing pipeline
- [x] Implement optic disc detection
- [x] Implement major vessel orientation extraction
- [x] Generate fundus-derived vessel density maps
- [x] Define a standard constraint data format
- [ ] Add OCTA-derived density maps if OCTA data become available

## Milestone 5: Image-Constrained Generation (In Progress)

- [x] Integrate extracted optic disc location as root position
- [x] Use extracted vessel orientations for initial branch angles
- [x] Incorporate density maps as branch depth modifiers
- [x] Incorporate density-guided branch direction
- [x] Incorporate density-based branch survival
- [x] Implement per-image constrained generation pipeline
- [x] Compare baseline, constrained, and density-aware generation results
- [x] Add ablation variants for depth-only, direction-only, and survival-only density rules
- [x] Tune density-aware rules to reduce over-pruning while preserving coverage gains
- [ ] Improve density matching under recovered terminal count

## Milestone 6: Advanced Branching Rules (Planned)

- [ ] Implement Murray's law-constrained branching (branch radii and angles)
- [ ] Explore optimization-based branch placement
- [ ] Explore asymmetric branching rules
- [ ] Compare branching rule variants quantitatively

## Milestone 7: Validation and Comparison (Planned)

- [ ] Segment vessels from healthy retinal images
- [ ] Extract topological features from real vascular networks
- [ ] Compare generated networks with real networks (density, branching angles, fractal dimension)
- [ ] Interpret density correlation and terminal density score against qualitative figures
- [x] Add terminal-over-density overlay figure
- [x] Add short-horizon density sampling for direction selection
- [ ] Produce final evaluation report and figures

## Milestone 8: Documentation and Publication (Future)

- [ ] Clean up codebase for reproducibility
- [ ] Write complete methodology section
- [ ] Prepare publication-quality figures
- [ ] Draft manuscript
