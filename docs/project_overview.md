# Project Overview

## Problem

Retinal vasculature forms a hierarchical branching network responsible for distributing blood across the retinal surface. Understanding how these networks achieve both spatial coverage and structural efficiency is relevant to ophthalmology, vascular biology, and computational modeling.

From a mathematical standpoint, the retinal vascular network can be represented as a spatial graph G = (V, E), where V denotes branching nodes and E denotes vessel segments. Each segment has geometric properties including length, orientation, and generation depth.

Two structural objectives arise naturally:

1. **Spatial coverage** — Terminal vessels should distribute across the retinal domain so that tissue is adequately supplied.
2. **Structural efficiency** — The total vessel length and branching complexity should be minimized subject to coverage requirements.

## Scope

This project studies **generative branching models** for retinal vascular networks. Rather than reconstructing vessels from images (as in segmentation or tracing approaches), the goal is to generate plausible vascular trees using explicit branching rules, constrained by structural information extracted from retinal imaging.

The key distinction is:
- **Input:** Image-derived constraints (optic disc location, major vessel orientation, regional vessel density)
- **Process:** Rule-based recursive branching with configurable parameters
- **Output:** Synthetic vascular networks evaluated for coverage and efficiency

## Image Modalities

Two imaging modalities provide structural constraints:

- **Fundus photography** — Provides information about major vessel layout, optic disc location, and overall vascular architecture
- **Optical coherence tomography angiography (OCTA)** — Provides detailed information about vessel density at different retinal layers

These images are used as **references and constraint sources**, not as reconstruction targets.

## Current Direction

The project currently has:

1. A **baseline recursive branching model** that generates vascular trees from a root node (optic disc), respecting retina boundary and macula avascular zone constraints
2. A **parameter experiment framework** that sweeps across branching parameters (length decay factor, branch angles) and evaluates networks using coverage and efficiency metrics
3. **15 healthy retinal images** available for future constraint extraction work

The next phase focuses on extracting quantitative constraints from the retinal images and incorporating them into the generative model.
