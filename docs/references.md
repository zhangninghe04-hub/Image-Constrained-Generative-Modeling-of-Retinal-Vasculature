# References

## Papers in This Repository

### Guedri, H., Bajahzar, A., & Belmabrouk, H. (2021)
**Three-Dimensional Modeling of the Retinal Vascular Tree via Fractal Interpolation**
*Computer Modeling in Engineering & Sciences*, 127(1), 59–77. DOI: 10.32604/cmes.2021.013632

**Relevance:** This paper models retinal vascular trees using centerline extraction and 3D fractal interpolation via Iterated Function Systems (IFS). It demonstrates that retinal vessels exhibit multiscale branching geometry that can be captured by fractal methods. The Douglas-Peucker algorithm is used for data reduction, and characteristic points (endpoints, bifurcation points, connective points) are classified using crossing numbers. This paper provides methodological background on how vascular tree geometry can be described mathematically, and informs the evaluation metrics for this project (e.g., fractal dimension, branching point classification).

**File:** `papers/TSP_CMES_13632.pdf`

---

### Zhang, N. (2026)
**Image-Constrained Generative Modeling of Retinal Vasculature** (Research Proposal)

**Relevance:** This is the founding document for the project. It defines the research question, reviews the relevant literature, establishes the mathematical framework (spatial graph G = (V, E), spatial coverage, structural efficiency), and proposes the image-constrained generative modeling approach.

**File:** `papers/proposal.pdf`

---

## Key References from the Literature Review

### Murray, C. D. (1926)
**The Physiological Principle of Minimum Work: I. The Vascular System and the Cost of Blood Volume**
*Proceedings of the National Academy of Sciences*, 12(3), 207–214.

**Relevance:** Murray's law interprets vessel bifurcation as a physiological optimization problem that minimizes the work required for blood flow and vessel maintenance. This is the theoretical foundation for understanding why vascular branching patterns follow predictable geometric relationships. The branching model in this project can be evaluated against Murray's law predictions.

### West, G. B., Brown, J. H., & Enquist, B. J. (1997)
**A General Model for the Origin of Allometric Scaling Laws in Biology**
*Science*, 276(5309), 122–126.

**Relevance:** Proposes a general framework for biological transport networks in which hierarchical branching structures emerge from the need to distribute resources efficiently across spatial domains. Vascular systems are modeled as spatial branching networks governed by scaling relations and optimization constraints. This provides the theoretical context for why branching rules produce networks with specific spatial properties.

### Yeh, F. et al.
**Shape-Grammar Approach for Generating Artificial Retinal Vascular Networks**

**Relevance:** Introduces a shape-grammar approach for generating artificial retinal vascular networks. Demonstrates that rule-based branching systems can reproduce realistic retinal vascular structures. This is a direct methodological precedent for the generative approach used in this project.

### Brown, A. et al. (2024)
**Physics-Informed Generative Adversarial Network for Synthesizing Retinal Vascular Networks**

**Relevance:** Proposes a physics-informed GAN that synthesizes retinal vascular networks while incorporating biophysical constraints such as vessel connectivity and flow structure. Demonstrates that realistic vascular networks can be generated from limited simulated data. This paper represents the data-driven end of the spectrum; the current project takes a more explicit, rule-based approach but shares the goal of generating anatomically plausible networks.
