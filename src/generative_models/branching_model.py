"""
Generative branching model for retinal vascular networks.

This module provides the core tree generation logic. The model generates
synthetic vascular trees using recursive branching rules with configurable
parameters and constraints derived from retinal images.

Two generator classes are provided:
- RetinalTreeGenerator: baseline model with geometric constraints only
- ConstrainedTreeGenerator: extends baseline with density-aware branching
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class Node:
    """A node in the vascular tree."""
    x: float
    y: float
    depth: int


@dataclass
class Edge:
    """A vessel segment connecting two nodes."""
    start: Tuple[float, float]
    end: Tuple[float, float]
    depth: int
    length: float


@dataclass
class TreeGeneratorConfig:
    """Configuration for the retinal tree generator.

    Parameters control the geometry and constraints of the generated
    vascular network. Can be populated manually or from image-derived
    constraints via `from_constraints()`.
    """
    retina_radius: float = 1.0
    root: Tuple[float, float] = (0.18, 0.0)

    initial_length: float = 0.23
    alpha: float = 0.72
    max_depth: int = 6

    base_angle_up: float = np.deg2rad(155)
    base_angle_down: float = np.deg2rad(205)

    branch_angle_mean: float = np.deg2rad(28)
    branch_angle_std: float = np.deg2rad(7)

    min_length: float = 0.03
    boundary_margin: float = 0.01

    macula_center: Tuple[float, float] = (-0.25, 0.0)
    macula_radius: float = 0.16

    # Density-aware branching (optional)
    density_map: Optional[np.ndarray] = None
    density_weight: float = 0.5
    density_depth_weight: float = 1.0
    density_direction_weight: float = 0.35
    density_survival_weight: float = 0.25
    density_candidate_angles: int = 5

    random_seed: Optional[int] = 42

    @classmethod
    def from_constraints(
        cls,
        constraints: dict,
        alpha: float = 0.72,
        max_depth: int = 6,
        branch_angle_mean_deg: float = 28,
        branch_angle_std_deg: float = 7,
        initial_length: float = 0.23,
        density_weight: float = 0.5,
        density_depth_weight: float = 1.0,
        density_direction_weight: float = 0.35,
        density_survival_weight: float = 0.25,
        density_candidate_angles: int = 5,
        random_seed: Optional[int] = 42,
    ) -> "TreeGeneratorConfig":
        """
        Create a config from image-derived constraints.

        Parameters
        ----------
        constraints : dict
            Output of extract_all_constraints(). Must contain a
            'normalized' key with sub-keys: root, macula_center,
            macula_radius, base_angle_up, base_angle_down.
        alpha, max_depth, branch_angle_mean_deg, etc.
            Branching parameters (not image-derived).
        density_weight : float
            Global density influence multiplier. Set to 0 to ignore density.
        density_depth_weight : float
            Relative weight for density-guided depth adjustment.
        density_direction_weight : float
            Relative weight for density-guided branch direction selection.
        density_survival_weight : float
            Relative weight for density-guided branch survival.
        density_candidate_angles : int
            Number of candidate directions sampled around each stochastic branch angle.
        random_seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        TreeGeneratorConfig
        """
        norm = constraints['normalized']
        density_map = constraints.get('vessel_density_map', None)

        return cls(
            retina_radius=1.0,
            root=norm['root'],
            initial_length=initial_length,
            alpha=alpha,
            max_depth=max_depth,
            base_angle_up=norm['base_angle_up'],
            base_angle_down=norm['base_angle_down'],
            branch_angle_mean=np.deg2rad(branch_angle_mean_deg),
            branch_angle_std=np.deg2rad(branch_angle_std_deg),
            macula_center=norm['macula_center'],
            macula_radius=norm['macula_radius'],
            density_map=density_map,
            density_weight=density_weight,
            density_depth_weight=density_depth_weight,
            density_direction_weight=density_direction_weight,
            density_survival_weight=density_survival_weight,
            density_candidate_angles=density_candidate_angles,
            random_seed=random_seed,
        )


# =============================================================================
# Geometry helpers
# =============================================================================

def segments_intersect(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    q1: Tuple[float, float],
    q2: Tuple[float, float],
) -> bool:
    """Check whether two line segments intersect."""

    def orientation(a, b, c) -> float:
        return (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])

    def on_segment(a, b, c) -> bool:
        return (
            min(a[0], c[0]) <= b[0] <= max(a[0], c[0])
            and min(a[1], c[1]) <= b[1] <= max(a[1], c[1])
        )

    o1 = orientation(p1, p2, q1)
    o2 = orientation(p1, p2, q2)
    o3 = orientation(q1, q2, p1)
    o4 = orientation(q1, q2, p2)

    if (o1 * o2 < 0) and (o3 * o4 < 0):
        return True

    if np.isclose(o1, 0) and on_segment(p1, q1, p2):
        return True
    if np.isclose(o2, 0) and on_segment(p1, q2, p2):
        return True
    if np.isclose(o3, 0) and on_segment(q1, p1, q2):
        return True
    if np.isclose(o4, 0) and on_segment(q1, p2, q2):
        return True

    return False


# =============================================================================
# Baseline generator
# =============================================================================

class RetinalTreeGenerator:
    """Generates synthetic retinal vascular trees using recursive branching.

    The generator places a root node at the optic disc position and grows
    branches recursively, applying geometric constraints (retina boundary,
    macula avascular zone, segment crossing prevention) at each step.
    """

    def __init__(self, config: TreeGeneratorConfig):
        self.config = config
        self.nodes: List[Node] = []
        self.edges: List[Edge] = []
        self.rng = np.random.default_rng(config.random_seed)

    def inside_retina(self, x: float, y: float) -> bool:
        r = np.sqrt(x**2 + y**2)
        return r <= (self.config.retina_radius - self.config.boundary_margin)

    def inside_macula(self, x: float, y: float) -> bool:
        cx, cy = self.config.macula_center
        return np.sqrt((x - cx) ** 2 + (y - cy) ** 2) <= self.config.macula_radius

    def segment_hits_macula(
        self, p1: Tuple[float, float], p2: Tuple[float, float]
    ) -> bool:
        """Check whether a segment intersects the macula avascular zone."""
        cx, cy = self.config.macula_center
        r = self.config.macula_radius
        x1, y1 = p1
        x2, y2 = p2
        dx, dy = x2 - x1, y2 - y1

        if np.isclose(dx, 0) and np.isclose(dy, 0):
            return self.inside_macula(x1, y1)

        t = max(0.0, min(1.0, ((cx - x1) * dx + (cy - y1) * dy) / (dx * dx + dy * dy)))
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy
        return np.sqrt((closest_x - cx) ** 2 + (closest_y - cy) ** 2) <= r

    def segment_crosses_existing(
        self, p1: Tuple[float, float], p2: Tuple[float, float]
    ) -> bool:
        """Check whether a new segment crosses any existing edge."""
        for edge in self.edges:
            q1, q2 = edge.start, edge.end
            if p1 == q1 or p1 == q2 or p2 == q1 or p2 == q2:
                continue
            if segments_intersect(p1, p2, q1, q2):
                return True
        return False

    def grow_branch(
        self, node: Node, angle: float, length: float, depth: int
    ) -> None:
        """Recursively grow a branch from the given node."""
        if depth > self.config.max_depth or length < self.config.min_length:
            return

        new_x = node.x + length * np.cos(angle)
        new_y = node.y + length * np.sin(angle)

        if not self.inside_retina(new_x, new_y):
            return

        start, end = (node.x, node.y), (new_x, new_y)

        if self.segment_hits_macula(start, end) or self.segment_crosses_existing(start, end):
            return

        child = Node(new_x, new_y, depth)
        self.nodes.append(child)
        self.edges.append(Edge(start=start, end=end, depth=depth, length=length))

        next_length = self.config.alpha * length
        delta_left = self.rng.normal(self.config.branch_angle_mean, self.config.branch_angle_std)
        delta_right = self.rng.normal(self.config.branch_angle_mean, self.config.branch_angle_std)

        self.grow_branch(child, angle + delta_left, next_length, depth + 1)
        self.grow_branch(child, angle - delta_right, next_length, depth + 1)

    def generate(self) -> None:
        """Generate the full vascular tree from the root node."""
        root = Node(self.config.root[0], self.config.root[1], 0)
        self.nodes = [root]
        self.edges = []

        self.grow_branch(root, self.config.base_angle_up, self.config.initial_length, depth=1)
        self.grow_branch(root, self.config.base_angle_down, self.config.initial_length, depth=1)

    def total_length(self) -> float:
        """Return the sum of all segment lengths."""
        return sum(e.length for e in self.edges)

    def terminal_nodes(self) -> List[Node]:
        """Return nodes that are not the start of any edge (leaf nodes)."""
        start_points = {e.start for e in self.edges}
        return [n for n in self.nodes if (n.x, n.y) not in start_points]


# =============================================================================
# Density-aware generator
# =============================================================================

class ConstrainedTreeGenerator(RetinalTreeGenerator):
    """Extends the baseline generator with density-aware branching.

    Density can influence growth in three separate ways:
    effective branch depth, candidate branch direction, and post-geometry
    branch survival. Each mechanism is separately weighted so the result
    pipeline can run ablation studies.
    """

    def _local_density(self, x: float, y: float) -> float:
        """
        Look up the local vessel density at normalized coordinates (x, y).

        Returns a value in [0, 1]. Returns 0.5 if no density map is set.
        """
        dmap = self.config.density_map
        if dmap is None:
            return 0.5

        grid_size = dmap.shape[0]
        r = self.config.retina_radius

        # Map (x, y) in [-r, r] to grid indices
        gj = int((x / r + 1) / 2 * grid_size)
        gi = int((1 - y / r) / 2 * grid_size)

        gi = max(0, min(grid_size - 1, gi))
        gj = max(0, min(grid_size - 1, gj))

        return float(dmap[gi, gj])

    def _effective_max_depth(self, x: float, y: float) -> int:
        """
        Compute the effective max depth at position (x, y) based on
        local vessel density.

        High-density regions get up to max_depth + 2 additional levels.
        Low-density regions may get reduced depth.
        """
        base_depth = self.config.max_depth
        density = self._local_density(x, y)
        w = self.config.density_weight * self.config.density_depth_weight

        # Density modulation: map density [0, 1] to depth adjustment [-1, +2]
        adjustment = int(round((density - 0.3) * 3 * w))
        return max(2, base_depth + adjustment)

    def _candidate_endpoint(
        self, node: Node, angle: float, length: float
    ) -> Tuple[float, float]:
        return (
            node.x + length * np.cos(angle),
            node.y + length * np.sin(angle),
        )

    def _select_density_guided_angle(
        self, node: Node, angle: float, length: float
    ) -> float:
        """Choose a nearby branch angle whose endpoint has higher density."""
        w = self.config.density_weight * self.config.density_direction_weight
        n_candidates = max(1, int(self.config.density_candidate_angles))
        if self.config.density_map is None or w <= 0 or n_candidates == 1:
            return angle

        offsets = np.linspace(
            -self.config.branch_angle_std,
            self.config.branch_angle_std,
            n_candidates,
        )

        best_angle = angle
        best_score = -np.inf
        for offset in offsets:
            candidate_angle = angle + offset
            end_x, end_y = self._candidate_endpoint(node, candidate_angle, length)
            if not self.inside_retina(end_x, end_y):
                continue
            density = self._local_density(end_x, end_y)
            angular_penalty = abs(offset) / (self.config.branch_angle_std + 1e-9)
            score = w * density - (1.0 - w) * 0.15 * angular_penalty
            if score > best_score:
                best_score = score
                best_angle = candidate_angle

        return best_angle

    def _density_survival_probability(self, x: float, y: float, depth: int) -> float:
        """Return the density-based survival probability for a candidate."""
        w = self.config.density_weight * self.config.density_survival_weight
        if self.config.density_map is None or w <= 0:
            return 1.0

        density = self._local_density(x, y)
        depth_fraction = depth / max(1, self.config.max_depth)
        base_survival = 0.96 - 0.08 * depth_fraction
        density_bonus = w * (density - 0.35)
        return float(np.clip(base_survival + density_bonus, 0.55, 0.99))

    def _branch_survives_density(self, x: float, y: float, depth: int) -> bool:
        """Apply density-based survival after geometry checks pass."""
        survival_probability = self._density_survival_probability(x, y, depth)
        return bool(self.rng.random() <= survival_probability)

    def grow_branch(
        self, node: Node, angle: float, length: float, depth: int
    ) -> None:
        """Recursively grow a branch with density-aware depth control."""
        effective_depth = self._effective_max_depth(node.x, node.y)

        if depth > effective_depth or length < self.config.min_length:
            return

        angle = self._select_density_guided_angle(node, angle, length)
        new_x, new_y = self._candidate_endpoint(node, angle, length)

        if not self.inside_retina(new_x, new_y):
            return

        start, end = (node.x, node.y), (new_x, new_y)

        if self.segment_hits_macula(start, end) or self.segment_crosses_existing(start, end):
            return

        if not self._branch_survives_density(new_x, new_y, depth):
            return

        child = Node(new_x, new_y, depth)
        self.nodes.append(child)
        self.edges.append(Edge(start=start, end=end, depth=depth, length=length))

        next_length = self.config.alpha * length
        delta_left = self.rng.normal(self.config.branch_angle_mean, self.config.branch_angle_std)
        delta_right = self.rng.normal(self.config.branch_angle_mean, self.config.branch_angle_std)

        self.grow_branch(child, angle + delta_left, next_length, depth + 1)
        self.grow_branch(child, angle - delta_right, next_length, depth + 1)
