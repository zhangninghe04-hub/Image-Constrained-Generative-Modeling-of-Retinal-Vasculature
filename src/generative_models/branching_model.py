"""
Generative branching model for retinal vascular networks.

This module provides the core tree generation logic, refactored from the
baseline notebook (notebooks/01_baseline_branching_model.ipynb) into
reusable classes. The model generates synthetic vascular trees using
recursive branching rules with configurable parameters and constraints.
"""

import numpy as np
from dataclasses import dataclass
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
    vascular network.
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

    random_seed: Optional[int] = 42


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
