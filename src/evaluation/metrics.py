"""
Evaluation metrics for generated retinal vascular networks.

Metrics assess spatial coverage, structural efficiency, and geometric
properties of synthetic vascular trees.
"""

import numpy as np
from typing import List, Tuple, Optional


# =============================================================================
# Nearest-neighbor analysis
# =============================================================================

def nearest_neighbor_distances(points: List[Tuple[float, float]]) -> np.ndarray:
    """
    Compute the nearest-neighbor distance for each point.

    Parameters
    ----------
    points : List[Tuple[float, float]]
        List of (x, y) coordinates.

    Returns
    -------
    np.ndarray
        Nearest-neighbor distance for each point.
    """
    pts = np.array(points)
    n = len(pts)
    if n < 2:
        return np.array([])

    # Compute pairwise distance matrix
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist_matrix, np.inf)

    return np.min(dist_matrix, axis=1)


def coverage_score(terminal_points: List[Tuple[float, float]]) -> float:
    """
    Compute the coverage score for a set of terminal node positions.

    The coverage score is the standard deviation of nearest-neighbor
    distances. Lower values indicate more uniform spatial coverage.

    Parameters
    ----------
    terminal_points : List[Tuple[float, float]]
        Positions of terminal (leaf) nodes.

    Returns
    -------
    float
        Standard deviation of nearest-neighbor distances.
    """
    nn = nearest_neighbor_distances(terminal_points)
    if len(nn) == 0:
        return float("nan")
    return float(np.std(nn))


def coverage_uniformity(terminal_points: List[Tuple[float, float]]) -> float:
    """
    Compute coverage uniformity as the coefficient of variation of
    nearest-neighbor distances. Lower = more uniform.

    Parameters
    ----------
    terminal_points : List[Tuple[float, float]]
        Positions of terminal (leaf) nodes.

    Returns
    -------
    float
        Coefficient of variation (std / mean) of NN distances.
    """
    nn = nearest_neighbor_distances(terminal_points)
    if len(nn) == 0 or np.mean(nn) == 0:
        return float("nan")
    return float(np.std(nn) / np.mean(nn))


# =============================================================================
# Structural efficiency
# =============================================================================

def total_vessel_length(edge_lengths: List[float]) -> float:
    """Sum of all edge lengths."""
    return sum(edge_lengths)


def length_per_terminal(
    edge_lengths: List[float],
    n_terminals: int,
) -> float:
    """
    Total vessel length divided by the number of terminal nodes.

    Measures the structural cost per unit of coverage. Lower values
    indicate more efficient networks.
    """
    if n_terminals == 0:
        return float("nan")
    return sum(edge_lengths) / n_terminals


# =============================================================================
# Fractal dimension
# =============================================================================

def fractal_dimension_box_counting(
    points: List[Tuple[float, float]],
    n_scales: int = 10,
    domain_radius: float = 1.0,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Estimate the fractal dimension of a point set using box counting.

    Parameters
    ----------
    points : List[Tuple[float, float]]
        Set of (x, y) coordinates (e.g., all node positions or
        sampled points along edges).
    n_scales : int
        Number of box sizes to test.
    domain_radius : float
        Radius of the bounding domain.

    Returns
    -------
    Tuple[float, np.ndarray, np.ndarray]
        (fractal_dimension, box_sizes, box_counts) where:
        - fractal_dimension: slope of log(N) vs log(1/epsilon)
        - box_sizes: array of epsilon values used
        - box_counts: array of box counts at each scale
    """
    pts = np.array(points)
    if len(pts) < 2:
        return float("nan"), np.array([]), np.array([])

    # Define box sizes from large to small
    max_size = 2 * domain_radius
    min_size = max_size / (2 ** n_scales)
    box_sizes = np.geomspace(max_size, min_size, n_scales)

    box_counts = []
    for eps in box_sizes:
        # Shift points so minimum is at origin
        shifted = pts - pts.min(axis=0)
        # Assign each point to a grid cell
        cell_indices = (shifted / eps).astype(int)
        # Count unique occupied cells
        unique_cells = set(map(tuple, cell_indices))
        box_counts.append(len(unique_cells))

    box_sizes = np.array(box_sizes)
    box_counts = np.array(box_counts, dtype=float)

    # Filter out scales where count is 0 or 1
    valid = box_counts > 1
    if valid.sum() < 2:
        return float("nan"), box_sizes, box_counts

    log_inv_eps = np.log(1.0 / box_sizes[valid])
    log_n = np.log(box_counts[valid])

    # Linear regression: log(N) = D * log(1/eps) + c
    coeffs = np.polyfit(log_inv_eps, log_n, 1)
    fractal_dim = coeffs[0]

    return float(fractal_dim), box_sizes, box_counts


# =============================================================================
# Branch angle analysis
# =============================================================================

def compute_branch_angles(
    edges: list,
) -> List[Tuple[float, float, float]]:
    """
    Compute branching angles at each bifurcation point.

    Parameters
    ----------
    edges : list
        List of Edge objects with .start, .end attributes (tuples).

    Returns
    -------
    List[Tuple[float, float, float]]
        For each bifurcation: (parent_angle, child1_angle, child2_angle)
        where angles are the absolute direction of each segment in radians.
    """
    # Build adjacency: for each start point, find all edges originating there
    from collections import defaultdict
    children_of = defaultdict(list)
    for edge in edges:
        children_of[edge.start].append(edge)

    # Find parent edge for each node
    parent_of = {}
    for edge in edges:
        parent_of[edge.end] = edge

    bifurcations = []
    for node, child_edges in children_of.items():
        if len(child_edges) != 2:
            continue

        # Get parent angle (direction arriving at this node)
        if node in parent_of:
            pe = parent_of[node]
            parent_angle = np.arctan2(
                pe.end[1] - pe.start[1],
                pe.end[0] - pe.start[0]
            )
        else:
            # Root node — no parent, use the mean direction of children
            parent_angle = None

        # Child angles
        child_angles = []
        for ce in child_edges:
            a = np.arctan2(
                ce.end[1] - ce.start[1],
                ce.end[0] - ce.start[0]
            )
            child_angles.append(a)

        if parent_angle is not None:
            bifurcations.append((parent_angle, child_angles[0], child_angles[1]))

    return bifurcations


def branch_angle_statistics(edges: list) -> dict:
    """
    Compute summary statistics on branching angles.

    Parameters
    ----------
    edges : list
        List of Edge objects with .start, .end attributes.

    Returns
    -------
    dict
        Keys:
        - 'n_bifurcations': number of bifurcation points
        - 'mean_total_angle': mean total bifurcation angle (degrees)
        - 'std_total_angle': std of total bifurcation angle
        - 'mean_asymmetry': mean absolute difference between child angles
        - 'angles_deg': list of individual (child1_delta, child2_delta) in degrees
    """
    bifs = compute_branch_angles(edges)

    if len(bifs) == 0:
        return {
            'n_bifurcations': 0,
            'mean_total_angle': float('nan'),
            'std_total_angle': float('nan'),
            'mean_asymmetry': float('nan'),
            'angles_deg': [],
        }

    total_angles = []
    asymmetries = []
    angle_pairs = []

    for parent_angle, c1, c2 in bifs:
        # Compute the angle each child deviates from parent direction
        delta1 = _angle_diff(c1, parent_angle)
        delta2 = _angle_diff(c2, parent_angle)

        total = abs(delta1) + abs(delta2)
        asym = abs(abs(delta1) - abs(delta2))

        total_angles.append(np.degrees(total))
        asymmetries.append(np.degrees(asym))
        angle_pairs.append((np.degrees(delta1), np.degrees(delta2)))

    return {
        'n_bifurcations': len(bifs),
        'mean_total_angle': float(np.mean(total_angles)),
        'std_total_angle': float(np.std(total_angles)),
        'mean_asymmetry': float(np.mean(asymmetries)),
        'angles_deg': angle_pairs,
    }


def _angle_diff(a: float, b: float) -> float:
    """Signed angular difference a - b, wrapped to [-pi, pi]."""
    d = a - b
    while d > np.pi:
        d -= 2 * np.pi
    while d < -np.pi:
        d += 2 * np.pi
    return d


# =============================================================================
# Density correlation
# =============================================================================

def density_correlation(
    terminal_points: List[Tuple[float, float]],
    target_density: np.ndarray,
    domain_radius: float = 1.0,
) -> float:
    """
    Compute the Pearson correlation between the spatial density of
    terminal nodes and a target vessel density map.

    Parameters
    ----------
    terminal_points : List[Tuple[float, float]]
        Positions of terminal nodes in normalized [-1, 1] coords.
    target_density : np.ndarray
        Target density map (grid_size x grid_size), values in [0, 1].
    domain_radius : float
        Radius of the domain.

    Returns
    -------
    float
        Pearson correlation coefficient in [-1, 1].
    """
    grid_size = target_density.shape[0]
    pts = np.array(terminal_points)

    if len(pts) == 0:
        return float("nan")

    # Build density map from terminal points
    generated_density = np.zeros((grid_size, grid_size), dtype=np.float64)

    for x, y in pts:
        # Map [-1, 1] to grid indices
        gi = int((1 - y / domain_radius) / 2 * grid_size)
        gj = int((x / domain_radius + 1) / 2 * grid_size)
        gi = max(0, min(grid_size - 1, gi))
        gj = max(0, min(grid_size - 1, gj))
        generated_density[gi, gj] += 1

    # Normalize
    max_val = generated_density.max()
    if max_val > 0:
        generated_density /= max_val

    # Flatten and compute correlation
    g_flat = generated_density.flatten()
    t_flat = target_density.flatten()

    # Only compare cells where target has nonzero density
    mask = t_flat > 0
    if mask.sum() < 3:
        return float("nan")

    corr = np.corrcoef(g_flat[mask], t_flat[mask])
    return float(corr[0, 1])
