"""
Visualization tools for retinal vascular networks.

Provides functions for plotting generated vascular trees, terminal node
distributions, and constraint overlays.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional


def plot_vascular_tree(
    edges: List[Tuple[Tuple[float, float], Tuple[float, float], int]],
    root: Tuple[float, float],
    retina_radius: float = 1.0,
    macula_center: Tuple[float, float] = (-0.25, 0.0),
    macula_radius: float = 0.16,
    title: str = "Retinal Vascular Tree",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot a generated vascular tree with anatomical constraints.

    Parameters
    ----------
    edges : List[Tuple[start, end, depth]]
        Each edge as ((x1,y1), (x2,y2), depth).
    root : Tuple[float, float]
        Root node (optic disc) position.
    retina_radius : float
        Radius of the retinal domain.
    macula_center : Tuple[float, float]
        Center of the macula avascular zone.
    macula_radius : float
        Radius of the macula avascular zone.
    title : str
        Plot title.
    save_path : str, optional
        If provided, save the figure to this path.
    """
    fig, ax = plt.subplots(figsize=(7, 7))

    retina = plt.Circle((0, 0), retina_radius, fill=False,
                         linestyle="--", linewidth=1.2, label="Retina boundary")
    ax.add_patch(retina)

    macula = plt.Circle(macula_center, macula_radius, fill=False,
                         linestyle=":", linewidth=1.5, label="Macula avascular zone")
    ax.add_patch(macula)

    ax.scatter([root[0]], [root[1]], s=60, zorder=5, label="Root / Optic Disc")

    for start, end, depth in edges:
        ax.plot([start[0], end[0]], [start[1], end[1]],
                linewidth=max(0.5, 2.5 - 0.25 * depth), color="darkred", alpha=0.8)

    ax.set_aspect("equal")
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_terminal_distribution(
    edges: List[Tuple[Tuple[float, float], Tuple[float, float], int]],
    terminal_points: List[Tuple[float, float]],
    root: Tuple[float, float],
    retina_radius: float = 1.0,
    macula_center: Tuple[float, float] = (-0.25, 0.0),
    macula_radius: float = 0.16,
    title: str = "Terminal Node Distribution",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot the vascular tree with terminal nodes highlighted.

    Parameters
    ----------
    edges : List[Tuple[start, end, depth]]
        Each edge as ((x1,y1), (x2,y2), depth).
    terminal_points : List[Tuple[float, float]]
        Positions of terminal (leaf) nodes.
    root : Tuple[float, float]
        Root node position.
    retina_radius, macula_center, macula_radius : float
        Anatomical constraint parameters.
    title : str
        Plot title.
    save_path : str, optional
        If provided, save the figure to this path.
    """
    fig, ax = plt.subplots(figsize=(7, 7))

    retina = plt.Circle((0, 0), retina_radius, fill=False, linestyle="--", linewidth=1.2)
    ax.add_patch(retina)
    macula = plt.Circle(macula_center, macula_radius, fill=False, linestyle=":", linewidth=1.5)
    ax.add_patch(macula)

    for start, end, depth in edges:
        ax.plot([start[0], end[0]], [start[1], end[1]], linewidth=0.8, color="gray", alpha=0.6)

    xs = [p[0] for p in terminal_points]
    ys = [p[1] for p in terminal_points]
    ax.scatter(xs, ys, s=30, zorder=5, label="Terminal nodes")
    ax.scatter([root[0]], [root[1]], s=60, zorder=5, label="Root / Optic Disc")

    ax.set_aspect("equal")
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_title(title)
    ax.legend()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
