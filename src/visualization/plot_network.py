"""
Visualization tools for retinal vascular networks.

Provides functions for plotting generated vascular trees, constraint
overlays, comparison panels, and evaluation results.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from typing import List, Tuple, Optional


def plot_vascular_tree(
    generator,
    title: str = "Retinal Vascular Tree",
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Axes:
    """
    Plot a generated vascular tree with anatomical constraints.

    Parameters
    ----------
    generator : RetinalTreeGenerator
        A generator that has already called .generate().
    title : str
        Plot title.
    ax : plt.Axes, optional
        Axes to draw on. If None, creates a new figure.
    save_path : str, optional
        If provided, save the figure to this path.
    show : bool
        Whether to call plt.show().

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))

    cfg = generator.config

    # Retina boundary
    ax.add_patch(Circle((0, 0), cfg.retina_radius, fill=False,
                        linestyle="--", linewidth=1.2, color="gray",
                        label="Retina boundary"))
    # Macula
    ax.add_patch(Circle(cfg.macula_center, cfg.macula_radius, fill=False,
                        linestyle=":", linewidth=1.5, color="orange",
                        label="Macula avascular zone"))
    # Root
    ax.scatter([cfg.root[0]], [cfg.root[1]], s=80, c="gold", edgecolors="black",
               zorder=5, label="Root / Optic Disc")

    # Edges colored by depth
    max_depth = max((e.depth for e in generator.edges), default=1)
    cmap = plt.cm.Reds
    for edge in generator.edges:
        color = cmap(0.3 + 0.7 * edge.depth / max_depth)
        lw = max(0.4, 3.0 - 0.4 * edge.depth)
        ax.plot([edge.start[0], edge.end[0]],
                [edge.start[1], edge.end[1]],
                linewidth=lw, color=color, alpha=0.85)

    ax.set_aspect("equal")
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right", fontsize=8)

    if save_path:
        ax.figure.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return ax


def plot_terminal_distribution(
    generator,
    title: str = "Terminal Node Distribution",
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Axes:
    """Plot the vascular tree with terminal nodes highlighted."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))

    cfg = generator.config
    terminals = generator.terminal_nodes()

    ax.add_patch(Circle((0, 0), cfg.retina_radius, fill=False,
                        linestyle="--", linewidth=1.2, color="gray"))
    ax.add_patch(Circle(cfg.macula_center, cfg.macula_radius, fill=False,
                        linestyle=":", linewidth=1.5, color="orange"))

    for edge in generator.edges:
        ax.plot([edge.start[0], edge.end[0]],
                [edge.start[1], edge.end[1]],
                linewidth=0.6, color="gray", alpha=0.4)

    xs = [n.x for n in terminals]
    ys = [n.y for n in terminals]
    ax.scatter(xs, ys, s=25, c="steelblue", zorder=5, label=f"Terminal nodes (n={len(terminals)})")
    ax.scatter([cfg.root[0]], [cfg.root[1]], s=80, c="gold", edgecolors="black",
               zorder=5, label="Root / Optic Disc")

    ax.set_aspect("equal")
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)

    if save_path:
        ax.figure.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return ax


def plot_constraints_overlay(
    image: np.ndarray,
    constraints: dict,
    title: str = "Extracted Constraints",
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Axes:
    """
    Overlay extracted constraints on the original fundus image.

    Parameters
    ----------
    image : np.ndarray
        Fundus image (RGB or BGR).
    constraints : dict
        Output of extract_all_constraints().
    title : str
        Plot title.
    """
    import cv2

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # Convert BGR to RGB if needed
    if image.ndim == 3 and image.shape[2] == 3:
        display_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        display_img = image

    ax.imshow(display_img)

    # Retina boundary
    rc = constraints['retina_center']
    rr = constraints['retina_radius']
    ax.add_patch(Circle(rc, rr, fill=False, color="cyan", linewidth=2,
                        linestyle="--", label="Retina boundary"))

    # Optic disc
    odc = constraints['optic_disc_center']
    odr = constraints['optic_disc_radius']
    ax.add_patch(Circle(odc, odr, fill=False, color="yellow", linewidth=2,
                        label=f"Optic disc (r={odr})"))
    ax.plot(*odc, 'y+', markersize=15, markeredgewidth=2)

    # Macula
    mc = constraints['macula_center']
    mr = constraints['macula_radius']
    ax.add_patch(Circle(mc, mr, fill=False, color="lime", linewidth=2,
                        linestyle=":", label=f"Macula (r={mr})"))
    ax.plot(*mc, 'g+', markersize=15, markeredgewidth=2)

    # Vessel orientation arrows
    angle_sup = constraints['angle_superior']
    angle_inf = constraints['angle_inferior']
    arrow_len = rr * 0.3

    for angle, label_text, color in [
        (angle_sup, "Superior arcade", "deepskyblue"),
        (angle_inf, "Inferior arcade", "coral"),
    ]:
        dx = arrow_len * np.cos(angle)
        dy = -arrow_len * np.sin(angle)  # Flip y for image coords
        ax.annotate("", xy=(odc[0] + dx, odc[1] + dy), xytext=odc,
                     arrowprops=dict(arrowstyle="->", color=color, lw=2.5))

    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.8)
    ax.axis("off")

    if save_path:
        ax.figure.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return ax


def plot_comparison(
    generator_baseline,
    generator_constrained,
    suptitle: str = "Baseline vs. Image-Constrained Generation",
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Side-by-side comparison of baseline and constrained generation.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    plot_vascular_tree(generator_baseline, title="Baseline (manual params)",
                       ax=axes[0], show=False)
    plot_vascular_tree(generator_constrained, title="Image-Constrained",
                       ax=axes[1], show=False)

    fig.suptitle(suptitle, fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()


def plot_density_map(
    density_map: np.ndarray,
    title: str = "Vessel Density Map",
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Axes:
    """Plot a vessel density map as a heatmap."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    im = ax.imshow(density_map, cmap="hot", interpolation="bilinear",
                   extent=[-1, 1, -1, 1], origin="lower")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.figure.colorbar(im, ax=ax, label="Relative density", shrink=0.8)

    if save_path:
        ax.figure.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return ax
