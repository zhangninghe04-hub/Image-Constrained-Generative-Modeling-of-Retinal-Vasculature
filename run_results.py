"""
Generate visual results and batch evaluation summaries.

Three representative fundus images are used for qualitative figures. All
available fundus images are used for the quantitative robustness check.
"""

import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from pathlib import Path

from src.constraint_extraction.extract_constraints import extract_all_constraints
from src.generative_models.branching_model import (
    TreeGeneratorConfig, RetinalTreeGenerator, ConstrainedTreeGenerator
)
from src.evaluation.metrics import (
    coverage_score, coverage_uniformity, length_per_terminal,
    fractal_dimension_box_counting, branch_angle_statistics, density_correlation,
    coverage_dispersion, occupied_grid_coverage, terminal_density_score
)

figures_dir = Path("figures")
figures_dir.mkdir(exist_ok=True)

SAMPLE_IMAGES = ["01_h.jpg", "05_h.jpg", "10_h.jpg"]
EVALUATION_IMAGES = [
    "01_h.jpg", "02_h.jpg", "03_h.jpg", "04_h.jpg", "05_h.jpg",
    "06_h.jpg", "07_h.jpg", "08_h.jpg", "09_h.jpg", "10_h.jpg",
    "11_h.jpg", "12_h.jpg", "13_h.jpg", "14_h.jpg", "15_h.jpg",
]

DENSITY_WEIGHT = 0.6

MODEL_SPECS = [
    {"label": "Baseline", "class": RetinalTreeGenerator, "from_constraints": False, "density_weight": 0.0, "depth": 0.0, "direction": 0.0, "survival": 0.0},
    {"label": "Constrained", "class": RetinalTreeGenerator, "from_constraints": True, "density_weight": 0.0, "depth": 0.0, "direction": 0.0, "survival": 0.0},
    {"label": "Density Depth Only", "class": ConstrainedTreeGenerator, "from_constraints": True, "density_weight": DENSITY_WEIGHT, "depth": 1.0, "direction": 0.0, "survival": 0.0},
    {"label": "Density Direction Only", "class": ConstrainedTreeGenerator, "from_constraints": True, "density_weight": DENSITY_WEIGHT, "depth": 0.0, "direction": 1.0, "survival": 0.0},
    {"label": "Density Survival Only", "class": ConstrainedTreeGenerator, "from_constraints": True, "density_weight": DENSITY_WEIGHT, "depth": 0.0, "direction": 0.0, "survival": 1.0},
    {"label": "Density-Aware", "class": ConstrainedTreeGenerator, "from_constraints": True, "density_weight": DENSITY_WEIGHT, "depth": 1.0, "direction": 1.0, "survival": 1.0},
]


def build_generator(spec, constraints=None):
    if spec["from_constraints"]:
        cfg = TreeGeneratorConfig.from_constraints(
            constraints,
            density_weight=spec["density_weight"],
            density_depth_weight=spec["depth"],
            density_direction_weight=spec["direction"],
            density_survival_weight=spec["survival"],
            random_seed=42,
        )
    else:
        cfg = TreeGeneratorConfig(random_seed=42)

    gen = spec["class"](cfg)
    gen.generate()
    return gen


def evaluate_generator(gen, image_name, model_name, target_density):
    terms = [(n.x, n.y) for n in gen.terminal_nodes()]
    pts = [(n.x, n.y) for n in gen.nodes]
    edge_lengths = [e.length for e in gen.edges]
    fd, _, _ = fractal_dimension_box_counting(pts)
    ba = branch_angle_statistics(gen.edges)
    return {
        "image": image_name,
        "model": model_name,
        "edges": len(gen.edges),
        "terminals": len(gen.terminal_nodes()),
        "length": gen.total_length(),
        "length_per_terminal": length_per_terminal(edge_lengths, len(terms)),
        "coverage_dispersion": coverage_dispersion(terms),
        "coverage_uniformity": coverage_uniformity(terms),
        "occupied_grid_coverage": occupied_grid_coverage(terms),
        "fractal_dim": fd,
        "branch_angle": ba["mean_total_angle"],
        "density_corr": density_correlation(terms, target_density),
        "terminal_density_score": terminal_density_score(terms, target_density),
    }

# ─────────────────────────────────────────────
# Helper: draw a single generator onto axes
# ─────────────────────────────────────────────
def draw_tree(ax, gen, title):
    cfg = gen.config
    ax.add_patch(Circle((0,0), cfg.retina_radius, fill=False,
                        linestyle="--", linewidth=1, color="#888888"))
    ax.add_patch(Circle(cfg.macula_center, cfg.macula_radius, fill=False,
                        linestyle=":", linewidth=1.2, color="#e67e22"))
    ax.scatter([cfg.root[0]], [cfg.root[1]], s=70, c="#f1c40f",
               edgecolors="black", zorder=6, linewidths=0.8)

    max_d = max((e.depth for e in gen.edges), default=1)
    cmap  = plt.cm.Reds
    for e in gen.edges:
        c  = cmap(0.25 + 0.75 * e.depth / max_d)
        lw = max(0.35, 2.8 - 0.38 * e.depth)
        ax.plot([e.start[0], e.end[0]], [e.start[1], e.end[1]],
                lw=lw, color=c, alpha=0.88)

    terms = gen.terminal_nodes()
    ax.set_aspect("equal"); ax.set_xlim(-1.12, 1.12); ax.set_ylim(-1.12, 1.12)
    ax.set_title(title, fontsize=9, pad=4)
    ax.axis("off")

    cov = coverage_dispersion([(n.x, n.y) for n in terms])
    occ = occupied_grid_coverage([(n.x, n.y) for n in terms])
    fd,_,_ = fractal_dimension_box_counting([(n.x, n.y) for n in gen.nodes])
    ax.text(0.02, 0.02,
            f"edges={len(gen.edges)}  terminals={len(terms)}\n"
            f"dispersion={cov:.4f}  grid={occ:.3f}  FD={fd:.3f}",
            transform=ax.transAxes, fontsize=7, va="bottom",
            color="#222222",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7, lw=0))


# ─────────────────────────────────────────────
# Figure 1: Constraint overlays (all 3 images)
# ─────────────────────────────────────────────
print("Generating Figure 1: constraint overlays...")
fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
fig.suptitle("Extracted Image Constraints", fontsize=13, y=1.01)

for ax, name in zip(axes, SAMPLE_IMAGES):
    img = cv2.imread(f"data/raw/healthy/{name}")
    c   = extract_all_constraints(img)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(rgb)

    rc, rr = c["retina_center"],  c["retina_radius"]
    od, or_ = c["optic_disc_center"], c["optic_disc_radius"]
    mc, mr  = c["macula_center"],  c["macula_radius"]

    ax.add_patch(Circle(rc, rr, fill=False, color="cyan",    lw=1.8, linestyle="--"))
    ax.add_patch(Circle(od, or_,fill=False, color="yellow",  lw=2.0))
    ax.add_patch(Circle(mc, mr, fill=False, color="#2ecc71", lw=2.0, linestyle=":"))
    ax.plot(*od, "y+", ms=14, mew=2)
    ax.plot(*mc, color="#2ecc71", marker="+", ms=14, mew=2, linestyle="none")

    # Orientation arrows
    alen = rr * 0.28
    for angle, col in [(c["angle_superior"],"deepskyblue"),
                       (c["angle_inferior"],"tomato")]:
        dx =  alen * np.cos(angle)
        dy = -alen * np.sin(angle)
        ax.annotate("", xy=(od[0]+dx, od[1]+dy), xytext=od,
                    arrowprops=dict(arrowstyle="->", color=col, lw=2.2))

    norm = c["normalized"]
    ax.set_title(
        f"{name}\n"
        f"OD=({norm['root'][0]:.2f}, {norm['root'][1]:.2f})  "
        f"sup={np.degrees(c['angle_superior']):.0f}°  "
        f"inf={np.degrees(c['angle_inferior']):.0f}°",
        fontsize=8.5
    )
    ax.axis("off")

# Legend
from matplotlib.lines import Line2D
legend_els = [
    Line2D([0],[0], color="cyan",       lw=1.8, linestyle="--", label="Retina boundary"),
    Line2D([0],[0], color="yellow",     lw=2.0, label="Optic disc"),
    Line2D([0],[0], color="#2ecc71",    lw=2.0, linestyle=":", label="Macula"),
    Line2D([0],[0], color="deepskyblue",lw=2.2, label="Superior arcade"),
    Line2D([0],[0], color="tomato",     lw=2.2, label="Inferior arcade"),
]
fig.legend(handles=legend_els, loc="lower center", ncol=5,
           fontsize=8.5, framealpha=0.9, bbox_to_anchor=(0.5, -0.04))

plt.tight_layout()
fig.savefig("figures/fig1_constraint_overlays.png", dpi=150, bbox_inches="tight")
plt.close()
print("  -> saved figures/fig1_constraint_overlays.png")


# ─────────────────────────────────────────────
# Figure 2: Model comparison (3 images × 3 models)
# ─────────────────────────────────────────────
print("Generating Figure 2: model comparison grid...")
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
fig.suptitle("Baseline  |  Image-Constrained  |  Density-Aware\n"
             "(one row per image)", fontsize=12, y=1.01)

col_titles = ["Baseline", "Image-Constrained", "Density-Aware"]
for ci, ct in enumerate(col_titles):
    axes[0, ci].set_title(ct, fontsize=11, fontweight="bold", pad=6)

for ri, name in enumerate(SAMPLE_IMAGES):
    img = cv2.imread(f"data/raw/healthy/{name}")
    cst = extract_all_constraints(img)

    bg = build_generator(MODEL_SPECS[0])
    cg = build_generator(MODEL_SPECS[1], cst)
    dg = build_generator(MODEL_SPECS[-1], cst)

    for ci, (gen, label) in enumerate([(bg,"Baseline"),
                                        (cg,"Constrained"),
                                        (dg,"Density-Aware")]):
        draw_tree(axes[ri, ci], gen, f"{name} — {label}")

    axes[ri, 0].set_ylabel(name, fontsize=9, labelpad=6)

plt.tight_layout(h_pad=1.5, w_pad=1.0)
fig.savefig("figures/fig2_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  -> saved figures/fig2_model_comparison.png")


# ─────────────────────────────────────────────
# Figure 3: Vessel density maps + terminal distribution
# ─────────────────────────────────────────────
print("Generating Figure 3: density maps and terminal distributions...")
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
fig.suptitle("Vessel Segmentation  |  Density Map  |  Terminal Node Distribution",
             fontsize=12, y=1.01)

for ri, name in enumerate(SAMPLE_IMAGES):
    img = cv2.imread(f"data/raw/healthy/{name}")
    cst = extract_all_constraints(img)
    gen = build_generator(MODEL_SPECS[-1], cst)

    # Col 0: vessel segmentation
    axes[ri,0].imshow(cst["vessel_mask"], cmap="gray")
    axes[ri,0].set_title(f"{name}\nVessel Segmentation", fontsize=8.5)
    axes[ri,0].axis("off")

    # Col 1: density map
    im = axes[ri,1].imshow(cst["vessel_density_map"], cmap="hot",
                            interpolation="bilinear", origin="lower",
                            extent=[-1,1,-1,1])
    axes[ri,1].set_title("Vessel Density Map", fontsize=8.5)
    axes[ri,1].set_xlabel("x"); axes[ri,1].set_ylabel("y")
    fig.colorbar(im, ax=axes[ri,1], shrink=0.75, label="Rel. density")

    # Col 2: terminal node scatter over tree
    ax = axes[ri,2]
    cfg2 = gen.config
    ax.add_patch(Circle((0,0), cfg2.retina_radius, fill=False,
                         linestyle="--", lw=1, color="#aaaaaa"))
    ax.add_patch(Circle(cfg2.macula_center, cfg2.macula_radius, fill=False,
                         linestyle=":", lw=1.2, color="#e67e22"))
    for e in gen.edges:
        ax.plot([e.start[0],e.end[0]], [e.start[1],e.end[1]],
                lw=0.5, color="#cccccc", alpha=0.6)
    terms = gen.terminal_nodes()
    ax.scatter([n.x for n in terms], [n.y for n in terms],
               s=18, c="steelblue", zorder=5, alpha=0.8)
    ax.scatter([cfg2.root[0]], [cfg2.root[1]], s=80, c="#f1c40f",
               edgecolors="black", zorder=6, lw=0.8)
    ax.set_aspect("equal"); ax.set_xlim(-1.12,1.12); ax.set_ylim(-1.12,1.12)
    ax.set_title(f"Terminal Distribution\n(n={len(terms)})", fontsize=8.5)
    ax.axis("off")

plt.tight_layout(h_pad=1.5, w_pad=1.0)
fig.savefig("figures/fig3_density_terminals.png", dpi=150, bbox_inches="tight")
plt.close()
print("  -> saved figures/fig3_density_terminals.png")


# ─────────────────────────────────────────────
# Figure 4: Evaluation summary bar charts
# ─────────────────────────────────────────────
print("Generating Figure 4: 15-image evaluation summary...")
import pandas as pd

all_rows = []
for name in EVALUATION_IMAGES:
    img = cv2.imread(f"data/raw/healthy/{name}")
    cst = extract_all_constraints(img)
    target_d = cst["vessel_density_map"]

    for spec in MODEL_SPECS:
        gen = build_generator(spec, cst)
        all_rows.append(evaluate_generator(gen, name, spec["label"], target_d))

df = pd.DataFrame(all_rows)
summary = df.groupby("model")[
    [
        "edges", "terminals", "length", "length_per_terminal",
        "coverage_dispersion", "coverage_uniformity", "occupied_grid_coverage",
        "fractal_dim", "branch_angle", "density_corr", "terminal_density_score",
    ]
].mean().round(3)
summary = summary.reindex([spec["label"] for spec in MODEL_SPECS])

# Save CSV
df.to_csv("results/evaluation_results.csv", index=False)
summary.to_csv("results/evaluation_summary.csv")
summary.loc[[spec["label"] for spec in MODEL_SPECS[2:]]].to_csv("results/ablation_summary.csv")
print("  -> saved results/evaluation_results.csv")

metrics = ["terminals", "length", "occupied_grid_coverage", "coverage_dispersion", "density_corr", "terminal_density_score"]
metric_labels = ["# Terminals", "Total Length", "Occupied Grid Coverage", "Coverage Dispersion", "Density Correlation", "Terminal Density Score"]
colors_bar = ["#5b9bd5", "#ed7d31", "#a5a5a5", "#ffc000", "#4472c4", "#70ad47"]
model_order = [spec["label"] for spec in MODEL_SPECS]

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle("Evaluation Metrics - Average over 15 Fundus Images", fontsize=13)

for ax, metric, mlabel in zip(axes.flat, metrics, metric_labels):
    vals = [summary.loc[m, metric] for m in model_order]
    bars = ax.bar(model_order, vals, color=colors_bar, edgecolor="white",
                  linewidth=0.8, width=0.55)
    for bar, val in zip(bars, vals):
        offset = max(abs(v) for v in vals) * 0.03 or 0.01
        va = "bottom" if val >= 0 else "top"
        y = val + offset if val >= 0 else val - offset
        ax.text(bar.get_x() + bar.get_width()/2, y,
                f"{val:.3f}", ha="center", va=va, fontsize=8.5, fontweight="bold")
    ax.set_title(mlabel, fontsize=10)
    ymin = min(0, min(vals))
    ymax = max(0, max(vals))
    pad = (ymax - ymin) * 0.18 or 0.1
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.axhline(0, color="#888888", linewidth=0.7, alpha=0.8)
    ax.spines[["top","right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=8.5)
    ax.tick_params(axis="y", labelsize=8)

plt.tight_layout()
fig.savefig("figures/fig4_evaluation_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("  -> saved figures/fig4_evaluation_summary.png")

print("\nAll done! Generated:")
for f in sorted(Path("figures").glob("fig*.png")):
    print(f"  {f}")
