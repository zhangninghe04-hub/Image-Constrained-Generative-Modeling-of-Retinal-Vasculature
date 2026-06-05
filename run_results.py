"""
Generate visual results and batch evaluation summaries.

The script uses only numpy, pandas, and Pillow for portability. It will use
the project's OpenCV-based constraint extraction when OpenCV is available, and
otherwise falls back to the lightweight fundus constraint extractor.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from src.constraint_extraction.extract_constraints import extract_all_constraints
from src.generative_models.branching_model import (
    ConstrainedTreeGenerator,
    RetinalTreeGenerator,
    TreeGeneratorConfig,
)
from src.evaluation.metrics import (
    branch_angle_statistics,
    coverage_dispersion,
    coverage_uniformity,
    density_correlation,
    density_lift_over_random,
    fractal_dimension_box_counting,
    length_per_terminal,
    matched_terminal_density_score,
    occupied_grid_coverage,
    terminal_density_score,
)


figures_dir = Path("figures")
results_dir = Path("results")
figures_dir.mkdir(exist_ok=True)
results_dir.mkdir(exist_ok=True)
ABLATION_RESULTS_PATH = "results/ablation_summary.csv"

SAMPLE_IMAGES = ["01_h.jpg", "05_h.jpg", "10_h.jpg"]
EVALUATION_IMAGES = [
    "01_h.jpg", "02_h.jpg", "03_h.jpg", "04_h.jpg", "05_h.jpg",
    "06_h.jpg", "07_h.jpg", "08_h.jpg", "09_h.jpg", "10_h.jpg",
    "11_h.jpg", "12_h.jpg", "13_h.jpg", "14_h.jpg", "15_h.jpg",
]

DENSITY_WEIGHT = 0.6
MODEL_ALPHA = 0.76
MODEL_MAX_DEPTH = 7
MODEL_INITIAL_LENGTH = 0.23
MATCHED_SAMPLE_SIZE = 12
MODEL_SPECS = [
    {"label": "Baseline", "class": RetinalTreeGenerator, "from_constraints": False, "density_weight": 0.0, "depth": 0.0, "direction": 0.0, "survival": 0.0},
    {"label": "Constrained", "class": RetinalTreeGenerator, "from_constraints": True, "density_weight": 0.0, "depth": 0.0, "direction": 0.0, "survival": 0.0},
    {"label": "Density Depth Only", "class": ConstrainedTreeGenerator, "from_constraints": True, "density_weight": DENSITY_WEIGHT, "depth": 0.75, "direction": 0.0, "survival": 0.0},
    {"label": "Density Direction Only", "class": ConstrainedTreeGenerator, "from_constraints": True, "density_weight": DENSITY_WEIGHT, "depth": 0.0, "direction": 0.70, "survival": 0.0},
    {"label": "Density Survival Only", "class": ConstrainedTreeGenerator, "from_constraints": True, "density_weight": DENSITY_WEIGHT, "depth": 0.0, "direction": 0.0, "survival": 0.05},
    {"label": "Density-Aware", "class": ConstrainedTreeGenerator, "from_constraints": True, "density_weight": DENSITY_WEIGHT, "depth": 0.75, "direction": 0.70, "survival": 0.0},
]

MODEL_ORDER = [spec["label"] for spec in MODEL_SPECS]


def load_image(name):
    path = Path("data/raw/healthy") / name
    return np.array(Image.open(path).convert("RGB"))


def build_generator(spec, constraints=None):
    if spec["from_constraints"]:
        cfg = TreeGeneratorConfig.from_constraints(
            constraints,
            alpha=MODEL_ALPHA,
            max_depth=MODEL_MAX_DEPTH,
            initial_length=MODEL_INITIAL_LENGTH,
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
    matched_sample_size = min(MATCHED_SAMPLE_SIZE, len(terms))
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
        "matched_terminal_density_score": matched_terminal_density_score(
            terms, target_density, sample_size=matched_sample_size
        ),
        "density_lift_over_random": density_lift_over_random(
            terms, target_density, sample_size=matched_sample_size
        ),
    }


def font(size=18):
    try:
        return ImageFont.truetype("Arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def paste_title(canvas, xy, text, size=18):
    draw = ImageDraw.Draw(canvas)
    draw.multiline_text(xy, text, fill=(20, 20, 20), font=font(size), spacing=4)


def normalize_to_panel(x, y, box):
    x0, y0, w, h = box
    px = x0 + int((x + 1.12) / 2.24 * w)
    py = y0 + int((1.12 - y) / 2.24 * h)
    return px, py


def draw_tree_panel(draw, gen, box, title):
    x0, y0, w, h = box
    draw.rectangle([x0, y0, x0 + w, y0 + h], fill=(250, 250, 250), outline=(220, 220, 220))
    cx, cy = normalize_to_panel(0, 0, box)
    radius = int(1.0 / 2.24 * min(w, h))
    draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], outline=(150, 150, 150), width=2)

    mcx, mcy = normalize_to_panel(*gen.config.macula_center, box)
    mr = int(gen.config.macula_radius / 2.24 * min(w, h))
    draw.ellipse([mcx - mr, mcy - mr, mcx + mr, mcy + mr], outline=(230, 126, 34), width=2)

    max_depth = max((e.depth for e in gen.edges), default=1)
    for e in gen.edges:
        p1 = normalize_to_panel(e.start[0], e.start[1], box)
        p2 = normalize_to_panel(e.end[0], e.end[1], box)
        shade = int(220 - 150 * e.depth / max_depth)
        width = max(1, int(4 - 0.45 * e.depth))
        draw.line([p1, p2], fill=(shade, 40, 35), width=width)

    rx, ry = normalize_to_panel(*gen.config.root, box)
    draw.ellipse([rx - 5, ry - 5, rx + 5, ry + 5], fill=(241, 196, 15), outline=(0, 0, 0))
    terms = gen.terminal_nodes()
    metrics = f"edges={len(gen.edges)} terminals={len(terms)}"
    draw.text((x0 + 8, y0 + 8), title, fill=(20, 20, 20), font=font(16))
    draw.text((x0 + 8, y0 + h - 26), metrics, fill=(40, 40, 40), font=font(13))


def heatmap_image(density, size):
    arr = np.clip(density, 0, 1)
    arr = np.kron(arr, np.ones((max(1, size // arr.shape[0]), max(1, size // arr.shape[1]))))
    arr = arr[:size, :size]
    red = (255 * arr).astype(np.uint8)
    green = (180 * np.maximum(arr - 0.25, 0) / 0.75).astype(np.uint8)
    blue = np.zeros_like(red)
    return Image.fromarray(np.dstack([red, green, blue]), "RGB").resize((size, size))


def vessel_mask_image(mask, size):
    arr = (mask > 0).astype(np.uint8) * 255
    return Image.fromarray(arr, "L").resize((size, size)).convert("RGB")


def draw_constraints_panel(base, constraints, title, size=(520, 420)):
    img = Image.fromarray(base).resize(size)
    draw = ImageDraw.Draw(img)
    sx = size[0] / base.shape[1]
    sy = size[1] / base.shape[0]

    def scale_circle(center, radius):
        x, y = center
        return int(x * sx), int(y * sy), int(radius * (sx + sy) / 2)

    for center, radius, color in [
        (constraints["retina_center"], constraints["retina_radius"], (0, 210, 255)),
        (constraints["optic_disc_center"], constraints["optic_disc_radius"], (255, 230, 0)),
        (constraints["macula_center"], constraints["macula_radius"], (46, 204, 113)),
    ]:
        x, y, r = scale_circle(center, radius)
        draw.ellipse([x - r, y - r, x + r, y + r], outline=color, width=3)

    od_x, od_y, _ = scale_circle(constraints["optic_disc_center"], constraints["optic_disc_radius"])
    alen = int(constraints["retina_radius"] * 0.28 * (sx + sy) / 2)
    for angle, color in [
        (constraints["angle_superior"], (0, 180, 255)),
        (constraints["angle_inferior"], (230, 90, 70)),
    ]:
        dx = int(alen * np.cos(angle))
        dy = int(-alen * np.sin(angle))
        draw.line([od_x, od_y, od_x + dx, od_y + dy], fill=color, width=4)

    draw.rectangle([0, 0, size[0], 42], fill=(255, 255, 255))
    draw.text((8, 8), title, fill=(20, 20, 20), font=font(16))
    return img


def figure_constraints():
    panels = []
    for name in SAMPLE_IMAGES:
        img = load_image(name)
        constraints = extract_all_constraints(img)
        norm = constraints["normalized"]
        title = (
            f"{name} OD=({norm['root'][0]:.2f},{norm['root'][1]:.2f}) "
            f"sup={np.degrees(constraints['angle_superior']):.0f} "
            f"inf={np.degrees(constraints['angle_inferior']):.0f}"
        )
        panels.append(draw_constraints_panel(img, constraints, title))

    canvas = Image.new("RGB", (1560, 470), "white")
    paste_title(canvas, (20, 14), "Extracted Image Constraints", 24)
    for i, panel in enumerate(panels):
        canvas.paste(panel, (i * 520, 50))
    canvas.save(figures_dir / "fig1_constraint_overlays.png")


def figure_model_comparison():
    cell_w, cell_h = 420, 420
    canvas = Image.new("RGB", (cell_w * 3, cell_h * 3 + 70), "white")
    paste_title(canvas, (20, 16), "Baseline | Image-Constrained | Density-Aware", 24)
    draw = ImageDraw.Draw(canvas)
    for ri, name in enumerate(SAMPLE_IMAGES):
        img = load_image(name)
        constraints = extract_all_constraints(img)
        gens = [
            ("Baseline", build_generator(MODEL_SPECS[0])),
            ("Constrained", build_generator(MODEL_SPECS[1], constraints)),
            ("Density-Aware", build_generator(MODEL_SPECS[-1], constraints)),
        ]
        for ci, (label, gen) in enumerate(gens):
            box = (ci * cell_w + 12, ri * cell_h + 70, cell_w - 24, cell_h - 24)
            draw_tree_panel(draw, gen, box, f"{name} - {label}")
    canvas.save(figures_dir / "fig2_model_comparison.png")


def figure_density_terminals():
    cell = 360
    canvas = Image.new("RGB", (cell * 3, cell * 3 + 70), "white")
    paste_title(canvas, (20, 16), "Vessel Segmentation | Density Map | Terminal Distribution", 24)
    draw = ImageDraw.Draw(canvas)
    for ri, name in enumerate(SAMPLE_IMAGES):
        img = load_image(name)
        constraints = extract_all_constraints(img)
        gen = build_generator(MODEL_SPECS[-1], constraints)
        y = ri * cell + 70
        canvas.paste(vessel_mask_image(constraints["vessel_mask"], cell - 24), (12, y + 12))
        canvas.paste(heatmap_image(constraints["vessel_density_map"], cell - 24), (cell + 12, y + 12))
        draw_tree_panel(draw, gen, (cell * 2 + 12, y + 12, cell - 24, cell - 24), f"{name} terminals")
        draw.text((18, y + 18), f"{name} segmentation", fill=(20, 20, 20), font=font(15))
        draw.text((cell + 18, y + 18), f"{name} density", fill=(255, 255, 255), font=font(15))
    canvas.save(figures_dir / "fig3_density_terminals.png")


def draw_terminal_overlay(draw, density, gen, box, title, color):
    x0, y0, w, h = box
    heat = heatmap_image(density, min(w, h)).resize((w, h))
    return heat


def figure_terminal_density_overlay():
    cell = 360
    canvas = Image.new("RGB", (cell * 3, cell * 3 + 70), "white")
    paste_title(canvas, (20, 16), "Terminal Nodes Over Fundus-Derived Density Map", 24)
    draw = ImageDraw.Draw(canvas)
    colors = {
        "Baseline": (60, 130, 220),
        "Constrained": (255, 210, 40),
        "Density-Aware": (80, 230, 120),
    }
    for ri, name in enumerate(SAMPLE_IMAGES):
        img = load_image(name)
        constraints = extract_all_constraints(img)
        density = constraints["vessel_density_map"]
        gens = [
            ("Baseline", build_generator(MODEL_SPECS[0])),
            ("Constrained", build_generator(MODEL_SPECS[1], constraints)),
            ("Density-Aware", build_generator(MODEL_SPECS[-1], constraints)),
        ]
        for ci, (label, gen) in enumerate(gens):
            x = ci * cell + 12
            y = ri * cell + 70
            panel = heatmap_image(density, cell - 24)
            canvas.paste(panel, (x, y + 12))
            panel_box = (x, y + 12, cell - 24, cell - 24)
            for term in gen.terminal_nodes():
                px, py = normalize_to_panel(term.x, term.y, panel_box)
                draw.ellipse([px - 4, py - 4, px + 4, py + 4], fill=colors[label], outline=(0, 0, 0))
            draw.text((x + 6, y + 18), f"{name} - {label}", fill=(255, 255, 255), font=font(15))
            draw.text((x + 6, y + cell - 34), f"terminals={len(gen.terminal_nodes())}", fill=(255, 255, 255), font=font(13))
    canvas.save(figures_dir / "fig5_terminal_density_overlay.png")


def figure_summary(summary):
    metrics = [
        ("terminals", "# Terminals"),
        ("length", "Total Length"),
        ("occupied_grid_coverage", "Occupied Grid Coverage"),
        ("matched_terminal_density_score", "Matched Density Score"),
        ("density_lift_over_random", "Density Lift vs Random"),
        ("terminal_density_score", "Terminal Density Score"),
    ]
    w, h = 1500, 900
    canvas = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(canvas)
    paste_title(canvas, (20, 18), "Evaluation Metrics - Average over 15 Fundus Images", 26)
    colors = [(91, 155, 213), (237, 125, 49), (165, 165, 165), (255, 192, 0), (68, 114, 196), (112, 173, 71)]
    panel_w, panel_h = 480, 360
    for idx, (metric, label) in enumerate(metrics):
        x0 = 30 + (idx % 3) * panel_w
        y0 = 80 + (idx // 3) * panel_h
        vals = [float(summary.loc[m, metric]) for m in MODEL_ORDER]
        finite = [v for v in vals if np.isfinite(v)]
        ymin = min(0, min(finite))
        ymax = max(0, max(finite))
        span = ymax - ymin if ymax > ymin else 1
        draw.text((x0, y0), label, fill=(20, 20, 20), font=font(18))
        axis_y = y0 + 270 if ymin >= 0 else y0 + int(270 - (0 - ymin) / span * 220)
        draw.line([x0 + 40, axis_y, x0 + panel_w - 30, axis_y], fill=(120, 120, 120), width=1)
        bar_w = 48
        gap = 18
        for i, (model, val) in enumerate(zip(MODEL_ORDER, vals)):
            bx = x0 + 50 + i * (bar_w + gap)
            by = y0 + 270 - int((val - ymin) / span * 220)
            top = min(by, axis_y)
            bottom = max(by, axis_y)
            draw.rectangle([bx, top, bx + bar_w, bottom], fill=colors[i], outline=(255, 255, 255))
            draw.text((bx - 4, min(by, axis_y) - 18), f"{val:.3f}", fill=(35, 35, 35), font=font(12))
            draw.text((bx - 10, y0 + 282), str(i + 1), fill=(35, 35, 35), font=font(12))
        legend = " ".join([f"{i+1}:{m.split()[0]}" for i, m in enumerate(MODEL_ORDER)])
        draw.text((x0 + 40, y0 + 310), legend, fill=(70, 70, 70), font=font(11))
    canvas.save(figures_dir / "fig4_evaluation_summary.png")


def main():
    print("Generating figures and evaluation results...")
    figure_constraints()
    figure_model_comparison()
    figure_density_terminals()
    figure_terminal_density_overlay()

    all_rows = []
    for name in EVALUATION_IMAGES:
        img = load_image(name)
        constraints = extract_all_constraints(img)
        target_density = constraints["vessel_density_map"]
        for spec in MODEL_SPECS:
            gen = build_generator(spec, constraints)
            all_rows.append(evaluate_generator(gen, name, spec["label"], target_density))

    df = pd.DataFrame(all_rows)
    summary = df.groupby("model")[
        [
            "edges", "terminals", "length", "length_per_terminal",
            "coverage_dispersion", "coverage_uniformity", "occupied_grid_coverage",
            "fractal_dim", "branch_angle", "density_corr", "terminal_density_score",
            "matched_terminal_density_score", "density_lift_over_random",
        ]
    ].mean().round(3)
    summary = summary.reindex(MODEL_ORDER)

    df.to_csv(results_dir / "evaluation_results.csv", index=False)
    summary.to_csv(results_dir / "evaluation_summary.csv")
    summary.loc[MODEL_ORDER[2:]].to_csv(ABLATION_RESULTS_PATH)
    figure_summary(summary)

    print(summary.to_string())
    print("Generated:")
    for path in sorted(figures_dir.glob("fig*.png")):
        print(f"  {path}")


if __name__ == "__main__":
    main()
