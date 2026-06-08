"""
Systematic parameter search for the fundus-only density-aware model.

The search focuses on the current modeling problem: recovering spatial reach
and terminal count in the constrained generator while preserving improved
terminal placement in high-density fundus regions.
"""

import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, ".")

from src.constraint_extraction.extract_constraints import extract_all_constraints
from src.evaluation.metrics import (
    coverage_dispersion,
    density_lift_over_random,
    matched_terminal_density_score,
    occupied_grid_coverage,
    terminal_density_score,
)
from src.generative_models.branching_model import ConstrainedTreeGenerator, TreeGeneratorConfig


EVALUATION_IMAGES = [
    "01_h.jpg", "02_h.jpg", "03_h.jpg", "04_h.jpg", "05_h.jpg",
    "06_h.jpg", "07_h.jpg", "08_h.jpg", "09_h.jpg", "10_h.jpg",
    "11_h.jpg", "12_h.jpg", "13_h.jpg", "14_h.jpg", "15_h.jpg",
]

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
PARAMETER_SEARCH_PATH = "results/parameter_search.csv"
BEST_PARAMETER_SUMMARY_PATH = "results/best_parameter_summary.md"

SEARCH_GRID = {
    "alpha": [0.76],
    "max_depth": [6, 7],
    "initial_length": [0.23, 0.26],
    "density_weight": [0.60],
    "density_depth_weight": [0.50, 0.75],
    "density_direction_weight": [0.50, 0.70, 0.90],
    "density_survival_weight": [0.00, 0.05],
    "density_angle_span": [1.0, 1.5],
    "density_horizon_weight": [0.0, 0.5, 1.0],
}

TARGET_TERMINALS = 22.0
MATCHED_SAMPLE_SIZE = 12


def load_image(name):
    path = Path("data/raw/healthy") / name
    return np.array(Image.open(path).convert("RGB"))


def precompute_constraints():
    constraints = {}
    for name in EVALUATION_IMAGES:
        constraints[name] = extract_all_constraints(load_image(name))
    return constraints


def generate_density_model(params, constraints):
    cfg = TreeGeneratorConfig.from_constraints(
        constraints,
        alpha=params["alpha"],
        max_depth=params["max_depth"],
        initial_length=params["initial_length"],
        density_weight=params["density_weight"],
        density_depth_weight=params["density_depth_weight"],
        density_direction_weight=params["density_direction_weight"],
        density_survival_weight=params["density_survival_weight"],
        density_angle_span=params["density_angle_span"],
        density_horizon_weight=params["density_horizon_weight"],
        random_seed=42,
    )
    gen = ConstrainedTreeGenerator(cfg)
    gen.generate()
    return gen


def evaluate_params(params, all_constraints):
    rows = []
    for image_name, constraints in all_constraints.items():
        gen = generate_density_model(params, constraints)
        terms = [(n.x, n.y) for n in gen.terminal_nodes()]
        density_map = constraints["vessel_density_map"]
        sample_size = min(MATCHED_SAMPLE_SIZE, len(terms))
        rows.append({
            "image": image_name,
            "edges": len(gen.edges),
            "terminals": len(terms),
            "length": gen.total_length(),
            "occupied_grid_coverage": occupied_grid_coverage(terms),
            "coverage_dispersion": coverage_dispersion(terms),
            "terminal_density_score": terminal_density_score(terms, density_map),
            "matched_terminal_density_score": matched_terminal_density_score(
                terms, density_map, sample_size=sample_size
            ),
            "density_lift_over_random": density_lift_over_random(
                terms, density_map, sample_size=sample_size
            ),
        })

    df = pd.DataFrame(rows)
    mean = df.mean(numeric_only=True).to_dict()
    return mean


def score_candidate(mean_metrics):
    terminals = mean_metrics["terminals"]
    terminal_penalty = abs(terminals - TARGET_TERMINALS) / TARGET_TERMINALS
    sparse_penalty = max(0.0, 20.0 - terminals) / 20.0
    dense_penalty = max(0.0, terminals - 25.0) / 25.0
    return (
        mean_metrics["occupied_grid_coverage"] * 3.0
        + mean_metrics["matched_terminal_density_score"] * 1.5
        + mean_metrics["density_lift_over_random"] * 2.0
        - mean_metrics["coverage_dispersion"] * 0.6
        - terminal_penalty * 0.45
        - sparse_penalty * 0.9
        - dense_penalty * 1.1
    )


def iter_grid(grid):
    keys = list(grid)
    for combo in product(*[grid[k] for k in keys]):
        params = {}
        for key, value in zip(keys, combo):
            params[key] = int(value) if key == "max_depth" else float(value)
        yield params


def main():
    print("Precomputing fundus constraints...")
    all_constraints = precompute_constraints()

    records = []
    total = np.prod([len(v) for v in SEARCH_GRID.values()])
    for i, params in enumerate(iter_grid(SEARCH_GRID), start=1):
        metrics = evaluate_params(params, all_constraints)
        score = score_candidate(metrics)
        record = {**params, **metrics, "search_score": score}
        records.append(record)
        if i % 50 == 0 or i == total:
            print(f"searched {i}/{total}")

    df = pd.DataFrame(records).sort_values("search_score", ascending=False)
    out_csv = Path(PARAMETER_SEARCH_PATH)
    df.to_csv(out_csv, index=False)

    best = df.iloc[0]
    summary = Path(BEST_PARAMETER_SUMMARY_PATH)
    summary.write_text(
        "# Best Parameter Search Result\n\n"
        f"Search rows: `{len(df)}`\n\n"
        "## Best Parameters\n\n"
        f"- alpha: `{best['alpha']:.2f}`\n"
        f"- max_depth: `{int(best['max_depth'])}`\n"
        f"- initial_length: `{best['initial_length']:.2f}`\n"
        f"- density_weight: `{best['density_weight']:.2f}`\n"
        f"- density_depth_weight: `{best['density_depth_weight']:.2f}`\n"
        f"- density_direction_weight: `{best['density_direction_weight']:.2f}`\n"
        f"- density_survival_weight: `{best['density_survival_weight']:.2f}`\n\n"
        f"- density_angle_span: `{best['density_angle_span']:.2f}`\n"
        f"- density_horizon_weight: `{best['density_horizon_weight']:.2f}`\n\n"
        "## Mean Metrics\n\n"
        f"- terminals: `{best['terminals']:.3f}`\n"
        f"- length: `{best['length']:.3f}`\n"
        f"- occupied_grid_coverage: `{best['occupied_grid_coverage']:.3f}`\n"
        f"- coverage_dispersion: `{best['coverage_dispersion']:.3f}`\n"
        f"- terminal_density_score: `{best['terminal_density_score']:.3f}`\n"
        f"- matched_terminal_density_score: `{best['matched_terminal_density_score']:.3f}`\n"
        f"- density_lift_over_random: `{best['density_lift_over_random']:.3f}`\n"
        f"- search_score: `{best['search_score']:.3f}`\n",
        encoding="utf-8",
    )

    print(df.head(10).to_string(index=False))
    print(f"Saved {out_csv}")
    print(f"Saved {summary}")


if __name__ == "__main__":
    main()
