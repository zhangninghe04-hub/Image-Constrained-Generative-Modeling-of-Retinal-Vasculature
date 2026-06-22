"""
Optimize a single connected-recovery refinement for terminal small vessels.

This script continues from the previous HRF comparison. The four-method stage
identified connected_recovery as the most sensitive small-vessel baseline.
This update keeps only that route and applies a softer structural constraint:
remove isolated vessel candidates far from the main tree while preserving
candidate branches near the main-vessel and endpoint neighborhoods.
"""

from __future__ import annotations

import argparse
import importlib.util
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, distance_transform_edt, find_objects, label
from skimage.morphology import remove_small_objects, skeletonize


@dataclass(frozen=True)
class Params:
    anchor_radius: int
    endpoint_radius: int
    min_area: int
    min_mean_vesselness: float
    min_max_vesselness: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize connected_recovery for terminal vessels.")
    parser.add_argument("--hrf-root", required=True)
    parser.add_argument("--base-script", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--figure-dir", required=True)
    parser.add_argument("--max-image-side", type=int, default=1200)
    return parser.parse_args()


def load_base_module(path: Path):
    spec = importlib.util.spec_from_file_location("hrf_base_methods", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def image_files(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"})


def pair_hrf(root: Path) -> list[tuple[Path, Path]]:
    masks = {p.stem: p for p in image_files(root / "masks")}
    return [(p, masks[p.stem]) for p in image_files(root / "images") if p.stem in masks]


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def neighbor_count(skel: np.ndarray) -> np.ndarray:
    padded = np.pad(skel.astype(np.uint8), 1)
    count = np.zeros_like(skel, dtype=np.uint8)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx or dy:
                count += padded[1 + dy : 1 + dy + skel.shape[0], 1 + dx : 1 + dx + skel.shape[1]]
    return count


def skeleton_endpoints(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    skel = skeletonize(mask > 0)
    return skel, skel & (neighbor_count(skel) == 1)


def disk(radius: int) -> np.ndarray:
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (x * x + y * y) <= radius * radius


def terminal_region(gt: np.ndarray, radius: int = 8) -> np.ndarray:
    _, endpoints = skeleton_endpoints(gt)
    return gt & binary_dilation(endpoints, structure=disk(radius))


def mask_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    tn = int((~pred & ~gt).sum())
    return {
        "dice": safe_div(2 * tp, 2 * tp + fp + fn),
        "iou": safe_div(tp, tp + fp + fn),
        "sensitivity": safe_div(tp, tp + fn),
        "specificity": safe_div(tn, tn + fp),
        "precision": safe_div(tp, tp + fp),
        "fp_pixels": fp,
        "fn_pixels": fn,
        "fp_over_gt_area": safe_div(fp, int(gt.sum())),
        "fn_over_gt_area": safe_div(fn, int(gt.sum())),
    }


def endpoint_metrics(pred: np.ndarray, gt: np.ndarray, radius: int = 10) -> dict[str, float]:
    _, pred_end = skeleton_endpoints(pred)
    _, gt_end = skeleton_endpoints(gt)
    gt_count = int(gt_end.sum())
    pred_count = int(pred_end.sum())
    if gt_count == 0 or pred_count == 0:
        return {
            "gt_endpoint_count": gt_count,
            "pred_endpoint_count": pred_count,
            "endpoint_recall": 0.0,
            "endpoint_precision": 0.0,
            "missed_endpoint_count": gt_count,
        }
    dist_to_pred = distance_transform_edt(~pred_end)
    dist_to_gt = distance_transform_edt(~gt_end)
    matched_gt = gt_end & (dist_to_pred <= radius)
    matched_pred = pred_end & (dist_to_gt <= radius)
    return {
        "gt_endpoint_count": gt_count,
        "pred_endpoint_count": pred_count,
        "endpoint_recall": safe_div(int(matched_gt.sum()), gt_count),
        "endpoint_precision": safe_div(int(matched_pred.sum()), pred_count),
        "missed_endpoint_count": int(gt_count - matched_gt.sum()),
    }


def skeleton_distance_f1(pred: np.ndarray, gt: np.ndarray, radius: int = 5) -> float:
    pred_skel, _ = skeleton_endpoints(pred)
    gt_skel, _ = skeleton_endpoints(gt)
    pred_count = int(pred_skel.sum())
    gt_count = int(gt_skel.sum())
    if pred_count == 0 or gt_count == 0:
        return 0.0
    dist_to_gt = distance_transform_edt(~gt_skel)
    dist_to_pred = distance_transform_edt(~pred_skel)
    pred_dist = dist_to_gt[pred_skel]
    gt_dist = dist_to_pred[gt_skel]
    precision = safe_div(int((pred_dist <= radius).sum()), pred_count)
    recall = safe_div(int((gt_dist <= radius).sum()), gt_count)
    return safe_div(2 * precision * recall, precision + recall)


def soft_connected_refinement(
    raw: np.ndarray,
    baseline: np.ndarray,
    vesselness: np.ndarray,
    params: Params,
    baseline_endpoints: np.ndarray | None = None,
) -> tuple[np.ndarray, int, int]:
    baseline = baseline.astype(bool)
    raw = raw.astype(bool)
    refined = baseline.copy()
    if baseline_endpoints is None:
        _, baseline_endpoints = skeleton_endpoints(baseline)
    anchor_zone = binary_dilation(baseline, iterations=params.anchor_radius)
    endpoint_zone = binary_dilation(baseline_endpoints, structure=disk(params.endpoint_radius))

    labeled, n_components = label(raw & ~baseline)
    objects = find_objects(labeled)
    removed_far = 0
    removed_weak = 0

    for component_id in range(1, n_components + 1):
        bbox = objects[component_id - 1]
        if bbox is None:
            continue
        component = labeled[bbox] == component_id
        area = int(component.sum())
        if area == 0:
            continue
        touches_anchor = bool((component & anchor_zone[bbox]).any())
        touches_endpoint_zone = bool((component & endpoint_zone[bbox]).any())
        comp_vals = vesselness[bbox][component]
        mean_v = float(comp_vals.mean()) if comp_vals.size else 0.0
        max_v = float(comp_vals.max()) if comp_vals.size else 0.0

        if not touches_anchor and not touches_endpoint_zone:
            removed_far += 1
            continue

        keep_by_size = area >= params.min_area and mean_v >= params.min_mean_vesselness
        keep_by_endpoint = touches_endpoint_zone and max_v >= params.min_max_vesselness
        keep_by_strength = max_v >= params.min_max_vesselness and touches_anchor

        if keep_by_size or keep_by_endpoint or keep_by_strength:
            refined[bbox] |= component
        else:
            removed_weak += 1

    refined = remove_small_objects(refined.astype(bool), min_size=12)
    return refined, removed_far, removed_weak


def evaluate_row(
    image: str,
    method: str,
    pred: np.ndarray,
    gt: np.ndarray,
    removed_far: int = 0,
    removed_weak: int = 0,
) -> dict[str, float | str]:
    term_gt = terminal_region(gt)
    row: dict[str, float | str] = {
        "image": image,
        "method": method,
        "removed_far_components": removed_far,
        "removed_weak_components": removed_weak,
        "gt_vessel_pixels": int(gt.sum()),
        "pred_vessel_pixels": int(pred.sum()),
    }
    row.update(mask_metrics(pred, gt))
    row.update({f"terminal_{k}": v for k, v in mask_metrics(pred & term_gt, term_gt).items()})
    row.update(endpoint_metrics(pred, gt))
    row["skeleton_distance_f1_radius_5"] = skeleton_distance_f1(pred, gt, radius=5)
    return row


def load_cases(hrf_root: Path, base_module, max_side: int) -> list[dict[str, object]]:
    cases = []
    pairs = pair_hrf(hrf_root)
    for idx, (image_path, mask_path) in enumerate(pairs, start=1):
        print(f"[{idx}/{len(pairs)}] {image_path.name}", flush=True)
        image = base_module.load_rgb(image_path, max_side)
        gt = base_module.load_mask(mask_path, image.shape[:2])
        baseline = base_module.segment_baseline(image)
        _, baseline_endpoints = skeleton_endpoints(baseline)
        vesselness, fov = base_module.vesselness_map(image)
        raw = base_module.segment_connected_recovery_from_maps(baseline, vesselness, fov)
        cases.append(
            {
                "name": image_path.name,
                "image": image,
                "gt": gt,
                "term_gt": terminal_region(gt),
                "baseline": baseline,
                "baseline_endpoints": baseline_endpoints,
                "vesselness": vesselness,
                "raw": raw,
            }
        )
    return cases


def param_grid() -> list[Params]:
    return [
        Params(10, 24, 18, 0.08, 0.24),
        Params(10, 32, 36, 0.08, 0.24),
        Params(10, 40, 36, 0.08, 0.24),
        Params(10, 32, 60, 0.10, 0.24),
        Params(10, 40, 60, 0.10, 0.24),
        Params(10, 40, 90, 0.12, 0.24),
    ]


def choose_best(cases: list[dict[str, object]]) -> tuple[Params, pd.DataFrame]:
    rows = []
    grid = param_grid()
    for idx, params in enumerate(grid, start=1):
        print(f"  candidate {idx}/{len(grid)}: {params}", flush=True)
        eval_rows = []
        for case in cases:
            pred, _, _ = soft_connected_refinement(
                case["raw"],
                case["baseline"],
                case["vesselness"],
                params,
                case["baseline_endpoints"],
            )
            metrics = mask_metrics(pred, case["gt"])
            metrics["terminal_sensitivity"] = mask_metrics(
                pred & case["term_gt"], case["term_gt"]
            )["sensitivity"]
            eval_rows.append(metrics)
        df = pd.DataFrame(eval_rows)
        row = df.mean(numeric_only=True).to_dict()
        row.update(params.__dict__)
        # Keep terminal sensitivity near the raw connected_recovery level while rewarding FP reduction.
        row["score"] = (
            1.45 * row["terminal_sensitivity"]
            + 0.95 * row["dice"]
            + 0.65 * row["precision"]
            - 0.18 * row["fp_over_gt_area"]
        )
        rows.append(row)
    grid_df = pd.DataFrame(rows)
    candidates = grid_df[grid_df["terminal_sensitivity"] >= 0.64].copy()
    if candidates.empty:
        candidates = grid_df.copy()
    best = candidates.sort_values(
        ["score", "terminal_sensitivity", "dice", "precision"],
        ascending=False,
    ).iloc[0]
    params = Params(
        int(best["anchor_radius"]),
        int(best["endpoint_radius"]),
        int(best["min_area"]),
        float(best["min_mean_vesselness"]),
        float(best["min_max_vesselness"]),
    )
    return params, grid_df


def make_summary_figure(summary: pd.DataFrame, out: Path) -> None:
    methods = ["baseline_raw", "connected_recovery_raw", "soft_structure_refined"]
    metrics = [
        ("dice", "Dice"),
        ("precision", "Precision"),
        ("sensitivity", "Sens."),
        ("terminal_sensitivity", "Terminal sens."),
        ("skeleton_distance_f1_radius_5", "Skel. F1 r=5"),
    ]
    rows = summary.set_index("method").loc[methods].reset_index()
    w, h = 1500, 760
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    draw.text((36, 26), "Connected recovery optimization for terminal small vessels", fill=(20, 20, 20))
    x0, y0, cw, ch = 90, 110, 1320, 450
    colors = [(64, 132, 214), (232, 144, 58), (72, 171, 112), (151, 99, 205), (88, 168, 190)]
    for t in np.linspace(0, 1, 6):
        y = y0 + ch - int(t * ch)
        draw.line([x0, y, x0 + cw, y], fill=(225, 225, 225))
        draw.text((38, y - 8), f"{t:.1f}", fill=(70, 70, 70))
    group_w = cw / len(rows)
    bar_w = 36
    for i, (_, row) in enumerate(rows.iterrows()):
        gx = x0 + i * group_w + 55
        for j, (col, _) in enumerate(metrics):
            val = float(row[col])
            bh = int(max(0, min(1, val)) * ch)
            bx = gx + j * (bar_w + 14)
            by = y0 + ch - bh
            draw.rectangle([bx, by, bx + bar_w, y0 + ch], fill=colors[j])
            draw.text((bx - 3, by - 18), f"{val:.2f}", fill=(30, 30, 30))
        draw.multiline_text((x0 + i * group_w + 20, y0 + ch + 22), str(row["method"]).replace("_", "\n"), fill=(30, 30, 30), spacing=2)
    lx, ly = 90, 670
    for j, (_, label_text) in enumerate(metrics):
        draw.rectangle([lx + j * 220, ly, lx + 15 + j * 220, ly + 15], fill=colors[j])
        draw.text((lx + 22 + j * 220, ly - 1), label_text, fill=(30, 30, 30))
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)


def mask_panel(mask: np.ndarray, size: int = 300) -> Image.Image:
    return Image.fromarray(mask.astype(np.uint8) * 255).resize((size, size), Image.Resampling.NEAREST).convert("RGB")


def overlay_panel(image: np.ndarray, mask: np.ndarray, gt: np.ndarray, size: int = 300) -> Image.Image:
    base = Image.fromarray(image).resize((size, size))
    mask_r = np.array(Image.fromarray(mask.astype(np.uint8) * 255).resize((size, size), Image.Resampling.NEAREST)) > 0
    gt_r = np.array(Image.fromarray(gt.astype(np.uint8) * 255).resize((size, size), Image.Resampling.NEAREST)) > 0
    arr = np.array(base).astype(np.float32)
    arr[mask_r] = arr[mask_r] * 0.45 + np.array([255, 255, 255]) * 0.55
    arr[gt_r] = arr[gt_r] * 0.65 + np.array([70, 220, 90]) * 0.35
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def make_example_figure(cases: list[dict[str, object]], per_image: pd.DataFrame, out: Path) -> None:
    chosen = (
        per_image[per_image["method"] == "soft_structure_refined"]
        .assign(balance=lambda d: d["terminal_sensitivity"] - 0.25 * d["fp_over_gt_area"])
        .sort_values("balance", ascending=False)
        .head(3)["image"]
        .tolist()
    )
    case_map = {case["name"]: case for case in cases}
    labels = ["Original", "GT", "Raw connected", "Soft refined", "Removed FP candidates", "FN after refined"]
    size = 260
    rows = []
    for name in chosen:
        case = case_map[name]
        pred, _, _ = soft_connected_refinement(
            case["raw"],
            case["baseline"],
            case["vesselness"],
            make_example_figure.params,
            case["baseline_endpoints"],
        )
        removed = case["raw"] & ~pred
        fn = case["gt"] & ~pred
        panels = [
            Image.fromarray(case["image"]).resize((size, size)),
            mask_panel(case["gt"], size),
            overlay_panel(case["image"], case["raw"], case["gt"], size),
            overlay_panel(case["image"], pred, case["gt"], size),
            mask_panel(removed, size),
            mask_panel(fn, size),
        ]
        row = Image.new("RGB", (size * len(panels), size + 62), "white")
        draw = ImageDraw.Draw(row)
        draw.text((0, 4), name, fill=(20, 20, 20))
        for i, (panel, label_text) in enumerate(zip(panels, labels)):
            draw.text((i * size + 6, 30), label_text, fill=(20, 20, 20))
            row.paste(panel, (i * size, 62))
        rows.append(row)
    canvas = Image.new("RGB", (size * len(labels), 40 + sum(r.height for r in rows) + 12 * (len(rows) - 1)), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((0, 10), "Soft structural refinement examples", fill=(20, 20, 20))
    y = 40
    for row in rows:
        canvas.paste(row, (0, y))
        y += row.height + 12
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    args = parse_args()
    hrf_root = Path(args.hrf_root)
    output_dir = Path(args.output_dir)
    figure_dir = Path(args.figure_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    base_module = load_base_module(Path(args.base_script))

    print("Loading HRF cases...")
    cases = load_cases(hrf_root, base_module, args.max_image_side)
    print("Searching parameters...")
    best_params, grid_df = choose_best(cases)
    print(f"Best params: {best_params}")

    rows = []
    for case in cases:
        rows.append(evaluate_row(case["name"], "baseline_raw", case["baseline"], case["gt"]))
        rows.append(evaluate_row(case["name"], "connected_recovery_raw", case["raw"], case["gt"]))
        refined, removed_far, removed_weak = soft_connected_refinement(
            case["raw"],
            case["baseline"],
            case["vesselness"],
            best_params,
            case["baseline_endpoints"],
        )
        rows.append(
            evaluate_row(
                case["name"],
                "soft_structure_refined",
                refined,
                case["gt"],
                removed_far,
                removed_weak,
            )
        )

    per_image = pd.DataFrame(rows)
    numeric = per_image.select_dtypes(include=[np.number]).columns
    summary = per_image.groupby("method")[numeric].mean().reset_index()
    grid_df.to_csv(output_dir / "soft_refinement_grid_search.csv", index=False)
    per_image.to_csv(output_dir / "soft_refinement_metrics.csv", index=False)
    summary.to_csv(output_dir / "soft_refinement_summary.csv", index=False)

    make_summary_figure(summary, figure_dir / "soft_refinement_summary.png")
    make_example_figure.params = best_params
    make_example_figure(cases, per_image, figure_dir / "soft_refinement_examples.png")

    cols = [
        "method",
        "dice",
        "sensitivity",
        "precision",
        "terminal_sensitivity",
        "fp_over_gt_area",
        "fn_over_gt_area",
        "endpoint_recall",
        "endpoint_precision",
        "skeleton_distance_f1_radius_5",
    ]
    print(summary[cols].round(3).to_string(index=False))
    print(f"Saved results to {output_dir}")
    print(f"Saved figures to {figure_dir}")


if __name__ == "__main__":
    main()
