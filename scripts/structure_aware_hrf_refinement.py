"""
Structure-aware HRF vessel refinement.

This script starts from the existing HRF segmentation methods and adds the
structural constraints discussed after the group meeting:

1. remove very short vessel components,
2. suppress candidates that are disconnected from the main vessel tree,
3. compare masks with skeleton-distance metrics in addition to pixel overlap.

The raw HRF images are expected locally and are not written to the repository.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, distance_transform_edt, find_objects, label
from skimage.morphology import closing, disk, remove_small_objects, skeletonize


METHODS = ["baseline", "multiscale_line", "clean_recovery", "connected_recovery"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply structure-aware filtering to HRF vessel masks.")
    parser.add_argument("--hrf-root", required=True, help="Folder containing HRF images/ and masks/.")
    parser.add_argument("--base-script", required=True, help="Path to compare_hrf_small_vessel_methods.py.")
    parser.add_argument("--output-dir", required=True, help="Directory for CSV outputs.")
    parser.add_argument("--figure-dir", required=True, help="Directory for figures.")
    parser.add_argument("--max-image-side", type=int, default=1600)
    parser.add_argument("--min-component-area", type=int, default=80)
    parser.add_argument("--min-skeleton-length", type=int, default=28)
    parser.add_argument("--connection-radius", type=int, default=8)
    parser.add_argument("--short-spur-iterations", type=int, default=6)
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


def prune_skeleton_spurs(skel: np.ndarray, iterations: int) -> np.ndarray:
    pruned = skel.copy()
    for _ in range(iterations):
        endpoints = pruned & (neighbor_count(pruned) == 1)
        if not endpoints.any():
            break
        pruned &= ~endpoints
    return pruned


def largest_component(mask: np.ndarray) -> np.ndarray:
    labeled, n_components = label(mask > 0)
    if n_components == 0:
        return np.zeros_like(mask, dtype=bool)
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    return labeled == int(np.argmax(counts))


def component_stats(mask: np.ndarray) -> tuple[np.ndarray, int, np.ndarray]:
    labeled, n_components = label(mask > 0)
    counts = np.bincount(labeled.ravel()) if n_components else np.array([0])
    return labeled, n_components, counts


def structure_filter_with_counts(mask: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, int, int]:
    cleaned = remove_small_objects(mask.astype(bool), min_size=args.min_component_area)
    cleaned = closing(cleaned, footprint=disk(1))
    main = largest_component(cleaned)
    if not main.any():
        return cleaned, 0, 0

    main_neighborhood = binary_dilation(main, iterations=args.connection_radius)
    labeled, n_components, counts = component_stats(cleaned)
    objects = find_objects(labeled)
    refined = main.copy()
    removed_short = 0
    removed_disconnected = 0

    for component_id in range(1, n_components + 1):
        area = int(counts[component_id])
        if area < args.min_component_area:
            removed_short += 1
            continue
        component = labeled == component_id
        if component.sum() == main.sum() and (component & main).any():
            continue
        touches_main = bool((component & main_neighborhood).any())

        if not touches_main:
            removed_disconnected += 1
            continue

        bbox = objects[component_id - 1]
        if bbox is None:
            removed_short += 1
            continue
        component_crop = labeled[bbox] == component_id
        skel_len = int(skeletonize(component_crop).sum())
        if skel_len < args.min_skeleton_length:
            removed_short += 1
            continue
        refined |= component

    refined = remove_small_objects(refined.astype(bool), min_size=args.min_component_area)
    return refined, removed_short, removed_disconnected


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


def terminal_region(gt: np.ndarray, endpoint_radius: int = 8) -> np.ndarray:
    _, endpoints = skeleton_endpoints(gt)
    return gt & binary_dilation(endpoints, structure=disk(endpoint_radius))


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


def skeleton_distance_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    pred_skel, _ = skeleton_endpoints(pred)
    gt_skel, _ = skeleton_endpoints(gt)
    pred_count = int(pred_skel.sum())
    gt_count = int(gt_skel.sum())
    if pred_count == 0 or gt_count == 0:
        return {
            "pred_to_gt_skeleton_distance_mean": np.nan,
            "pred_to_gt_skeleton_distance_p95": np.nan,
            "gt_to_pred_skeleton_distance_mean": np.nan,
            "gt_to_pred_skeleton_distance_p95": np.nan,
            "skeleton_distance_f1_radius_3": 0.0,
            "skeleton_distance_f1_radius_5": 0.0,
        }

    dist_to_gt = distance_transform_edt(~gt_skel)
    dist_to_pred = distance_transform_edt(~pred_skel)
    pred_dist = dist_to_gt[pred_skel]
    gt_dist = dist_to_pred[gt_skel]

    out = {
        "pred_to_gt_skeleton_distance_mean": float(np.mean(pred_dist)),
        "pred_to_gt_skeleton_distance_p95": float(np.percentile(pred_dist, 95)),
        "gt_to_pred_skeleton_distance_mean": float(np.mean(gt_dist)),
        "gt_to_pred_skeleton_distance_p95": float(np.percentile(gt_dist, 95)),
    }
    for radius in (3, 5):
        precision = safe_div(int((pred_dist <= radius).sum()), pred_count)
        recall = safe_div(int((gt_dist <= radius).sum()), gt_count)
        out[f"skeleton_precision_radius_{radius}"] = precision
        out[f"skeleton_recall_radius_{radius}"] = recall
        out[f"skeleton_distance_f1_radius_{radius}"] = safe_div(2 * precision * recall, precision + recall)
    return out


def skeleton_overlap_metrics(pred: np.ndarray, gt: np.ndarray, spur_iterations: int) -> dict[str, float]:
    pred_skel, _ = skeleton_endpoints(pred)
    gt_skel, _ = skeleton_endpoints(gt)
    pred_pruned = prune_skeleton_spurs(pred_skel, spur_iterations)
    gt_pruned = prune_skeleton_spurs(gt_skel, spur_iterations)
    pred_components, pred_count = label(pred_skel)
    gt_components, gt_count = label(gt_skel)
    pred_pruned_components, pred_pruned_count = label(pred_pruned)
    gt_pruned_components, gt_pruned_count = label(gt_pruned)
    return {
        "pred_skeleton_pixels": int(pred_skel.sum()),
        "gt_skeleton_pixels": int(gt_skel.sum()),
        "skeleton_overlap": safe_div(int((pred_skel & gt_skel).sum()), int(gt_skel.sum())),
        "pred_skeleton_components": int(pred_count),
        "gt_skeleton_components": int(gt_count),
        "component_ratio": safe_div(float(pred_count), float(gt_count)),
        "pruned_pred_skeleton_pixels": int(pred_pruned.sum()),
        "pruned_gt_skeleton_pixels": int(gt_pruned.sum()),
        "pruned_skeleton_overlap": safe_div(int((pred_pruned & gt_pruned).sum()), int(gt_pruned.sum())),
        "pruned_pred_skeleton_components": int(pred_pruned_count),
        "pruned_gt_skeleton_components": int(gt_pruned_count),
        "pruned_component_ratio": safe_div(float(pred_pruned_count), float(gt_pruned_count)),
    }


def evaluate_prediction(
    image_name: str,
    image_shape: tuple[int, int],
    gt: np.ndarray,
    pred: np.ndarray,
    method: str,
    refinement: str,
    removed_short: int,
    removed_disconnected: int,
    args: argparse.Namespace,
) -> dict[str, float | str]:
    term_gt = terminal_region(gt)
    term_pred = pred & term_gt
    row: dict[str, float | str] = {
        "dataset": "HRF",
        "image": image_name,
        "method": method,
        "refinement": refinement,
        "height": image_shape[0],
        "width": image_shape[1],
        "gt_vessel_pixels": int(gt.sum()),
        "pred_vessel_pixels": int(pred.sum()),
        "removed_short_components": removed_short,
        "removed_disconnected_components": removed_disconnected,
    }
    row.update(mask_metrics(pred, gt))
    row.update({f"terminal_{k}": v for k, v in mask_metrics(term_pred, term_gt).items()})
    row.update(endpoint_metrics(pred, gt))
    row.update(skeleton_overlap_metrics(pred, gt, args.short_spur_iterations))
    row.update(skeleton_distance_metrics(pred, gt))
    return row


def make_summary_chart(summary: pd.DataFrame, out: Path) -> None:
    metrics = [
        ("dice", "Dice"),
        ("precision", "Precision"),
        ("terminal_sensitivity", "Terminal sens."),
        ("skeleton_distance_f1_radius_5", "Skeleton F1 r=5"),
    ]
    rows = summary[summary["refinement"].isin(["raw", "structure_filtered"])].copy()
    rows["label"] = rows["method"] + "\n" + rows["refinement"].map({"raw": "raw", "structure_filtered": "filtered"})
    rows = rows.sort_values(["method", "refinement"])

    w, h = 1600, 780
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    draw.text((36, 24), "Structure-aware filtering: HRF method comparison", fill=(20, 20, 20))
    x0, y0, cw, ch = 80, 100, 1450, 450
    colors = [(65, 130, 210), (230, 140, 55), (80, 170, 110), (150, 95, 200)]
    for t in np.linspace(0, 1, 6):
        y = y0 + ch - int(t * ch)
        draw.line([x0, y, x0 + cw, y], fill=(225, 225, 225))
        draw.text((30, y - 8), f"{t:.1f}", fill=(70, 70, 70))

    group_w = cw / len(rows)
    bar_w = 30
    for i, (_, row) in enumerate(rows.iterrows()):
        gx = x0 + i * group_w + group_w * 0.08
        for j, (col, _) in enumerate(metrics):
            val = float(row[col])
            bh = int(max(0, min(1, val)) * ch)
            bx = gx + j * (bar_w + 8)
            by = y0 + ch - bh
            draw.rectangle([bx, by, bx + bar_w, y0 + ch], fill=colors[j])
            draw.text((bx - 3, by - 16), f"{val:.2f}", fill=(30, 30, 30))
        draw.multiline_text((x0 + i * group_w + 4, y0 + ch + 18), row["label"], fill=(30, 30, 30), spacing=2)

    lx, ly = 90, 690
    for j, (_, label_text) in enumerate(metrics):
        draw.rectangle([lx, ly + j * 22, lx + 15, ly + 15 + j * 22], fill=colors[j])
        draw.text((lx + 22, ly - 1 + j * 22), label_text, fill=(30, 30, 30))
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)


def overlay_comparison(image: np.ndarray, raw: np.ndarray, filtered: np.ndarray, gt: np.ndarray, title: str, size: int = 330) -> Image.Image:
    image_pil = Image.fromarray(image).resize((size, size))
    gt_pil = Image.fromarray(gt.astype(np.uint8) * 255).resize((size, size), Image.Resampling.NEAREST)
    raw_pil = Image.fromarray(raw.astype(np.uint8) * 255).resize((size, size), Image.Resampling.NEAREST)
    filtered_pil = Image.fromarray(filtered.astype(np.uint8) * 255).resize((size, size), Image.Resampling.NEAREST)
    gt_arr = np.array(gt_pil) > 0
    raw_arr = np.array(raw_pil) > 0
    filtered_arr = np.array(filtered_pil) > 0

    panels = []
    for mask, label_text in [
        (gt_arr, "GT"),
        (raw_arr, "Raw"),
        (filtered_arr, "Filtered"),
        (raw_arr & ~filtered_arr, "Removed by filter"),
        (filtered_arr & ~gt_arr, "Remaining FP"),
    ]:
        if label_text == "GT":
            arr = np.zeros((size, size, 3), dtype=np.uint8)
            arr[mask] = (255, 255, 255)
        elif label_text in {"Raw", "Filtered"}:
            arr = np.array(image_pil).astype(np.float32)
            arr[mask] = arr[mask] * 0.45 + np.array([255, 255, 255]) * 0.55
            arr[gt_arr] = arr[gt_arr] * 0.65 + np.array([70, 220, 90]) * 0.35
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        else:
            arr = np.zeros((size, size, 3), dtype=np.uint8)
            arr[mask] = (255, 255, 255)
        panel = Image.fromarray(arr)
        canvas = Image.new("RGB", (size, size + 34), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 9), label_text, fill=(20, 20, 20))
        canvas.paste(panel, (0, 34))
        panels.append(canvas)

    row = Image.new("RGB", (size * len(panels), size + 68), "white")
    draw = ImageDraw.Draw(row)
    draw.text((0, 4), title, fill=(20, 20, 20))
    x = 0
    for panel in panels:
        row.paste(panel, (x, 34))
        x += size
    return row


def make_example_figure(records: pd.DataFrame, hrf_root: Path, base_module, args: argparse.Namespace, out: Path) -> None:
    base = records[(records["refinement"] == "raw") & (records["method"] == "connected_recovery")].copy()
    base["fp_load"] = base["fp_over_gt_area"] + base["removed_short_components"] * 0.0
    chosen = base.sort_values("fp_over_gt_area", ascending=False)["image"].head(3).tolist()
    rows = []
    for image_name in chosen:
        image_path = hrf_root / "images" / image_name
        mask_path = next((hrf_root / "masks").glob(Path(image_name).stem + ".*"))
        image = base_module.load_rgb(image_path, args.max_image_side)
        gt = base_module.load_mask(mask_path, image.shape[:2])
        raw = base_module.segment_all_methods(image)["connected_recovery"]
        filtered, _, _ = structure_filter_with_counts(raw, args)
        rows.append(overlay_comparison(image, raw, filtered, gt, f"{image_name} | connected_recovery raw vs structure-filtered"))

    w = max(r.width for r in rows)
    h = 48 + sum(r.height for r in rows) + 12 * (len(rows) - 1)
    canvas = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((0, 10), "Structure-aware filtering examples", fill=(20, 20, 20))
    y = 48
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

    records = []
    pairs = pair_hrf(hrf_root)
    print(f"HRF paired images: {len(pairs)}")
    for idx, (image_path, mask_path) in enumerate(pairs, start=1):
        print(f"[{idx}/{len(pairs)}] {image_path.name}", flush=True)
        image = base_module.load_rgb(image_path, args.max_image_side)
        gt = base_module.load_mask(mask_path, image.shape[:2])
        raw_predictions = base_module.segment_all_methods(image)
        for method in METHODS:
            raw = raw_predictions[method]
            records.append(
                evaluate_prediction(image_path.name, image.shape[:2], gt, raw, method, "raw", 0, 0, args)
            )
            filtered, removed_short, removed_disconnected = structure_filter_with_counts(raw, args)
            records.append(
                evaluate_prediction(
                    image_path.name,
                    image.shape[:2],
                    gt,
                    filtered,
                    method,
                    "structure_filtered",
                    removed_short,
                    removed_disconnected,
                    args,
                )
            )

    df = pd.DataFrame(records)
    metrics_path = output_dir / "structure_aware_metrics.csv"
    summary_path = output_dir / "structure_aware_summary.csv"
    df.to_csv(metrics_path, index=False)
    numeric = df.select_dtypes(include=[np.number]).columns
    summary = df.groupby(["method", "refinement"])[numeric].mean().reset_index()
    summary.to_csv(summary_path, index=False)
    make_summary_chart(summary, figure_dir / "structure_aware_method_comparison.png")
    make_example_figure(df, hrf_root, base_module, args, figure_dir / "structure_aware_filter_examples.png")

    cols = [
        "method",
        "refinement",
        "dice",
        "sensitivity",
        "precision",
        "terminal_sensitivity",
        "fp_over_gt_area",
        "fn_over_gt_area",
        "skeleton_distance_f1_radius_5",
        "pred_to_gt_skeleton_distance_mean",
        "removed_short_components",
        "removed_disconnected_components",
    ]
    print(summary[cols].round(3).to_string(index=False))
    print(f"Saved {metrics_path}")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
