"""
Compare small-vessel segmentation methods on the HRF ground-truth dataset.

This script keeps the project focus on retinal vascular modeling: segmentation
is evaluated because vessel masks, terminal branches, and skeleton connectivity
are required image-derived constraints for later vascular tree modeling.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, distance_transform_edt, label
from skimage.morphology import remove_small_objects, skeletonize


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare HRF small-vessel segmentation methods against ground truth."
    )
    parser.add_argument(
        "--hrf-root",
        default="data/test_data/HRF",
        help="HRF folder containing images/ and masks/.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/hrf_small_vessel_methods",
        help="Directory for result CSV files.",
    )
    parser.add_argument(
        "--figure-dir",
        default="figures",
        help="Directory for comparison figures.",
    )
    parser.add_argument("--endpoint-radius", type=int, default=8)
    parser.add_argument("--endpoint-match-radius", type=int, default=10)
    parser.add_argument(
        "--max-image-side",
        type=int,
        default=0,
        help="Optional resize limit for faster exploratory runs. Default keeps full resolution.",
    )
    return parser.parse_args()


def image_files(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def pair_hrf(root: Path) -> list[tuple[Path, Path]]:
    masks = {p.stem: p for p in image_files(root / "masks")}
    return [(p, masks[p.stem]) for p in image_files(root / "images") if p.stem in masks]


def load_rgb(path: Path, max_side: int = 0) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if max_side and max(image.size) > max_side:
        scale = max_side / max(image.size)
        new_size = (int(image.size[0] * scale), int(image.size[1] * scale))
        image = image.resize(new_size, Image.Resampling.BILINEAR)
    return np.array(image)


def load_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    mask = Image.open(path).convert("L")
    if mask.size != (shape[1], shape[0]):
        mask = mask.resize((shape[1], shape[0]), Image.Resampling.NEAREST)
    return np.array(mask) > 0


def preprocess_green(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    green = image[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(green)
    background = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=12, sigmaY=12)
    dark_vessels = cv2.subtract(background, enhanced)
    dark_vessels = cv2.normalize(dark_vessels, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    fov = green > max(8, int(np.percentile(green, 5)))
    return green, dark_vessels, fov


def segment_baseline(image: np.ndarray) -> np.ndarray:
    _, dark_vessels, fov = preprocess_green(image)
    blur = cv2.GaussianBlur(dark_vessels, (3, 3), 0)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, -2
    )
    pred = ((otsu > 0) | (adaptive > 0)) & binary_dilation(fov, iterations=2)
    return remove_small_objects(pred.astype(bool), min_size=24)


def _line_kernel(length: int, angle_deg: float) -> np.ndarray:
    kernel = np.zeros((length, length), dtype=np.float32)
    center = length // 2
    angle = np.deg2rad(angle_deg)
    dx = np.cos(angle)
    dy = np.sin(angle)
    half = length // 2
    for t in range(-half, half + 1):
        x = int(round(center + t * dx))
        y = int(round(center + t * dy))
        if 0 <= x < length and 0 <= y < length:
            kernel[y, x] = 1.0
    kernel /= max(kernel.sum(), 1.0)
    kernel -= kernel.mean()
    return kernel


def vesselness_map(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _, dark_vessels, fov = preprocess_green(image)
    norm = cv2.GaussianBlur(dark_vessels, (3, 3), 0).astype(np.float32) / 255.0
    responses = []
    for length in (9, 15, 21):
        for angle in range(0, 180, 30):
            response = cv2.filter2D(norm, cv2.CV_32F, _line_kernel(length, angle))
            responses.append(response)
    vesselness = np.maximum.reduce(responses)
    vesselness[vesselness < 0] = 0
    if float(vesselness.max()) > 0:
        vesselness = vesselness / float(vesselness.max())
    return vesselness, fov


def segment_multiscale_line_from_maps(
    baseline: np.ndarray, vesselness: np.ndarray, fov: np.ndarray
) -> np.ndarray:
    vals = vesselness[fov]
    threshold = np.percentile(vals, 92.0) if vals.size else 1.0
    thin_candidates = (vesselness >= threshold) & fov
    pred = baseline | thin_candidates
    pred = remove_small_objects(pred.astype(bool), min_size=18)
    return pred


def segment_connected_recovery_from_maps(
    baseline: np.ndarray, vesselness: np.ndarray, fov: np.ndarray
) -> np.ndarray:
    vals = vesselness[fov]
    if vals.size == 0:
        return baseline
    high = vesselness >= np.percentile(vals, 94.0)
    low = vesselness >= np.percentile(vals, 88.0)
    near_existing = binary_dilation(baseline, iterations=3)
    connected_candidates = low & near_existing & fov
    pred = baseline | high | connected_candidates
    pred = remove_small_objects(pred.astype(bool), min_size=18)
    return pred


def segment_clean_recovery_from_maps(
    baseline: np.ndarray, vesselness: np.ndarray, fov: np.ndarray
) -> np.ndarray:
    vals = vesselness[fov]
    if vals.size == 0:
        return baseline

    seed = (vesselness >= np.percentile(vals, 95.0)) & fov
    candidates = (vesselness >= np.percentile(vals, 90.0)) & binary_dilation(
        baseline, iterations=2
    ) & fov
    labeled, n_components = label(candidates)
    anchor = binary_dilation(baseline, iterations=1)
    clean = baseline.copy()

    for component_id in range(1, n_components + 1):
        component = labeled == component_id
        size = int(component.sum())
        if size < 12:
            continue
        if not (component & anchor).any():
            continue
        if not (component & seed).any():
            continue
        clean |= component

    return remove_small_objects(clean.astype(bool), min_size=24)


METHOD_NAMES = ["baseline", "multiscale_line", "connected_recovery", "clean_recovery"]


def segment_all_methods(image: np.ndarray) -> dict[str, np.ndarray]:
    baseline = segment_baseline(image)
    vesselness, fov = vesselness_map(image)
    return {
        "baseline": baseline,
        "multiscale_line": segment_multiscale_line_from_maps(baseline, vesselness, fov),
        "connected_recovery": segment_connected_recovery_from_maps(baseline, vesselness, fov),
        "clean_recovery": segment_clean_recovery_from_maps(baseline, vesselness, fov),
    }


def skeleton_endpoints(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    skel = skeletonize(mask > 0)
    padded = np.pad(skel.astype(np.uint8), 1)
    count = np.zeros_like(skel, dtype=np.uint8)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx or dy:
                count += padded[1 + dy : 1 + dy + skel.shape[0], 1 + dx : 1 + dx + skel.shape[1]]
    return skel, skel & (count == 1)


def prune_skeleton_spurs(skel: np.ndarray, iterations: int = 8) -> np.ndarray:
    pruned = skel.copy()
    for _ in range(iterations):
        _, endpoints = skeleton_endpoints(pruned)
        if not endpoints.any():
            break
        pruned = pruned & ~endpoints
    return pruned


def disk(radius: int) -> np.ndarray:
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (x * x + y * y) <= radius * radius


def terminal_region(mask: np.ndarray, endpoints: np.ndarray, radius: int) -> np.ndarray:
    if endpoints.sum() == 0:
        return np.zeros_like(mask, dtype=bool)
    return mask & binary_dilation(endpoints, structure=disk(radius))


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


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
    }


def endpoint_metrics(pred_end: np.ndarray, gt_end: np.ndarray, radius: int) -> dict[str, float]:
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


def component_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    pred_skel, _ = skeleton_endpoints(pred)
    gt_skel, _ = skeleton_endpoints(gt)
    pred_pruned = prune_skeleton_spurs(pred_skel)
    gt_pruned = prune_skeleton_spurs(gt_skel)
    pred_components, pred_count = label(pred_skel)
    gt_components, gt_count = label(gt_skel)
    pred_pruned_components, pred_pruned_count = label(pred_pruned)
    gt_pruned_components, gt_pruned_count = label(gt_pruned)
    overlap = pred_skel & gt_skel
    pruned_overlap = pred_pruned & gt_pruned
    return {
        "pred_skeleton_pixels": int(pred_skel.sum()),
        "gt_skeleton_pixels": int(gt_skel.sum()),
        "skeleton_overlap": safe_div(int(overlap.sum()), int(gt_skel.sum())),
        "pred_skeleton_components": int(pred_count),
        "gt_skeleton_components": int(gt_count),
        "component_ratio": safe_div(float(pred_count), float(gt_count)),
        "pruned_pred_skeleton_pixels": int(pred_pruned.sum()),
        "pruned_gt_skeleton_pixels": int(gt_pruned.sum()),
        "pruned_skeleton_overlap": safe_div(
            int(pruned_overlap.sum()), int(gt_pruned.sum())
        ),
        "pruned_pred_skeleton_components": int(pred_pruned_count),
        "pruned_gt_skeleton_components": int(gt_pruned_count),
        "pruned_component_ratio": safe_div(
            float(pred_pruned_count), float(gt_pruned_count)
        ),
    }


def evaluate_prediction(
    image_name: str,
    image_shape: tuple[int, int],
    gt: np.ndarray,
    pred: np.ndarray,
    method: str,
    args: argparse.Namespace,
) -> dict[str, float | str]:
    gt_skel, gt_end = skeleton_endpoints(gt)
    pred_skel, pred_end = skeleton_endpoints(pred)
    term_gt = terminal_region(gt, gt_end, args.endpoint_radius)
    term_pred = pred & binary_dilation(gt_end, structure=disk(args.endpoint_radius))

    row: dict[str, float | str] = {
        "dataset": "HRF",
        "image": image_name,
        "method": method,
        "height": image_shape[0],
        "width": image_shape[1],
        "gt_vessel_pixels": int(gt.sum()),
        "pred_vessel_pixels": int(pred.sum()),
    }
    row.update(mask_metrics(pred, gt))
    row.update({f"terminal_{k}": v for k, v in mask_metrics(term_pred, term_gt).items()})
    row.update(endpoint_metrics(pred_end, gt_end, args.endpoint_match_radius))
    row.update(component_metrics(pred, gt))
    return row


def make_method_chart(summary: pd.DataFrame, out: Path) -> None:
    metrics = [
        ("dice", "Dice"),
        ("terminal_sensitivity", "Terminal sens."),
        ("endpoint_recall", "Endpoint recall"),
        ("pruned_skeleton_overlap", "Pruned skel. overlap"),
    ]
    methods = summary["method"].tolist()
    colors = [(80, 130, 210), (235, 150, 65), (90, 170, 110), (160, 105, 200)]
    w, h = 1200, 720
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    draw.text((36, 24), "HRF small-vessel segmentation method comparison", fill=(20, 20, 20))
    x0, y0, cw, ch = 90, 90, 1040, 430
    for t in np.linspace(0, 1, 6):
        y = y0 + ch - int(t * ch)
        draw.line([x0, y, x0 + cw, y], fill=(225, 225, 225))
        draw.text((44, y - 8), f"{t:.1f}", fill=(80, 80, 80))
    group_w = cw / len(methods)
    bar_w = 48
    for i, method in enumerate(methods):
        gx = x0 + i * group_w + group_w * 0.18
        for j, (col, _) in enumerate(metrics):
            val = float(summary.iloc[i][col])
            bh = int(val * ch)
            bx = gx + j * (bar_w + 12)
            by = y0 + ch - bh
            draw.rectangle([bx, by, bx + bar_w, y0 + ch], fill=colors[j])
            draw.text((bx - 2, by - 18), f"{val:.2f}", fill=(40, 40, 40))
        draw.text((x0 + i * group_w + 22, y0 + ch + 20), method, fill=(30, 30, 30))
    lx, ly = 90, 585
    for j, (_, label_text) in enumerate(metrics):
        draw.rectangle([lx, ly + j * 24, lx + 16, ly + 16 + j * 24], fill=colors[j])
        draw.text((lx + 24, ly - 1 + j * 24), label_text, fill=(30, 30, 30))
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)


def overlay_panel(image: np.ndarray, gt: np.ndarray, pred: np.ndarray, title: str, size: int = 360) -> Image.Image:
    base = Image.fromarray(image).resize((size, size))
    gt_r = Image.fromarray(gt.astype(np.uint8) * 255).resize((size, size), Image.Resampling.NEAREST)
    pred_r = Image.fromarray(pred.astype(np.uint8) * 255).resize((size, size), Image.Resampling.NEAREST)
    arr = np.array(base).astype(np.float32)
    gt_arr = np.array(gt_r) > 0
    pred_arr = np.array(pred_r) > 0
    missed = gt_arr & ~pred_arr
    false_pos = pred_arr & ~gt_arr
    arr[gt_arr] = arr[gt_arr] * 0.65 + np.array([60, 210, 80]) * 0.35
    arr[false_pos] = np.array([230, 80, 70])
    arr[missed] = np.array([255, 230, 0])
    panel = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(panel)
    draw.rectangle([0, 0, size, 36], fill=(255, 255, 255))
    draw.text((8, 9), title, fill=(20, 20, 20))
    return panel


def make_example_figure(df: pd.DataFrame, hrf_root: Path, args: argparse.Namespace, out: Path) -> None:
    baseline = df[df["method"] == "baseline"].sort_values("terminal_sensitivity").head(3)
    if baseline.empty:
        return
    panel_size = 330
    methods = METHOD_NAMES
    canvas = Image.new("RGB", (len(methods) * panel_size, len(baseline) * panel_size + 60), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 14), "HRF method examples: green=GT, yellow=missed GT, red=false positives", fill=(20, 20, 20))
    for row_i, (_, row) in enumerate(baseline.iterrows()):
        image_path = hrf_root / "images" / row["image"]
        mask_path = next((hrf_root / "masks").glob(Path(row["image"]).stem + ".*"))
        image = load_rgb(image_path, args.max_image_side)
        gt = load_mask(mask_path, image.shape[:2])
        predictions = segment_all_methods(image)
        for col_i, method in enumerate(methods):
            pred = predictions[method]
            panel = overlay_panel(image, gt, pred, f"{row['image']} - {method}", panel_size)
            canvas.paste(panel, (col_i * panel_size, 60 + row_i * panel_size))
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)


def main() -> None:
    args = parse_args()
    hrf_root = Path(args.hrf_root)
    output_dir = Path(args.output_dir)
    figure_dir = Path(args.figure_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    pairs = pair_hrf(hrf_root)
    print(f"HRF paired images: {len(pairs)}")
    records = []
    for image_path, mask_path in pairs:
        image = load_rgb(image_path, args.max_image_side)
        gt = load_mask(mask_path, image.shape[:2])
        predictions = segment_all_methods(image)
        for method, pred in predictions.items():
            records.append(
                evaluate_prediction(image_path.name, image.shape[:2], gt, pred, method, args)
            )

    df = pd.DataFrame(records)
    metrics_path = output_dir / "hrf_method_metrics.csv"
    summary_path = output_dir / "hrf_method_summary.csv"
    df.to_csv(metrics_path, index=False)
    numeric = df.select_dtypes(include=[np.number]).columns
    summary = df.groupby("method")[numeric].mean().reset_index()
    summary.to_csv(summary_path, index=False)

    make_method_chart(summary, figure_dir / "hrf_small_vessel_method_comparison.png")
    make_example_figure(df, hrf_root, args, figure_dir / "hrf_small_vessel_method_examples.png")

    cols = [
        "method", "dice", "sensitivity", "precision", "terminal_sensitivity",
        "endpoint_recall", "missed_endpoint_count", "skeleton_overlap",
        "pruned_skeleton_overlap", "pred_skeleton_components",
        "pruned_pred_skeleton_components",
    ]
    print(summary[cols].round(3).to_string(index=False))
    print(f"Saved {metrics_path}")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
