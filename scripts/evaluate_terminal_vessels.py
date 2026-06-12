"""
Evaluate fundus vessel segmentation with emphasis on terminal small vessels.

The script expects a dataset root with subdirectories in the form:

    DATA_ROOT/DATASET_NAME/images/*
    DATA_ROOT/DATASET_NAME/masks/*

It runs a lightweight fundus vessel baseline, compares it with the provided
vessel masks, and reports both whole-mask metrics and terminal-vessel metrics.
Terminal regions are defined from ground-truth skeleton endpoints and a local
neighborhood around those endpoints.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_dilation, distance_transform_edt
from skimage.morphology import remove_small_objects, skeletonize


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate small terminal vessel identification on fundus datasets."
    )
    parser.add_argument(
        "--data-root",
        default="data/test_data",
        help="Root directory containing dataset/images and dataset/masks folders.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/terminal_vessel_evaluation",
        help="Directory for metric CSV files.",
    )
    parser.add_argument(
        "--figure-path",
        default="figures/terminal_vessel_examples.png",
        help="Path for a visual summary of representative missed terminal vessels.",
    )
    parser.add_argument(
        "--endpoint-radius",
        type=int,
        default=8,
        help="Pixel radius used for terminal-vessel endpoint neighborhoods.",
    )
    parser.add_argument(
        "--endpoint-match-radius",
        type=int,
        default=10,
        help="Pixel radius used to match predicted and reference endpoints.",
    )
    parser.add_argument(
        "--max-figure-cases",
        type=int,
        default=6,
        help="Number of low terminal-recall cases to show in the summary figure.",
    )
    parser.add_argument(
        "--max-image-side",
        type=int,
        default=0,
        help="Optionally resize images so the longest side is at most this value.",
    )
    return parser.parse_args()


def list_images(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def pair_image_and_mask(dataset_dir: Path) -> Iterable[tuple[Path, Path]]:
    image_dir = dataset_dir / "images"
    mask_dir = dataset_dir / "masks"
    if not image_dir.exists() or not mask_dir.exists():
        return []

    masks = {p.stem: p for p in list_images(mask_dir)}
    pairs = []
    for image_path in list_images(image_dir):
        mask_path = masks.get(image_path.stem)
        if mask_path is not None:
            pairs.append((image_path, mask_path))
    return pairs


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
    arr = np.array(mask)
    return arr > 0


def segment_fundus_vessels(image: np.ndarray) -> np.ndarray:
    """Lightweight baseline for dark retinal vessel segmentation."""
    green = image[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(green)

    background = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=12, sigmaY=12)
    dark_vessels = cv2.subtract(background, enhanced)
    dark_vessels = cv2.normalize(dark_vessels, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    blur = cv2.GaussianBlur(dark_vessels, (3, 3), 0)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        -2,
    )
    pred = (otsu > 0) | (adaptive > 0)

    # Restrict to the bright retinal field and remove isolated noise.
    fov = green > max(8, int(np.percentile(green, 5)))
    pred &= binary_dilation(fov, iterations=2)
    pred = remove_small_objects(pred.astype(bool), min_size=24)
    return pred.astype(bool)


def skeleton_endpoints(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    skel = skeletonize(mask > 0)
    padded = np.pad(skel.astype(np.uint8), 1)
    neighbor_count = np.zeros_like(skel, dtype=np.uint8)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            neighbor_count += padded[1 + dy : 1 + dy + skel.shape[0], 1 + dx : 1 + dx + skel.shape[1]]
    endpoints = skel & (neighbor_count == 1)
    return skel, endpoints


def disk_structure(radius: int) -> np.ndarray:
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (x * x + y * y) <= radius * radius


def terminal_region(mask: np.ndarray, endpoints: np.ndarray, radius: int) -> np.ndarray:
    if endpoints.sum() == 0:
        return np.zeros_like(mask, dtype=bool)
    dilated = binary_dilation(endpoints, structure=disk_structure(radius))
    return (mask > 0) & dilated


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


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


def endpoint_metrics(
    pred_endpoints: np.ndarray,
    gt_endpoints: np.ndarray,
    match_radius: int,
) -> dict[str, float]:
    gt_count = int(gt_endpoints.sum())
    pred_count = int(pred_endpoints.sum())
    if gt_count == 0 or pred_count == 0:
        return {
            "gt_endpoint_count": gt_count,
            "pred_endpoint_count": pred_count,
            "endpoint_recall": 0.0,
            "endpoint_precision": 0.0,
            "missed_endpoint_count": gt_count,
        }

    dist_to_pred = distance_transform_edt(~pred_endpoints)
    dist_to_gt = distance_transform_edt(~gt_endpoints)
    matched_gt = gt_endpoints & (dist_to_pred <= match_radius)
    matched_pred = pred_endpoints & (dist_to_gt <= match_radius)
    return {
        "gt_endpoint_count": gt_count,
        "pred_endpoint_count": pred_count,
        "endpoint_recall": safe_div(int(matched_gt.sum()), gt_count),
        "endpoint_precision": safe_div(int(matched_pred.sum()), pred_count),
        "missed_endpoint_count": int(gt_count - matched_gt.sum()),
    }


def evaluate_pair(
    dataset: str,
    image_path: Path,
    mask_path: Path,
    endpoint_radius: int,
    endpoint_match_radius: int,
    max_image_side: int = 0,
) -> dict[str, float | str]:
    image = load_rgb(image_path, max_image_side)
    gt = load_mask(mask_path, image.shape[:2])
    pred = segment_fundus_vessels(image)

    gt_skel, gt_endpoints = skeleton_endpoints(gt)
    pred_skel, pred_endpoints = skeleton_endpoints(pred)
    term_gt = terminal_region(gt, gt_endpoints, endpoint_radius)
    term_pred = pred & binary_dilation(gt_endpoints, structure=disk_structure(endpoint_radius))

    row: dict[str, float | str] = {
        "dataset": dataset,
        "image": image_path.name,
        "height": image.shape[0],
        "width": image.shape[1],
        "gt_vessel_pixels": int(gt.sum()),
        "pred_vessel_pixels": int(pred.sum()),
    }
    row.update(mask_metrics(pred, gt))
    terminal_scores = mask_metrics(term_pred, term_gt)
    row.update({f"terminal_{k}": v for k, v in terminal_scores.items()})
    row.update(endpoint_metrics(pred_endpoints, gt_endpoints, endpoint_match_radius))
    return row


def draw_case_panel(
    row: pd.Series,
    data_root: Path,
    endpoint_radius: int,
    max_image_side: int,
    size: int = 330,
) -> Image.Image:
    image_path = data_root / row["dataset"] / "images" / row["image"]
    mask_path = next((data_root / row["dataset"] / "masks").glob(Path(row["image"]).stem + ".*"))
    image = load_rgb(image_path, max_image_side)
    gt = load_mask(mask_path, image.shape[:2])
    pred = segment_fundus_vessels(image)
    _, gt_endpoints = skeleton_endpoints(gt)
    term_gt = terminal_region(gt, gt_endpoints, endpoint_radius)
    missed = term_gt & ~pred

    base = Image.fromarray(image).resize((size, size))
    overlay = np.array(base).astype(np.float32)
    gt_r = Image.fromarray(gt.astype(np.uint8) * 255).resize((size, size), Image.Resampling.NEAREST)
    pred_r = Image.fromarray(pred.astype(np.uint8) * 255).resize((size, size), Image.Resampling.NEAREST)
    missed_r = Image.fromarray(missed.astype(np.uint8) * 255).resize((size, size), Image.Resampling.NEAREST)
    gt_arr = np.array(gt_r) > 0
    pred_arr = np.array(pred_r) > 0
    missed_arr = np.array(missed_r) > 0

    overlay[gt_arr] = overlay[gt_arr] * 0.55 + np.array([50, 210, 80]) * 0.45
    overlay[pred_arr] = overlay[pred_arr] * 0.55 + np.array([230, 70, 60]) * 0.45
    overlay[missed_arr] = np.array([255, 230, 0])

    panel = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(panel)
    label = (
        f"{row['dataset']} / {row['image']}\n"
        f"terminal recall={row['terminal_sensitivity']:.3f}, "
        f"endpoint recall={row['endpoint_recall']:.3f}"
    )
    draw.rectangle([0, 0, size, 44], fill=(255, 255, 255))
    draw.multiline_text((8, 6), label, fill=(20, 20, 20), spacing=2)
    return panel


def make_summary_figure(
    df: pd.DataFrame,
    data_root: Path,
    figure_path: Path,
    endpoint_radius: int,
    max_image_side: int,
    max_cases: int,
) -> None:
    candidates = df.sort_values(["terminal_sensitivity", "endpoint_recall"]).head(max_cases)
    if candidates.empty:
        return
    panel_size = 330
    cols = min(3, len(candidates))
    rows = int(np.ceil(len(candidates) / cols))
    canvas = Image.new("RGB", (cols * panel_size, rows * panel_size + 58), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((14, 14), "Representative missed terminal small vessels", fill=(20, 20, 20))
    draw.text((14, 34), "Green=reference mask, red=baseline prediction, yellow=missed terminal region", fill=(80, 80, 80))
    for i, (_, row) in enumerate(candidates.iterrows()):
        panel = draw_case_panel(row, data_root, endpoint_radius, max_image_side, size=panel_size)
        x = (i % cols) * panel_size
        y = 58 + (i // cols) * panel_size
        canvas.paste(panel, (x, y))
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(figure_path)


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    figure_path = Path(args.figure_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    datasets = sorted(p for p in data_root.iterdir() if p.is_dir())
    for dataset_dir in datasets:
        pairs = pair_image_and_mask(dataset_dir)
        print(f"{dataset_dir.name}: {len(pairs)} paired images")
        for image_path, mask_path in pairs:
            records.append(
                evaluate_pair(
                    dataset_dir.name,
                    image_path,
                    mask_path,
                    args.endpoint_radius,
                    args.endpoint_match_radius,
                    args.max_image_side,
                )
            )

    df = pd.DataFrame(records)
    metrics_path = output_dir / "terminal_vessel_metrics.csv"
    summary_path = output_dir / "terminal_vessel_summary.csv"
    df.to_csv(metrics_path, index=False)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    summary = df.groupby("dataset")[numeric_cols].mean().reset_index()
    overall = pd.DataFrame([{"dataset": "OVERALL", **df[numeric_cols].mean().to_dict()}])
    summary = pd.concat([summary, overall], ignore_index=True)
    summary.to_csv(summary_path, index=False)

    make_summary_figure(
        df,
        data_root,
        figure_path,
        args.endpoint_radius,
        args.max_image_side,
        args.max_figure_cases,
    )

    print(f"Saved {metrics_path}")
    print(f"Saved {summary_path}")
    print(f"Saved {figure_path}")
    print(summary[[
        "dataset", "dice", "iou", "sensitivity", "terminal_sensitivity",
        "endpoint_recall", "missed_endpoint_count"
    ]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
