"""
Hierarchical terminal-vessel refinement for HRF vessel segmentation.

This experiment follows the latest refinement plan:

1. keep a conservative baseline mask as the main-vessel layer,
2. recover terminal small-vessel candidates locally around baseline endpoints,
3. use branch-level skeleton checks and direction continuity as soft supports,
4. report skeleton-distance metrics in addition to pixel overlap metrics.

The goal is not to replace GT evaluation, but to make the refinement and
evaluation more consistent with terminal-vessel structure.
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
class SoftParams:
    anchor_radius: int = 10
    endpoint_radius: int = 40
    min_area: int = 90
    min_mean_vesselness: float = 0.12
    min_max_vesselness: float = 0.24


@dataclass(frozen=True)
class HierarchicalParams:
    anchor_radius: int = 5
    endpoint_radius: int = 46
    nonterminal_min_area: int = 120
    terminal_min_area: int = 18
    terminal_min_skeleton_length: int = 8
    min_mean_vesselness: float = 0.08
    min_max_vesselness: float = 0.22
    min_direction_cosine: float = -0.35


@dataclass(frozen=True)
class BranchGraphParams:
    anchor_radius: int = 5
    endpoint_radius: int = 54
    nonterminal_min_area: int = 130
    terminal_min_area: int = 12
    terminal_min_branch_length: int = 6
    nonterminal_min_branch_length: int = 16
    min_mean_vesselness: float = 0.04
    min_max_vesselness: float = 0.16
    min_terminal_direction_cosine: float = -0.45
    min_nonterminal_direction_cosine: float = -0.25
    max_graph_component_area: int = 1800


@dataclass(frozen=True)
class SoftGraphScoreParams:
    anchor_radius: int = 10
    endpoint_radius: int = 44
    min_area: int = 12
    min_mean_vesselness: float = 0.04
    min_max_vesselness: float = 0.18
    terminal_score_threshold: float = 1.95
    nonterminal_score_threshold: float = 2.45
    direction_penalty_threshold: float = -0.55


@dataclass(frozen=True)
class TreeIterativeParams:
    layers: int = 3
    layer_radius_step: int = 12
    bridge_radius: int = 5
    min_area: int = 10
    min_branch_length: int = 7
    strong_branch_length: int = 16
    min_mean_vesselness: float = 0.04
    min_max_vesselness: float = 0.16
    min_dark_contrast: float = 0.016
    terminal_score_threshold: float = 2.15
    nonterminal_score_threshold: float = 2.55
    recovery_radius: int = 2


@dataclass(frozen=True)
class BranchFeature:
    length: int
    start_y: int
    start_x: int
    end_y: int
    end_x: int
    far_y: int
    far_x: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run hierarchical terminal-vessel refinement on HRF.")
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


def disk(radius: int) -> np.ndarray:
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (x * x + y * y) <= radius * radius


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


def terminal_region(gt: np.ndarray, radius: int = 8) -> np.ndarray:
    _, endpoints = skeleton_endpoints(gt)
    return gt & binary_dilation(endpoints, structure=disk(radius))


def mask_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
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


def skeleton_distance_metrics(pred: np.ndarray, gt: np.ndarray, prefix: str = "") -> dict[str, float]:
    pred_skel, _ = skeleton_endpoints(pred)
    gt_skel, _ = skeleton_endpoints(gt)
    pred_count = int(pred_skel.sum())
    gt_count = int(gt_skel.sum())
    keys = [
        "pred_to_gt_skeleton_distance_mean",
        "pred_to_gt_skeleton_distance_p95",
        "gt_to_pred_skeleton_distance_mean",
        "gt_to_pred_skeleton_distance_p95",
        "skeleton_precision_radius_3",
        "skeleton_recall_radius_3",
        "skeleton_distance_f1_radius_3",
        "skeleton_precision_radius_5",
        "skeleton_recall_radius_5",
        "skeleton_distance_f1_radius_5",
    ]
    if pred_count == 0 or gt_count == 0:
        return {prefix + key: 0.0 for key in keys}

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
    return {prefix + key: value for key, value in out.items()}


def endpoint_direction_map(skel: np.ndarray, endpoints: np.ndarray, radius: int = 14) -> tuple[np.ndarray, np.ndarray]:
    endpoint_coords = np.argwhere(endpoints)
    vectors = []
    for y, x in endpoint_coords:
        y0, y1 = max(0, y - radius), min(skel.shape[0], y + radius + 1)
        x0, x1 = max(0, x - radius), min(skel.shape[1], x + radius + 1)
        local = np.argwhere(skel[y0:y1, x0:x1])
        if len(local) < 2:
            vectors.append((0.0, 0.0))
            continue
        local = local + np.array([y0, x0])
        dist = np.sqrt((local[:, 0] - y) ** 2 + (local[:, 1] - x) ** 2)
        support = local[(dist > 2) & (dist <= radius)]
        if len(support) == 0:
            vectors.append((0.0, 0.0))
            continue
        centroid = support.mean(axis=0)
        vec = np.array([y, x], dtype=float) - centroid
        norm = float(np.linalg.norm(vec))
        vectors.append(tuple(vec / norm) if norm else (0.0, 0.0))
    return endpoint_coords, np.array(vectors, dtype=float)


def component_direction_cosine(
    component_coords: np.ndarray,
    endpoint_coords: np.ndarray,
    endpoint_vectors: np.ndarray,
) -> float:
    if len(endpoint_coords) == 0 or len(component_coords) == 0:
        return 0.0
    centroid = component_coords.mean(axis=0)
    dists = np.sqrt(((endpoint_coords - centroid) ** 2).sum(axis=1))
    idx = int(np.argmin(dists))
    endpoint = endpoint_coords[idx].astype(float)
    direction = endpoint_vectors[idx]
    if float(np.linalg.norm(direction)) == 0:
        return 0.0
    comp_vec = centroid - endpoint
    norm = float(np.linalg.norm(comp_vec))
    if norm == 0:
        return 0.0
    return float(np.dot(comp_vec / norm, direction))


def skeleton_branches(
    component: np.ndarray,
    bbox: tuple[slice, slice],
    component_skel: np.ndarray | None = None,
) -> list[BranchFeature]:
    skel = component_skel if component_skel is not None else skeletonize(component > 0)
    if not skel.any():
        return []
    branches: list[BranchFeature] = []
    junctions = skel & (neighbor_count(skel) >= 3)
    branch_mask = skel & ~junctions
    labeled, n_branches = label(branch_mask)
    objects = find_objects(labeled)

    for branch_id in range(1, n_branches + 1):
        branch_bbox = objects[branch_id - 1]
        if branch_bbox is None:
            continue
        coords = np.argwhere(labeled[branch_bbox] == branch_id)
        if len(coords) < 2:
            continue
        coords = coords + np.array([branch_bbox[0].start, branch_bbox[1].start])
        centroid = coords.mean(axis=0)
        p1 = coords[int(np.argmax(((coords - centroid) ** 2).sum(axis=1)))]
        p2 = coords[int(np.argmax(((coords - p1) ** 2).sum(axis=1)))]
        branches.append(
            BranchFeature(
                length=int(len(coords)),
                start_y=int(p1[0] + bbox[0].start),
                start_x=int(p1[1] + bbox[1].start),
                end_y=int(p2[0] + bbox[0].start),
                end_x=int(p2[1] + bbox[1].start),
                far_y=int(p2[0] + bbox[0].start),
                far_x=int(p2[1] + bbox[1].start),
            )
        )
    return branches


def branch_graph_direction_score(
    branches: list[BranchFeature],
    endpoint_coords: np.ndarray,
    endpoint_vectors: np.ndarray,
) -> float:
    if not branches or len(endpoint_coords) == 0:
        return 0.0
    best_score = -1.0
    for branch in branches:
        ends = np.array([[branch.start_y, branch.start_x], [branch.end_y, branch.end_x]], dtype=float)
        for end_idx, near_end in enumerate(ends):
            dists = np.sqrt(((endpoint_coords - near_end) ** 2).sum(axis=1))
            endpoint_idx = int(np.argmin(dists))
            endpoint = endpoint_coords[endpoint_idx].astype(float)
            direction = endpoint_vectors[endpoint_idx]
            if float(np.linalg.norm(direction)) == 0:
                continue
            far_end = ends[1 - end_idx]
            candidate_vec = far_end - endpoint
            norm = float(np.linalg.norm(candidate_vec))
            if norm == 0:
                continue
            score = float(np.dot(candidate_vec / norm, direction))
            best_score = max(best_score, score)
    return best_score


def branch_graph_stats(
    component: np.ndarray,
    bbox: tuple[slice, slice],
    endpoint_coords: np.ndarray,
    endpoint_vectors: np.ndarray,
    component_skel: np.ndarray | None = None,
) -> dict[str, float]:
    skel = component_skel if component_skel is not None else skeletonize(component > 0)
    skel_coords = np.argwhere(skel)
    if len(skel_coords) == 0:
        return {
            "branch_count": 0.0,
            "longest_branch_length": 0.0,
            "mean_branch_length": 0.0,
            "branch_direction_cosine": 0.0,
        }
    neighbor = neighbor_count(skel)
    endpoints = skel & (neighbor == 1)
    junctions = skel & (neighbor >= 3)
    endpoint_count = int(endpoints.sum())
    junction_count = int(junctions.sum())
    branch_count = max(1.0, endpoint_count / 2.0 + junction_count)
    global_coords = skel_coords + np.array([bbox[0].start, bbox[1].start])
    length = float(len(global_coords))

    direction_score = 0.0
    if len(endpoint_coords) > 0:
        centroid = global_coords.mean(axis=0)
        endpoint_idx = int(np.argmin(((endpoint_coords - centroid) ** 2).sum(axis=1)))
        endpoint = endpoint_coords[endpoint_idx].astype(float)
        direction = endpoint_vectors[endpoint_idx]
        if float(np.linalg.norm(direction)) != 0:
            far_point = global_coords[int(np.argmax(((global_coords - endpoint) ** 2).sum(axis=1)))].astype(float)
            candidate_vec = far_point - endpoint
            norm = float(np.linalg.norm(candidate_vec))
            if norm:
                direction_score = float(np.dot(candidate_vec / norm, direction))

    return {
        "branch_count": float(branch_count),
        "longest_branch_length": length,
        "mean_branch_length": safe_div(length, branch_count),
        "branch_direction_cosine": direction_score,
    }


def local_dark_line_features(image: np.ndarray, component: np.ndarray) -> tuple[float, float]:
    green = image[:, :, 1].astype(float) / 255.0
    component = component.astype(bool)
    if not component.any():
        return 0.0, 0.0
    near = binary_dilation(component, structure=disk(2))
    ring = binary_dilation(component, structure=disk(7)) & ~near
    if not ring.any():
        ring = ~component
    vessel_mean = float(green[component].mean())
    background_mean = float(green[ring].mean()) if ring.any() else vessel_mean
    contrast = background_mean - vessel_mean
    dark_score = max(0.0, contrast)
    return dark_score, vessel_mean


def expand_bbox(bbox: tuple[slice, slice], shape: tuple[int, int], pad: int) -> tuple[slice, slice]:
    y0 = max(0, bbox[0].start - pad)
    y1 = min(shape[0], bbox[0].stop + pad)
    x0 = max(0, bbox[1].start - pad)
    x1 = min(shape[1], bbox[1].stop + pad)
    return slice(y0, y1), slice(x0, x1)


def paste_local(local: np.ndarray, bbox: tuple[slice, slice], shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    out[bbox] = local
    return out


def soft_refinement(
    raw: np.ndarray,
    baseline: np.ndarray,
    vesselness: np.ndarray,
    baseline_endpoints: np.ndarray,
    params: SoftParams,
) -> tuple[np.ndarray, dict[str, int]]:
    refined = baseline.astype(bool).copy()
    anchor_zone = binary_dilation(baseline, iterations=params.anchor_radius)
    endpoint_zone = binary_dilation(baseline_endpoints, structure=disk(params.endpoint_radius))
    labeled, n_components = label(raw & ~baseline)
    objects = find_objects(labeled)
    counts = {"removed_far_components": 0, "removed_weak_components": 0, "kept_components": 0}

    for component_id in range(1, n_components + 1):
        bbox = objects[component_id - 1]
        if bbox is None:
            continue
        component = labeled[bbox] == component_id
        area = int(component.sum())
        if area == 0:
            continue
        touches_anchor = bool((component & anchor_zone[bbox]).any())
        touches_endpoint = bool((component & endpoint_zone[bbox]).any())
        comp_vals = vesselness[bbox][component]
        mean_v = float(comp_vals.mean()) if comp_vals.size else 0.0
        max_v = float(comp_vals.max()) if comp_vals.size else 0.0
        if not touches_anchor and not touches_endpoint:
            counts["removed_far_components"] += 1
            continue
        keep_by_size = area >= params.min_area and mean_v >= params.min_mean_vesselness
        keep_by_endpoint = touches_endpoint and max_v >= params.min_max_vesselness
        keep_by_strength = touches_anchor and max_v >= params.min_max_vesselness
        if keep_by_size or keep_by_endpoint or keep_by_strength:
            refined[bbox] |= component
            counts["kept_components"] += 1
        else:
            counts["removed_weak_components"] += 1

    return remove_small_objects(refined.astype(bool), min_size=12), counts


def hierarchical_refinement(
    raw: np.ndarray,
    baseline: np.ndarray,
    vesselness: np.ndarray,
    baseline_skel: np.ndarray,
    baseline_endpoints: np.ndarray,
    params: HierarchicalParams,
) -> tuple[np.ndarray, dict[str, int]]:
    main_vessel = remove_small_objects(baseline.astype(bool), min_size=24)
    refined = main_vessel.copy()
    anchor_zone = binary_dilation(main_vessel, iterations=params.anchor_radius)
    endpoint_zone = binary_dilation(baseline_endpoints, structure=disk(params.endpoint_radius))
    endpoint_coords, endpoint_vectors = endpoint_direction_map(baseline_skel, baseline_endpoints)
    support_zone = endpoint_zone
    candidate_mask = raw & ~main_vessel & support_zone
    candidate_skel = skeletonize(candidate_mask)
    labeled, n_components = label(candidate_mask)
    objects = find_objects(labeled)
    counts = {
        "kept_terminal_components": 0,
        "kept_nonterminal_components": 0,
        "removed_far_components": 0,
        "removed_weak_components": 0,
        "removed_direction_components": 0,
        "removed_short_components": 0,
    }

    for component_id in range(1, n_components + 1):
        bbox = objects[component_id - 1]
        if bbox is None:
            continue
        component = labeled[bbox] == component_id
        area = int(component.sum())
        if area == 0:
            continue
        comp_vals = vesselness[bbox][component]
        mean_v = float(comp_vals.mean()) if comp_vals.size else 0.0
        max_v = float(comp_vals.max()) if comp_vals.size else 0.0
        touches_anchor = bool((component & anchor_zone[bbox]).any())
        touches_endpoint = bool((component & endpoint_zone[bbox]).any())

        if not touches_anchor and not touches_endpoint:
            counts["removed_far_components"] += 1
            continue

        skel_len = int(skeletonize(component).sum())
        local_coords = np.argwhere(component) + np.array([bbox[0].start, bbox[1].start])
        direction_cos = component_direction_cosine(local_coords, endpoint_coords, endpoint_vectors)

        terminal_candidate = touches_endpoint
        terminal_supported = (
            terminal_candidate
            and area >= params.terminal_min_area
            and skel_len >= params.terminal_min_skeleton_length
            and max_v >= params.min_max_vesselness
            and direction_cos >= params.min_direction_cosine
        )
        nonterminal_supported = (
            touches_anchor
            and area >= params.nonterminal_min_area
            and mean_v >= params.min_mean_vesselness
            and max_v >= params.min_max_vesselness
        )

        if terminal_supported:
            refined[bbox] |= component
            counts["kept_terminal_components"] += 1
        elif nonterminal_supported:
            refined[bbox] |= component
            counts["kept_nonterminal_components"] += 1
        else:
            if terminal_candidate and direction_cos < params.min_direction_cosine:
                counts["removed_direction_components"] += 1
            elif skel_len < params.terminal_min_skeleton_length:
                counts["removed_short_components"] += 1
            else:
                counts["removed_weak_components"] += 1

    return remove_small_objects(refined.astype(bool), min_size=12), counts


def branch_graph_refinement(
    raw: np.ndarray,
    baseline: np.ndarray,
    vesselness: np.ndarray,
    baseline_skel: np.ndarray,
    baseline_endpoints: np.ndarray,
    params: BranchGraphParams,
) -> tuple[np.ndarray, dict[str, int]]:
    main_vessel = remove_small_objects(baseline.astype(bool), min_size=24)
    refined = main_vessel.copy()
    anchor_zone = binary_dilation(main_vessel, iterations=params.anchor_radius)
    endpoint_zone = binary_dilation(baseline_endpoints, structure=disk(params.endpoint_radius))
    endpoint_coords, endpoint_vectors = endpoint_direction_map(baseline_skel, baseline_endpoints)
    support_zone = anchor_zone | endpoint_zone
    candidate_mask = raw & ~main_vessel & support_zone & (vesselness >= params.min_mean_vesselness)
    candidate_skel = skeletonize(candidate_mask)
    labeled, n_components = label(candidate_skel)
    objects = find_objects(labeled)
    counts = {
        "kept_terminal_components": 0,
        "kept_nonterminal_components": 0,
        "removed_far_components": 0,
        "removed_weak_components": 0,
        "removed_direction_components": 0,
        "removed_short_components": 0,
        "branch_graph_components": 0,
    }
    kept_skel = np.zeros_like(candidate_skel, dtype=bool)

    for component_id in range(1, n_components + 1):
        bbox = objects[component_id - 1]
        if bbox is None:
            continue
        skel_component = labeled[bbox] == component_id
        branch_len = int(skel_component.sum())
        if branch_len == 0:
            continue
        support_component = binary_dilation(skel_component, iterations=1) & candidate_mask[bbox]
        area = int(support_component.sum())
        comp_vals = vesselness[bbox][support_component]
        mean_v = float(comp_vals.mean()) if comp_vals.size else 0.0
        max_v = float(comp_vals.max()) if comp_vals.size else 0.0
        touches_anchor = bool((skel_component & anchor_zone[bbox]).any())
        touches_endpoint = bool((skel_component & endpoint_zone[bbox]).any())

        if not touches_anchor and not touches_endpoint:
            counts["removed_far_components"] += 1
            continue
        if max_v < params.min_max_vesselness:
            counts["removed_weak_components"] += 1
            continue
        if touches_endpoint and area < params.terminal_min_area:
            counts["removed_short_components"] += 1
            continue
        if not touches_endpoint and area < params.nonterminal_min_area:
            counts["removed_weak_components"] += 1
            continue

        stats = branch_graph_stats(support_component, bbox, endpoint_coords, endpoint_vectors, skel_component)
        counts["branch_graph_components"] += 1
        longest_branch = stats["longest_branch_length"]
        direction_cos = stats["branch_direction_cosine"]

        terminal_candidate = touches_endpoint
        terminal_supported = (
            terminal_candidate
            and area >= params.terminal_min_area
            and longest_branch >= params.terminal_min_branch_length
            and max_v >= params.min_max_vesselness
            and direction_cos >= params.min_terminal_direction_cosine
        )
        nonterminal_supported = (
            touches_anchor
            and area >= params.nonterminal_min_area
            and longest_branch >= params.nonterminal_min_branch_length
            and mean_v >= params.min_mean_vesselness
            and max_v >= params.min_max_vesselness
        )

        if terminal_supported:
            kept_skel[bbox] |= skel_component
            counts["kept_terminal_components"] += 1
        elif nonterminal_supported:
            kept_skel[bbox] |= skel_component
            counts["kept_nonterminal_components"] += 1
        else:
            if longest_branch < params.terminal_min_branch_length:
                counts["removed_short_components"] += 1
            elif direction_cos < params.min_terminal_direction_cosine and terminal_candidate:
                counts["removed_direction_components"] += 1
            else:
                counts["removed_weak_components"] += 1

    recovered = candidate_mask & binary_dilation(kept_skel, iterations=2)
    refined |= recovered
    return remove_small_objects(refined.astype(bool), min_size=12), counts


def soft_graph_score_refinement(
    raw: np.ndarray,
    baseline: np.ndarray,
    vesselness: np.ndarray,
    baseline_skel: np.ndarray,
    baseline_endpoints: np.ndarray,
    params: SoftGraphScoreParams,
) -> tuple[np.ndarray, dict[str, int]]:
    main_vessel = remove_small_objects(baseline.astype(bool), min_size=24)
    refined = main_vessel.copy()
    anchor_zone = binary_dilation(main_vessel, iterations=params.anchor_radius)
    endpoint_zone = binary_dilation(baseline_endpoints, structure=disk(params.endpoint_radius))
    endpoint_coords, endpoint_vectors = endpoint_direction_map(baseline_skel, baseline_endpoints)
    labeled, n_components = label(raw & ~main_vessel)
    objects = find_objects(labeled)
    counts = {
        "kept_terminal_components": 0,
        "kept_nonterminal_components": 0,
        "removed_far_components": 0,
        "removed_weak_components": 0,
        "removed_direction_components": 0,
        "removed_short_components": 0,
        "soft_graph_scored_components": 0,
    }

    for component_id in range(1, n_components + 1):
        bbox = objects[component_id - 1]
        if bbox is None:
            continue
        component = labeled[bbox] == component_id
        area = int(component.sum())
        if area < params.min_area:
            counts["removed_short_components"] += 1
            continue

        touches_anchor = bool((component & anchor_zone[bbox]).any())
        touches_endpoint = bool((component & endpoint_zone[bbox]).any())
        if not touches_anchor and not touches_endpoint:
            counts["removed_far_components"] += 1
            continue

        comp_vals = vesselness[bbox][component]
        mean_v = float(comp_vals.mean()) if comp_vals.size else 0.0
        max_v = float(comp_vals.max()) if comp_vals.size else 0.0
        if max_v < params.min_max_vesselness and mean_v < params.min_mean_vesselness:
            counts["removed_weak_components"] += 1
            continue

        component_skel = skeletonize(component)
        stats = branch_graph_stats(component, bbox, endpoint_coords, endpoint_vectors, component_skel)
        branch_len = stats["longest_branch_length"]
        direction_cos = stats["branch_direction_cosine"]
        counts["soft_graph_scored_components"] += 1

        support_score = 0.0
        support_score += 0.95 if touches_endpoint else 0.0
        support_score += 0.65 if touches_anchor else 0.0
        vessel_score = 0.65 * min(max_v / 0.24, 1.0) + 0.35 * min(mean_v / 0.12, 1.0)
        size_score = 0.45 * min(area / 90.0, 1.0)
        branch_score = 0.45 * min(branch_len / 18.0, 1.0)
        direction_score = 0.35 * max(0.0, (direction_cos + 1.0) / 2.0)
        score = support_score + vessel_score + size_score + branch_score + direction_score

        terminal_candidate = touches_endpoint
        threshold = params.terminal_score_threshold if terminal_candidate else params.nonterminal_score_threshold
        if direction_cos < params.direction_penalty_threshold and branch_len < 18:
            score -= 0.35

        if score >= threshold:
            recovered_component = component & binary_dilation(component_skel, iterations=2)
            refined[bbox] |= recovered_component
            if terminal_candidate:
                counts["kept_terminal_components"] += 1
            else:
                counts["kept_nonterminal_components"] += 1
        elif direction_cos < params.direction_penalty_threshold:
            counts["removed_direction_components"] += 1
        elif branch_len < 6:
            counts["removed_short_components"] += 1
        else:
            counts["removed_weak_components"] += 1

    return remove_small_objects(refined.astype(bool), min_size=12), counts


def tree_iterative_refinement(
    raw: np.ndarray,
    baseline: np.ndarray,
    vesselness: np.ndarray,
    image: np.ndarray,
    params: TreeIterativeParams,
) -> tuple[np.ndarray, dict[str, int]]:
    refined = remove_small_objects(baseline.astype(bool), min_size=24)
    candidate_pool = (raw & ~refined) & (vesselness >= params.min_mean_vesselness)
    counts = {
        "iterative_layers": params.layers,
        "tree_connected_components": 0,
        "kept_terminal_components": 0,
        "kept_nonterminal_components": 0,
        "removed_far_components": 0,
        "removed_weak_components": 0,
        "removed_short_components": 0,
        "removed_low_contrast_components": 0,
    }

    for layer_idx in range(params.layers):
        current_skel, current_endpoints = skeleton_endpoints(refined)
        endpoint_zone = binary_dilation(current_endpoints, structure=disk(params.layer_radius_step * (layer_idx + 1)))
        tree_zone = binary_dilation(refined, iterations=params.bridge_radius + layer_idx * 2)
        search_zone = endpoint_zone | tree_zone
        layer_candidates = candidate_pool & search_zone
        labeled, n_components = label(layer_candidates)
        objects = find_objects(labeled)
        if n_components == 0:
            continue
        endpoint_coords, endpoint_vectors = endpoint_direction_map(current_skel, current_endpoints)
        layer_kept = np.zeros_like(refined, dtype=bool)

        for component_id in range(1, n_components + 1):
            bbox = objects[component_id - 1]
            if bbox is None:
                continue
            component = labeled[bbox] == component_id
            area = int(component.sum())
            if area < params.min_area:
                counts["removed_short_components"] += 1
                continue

            bridge_bbox = expand_bbox(bbox, refined.shape, params.bridge_radius)
            local_component = np.zeros((bridge_bbox[0].stop - bridge_bbox[0].start, bridge_bbox[1].stop - bridge_bbox[1].start), dtype=bool)
            y_offset = bbox[0].start - bridge_bbox[0].start
            x_offset = bbox[1].start - bridge_bbox[1].start
            local_component[y_offset : y_offset + component.shape[0], x_offset : x_offset + component.shape[1]] = component
            connected_to_tree = bool((binary_dilation(local_component, iterations=params.bridge_radius) & refined[bridge_bbox]).any())
            if not connected_to_tree:
                counts["removed_far_components"] += 1
                continue

            component_skel = skeletonize(component)
            branch_stats = branch_graph_stats(component, bbox, endpoint_coords, endpoint_vectors, component_skel)
            branch_len = branch_stats["longest_branch_length"]
            direction_cos = branch_stats["branch_direction_cosine"]
            touches_endpoint = bool((component & endpoint_zone[bbox]).any())
            comp_vals = vesselness[bbox][component]
            mean_v = float(comp_vals.mean()) if comp_vals.size else 0.0
            max_v = float(comp_vals.max()) if comp_vals.size else 0.0
            dark_contrast, vessel_mean = local_dark_line_features(image[bbox], component)
            counts["tree_connected_components"] += 1

            if branch_len < params.min_branch_length and dark_contrast < params.min_dark_contrast:
                counts["removed_short_components"] += 1
                continue
            if max_v < params.min_max_vesselness and dark_contrast < params.min_dark_contrast:
                counts["removed_low_contrast_components"] += 1
                continue

            connection_score = 0.85
            terminal_score = 0.80 if touches_endpoint else 0.0
            vessel_score = 0.55 * min(max_v / 0.24, 1.0) + 0.30 * min(mean_v / 0.12, 1.0)
            contrast_score = 0.55 * min(max(dark_contrast, 0.0) / 0.05, 1.0)
            branch_score = 0.45 * min(branch_len / params.strong_branch_length, 1.0)
            direction_score = 0.25 * max(0.0, (direction_cos + 1.0) / 2.0)
            layer_penalty = 0.08 * layer_idx
            score = connection_score + terminal_score + vessel_score + contrast_score + branch_score + direction_score - layer_penalty
            threshold = params.terminal_score_threshold if touches_endpoint else params.nonterminal_score_threshold

            if score >= threshold:
                recovered = component & binary_dilation(component_skel, iterations=params.recovery_radius)
                layer_kept[bbox] |= recovered
                if touches_endpoint:
                    counts["kept_terminal_components"] += 1
                else:
                    counts["kept_nonterminal_components"] += 1
            else:
                counts["removed_weak_components"] += 1

        if not layer_kept.any():
            continue
        refined |= layer_kept
        candidate_pool &= ~binary_dilation(layer_kept, iterations=1)

    return remove_small_objects(refined.astype(bool), min_size=12), counts


def evaluate_row(
    image: str,
    method: str,
    pred: np.ndarray,
    gt: np.ndarray,
    term_gt: np.ndarray,
    counts: dict[str, int] | None = None,
) -> dict[str, float | str]:
    counts = counts or {}
    row: dict[str, float | str] = {
        "image": image,
        "method": method,
        "gt_vessel_pixels": int(gt.sum()),
        "pred_vessel_pixels": int(pred.sum()),
    }
    for key, value in counts.items():
        row[key] = int(value)
    row.update(mask_metrics(pred, gt))
    row.update({f"terminal_{key}": value for key, value in mask_metrics(pred & term_gt, term_gt).items()})
    row.update(endpoint_metrics(pred, gt))
    row.update(skeleton_distance_metrics(pred, gt))
    row.update(skeleton_distance_metrics(pred & term_gt, term_gt, prefix="terminal_"))
    return row


def load_cases(hrf_root: Path, base_module, max_side: int) -> list[dict[str, object]]:
    cases = []
    pairs = pair_hrf(hrf_root)
    for idx, (image_path, mask_path) in enumerate(pairs, start=1):
        print(f"[{idx}/{len(pairs)}] {image_path.name}", flush=True)
        image = base_module.load_rgb(image_path, max_side)
        gt = base_module.load_mask(mask_path, image.shape[:2])
        baseline = base_module.segment_baseline(image)
        baseline_skel, baseline_endpoints = skeleton_endpoints(baseline)
        vesselness, fov = base_module.vesselness_map(image)
        connected_raw = base_module.segment_connected_recovery_from_maps(baseline, vesselness, fov)
        cases.append(
            {
                "name": image_path.name,
                "image": image,
                "gt": gt,
                "term_gt": terminal_region(gt),
                "baseline": baseline,
                "baseline_skel": baseline_skel,
                "baseline_endpoints": baseline_endpoints,
                "vesselness": vesselness,
                "connected_raw": connected_raw,
            }
        )
    return cases


def make_summary_figure(summary: pd.DataFrame, out: Path) -> None:
    order = [
        "baseline_raw",
        "connected_recovery_raw",
        "soft_structure_refined",
        "hierarchical_terminal_refined",
        "branch_graph_refined",
        "soft_graph_score_refined",
        "tree_iterative_refined",
    ]
    metrics = [
        ("dice", "Dice"),
        ("precision", "Precision"),
        ("terminal_sensitivity", "Terminal sens."),
        ("skeleton_distance_f1_radius_5", "Skel. F1 r=5"),
        ("terminal_skeleton_distance_f1_radius_5", "Terminal skel. F1"),
    ]
    rows = summary.set_index("method").loc[order].reset_index()
    w, h = 2250, 780
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    draw.text((36, 26), "Hierarchical terminal refinement with skeleton-distance evaluation", fill=(20, 20, 20))
    x0, y0, cw, ch = 90, 112, 2050, 450
    colors = [(64, 132, 214), (232, 144, 58), (151, 99, 205), (88, 168, 190), (80, 170, 110)]
    for t in np.linspace(0, 1, 6):
        y = y0 + ch - int(t * ch)
        draw.line([x0, y, x0 + cw, y], fill=(225, 225, 225))
        draw.text((38, y - 8), f"{t:.1f}", fill=(70, 70, 70))
    group_w = cw / len(rows)
    bar_w = 28
    for i, (_, row) in enumerate(rows.iterrows()):
        gx = x0 + i * group_w + 38
        for j, (col, _) in enumerate(metrics):
            val = float(row[col])
            bh = int(max(0, min(1, val)) * ch)
            bx = gx + j * (bar_w + 12)
            by = y0 + ch - bh
            draw.rectangle([bx, by, bx + bar_w, y0 + ch], fill=colors[j])
            draw.text((bx - 3, by - 17), f"{val:.2f}", fill=(30, 30, 30))
        draw.multiline_text(
            (x0 + i * group_w + 8, y0 + ch + 22),
            str(row["method"]).replace("_", "\n"),
            fill=(30, 30, 30),
            spacing=2,
        )
    lx, ly = 90, 676
    for j, (_, label_text) in enumerate(metrics):
        x = lx + (j % 3) * 330
        y = ly + (j // 3) * 28
        draw.rectangle([x, y, x + 15, y + 15], fill=colors[j])
        draw.text((x + 22, y - 1), label_text, fill=(30, 30, 30))
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)


def mask_panel(mask: np.ndarray, size: int = 260) -> Image.Image:
    return Image.fromarray(mask.astype(np.uint8) * 255).resize((size, size), Image.Resampling.NEAREST).convert("RGB")


def overlay_panel(image: np.ndarray, mask: np.ndarray, gt: np.ndarray, size: int = 260) -> Image.Image:
    base = Image.fromarray(image).resize((size, size))
    mask_r = np.array(Image.fromarray(mask.astype(np.uint8) * 255).resize((size, size), Image.Resampling.NEAREST)) > 0
    gt_r = np.array(Image.fromarray(gt.astype(np.uint8) * 255).resize((size, size), Image.Resampling.NEAREST)) > 0
    arr = np.array(base).astype(np.float32)
    arr[mask_r] = arr[mask_r] * 0.45 + np.array([255, 255, 255]) * 0.55
    arr[gt_r] = arr[gt_r] * 0.65 + np.array([70, 220, 90]) * 0.35
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def make_example_figure(cases: list[dict[str, object]], per_image: pd.DataFrame, out: Path) -> None:
    chosen = (
        per_image[per_image["method"] == "tree_iterative_refined"]
        .assign(balance=lambda d: d["terminal_sensitivity"] + 0.3 * d["precision"] - 0.1 * d["fp_over_gt_area"])
        .sort_values("balance", ascending=False)
        .head(3)["image"]
        .tolist()
    )
    case_map = {case["name"]: case for case in cases}
    labels = ["Original", "GT", "Connected raw", "Tree iterative", "Recovered candidates", "FN after refine"]
    size = 245
    rows = []
    params = TreeIterativeParams()
    for name in chosen:
        case = case_map[name]
        refined, _ = tree_iterative_refinement(
            case["connected_raw"],
            case["baseline"],
            case["vesselness"],
            case["image"],
            params,
        )
        recovered = refined & ~case["baseline"]
        fn = case["gt"] & ~refined
        panels = [
            Image.fromarray(case["image"]).resize((size, size)),
            mask_panel(case["gt"], size),
            overlay_panel(case["image"], case["connected_raw"], case["gt"], size),
            overlay_panel(case["image"], refined, case["gt"], size),
            mask_panel(recovered, size),
            mask_panel(fn, size),
        ]
        row = Image.new("RGB", (size * len(panels), size + 62), "white")
        draw = ImageDraw.Draw(row)
        draw.text((0, 4), name, fill=(20, 20, 20))
        for i, (panel, label_text) in enumerate(zip(panels, labels)):
            draw.text((i * size + 6, 30), label_text, fill=(20, 20, 20))
            row.paste(panel, (i * size, 62))
        rows.append(row)
    canvas = Image.new("RGB", (size * len(labels), 42 + sum(r.height for r in rows) + 12 * (len(rows) - 1)), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((0, 10), "Tree-connected iterative terminal refinement examples", fill=(20, 20, 20))
    y = 42
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
    soft_params = SoftParams()
    hierarchical_params = HierarchicalParams()
    branch_graph_params = BranchGraphParams()
    soft_graph_score_params = SoftGraphScoreParams()
    tree_iterative_params = TreeIterativeParams()

    rows = []
    for idx, case in enumerate(cases, start=1):
        print(f"Refining {idx}/{len(cases)}: {case['name']}", flush=True)
        rows.append(evaluate_row(case["name"], "baseline_raw", case["baseline"], case["gt"], case["term_gt"]))
        rows.append(
            evaluate_row(
                case["name"],
                "connected_recovery_raw",
                case["connected_raw"],
                case["gt"],
                case["term_gt"],
            )
        )
        soft, soft_counts = soft_refinement(
            case["connected_raw"],
            case["baseline"],
            case["vesselness"],
            case["baseline_endpoints"],
            soft_params,
        )
        rows.append(evaluate_row(case["name"], "soft_structure_refined", soft, case["gt"], case["term_gt"], soft_counts))
        hierarchical, hierarchical_counts = hierarchical_refinement(
            case["connected_raw"],
            case["baseline"],
            case["vesselness"],
            case["baseline_skel"],
            case["baseline_endpoints"],
            hierarchical_params,
        )
        rows.append(
            evaluate_row(
                case["name"],
                "hierarchical_terminal_refined",
                hierarchical,
                case["gt"],
                case["term_gt"],
                hierarchical_counts,
            )
        )
        branch_graph, branch_graph_counts = branch_graph_refinement(
            case["connected_raw"],
            case["baseline"],
            case["vesselness"],
            case["baseline_skel"],
            case["baseline_endpoints"],
            branch_graph_params,
        )
        rows.append(
            evaluate_row(
                case["name"],
                "branch_graph_refined",
                branch_graph,
                case["gt"],
                case["term_gt"],
                branch_graph_counts,
            )
        )
        soft_graph_score, soft_graph_score_counts = soft_graph_score_refinement(
            case["connected_raw"],
            case["baseline"],
            case["vesselness"],
            case["baseline_skel"],
            case["baseline_endpoints"],
            soft_graph_score_params,
        )
        rows.append(
            evaluate_row(
                case["name"],
                "soft_graph_score_refined",
                soft_graph_score,
                case["gt"],
                case["term_gt"],
                soft_graph_score_counts,
            )
        )
        tree_iterative, tree_iterative_counts = tree_iterative_refinement(
            case["connected_raw"],
            case["baseline"],
            case["vesselness"],
            case["image"],
            tree_iterative_params,
        )
        rows.append(
            evaluate_row(
                case["name"],
                "tree_iterative_refined",
                tree_iterative,
                case["gt"],
                case["term_gt"],
                tree_iterative_counts,
            )
        )

    per_image = pd.DataFrame(rows).fillna(0)
    numeric = per_image.select_dtypes(include=[np.number]).columns
    summary = per_image.groupby("method")[numeric].mean().reset_index()
    per_image.to_csv(output_dir / "hierarchical_terminal_metrics.csv", index=False)
    summary.to_csv(output_dir / "hierarchical_terminal_summary.csv", index=False)
    make_summary_figure(summary, figure_dir / "hierarchical_terminal_summary.png")
    make_example_figure(cases, per_image, figure_dir / "hierarchical_terminal_examples.png")

    cols = [
        "method",
        "dice",
        "sensitivity",
        "precision",
        "terminal_sensitivity",
        "fp_over_gt_area",
        "fn_over_gt_area",
        "skeleton_distance_f1_radius_5",
        "terminal_skeleton_distance_f1_radius_5",
        "endpoint_recall",
        "endpoint_precision",
    ]
    print(summary[cols].round(3).to_string(index=False))
    print(f"Saved results to {output_dir}")
    print(f"Saved figures to {figure_dir}")


if __name__ == "__main__":
    main()
