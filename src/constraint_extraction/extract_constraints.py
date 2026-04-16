"""
Constraint extraction from retinal fundus images.

This module extracts structural constraints from fundus images that are
used to parameterize the generative branching model:
- Retina boundary (field of view)
- Optic disc location and radius
- Macula (fovea) location
- Major vessel orientations (superior/inferior temporal arcades)
- Regional vessel density map
"""

import numpy as np
import cv2
from typing import Tuple, Optional, Dict, Any


# =============================================================================
# Retina field of view
# =============================================================================

def detect_retina_boundary(image: np.ndarray) -> Tuple[Tuple[int, int], int]:
    """
    Detect the circular retina field of view from a fundus image.

    The fundus image has a circular bright region on a black background.
    We threshold, find the largest contour, and fit a minimum enclosing circle.

    Parameters
    ----------
    image : np.ndarray
        Fundus image (BGR or RGB), shape (H, W, 3).

    Returns
    -------
    Tuple[Tuple[int, int], int]
        ((center_x, center_y), radius) in pixel coordinates.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    _, mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)

    # Clean up with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        h, w = gray.shape[:2]
        return (w // 2, h // 2), min(w, h) // 2

    largest = max(contours, key=cv2.contourArea)
    (cx, cy), radius = cv2.minEnclosingCircle(largest)
    return (int(cx), int(cy)), int(radius)


# =============================================================================
# Optic disc detection
# =============================================================================

def detect_optic_disc(
    image: np.ndarray,
    retina_center: Optional[Tuple[int, int]] = None,
    retina_radius: Optional[int] = None,
) -> Tuple[Tuple[int, int], int]:
    """
    Detect the optic disc from a fundus image.

    The optic disc is the brightest, roughly circular region in the image.
    We use channel-based intensity analysis with morphological filtering.

    Parameters
    ----------
    image : np.ndarray
        Fundus image (BGR), shape (H, W, 3).
    retina_center : Tuple[int, int], optional
        Center of the retina FOV. If None, auto-detected.
    retina_radius : int, optional
        Radius of the retina FOV. If None, auto-detected.

    Returns
    -------
    Tuple[Tuple[int, int], int]
        ((center_x, center_y), radius) of the optic disc in pixel coordinates.
    """
    if retina_center is None or retina_radius is None:
        retina_center, retina_radius = detect_retina_boundary(image)

    # Create retina mask to exclude background
    h, w = image.shape[:2]
    retina_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(retina_mask, retina_center, int(retina_radius * 0.95), 255, -1)

    # Use the red channel (strongest signal for optic disc in fundus images)
    # combined with value channel from HSV
    if image.ndim == 3:
        red_channel = image[:, :, 2] if image.shape[2] == 3 else image[:, :, 0]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        value_channel = hsv[:, :, 2]
        # Combine red and value channels
        combined = cv2.addWeighted(red_channel, 0.5, value_channel, 0.5, 0)
    else:
        combined = image.copy()

    # Apply retina mask
    combined = cv2.bitwise_and(combined, combined, mask=retina_mask)

    # Heavy blur to smooth out vessels and find the brightest blob
    ksize = max(31, retina_radius // 5)
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    blurred = cv2.GaussianBlur(combined, (ksize, ksize), 0)

    # Threshold to find the brightest region — use a high threshold
    max_val = blurred.max()
    _, bright_mask = cv2.threshold(blurred, int(max_val * 0.92), 255, cv2.THRESH_BINARY)

    # Clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # Fallback: use the single brightest point
        _, _, _, max_loc = cv2.minMaxLoc(blurred, mask=retina_mask)
        return max_loc, retina_radius // 10

    # The optic disc is roughly 1/6 to 1/5 of the retina diameter.
    # Filter contours by expected size range.
    expected_min_r = retina_radius * 0.04
    expected_max_r = retina_radius * 0.15
    expected_min_area = np.pi * expected_min_r ** 2
    expected_max_area = np.pi * expected_max_r ** 2

    best_contour = None
    best_score = -1
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < expected_min_area * 0.3:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)

        # Penalize contours that are way too large
        if area > expected_max_area * 2:
            size_penalty = expected_max_area / area
        else:
            size_penalty = 1.0

        score = area * circularity * size_penalty
        if score > best_score:
            best_score = score
            best_contour = cnt

    if best_contour is None:
        _, _, _, max_loc = cv2.minMaxLoc(blurred, mask=retina_mask)
        return max_loc, retina_radius // 10

    (cx, cy), radius = cv2.minEnclosingCircle(best_contour)

    # Clamp radius to expected range
    radius = max(radius, expected_min_r)
    radius = min(radius, expected_max_r)

    return (int(cx), int(cy)), max(int(radius), 5)


# =============================================================================
# Macula (fovea) detection
# =============================================================================

def detect_macula(
    image: np.ndarray,
    optic_disc_center: Tuple[int, int],
    retina_center: Tuple[int, int],
    retina_radius: int,
) -> Tuple[Tuple[int, int], int]:
    """
    Detect the macula (foveal) region from a fundus image.

    The macula is a dark region roughly 2-2.5 disc diameters temporal
    to the optic disc, near the image center.

    Parameters
    ----------
    image : np.ndarray
        Fundus image (BGR).
    optic_disc_center : Tuple[int, int]
        Detected optic disc center (x, y).
    retina_center : Tuple[int, int]
        Retina FOV center.
    retina_radius : int
        Retina FOV radius.

    Returns
    -------
    Tuple[Tuple[int, int], int]
        ((center_x, center_y), radius) of the macula.
    """
    h, w = image.shape[:2]
    od_x, od_y = optic_disc_center

    # The macula is on the opposite side of the optic disc relative to
    # the retina center, at roughly 2-2.5 disc diameters distance
    # Direction from OD to retina center
    dx = retina_center[0] - od_x
    dy = retina_center[1] - od_y
    dist = np.sqrt(dx**2 + dy**2)

    if dist < 1:
        # OD is at center, estimate macula position
        macula_x = retina_center[0]
        macula_y = retina_center[1]
    else:
        # Move from OD in the direction of retina center, overshoot slightly
        scale = min(dist * 1.8, retina_radius * 0.6) / dist
        macula_x = int(od_x + dx * scale)
        macula_y = int(od_y + dy * scale)

    # Search for the darkest region near the estimated position
    # Create a search mask around the estimated macula location
    search_radius = int(retina_radius * 0.25)
    search_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(search_mask, (macula_x, macula_y), search_radius, 255, -1)

    # Also mask out the retina boundary
    retina_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(retina_mask, retina_center, int(retina_radius * 0.9), 255, -1)
    search_mask = cv2.bitwise_and(search_mask, retina_mask)

    # Use green channel (best contrast for macula)
    green = image[:, :, 1] if image.ndim == 3 else image

    # Blur and find the darkest point in the search region
    blurred = cv2.GaussianBlur(green, (31, 31), 0)
    # Invert so we can find the "brightest" (actually darkest) point
    inverted = cv2.bitwise_not(blurred)
    inverted = cv2.bitwise_and(inverted, inverted, mask=search_mask)

    _, _, _, max_loc = cv2.minMaxLoc(inverted, mask=search_mask)

    # Macula radius is typically about 1/6 of retina radius
    macula_radius = int(retina_radius * 0.12)

    return max_loc, macula_radius


# =============================================================================
# Vessel segmentation (for orientation and density extraction)
# =============================================================================

def segment_vessels(
    image: np.ndarray,
    retina_center: Optional[Tuple[int, int]] = None,
    retina_radius: Optional[int] = None,
) -> np.ndarray:
    """
    Segment vessels from a fundus image using morphological and
    Frangi-like filtering on the green channel.

    Parameters
    ----------
    image : np.ndarray
        Fundus image (BGR).
    retina_center : Tuple[int, int], optional
        Retina FOV center.
    retina_radius : int, optional
        Retina FOV radius.

    Returns
    -------
    np.ndarray
        Binary vessel mask, shape (H, W), dtype uint8 (0 or 255).
    """
    if retina_center is None or retina_radius is None:
        retina_center, retina_radius = detect_retina_boundary(image)

    h, w = image.shape[:2]
    retina_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(retina_mask, retina_center, int(retina_radius * 0.95), 255, -1)

    # Green channel has best vessel contrast
    green = image[:, :, 1] if image.ndim == 3 else image

    # CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(green)

    # Morphological top-hat to extract dark structures (vessels)
    # on a bright background
    kernel_size = max(15, retina_radius // 20)
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    background = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
    vessel_signal = cv2.subtract(background, enhanced)

    # Apply retina mask
    vessel_signal = cv2.bitwise_and(vessel_signal, vessel_signal, mask=retina_mask)

    # Threshold using Otsu's method
    _, vessel_mask = cv2.threshold(vessel_signal, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Clean up small noise
    clean_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    vessel_mask = cv2.morphologyEx(vessel_mask, cv2.MORPH_OPEN, clean_kernel)

    # Apply retina mask again
    vessel_mask = cv2.bitwise_and(vessel_mask, vessel_mask, mask=retina_mask)

    return vessel_mask


# =============================================================================
# Major vessel orientation
# =============================================================================

def extract_vessel_orientation(
    vessel_mask: np.ndarray,
    optic_disc_center: Tuple[int, int],
    optic_disc_radius: int,
    retina_radius: int,
) -> Tuple[float, float]:
    """
    Extract the major vessel orientations (superior and inferior arcades).

    Analyzes the angular distribution of vessel pixels in an annular region
    around the optic disc to find the two dominant directions.

    Parameters
    ----------
    vessel_mask : np.ndarray
        Binary vessel segmentation mask (0/255).
    optic_disc_center : Tuple[int, int]
        Optic disc center (x, y).
    optic_disc_radius : int
        Optic disc radius.
    retina_radius : int
        Retina FOV radius.

    Returns
    -------
    Tuple[float, float]
        (angle_superior, angle_inferior) in radians.
        Angles measured from the positive x-axis, counterclockwise.
    """
    h, w = vessel_mask.shape[:2]
    od_x, od_y = optic_disc_center

    # Analyze vessel pixels in an annular region around the optic disc
    inner_r = int(optic_disc_radius * 1.5)
    outer_r = int(optic_disc_radius * 4.0)
    outer_r = min(outer_r, retina_radius - 10)

    # Get vessel pixel coordinates in the annular region
    ys, xs = np.where(vessel_mask > 0)
    dx = xs - od_x
    dy = -(ys - od_y)  # Flip y for standard math convention

    distances = np.sqrt(dx**2 + dy**2)
    in_annulus = (distances >= inner_r) & (distances <= outer_r)

    if np.sum(in_annulus) < 10:
        # Not enough vessel pixels; return default arcade angles
        return (np.deg2rad(155), np.deg2rad(205))

    angles = np.arctan2(dy[in_annulus], dx[in_annulus])  # [-pi, pi]

    # Build angular histogram
    n_bins = 72  # 5-degree bins
    hist, bin_edges = np.histogram(angles, bins=n_bins, range=(-np.pi, np.pi))

    # Smooth the histogram
    from scipy.ndimage import uniform_filter1d
    hist_smooth = uniform_filter1d(hist.astype(float), size=5, mode='wrap')

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Find the two dominant peaks — one in upper half (0, pi), one in lower (-pi, 0)
    upper_mask = bin_centers > 0
    lower_mask = bin_centers < 0

    if np.any(hist_smooth[upper_mask] > 0):
        upper_idx = np.argmax(hist_smooth * upper_mask)
        angle_superior = bin_centers[upper_idx]
    else:
        angle_superior = np.deg2rad(155)

    if np.any(hist_smooth[lower_mask] > 0):
        lower_idx = np.argmax(hist_smooth * lower_mask)
        angle_inferior = bin_centers[lower_idx]
    else:
        angle_inferior = np.deg2rad(205)

    return (float(angle_superior), float(angle_inferior))


# =============================================================================
# Vessel density map
# =============================================================================

def compute_vessel_density_map(
    vessel_mask: np.ndarray,
    retina_center: Tuple[int, int],
    retina_radius: int,
    grid_size: int = 20,
) -> np.ndarray:
    """
    Compute a regional vessel density map from a segmented vessel mask.

    Divides the retinal field into a grid and computes the fraction of
    vessel pixels in each cell.

    Parameters
    ----------
    vessel_mask : np.ndarray
        Binary vessel mask (0/255).
    retina_center : Tuple[int, int]
        Retina FOV center.
    retina_radius : int
        Retina FOV radius.
    grid_size : int
        Number of grid cells along each axis.

    Returns
    -------
    np.ndarray
        Density map of shape (grid_size, grid_size) with values in [0, 1].
        Cells outside the retina have value 0.
    """
    cx, cy = retina_center
    r = retina_radius

    density = np.zeros((grid_size, grid_size), dtype=np.float64)
    binary = (vessel_mask > 0).astype(np.float64)

    for i in range(grid_size):
        for j in range(grid_size):
            # Map grid cell (i, j) to pixel coordinates
            x_start = int(cx - r + j * (2 * r / grid_size))
            x_end = int(cx - r + (j + 1) * (2 * r / grid_size))
            y_start = int(cy - r + i * (2 * r / grid_size))
            y_end = int(cy - r + (i + 1) * (2 * r / grid_size))

            # Clamp to image bounds
            x_start = max(0, x_start)
            x_end = min(vessel_mask.shape[1], x_end)
            y_start = max(0, y_start)
            y_end = min(vessel_mask.shape[0], y_end)

            if x_end <= x_start or y_end <= y_start:
                continue

            cell = binary[y_start:y_end, x_start:x_end]
            total_pixels = cell.size
            if total_pixels == 0:
                continue

            density[i, j] = cell.sum() / total_pixels

    # Normalize to [0, 1]
    max_density = density.max()
    if max_density > 0:
        density = density / max_density

    return density


# =============================================================================
# High-level extraction: all constraints from one image
# =============================================================================

def extract_all_constraints(
    image: np.ndarray,
    density_grid_size: int = 20,
) -> Dict[str, Any]:
    """
    Extract all structural constraints from a single fundus image.

    Returns a dictionary of constraints ready to parameterize
    the generative branching model.

    Parameters
    ----------
    image : np.ndarray
        Fundus image (BGR), shape (H, W, 3).
    density_grid_size : int
        Grid resolution for the vessel density map.

    Returns
    -------
    dict
        Keys:
        - 'retina_center': (x, y) pixel coords
        - 'retina_radius': int
        - 'optic_disc_center': (x, y) pixel coords
        - 'optic_disc_radius': int
        - 'macula_center': (x, y) pixel coords
        - 'macula_radius': int
        - 'angle_superior': float (radians)
        - 'angle_inferior': float (radians)
        - 'vessel_density_map': np.ndarray (grid_size x grid_size)
        - 'vessel_mask': np.ndarray (H x W, binary)
        - 'normalized': dict with all positions in normalized [-1, 1] coords
    """
    # Step 1: Retina boundary
    retina_center, retina_radius = detect_retina_boundary(image)

    # Step 2: Optic disc
    od_center, od_radius = detect_optic_disc(image, retina_center, retina_radius)

    # Step 3: Macula
    mac_center, mac_radius = detect_macula(image, od_center, retina_center, retina_radius)

    # Step 4: Vessel segmentation
    vessel_mask = segment_vessels(image, retina_center, retina_radius)

    # Step 5: Vessel orientation
    angle_sup, angle_inf = extract_vessel_orientation(
        vessel_mask, od_center, od_radius, retina_radius
    )

    # Step 6: Vessel density map
    density_map = compute_vessel_density_map(
        vessel_mask, retina_center, retina_radius, density_grid_size
    )

    # Step 7: Normalize positions to [-1, 1] coordinate system
    # (centered on retina center, scaled by retina radius)
    def normalize_point(px, py):
        nx = (px - retina_center[0]) / retina_radius
        ny = -(py - retina_center[1]) / retina_radius  # Flip y
        return (float(nx), float(ny))

    od_norm = normalize_point(*od_center)
    mac_norm = normalize_point(*mac_center)

    return {
        # Pixel coordinates
        'retina_center': retina_center,
        'retina_radius': retina_radius,
        'optic_disc_center': od_center,
        'optic_disc_radius': od_radius,
        'macula_center': mac_center,
        'macula_radius': mac_radius,
        'angle_superior': angle_sup,
        'angle_inferior': angle_inf,
        'vessel_density_map': density_map,
        'vessel_mask': vessel_mask,
        # Normalized coordinates (for the generative model)
        'normalized': {
            'retina_radius': 1.0,
            'root': od_norm,
            'macula_center': mac_norm,
            'macula_radius': float(mac_radius / retina_radius),
            'base_angle_up': angle_sup,
            'base_angle_down': angle_inf,
        },
    }
