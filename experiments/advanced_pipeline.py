"""
Advanced vessel segmentation pipeline with:
- Hysteresis thresholding
- Gabor filter bank
- Morphological reconstruction & cleanup
- Multi-source fusion
- Per-image adaptive threshold
- B-COSFIRE-inspired filter
"""

import cv2
import numpy as np
from skimage.morphology import skeletonize, remove_small_holes, remove_small_objects, binary_closing, binary_opening
from skimage.filters import threshold_otsu, threshold_triangle
from scipy import ndimage as ndi
from typing import List, Tuple, Optional, Dict

from vessel_pipeline import *


# ── 1. HYSTERESIS THRESHOLDING ─────────────────────────────────────────

def hysteresis_threshold(
    soft: np.ndarray,
    fov: np.ndarray,
    high_threshold: float = 0.5,
    low_threshold: float = 0.2,
    fov_erode_px: int = 8,
) -> np.ndarray:
    """
    Two-threshold hysteresis:
    - High: strong vessel pixels (sure vessel)
    - Low: weak vessel pixels, only kept if connected to a sure vessel
    
    This dramatically improves both sensitivity AND precision.
    """
    inner_fov = erode_mask(fov, int(fov_erode_px))
    working = soft.astype(np.float32).copy()
    working[~inner_fov] = 0.0
    
    # Normalize to 0-1 within FOV for consistent thresholds
    if np.any(inner_fov):
        low_val = np.percentile(working[inner_fov], 1)
        high_val = np.percentile(working[inner_fov], 99)
        if high_val > low_val:
            working = np.clip((working - low_val) / (high_val - low_val), 0, 1)
    
    strong = (working >= high_threshold) & inner_fov
    weak = (working >= low_threshold) & inner_fov & ~strong
    
    # Label all weak+strong components
    combined = np.zeros(working.shape, dtype=bool)
    combined[strong | weak] = True
    
    # Connected components on combined
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined.astype(np.uint8), 8)
    
    # Keep components that contain at least one strong pixel
    output = np.zeros(working.shape, dtype=bool)
    for label in range(1, n_labels):
        component = labels == label
        if np.any(strong[component]):
            output[component] = True
    
    return output & inner_fov


def adaptive_hysteresis(
    soft: np.ndarray,
    fov: np.ndarray,
    fov_erode_px: int = 8,
) -> np.ndarray:
    """
    Hysteresis with automatically chosen thresholds based on the
    vesselness histogram. High threshold at percentile where the
    histogram flattens, low threshold at the triangle threshold.
    """
    inner_fov = erode_mask(fov, int(fov_erode_px))
    working = soft.astype(np.float32).copy()
    working[~inner_fov] = 0.0
    
    values = working[inner_fov]
    values = values[values > 0]
    if values.size < 100:
        return working > 0.5
    
    # High threshold: top 5-8% by default (strong sure-vessel responses)
    high_th = float(np.percentile(values, 92))
    
    # Low threshold: use triangle or percentile
    try:
        low_th = float(threshold_triangle(values))
    except ValueError:
        low_th = float(np.percentile(values, 70))
    
    # Ensure high > low
    if high_th <= low_th:
        high_th = float(np.percentile(values, 92))
        low_th = max(float(np.percentile(values, 60)), high_th * 0.5)
    
    return hysteresis_threshold(soft, fov, high_threshold=high_th, low_threshold=low_th, fov_erode_px=fov_erode_px)


# ── 2. GABOR FILTER BANK ───────────────────────────────────────────────

def gabor_kernel_2d(size: int, theta: float, sigma: float, lambd: float, gamma: float = 0.5, psi: float = 0):
    """Create a single Gabor kernel."""
    center = size // 2
    y, x = np.ogrid[-center:size-center, -center:size-center]
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)
    gb = np.exp(-0.5 * (x_theta**2 / sigma**2 + y_theta**2 * gamma**2 / sigma**2))
    gb *= np.cos(2 * np.pi * x_theta / lambd + psi)
    gb -= gb.mean()
    return gb.astype(np.float32)


def gabor_filter_response(inverted_float: np.ndarray, fov: np.ndarray) -> np.ndarray:
    """
    Multi-scale, multi-orientation Gabor filter bank.
    Complements matched filter response - better at fine vessel detection.
    """
    response = np.zeros(inverted_float.shape, dtype=np.float32)
    
    # Scales and orientations for vessel detection
    sigmas = [1.5, 2.5, 3.5, 5.0]
    lambdas = [3.0, 5.0, 7.0, 10.0]
    angles = range(0, 180, 15)
    
    for sigma, lambd in zip(sigmas, lambdas):
        size = int(6 * sigma)
        if size % 2 == 0:
            size += 1
        size = max(3, size)
        
        for angle in angles:
            theta = np.deg2rad(angle)
            kernel = gabor_kernel_2d(size, theta, sigma, lambd)
            filtered = cv2.filter2D(inverted_float, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT)
            response = np.maximum(response, filtered)
    
    response[~fov] = 0
    return normalize01(response, fov)


# ── 3. MORPHOLOGICAL POSTPROCESSING ────────────────────────────────────

def morphological_cleanup(
    binary: np.ndarray,
    fov: np.ndarray,
    min_vessel_area: int = 8,
    close_radius: int = 2,
    remove_small_islands: bool = True,
    keep_elongated_only: bool = False,
) -> np.ndarray:
    """
    Advanced morphological cleanup for vessel masks.
    
    1. Remove small speckle (isolated pixels)
    2. Close small gaps in vessels (morphological closing)
    3. Optionally filter by elongation (vessels are long and thin)
    4. Remove FOV edge artifacts
    """
    output = binary.copy()
    output[~fov] = 0
    
    # Step 1: Remove very small components (speckle noise)
    if remove_small_islands:
        output = keep_components_at_least(output, min_vessel_area)
    
    # Step 2: Close small gaps (but don't over-fill)
    if close_radius > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_radius + 1, 2 * close_radius + 1))
        output = cv2.morphologyEx(output.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
    
    # Step 3: Remove edge-touching artifacts
    output = remove_border_components(output, border_px=3)
    
    # Step 4: Optional elongation filter
    if keep_elongated_only:
        output = keep_elongated_components(output, min_aspect=2.0, min_area=min_vessel_area)
    
    return output & fov


def keep_elongated_components(
    mask: np.ndarray,
    min_aspect: float = 2.0,
    min_area: int = 8,
) -> np.ndarray:
    """
    Keep only connected components that are elongated (vessel-like).
    Vessels are long and thin, noise tends to be round.
    """
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n_labels <= 1:
        return mask.astype(bool)
    
    output = np.zeros(mask.shape, dtype=bool)
    for label in range(1, n_labels):
        x, y, w, h, area = stats[label]
        if area < min_area:
            continue
        # Elongation: aspect ratio of bounding box
        short_side = max(1, min(w, h))
        aspect = max(w, h) / short_side
        # Also check pixel-wise: perimeter vs area
        # A vessel should have high perimeter/area ratio
        if aspect >= min_aspect:
            output[labels == label] = True
    
    return output


def vessel_connectivity_refinement(
    binary: np.ndarray,
    fov: np.ndarray,
) -> np.ndarray:
    """
    Use morphological reconstruction to connect broken vessel segments.
    """
    # Dilate to connect nearby segments
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(binary.astype(np.uint8), kernel, iterations=1).astype(bool)
    
    # Skeletonize to thin the connections back
    skeleton = skeletonize(dilated).astype(bool)
    
    # Remove small skeleton fragments
    skeleton = keep_components_at_least(skeleton, 10)
    
    return skeleton & fov


# ── 4. MULTI-SOURCE FUSION ─────────────────────────────────────────────

def fuse_multiple_sources(
    rgb: np.ndarray,
    fov: np.ndarray,
    config: VesselPipelineConfig,
) -> np.ndarray:
    """
    Fuse multiple channel sources at the combined-map level.
    
    Uses:
    - C6G6 (CLAHE L*6 + Green6): good general-purpose
    - FlatSub+G6: good for fine detail
    - Green channel residual: good for vessel-specific contrast
    
    Returns fused soft vesselness map.
    """
    # Get combined maps from each source
    res_c6 = process_channel_source(rgb, fov, config, 'C6G6')
    c6_combined = res_c6['combined']
    
    res_flat = process_channel_source(rgb, fov, config, 'FlatSub+G6')
    flat_combined = res_flat['combined']
    
    # Also get the raw green channel with simpler processing
    green = rgb[:, :, 1].astype(np.float32)
    green[~fov] = 0
    
    # CLAHE on green
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    green_clahe = clahe.apply(green.astype(np.uint8))
    green_clahe[~fov] = 0
    green_float = normalize01(green_clahe.astype(np.float32), fov)
    
    # Run vessel filters on green only (simpler, complementary)
    inv_green = 255 - np.clip(green_clahe, 0, 255).astype(np.uint8)
    inv_green_float = normalize01(inv_green.astype(np.float32), fov)
    green_matched = matched_filter_response(inv_green_float, fov)
    green_jerman = jerman_vesselness(inv_green_float, fov)
    
    green_combined = normalize01(0.6 * green_matched + 0.4 * green_jerman, fov)
    
    # Fuse: weighted combination
    fused = normalize01(
        0.45 * c6_combined + 0.35 * flat_combined + 0.20 * green_combined,
        fov
    )
    
    return fused


def compute_best_threshold_per_image(
    soft: np.ndarray,
    fov: np.ndarray,
    target_density_range: Tuple[float, float] = (0.06, 0.16),
) -> float:
    """
    Automatically pick the best threshold percentile for this specific image.
    Uses the vesselness histogram shape to find the elbow point.
    """
    inner_fov = erode_mask(fov, 8)
    values = soft[inner_fov]
    values = values[values > 0]
    if values.size < 100:
        return 0.5
    
    # Sort values, get cumulative distribution
    sorted_vals = np.sort(values)
    cumsum = np.cumsum(sorted_vals)
    cumsum = cumsum / cumsum[-1]
    
    # Find the knee/elbow: where the cumulative sum starts to flatten
    # This is where we transition from background noise to vessel signal
    n = len(sorted_vals)
    x = np.arange(n) / n
    
    # Distance from line (0,0) to (1,1) for the cumulative curve
    d = np.abs(cumsum - x)
    knee_idx = np.argmax(d)
    
    # The threshold value at the knee
    knee_th = sorted_vals[knee_idx]
    
    # Also check what percentile this corresponds to
    knee_pct = 100.0 * (1.0 - knee_idx / n)
    
    # Clamp to reasonable range
    knee_pct = max(target_density_range[0] * 100, min(target_density_range[1] * 100, knee_pct))
    
    return knee_pct / 100.0


def threshold_with_elbow(
    soft: np.ndarray,
    fov: np.ndarray,
    fov_erode_px: int = 8,
) -> np.ndarray:
    """
    Threshold using automatic elbow/knee detection on the vesselness histogram.
    """
    inner_fov = erode_mask(fov, int(fov_erode_px))
    working = soft.astype(np.float32).copy()
    working[~inner_fov] = 0.0
    
    # Use the elbow method
    target_density = compute_best_threshold_per_image(soft, fov)
    
    # Apply percentile threshold at the computed density
    return threshold_response_map(soft, fov, method='percentile',
                                   target_density=target_density,
                                   fov_erode_px=fov_erode_px)


# ── 5. B-COSFIRE INSPIRED FILTER ───────────────────────────────────────

def cosfire_filter_response(
    inverted_float: np.ndarray,
    fov: np.ndarray,
) -> np.ndarray:
    """
    Simplified B-COSFIRE-inspired filter.
    
    B-COSFIRE works by:
    1. Finding points of high DoG response around concentric circles
    2. Learning the spatial configuration of these points
    3. Shifting filter responses to align and combine them
    
    Our simplified version:
    1. Multi-scale DoG (Difference of Gaussian)
    2. Maximum response across orientations at each scale
    3. Blur and shift to allow tolerance
    """
    response = np.zeros(inverted_float.shape, dtype=np.float32)
    
    # Multi-scale DoG
    for sigma in [0.8, 1.2, 1.8, 2.5, 3.5]:
        k = 1.4  # ratio between inner and outer Gaussian
        inner = cv2.GaussianBlur(inverted_float, (0, 0), sigmaX=sigma)
        outer = cv2.GaussianBlur(inverted_float, (0, 0), sigmaX=sigma * k)
        dog = inner - outer
        dog = np.maximum(dog, 0)  # half-wave rectification
        response = np.maximum(response, dog)
    
    # Blur to allow some spatial tolerance (as in B-COSFIRE)
    response = cv2.GaussianBlur(response, (0, 0), sigmaX=1.5)
    response[~fov] = 0
    return normalize01(response, fov)


# ── 6. ENHANCED HYBRID PIPELINE ────────────────────────────────────────

def enhanced_combined(
    rgb: np.ndarray,
    fov: np.ndarray,
    config: VesselPipelineConfig,
) -> Dict[str, np.ndarray]:
    """
    Run ALL enhancement methods and return a dictionary of results.
    """
    results = {}
    
    # Standard Almeida pipeline on C6G6
    res_c6 = process_channel_source(rgb, fov, config, 'C6G6')
    results['C6_combined'] = res_c6['combined']
    results['C6_TriClean'] = res_c6['Triangle_clean']
    
    # Gabor filter on C6G6 inverted
    c6_channel = cielab_green_clahe_source(rgb, fov, l_clip=6.0, green_clip=6.0)
    c6_inverted = 255 - c6_channel
    c6_inverted_float = normalize01(c6_inverted.astype(np.float32), fov)
    results['C6_Gabor'] = gabor_filter_response(c6_inverted_float, fov)
    
    # DoG (B-COSFIRE inspired) on C6G6
    results['C6_DoG'] = cosfire_filter_response(c6_inverted_float, fov)
    
    # Fused: Almeida + Gabor
    results['C6_Almeida_Gabor'] = normalize01(
        0.7 * res_c6['combined'] + 0.3 * results['C6_Gabor'], fov
    )
    
    # Fused: Almeida + DoG + Gabor
    results['C6_Triple'] = normalize01(
        0.5 * res_c6['combined'] + 0.25 * results['C6_Gabor'] + 0.25 * results['C6_DoG'],
        fov
    )
    
    # Apply hysteresis to each
    for key in ['C6_combined', 'C6_Almeida_Gabor', 'C6_Triple']:
        hyst = adaptive_hysteresis(results[key], fov, fov_erode_px=8)
        results[f'{key}_hyst'] = hyst
    
    # Apply morphological cleanup to hysteresis
    for key in ['C6_combined', 'C6_Almeida_Gabor', 'C6_Triple']:
        hyst = adaptive_hysteresis(results[key], fov, fov_erode_px=8)
        cleaned = morphological_cleanup(hyst, fov, min_vessel_area=4, close_radius=2)
        results[f'{key}_hyst_clean'] = cleaned
    
    # Multi-source fusion
    fused = fuse_multiple_sources(rgb, fov, config)
    results['MultiFused'] = fused
    results['MultiFused_hyst'] = adaptive_hysteresis(fused, fov)
    results['MultiFused_hyst_clean'] = morphological_cleanup(
        results['MultiFused_hyst'], fov, min_vessel_area=4, close_radius=2
    )
    
    # Elbow threshold on C6G6 combined
    results['C6_elbow'] = threshold_with_elbow(res_c6['combined'], fov)
    results['C6_elbow_clean'] = morphological_cleanup(
        results['C6_elbow'], fov, min_vessel_area=4, close_radius=2
    )
    
    # Best P10 baseline for comparison
    results['C6_P10'] = res_c6['P10']
    results['C6_TriClean'] = res_c6['Triangle_clean']
    
    return results
