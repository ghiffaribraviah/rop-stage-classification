from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import frangi, threshold_sauvola
from skimage.morphology import skeletonize

from almeida_workflow_visualization import (
    AGRAWAL_ROOT,
    DEFAULT_OUTPUT,
    IMAGE_EXTENSIONS,
    ZHAO_ROOT,
    add_label,
    agrawal_rows,
    bsgmf_response,
    enhance_cielab,
    estimate_fov_mask,
    estimate_background_mean,
    jerman_vesselness,
    jerman_vesselness_with_scales,
    keep_components_at_least,
    line_kernel,
    modified_tophat,
    normalize01,
    normalize_green_with_background,
    read_binary_mask,
    read_rgb,
    remove_optic_disc_artifact,
    resize_max_side,
    triangle_threshold,
    triangle_threshold_value,
    uint8_image,
)


DEFAULT_THRESHOLD_OUTPUT = DEFAULT_OUTPUT.with_name("debug_threshold_workflow_15_samples.jpg")


def suppress_fov_border(fov: np.ndarray, radius: int | None = None) -> np.ndarray:
    if not np.any(fov):
        return fov.astype(bool)
    fov = fov.astype(bool)
    height, width = fov.shape
    min_side = max(1, min(height, width))
    if radius is None:
        radius = int(round(min_side * 0.075))
        radius = int(np.clip(radius, 18, 60))
    else:
        radius = max(1, int(radius))

    distance = cv2.distanceTransform(fov.astype(np.uint8), cv2.DIST_L2, 5)
    inner = (distance > float(radius)) & fov
    if int(inner.sum()) >= int(0.42 * fov.sum()):
        return inner.astype(bool)

    # Very small or narrow crops should not lose most of their valid area.
    fallback_radius = max(3, int(round(radius * 0.5)))
    inner = (distance > float(fallback_radius)) & fov
    return inner.astype(bool) if np.any(inner) else fov


def exclude_fov_outer_line(fov: np.ndarray, radius: int | None = None) -> np.ndarray:
    if not np.any(fov):
        return fov.astype(bool)
    fov = fov.astype(bool)
    min_side = max(1, min(fov.shape))
    if radius is None:
        radius = int(round(min_side * 0.055))
        radius = int(np.clip(radius, 16, 48))
    distance = cv2.distanceTransform(fov.astype(np.uint8), cv2.DIST_L2, 5)
    inner = (distance > float(max(1, radius))) & fov
    return inner.astype(bool) if int(inner.sum()) >= int(0.55 * fov.sum()) else suppress_fov_border(fov, radius=max(3, radius // 2))


def fov_border_band(fov: np.ndarray, radius: int | None = None) -> np.ndarray:
    if not np.any(fov):
        return np.zeros(fov.shape, dtype=bool)
    fov = fov.astype(bool)
    min_side = max(1, min(fov.shape))
    if radius is None:
        radius = int(round(min_side * 0.105))
        radius = int(np.clip(radius, 28, 90))
    distance = cv2.distanceTransform(fov.astype(np.uint8), cv2.DIST_L2, 5)
    return (distance <= float(max(1, radius))) & fov


def estimate_aperture_rim_mask(rgb: np.ndarray, fov: np.ndarray) -> np.ndarray:
    if not np.any(fov):
        return np.zeros(fov.shape, dtype=bool)

    fov = fov.astype(bool)
    min_side = max(1, min(fov.shape))
    distance = cv2.distanceTransform(fov.astype(np.uint8), cv2.DIST_L2, 5)
    search_radius = int(np.clip(round(min_side * 0.16), 48, 128))
    contact_radius = int(np.clip(round(min_side * 0.035), 10, 32))
    near_border = (distance <= float(search_radius)) & fov

    filled_rgb = fill_outside_fov_nearest(rgb, fov)
    lab = cv2.cvtColor(filled_rgb, cv2.COLOR_RGB2LAB)
    l_channel = lab[:, :, 0].astype(np.float32)
    gx = cv2.Sobel(l_channel, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(l_channel, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt(gx * gx + gy * gy)
    gradient = normalize01(gradient, near_border)

    if not np.any(near_border):
        return np.zeros(fov.shape, dtype=bool)
    threshold = max(0.18, float(np.percentile(gradient[near_border], 82.0)))
    candidates = (gradient >= threshold) & near_border
    candidates = cv2.morphologyEx(
        candidates.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    ).astype(bool)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidates.astype(np.uint8), 8)
    rim = np.zeros(fov.shape, dtype=bool)
    min_area = max(10, int(round(min_side * 0.012)))
    boundary_contact = distance <= float(contact_radius)
    for label in range(1, n_labels):
        component = labels == label
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        if not np.any(component & boundary_contact):
            continue
        component_distance = distance[component]
        if float(component_distance.mean()) > float(search_radius) * 0.40:
            continue
        if float(component_distance.max()) > float(search_radius) * 0.90:
            continue
        rim[component] = True

    dilate_radius = int(np.clip(round(min_side * 0.008), 3, 7))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_radius + 1, 2 * dilate_radius + 1))
    rim = cv2.dilate(rim.astype(np.uint8), kernel, iterations=1).astype(bool)
    return rim & near_border & fov


def fov_edge_weight(fov: np.ndarray, ramp_px: int | None = None, zero_px: int | None = None) -> np.ndarray:
    if not np.any(fov):
        return np.zeros(fov.shape, dtype=np.float32)
    min_side = max(1, min(fov.shape))
    if zero_px is None:
        zero_px = int(np.clip(round(min_side * 0.012), 5, 14))
    if ramp_px is None:
        ramp_px = int(np.clip(round(min_side * 0.025), 10, 24))
    distance = cv2.distanceTransform(fov.astype(np.uint8), cv2.DIST_L2, 5)
    weight = np.clip((distance - float(zero_px)) / float(max(1, ramp_px)), 0.0, 1.0).astype(np.float32)
    weight[~fov.astype(bool)] = 0
    return weight


def fill_outside_fov_nearest(image: np.ndarray, fov: np.ndarray) -> np.ndarray:
    if np.all(fov) or not np.any(fov):
        return image.copy()
    invalid = ~fov.astype(bool)
    _, indices = ndi.distance_transform_edt(invalid, return_indices=True)
    if image.ndim == 2:
        filled = image[indices[0], indices[1]]
    else:
        filled = image[indices[0], indices[1], :]
    return filled.astype(image.dtype, copy=False)


def dark_source_base(source: np.ndarray, fov: np.ndarray) -> np.ndarray:
    dark = 255.0 - source.astype(np.float32)
    dark[~fov] = 0
    dark = response_percentile_floor(dark, fov, floor_pct=35.0, high_pct=99.2, gamma=0.95)
    dark = cv2.GaussianBlur(dark, (0, 0), sigmaX=0.35, sigmaY=0.35, borderType=cv2.BORDER_REFLECT)
    dark[~fov] = 0
    return normalize01(dark, fov)


def local_clahe_channel(
    channel: np.ndarray,
    fov: np.ndarray,
    clip_limit: float = 6.0,
    tile_grid_size: tuple[int, int] = (16, 16),
) -> np.ndarray:
    output = np.clip(channel, 0, 255).astype(np.uint8)
    output = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tile_grid_size).apply(output)
    output[~fov] = 0
    return output


def cielab_green_clahe_source(rgb: np.ndarray, fov: np.ndarray) -> np.ndarray:
    return cielab_green_clahe_source_maps(rgb, fov)["source"]


def percentile_stretch_uint8(
    channel: np.ndarray,
    fov: np.ndarray,
    low_pct: float = 1.0,
    high_pct: float = 99.0,
) -> np.ndarray:
    output = np.zeros(channel.shape, dtype=np.uint8)
    if not np.any(fov):
        return output
    values = channel[fov]
    low, high = np.percentile(values, [float(low_pct), float(high_pct)])
    if high <= low:
        low, high = float(values.min()), float(values.max())
    if high <= low:
        return output
    stretched = np.clip((channel.astype(np.float32) - float(low)) / float(high - low), 0.0, 1.0)
    output[fov] = np.round(stretched[fov] * 255.0).astype(np.uint8)
    return output


def line_structuring_element(length: int, angle_deg: int, width: int = 1) -> np.ndarray:
    length = max(3, int(length) | 1)
    width = max(1, int(width))
    kernel = np.zeros((length, length), dtype=np.uint8)
    center = length // 2
    radians = np.deg2rad(float(angle_deg))
    half = center
    dx = int(round(np.cos(radians) * half))
    dy = int(round(np.sin(radians) * half))
    cv2.line(kernel, (center - dx, center - dy), (center + dx, center + dy), 1, width, cv2.LINE_AA)
    return (kernel > 0).astype(np.uint8)


def multiscale_line_blackhat(channel: np.ndarray, fov: np.ndarray) -> np.ndarray:
    source = np.clip(channel, 0, 255).astype(np.uint8)
    response = np.zeros(source.shape, dtype=np.float32)
    for length, width in ((11, 1), (17, 1), (23, 2)):
        scale_response = np.zeros(source.shape, dtype=np.float32)
        for angle in range(0, 180, 15):
            kernel = line_structuring_element(length, angle, width=width)
            closed = cv2.morphologyEx(source, cv2.MORPH_CLOSE, kernel)
            blackhat = cv2.subtract(closed, source).astype(np.float32)
            scale_response = np.maximum(scale_response, blackhat)
        response = np.maximum(response, scale_response)
    response[~fov] = 0
    response = cv2.medianBlur(percentile_stretch_uint8(response, fov, low_pct=45.0, high_pct=99.5), 3)
    response[~fov] = 0
    return response


def cielab_green_clahe_source_maps(
    rgb: np.ndarray,
    fov: np.ndarray,
    stats_fov: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    stats_fov = fov.astype(bool) if stats_fov is None else stats_fov.astype(bool)
    stats_fov &= fov.astype(bool)
    # Almeida-style preprocessing up to the normalized green image:
    # RGB -> CIELAB L* CLAHE -> enhanced green -> 30x30 mean background correction.
    working = fill_outside_fov_nearest(rgb, fov)
    working_masked = working.copy()
    working_masked[~fov] = 0
    lab = cv2.cvtColor(working, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    l_clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(l_channel)
    enhanced = cv2.cvtColor(cv2.merge([l_clahe, a_channel, b_channel]), cv2.COLOR_LAB2RGB)
    enhanced_masked = enhanced.copy()
    enhanced_masked[~fov] = 0
    green = enhanced[:, :, 1].copy()
    green_display = green.copy()
    green_display[~fov] = 0

    background = estimate_background_mean(green, stats_fov)
    background_uint8 = percentile_stretch_uint8(background, stats_fov, low_pct=1.0, high_pct=99.0)
    source = normalize_green_with_background(green, background, stats_fov)
    if np.any(stats_fov):
        neutral = int(np.median(source[stats_fov]))
        source[~stats_fov] = neutral
    source_filter = fill_outside_fov_nearest(source, stats_fov)
    vessel_blackhat = multiscale_line_blackhat(source_filter, stats_fov)
    source_view = source_filter.copy()
    source_display = source_filter.copy()
    source_display[~stats_fov] = 0
    source_view[~stats_fov] = 0
    lab_l = l_channel.copy()
    lab_l[~fov] = 0
    lab_l_clahe = l_clahe.copy()
    lab_l_clahe[~fov] = 0
    return {
        "bilateral_rgb": working_masked,
        "lab_l": lab_l,
        "lab_l_clahe": lab_l_clahe,
        "lab_enhanced_rgb": enhanced_masked,
        "green": green_display,
        "green_background": background_uint8,
        "green_flattened": source_display,
        "green_denoised": source_display,
        "green_clahe": source_display,
        "vessel_blackhat": vessel_blackhat,
        "source_vessel_view": source_view,
        "source": source_display,
        "source_filter": source_filter,
    }


def masked_median(channel: np.ndarray, fov: np.ndarray) -> float:
    return float(np.median(channel[fov])) if np.any(fov) else 0.0


def fill_outside_fov(channel: np.ndarray, fov: np.ndarray) -> np.ndarray:
    filled = channel.astype(np.float32).copy()
    if np.any(fov):
        filled[~fov] = masked_median(channel, fov)
    return filled


def local_dark_zscore_response(
    channel: np.ndarray,
    fov: np.ndarray,
    kernel_size: int,
    min_std: float = 3.0,
    z_cap: float = 3.0,
) -> np.ndarray:
    filled = fill_outside_fov(channel, fov)
    kernel_size = max(3, int(kernel_size) | 1)
    mean = cv2.blur(filled, (kernel_size, kernel_size), borderType=cv2.BORDER_REFLECT)
    mean_sq = cv2.blur(filled * filled, (kernel_size, kernel_size), borderType=cv2.BORDER_REFLECT)
    std = np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))
    dark = np.maximum(mean - filled, 0.0)
    zscore = dark / np.maximum(std, float(min_std))
    zscore = np.clip(zscore / float(z_cap), 0.0, 1.0)
    zscore[~fov] = 0
    return zscore.astype(np.float32)


def multiscale_local_dark_zscore_response(channel: np.ndarray, fov: np.ndarray) -> np.ndarray:
    maps = [
        local_dark_zscore_response(channel, fov, kernel_size=15, min_std=2.5, z_cap=2.5),
        local_dark_zscore_response(channel, fov, kernel_size=31, min_std=3.0, z_cap=3.0),
        local_dark_zscore_response(channel, fov, kernel_size=51, min_std=4.0, z_cap=3.2),
    ]
    response = np.max(np.stack(maps, axis=0), axis=0)
    response = cv2.medianBlur(uint8_image(response), 3).astype(np.float32) / 255.0
    response = cv2.GaussianBlur(response, (0, 0), sigmaX=0.35, sigmaY=0.35, borderType=cv2.BORDER_REFLECT)
    response[~fov] = 0
    return response_percentile_floor(response, fov, floor_pct=20.0, high_pct=99.2, gamma=1.05)


def matched_filter_response(response: np.ndarray, fov: np.ndarray, size: int, sigma: float) -> np.ndarray:
    response = fill_outside_fov(response, fov)
    output = np.zeros(response.shape, dtype=np.float32)
    for angle in range(0, 180, 15):
        filtered = cv2.filter2D(
            response.astype(np.float32),
            cv2.CV_32F,
            line_kernel(size, angle, sigma),
            borderType=cv2.BORDER_REFLECT,
        )
        output = np.maximum(output, filtered)
    output[~fov] = 0
    return normalize01(output, fov)


def coherence_gate(source_channel: np.ndarray, fov: np.ndarray, sigma: float = 1.4) -> np.ndarray:
    source = fill_outside_fov(source_channel, fov)
    source = cv2.GaussianBlur(source, (0, 0), sigmaX=0.8, sigmaY=0.8, borderType=cv2.BORDER_REFLECT)
    gx = cv2.Sobel(source, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(source, cv2.CV_32F, 0, 1, ksize=3)
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
    trace = jxx + jyy
    delta = np.sqrt(np.maximum((jxx - jyy) ** 2 + 4.0 * jxy * jxy, 0.0))
    gate = delta / (trace + 1e-6)
    gate = cv2.medianBlur(uint8_image(gate), 3).astype(np.float32) / 255.0
    gate[~fov] = 0
    return np.clip(gate, 0.0, 1.0).astype(np.float32)


def response_percentile_floor(
    response: np.ndarray,
    fov: np.ndarray,
    floor_pct: float,
    high_pct: float,
    gamma: float = 1.0,
) -> np.ndarray:
    values = response[fov]
    if values.size == 0:
        return np.zeros(response.shape, dtype=np.float32)
    low, high = np.percentile(values, [float(floor_pct), float(high_pct)])
    if high <= low:
        return normalize01(response, fov)
    output = np.clip((response.astype(np.float32) - float(low)) / float(high - low), 0.0, 1.0)
    output = np.power(output, float(gamma))
    output[~fov] = 0
    return output.astype(np.float32)


def smooth_response_preserve_edges(response: np.ndarray, fov: np.ndarray) -> np.ndarray:
    median = cv2.medianBlur(uint8_image(response), 3).astype(np.float32) / 255.0
    blurred = cv2.GaussianBlur(response.astype(np.float32), (0, 0), sigmaX=0.45, sigmaY=0.45)
    output = normalize01(0.55 * response + 0.25 * median + 0.20 * blurred, fov)
    output[~fov] = 0
    return output.astype(np.float32)


def denoise_confidence_map(
    response: np.ndarray,
    fov: np.ndarray,
    floor_pct: float,
    high_pct: float,
    gamma: float = 1.0,
) -> np.ndarray:
    median = cv2.medianBlur(uint8_image(response), 3).astype(np.float32) / 255.0
    blurred = cv2.GaussianBlur(response.astype(np.float32), (0, 0), sigmaX=0.50, sigmaY=0.50)
    denoised = 0.50 * response + 0.32 * median + 0.18 * blurred
    denoised[~fov] = 0
    return response_percentile_floor(denoised, fov, floor_pct=floor_pct, high_pct=high_pct, gamma=gamma)


def line_supported_response(
    response: np.ndarray,
    support: np.ndarray,
    coherence: np.ndarray,
    fov: np.ndarray,
) -> np.ndarray:
    line_confidence = normalize01(support * (0.35 + 0.65 * coherence), fov)
    gated = response * np.clip(0.18 + 0.82 * line_confidence, 0.0, 1.0)
    return response_percentile_floor(gated, fov, floor_pct=44.0, high_pct=99.3, gamma=1.05)


def connected_support_mask(
    support: np.ndarray,
    coherence: np.ndarray,
    fov: np.ndarray,
    strong_pct: float = 90.0,
    weak_pct: float = 66.0,
) -> np.ndarray:
    if not np.any(fov):
        return np.zeros(support.shape, dtype=bool)

    strong_floor = active_percentile(support, fov, strong_pct)
    weak_floor = active_percentile(support, fov, weak_pct)
    coherence_floor = max(0.04, float(np.percentile(coherence[fov], 35.0)))
    high_support_floor = active_percentile(support, fov, 78.0)
    strong = (support >= strong_floor) & (coherence >= coherence_floor) & fov
    strong = shape_aware_component_clean(strong, min_area=10, min_elongated_area=4, min_axis_ratio=1.7)
    weak = (support >= weak_floor) & ((coherence >= coherence_floor) | (support >= high_support_floor)) & fov

    labels, count = ndi.label(weak)
    if count == 0:
        return strong
    strong_labels = np.unique(labels[strong])
    strong_labels = strong_labels[strong_labels > 0]
    connected = np.isin(labels, strong_labels) & fov
    return shape_aware_component_clean(connected, min_area=16, min_elongated_area=5, min_axis_ratio=1.8)


def connected_support_response(
    support: np.ndarray,
    coherence: np.ndarray,
    fov: np.ndarray,
    strong_pct: float = 90.0,
    weak_pct: float = 66.0,
    floor_pct: float = 45.0,
    high_pct: float = 99.2,
    gamma: float = 0.95,
    soft_floor: float = 0.04,
    soft_gain: float = 0.96,
    sigma: float = 0.65,
) -> np.ndarray:
    mask = connected_support_mask(support, coherence, fov, strong_pct=strong_pct, weak_pct=weak_pct)
    softened = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
    connected = support * np.clip(float(soft_floor) + float(soft_gain) * softened, 0.0, 1.0)
    connected[~fov] = 0
    return response_percentile_floor(connected, fov, floor_pct=floor_pct, high_pct=high_pct, gamma=gamma)


def soft_connected_threshold(response: np.ndarray, fov: np.ndarray) -> np.ndarray:
    values = response[fov]
    if values.size == 0:
        return np.zeros(response.shape, dtype=bool)

    strong_threshold = triangle_threshold_value(response, fov, nbins=4)
    active_values = values[values > max(0.01, float(np.percentile(values, 35.0)))]
    if active_values.size == 0:
        active_values = values
    weak_threshold = max(
        float(strong_threshold) * 0.55,
        float(np.percentile(active_values, 45.0)),
    )
    strong = (response >= float(strong_threshold)) & fov
    weak = (response >= float(weak_threshold)) & fov
    if not np.any(strong):
        return weak.astype(bool)

    labels, count = ndi.label(weak)
    if count == 0:
        return strong.astype(bool)
    strong_labels = np.unique(labels[strong])
    strong_labels = strong_labels[strong_labels > 0]
    connected = np.isin(labels, strong_labels) & fov

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    connected = cv2.morphologyEx(connected.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1).astype(bool)
    connected &= fov
    return connected


def vessel_aware_hysteresis(
    response: np.ndarray,
    bsgmf: np.ndarray,
    top_hat: np.ndarray,
    frangi_map: np.ndarray,
    jerman: np.ndarray,
    fov: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not np.any(fov):
        empty = np.zeros(response.shape, dtype=bool)
        return empty, empty, empty

    evidence = normalize01(
        0.25 * bsgmf.astype(np.float32)
        + 0.15 * top_hat.astype(np.float32)
        + 0.30 * frangi_map.astype(np.float32)
        + 0.30 * jerman.astype(np.float32),
        fov,
    )
    active_values = response[fov]
    active_values = active_values[active_values > max(0.01, float(np.percentile(active_values, 30.0)))]
    if active_values.size == 0:
        active_values = response[fov]

    strong_threshold = max(
        triangle_threshold_value(response, fov, nbins=4) * 0.95,
        float(np.percentile(active_values, 68.0)),
    )
    weak_threshold = max(
        float(strong_threshold) * 0.56,
        float(np.percentile(active_values, 42.0)),
    )
    evidence_floor = max(0.12, float(np.percentile(evidence[fov], 58.0)))
    high_evidence = max(0.22, float(np.percentile(evidence[fov], 76.0)))

    strong = (response >= strong_threshold) & (evidence >= evidence_floor) & fov
    weak = (
        (response >= weak_threshold)
        & ((evidence >= evidence_floor) | ((response >= strong_threshold * 0.82) & (evidence >= high_evidence * 0.70)))
        & fov
    )

    if not np.any(strong):
        return weak.astype(bool), strong.astype(bool), weak.astype(bool)

    labels, count = ndi.label(weak)
    if count == 0:
        return strong.astype(bool), strong.astype(bool), weak.astype(bool)
    strong_labels = np.unique(labels[strong])
    strong_labels = strong_labels[strong_labels > 0]
    mask = np.isin(labels, strong_labels) & fov
    return mask, strong.astype(bool), weak.astype(bool)


def threshold_response_maps(rgb: np.ndarray, fov: np.ndarray) -> dict[str, np.ndarray]:
    object_fov = fov.astype(bool)
    outer_circle = fov_border_band(object_fov)
    analysis_fov = object_fov
    preprocessing_fov = object_fov
    valid_fov = object_fov & ~outer_circle
    source_maps = cielab_green_clahe_source_maps(rgb, object_fov, stats_fov=object_fov)
    source = source_maps["source_filter"]
    inverted = 255 - source
    inverted[~analysis_fov] = 0
    vessel_base = normalize01(inverted.astype(np.float32), analysis_fov)
    vessel_base[~analysis_fov] = 0
    vessel_base_filter = fill_outside_fov(vessel_base, analysis_fov)
    bsgmf = bsgmf_response(vessel_base_filter, analysis_fov)
    bsgmf[~analysis_fov] = 0
    top_hat = modified_tophat(fill_outside_fov(bsgmf, analysis_fov), analysis_fov)
    top_hat[~analysis_fov] = 0
    top_hat_filter = fill_outside_fov(top_hat, analysis_fov)
    frangi_map = normalize01(
        frangi(top_hat_filter.astype(np.float32), sigmas=(1, 3, 5, 7), alpha=0.5, beta=15.0, black_ridges=False),
        valid_fov,
    )
    frangi_map[~valid_fov] = 0
    jerman = jerman_vesselness(top_hat_filter, analysis_fov)
    jerman[~valid_fov] = 0
    bsgmf[~valid_fov] = 0
    top_hat[~valid_fov] = 0
    combined = normalize01(0.70 * frangi_map + 0.30 * jerman, valid_fov)
    combined[~valid_fov] = 0
    border_weight = fov_edge_weight(analysis_fov, ramp_px=42, zero_px=6)
    border_weight[~valid_fov] = 0
    vessel_probability = normalize01(
        (0.78 * combined + 0.12 * bsgmf + 0.10 * top_hat) * border_weight,
        valid_fov,
    )
    vessel_probability[~valid_fov] = 0
    triangle, strong_seeds, weak_candidates = vessel_aware_hysteresis(
        vessel_probability,
        bsgmf,
        top_hat,
        frangi_map,
        jerman,
        valid_fov,
    )
    triangle &= valid_fov
    area_clean = shape_aware_component_clean(
        triangle,
        min_area=28,
        min_elongated_area=5,
        min_axis_ratio=1.8,
        min_skeleton_length=4,
    )
    area_clean &= valid_fov
    od_clean, od_marker = remove_optic_disc_artifact(area_clean, rgb, valid_fov)
    od_clean &= valid_fov
    skeleton = skeletonize(od_clean & valid_fov).astype(bool)
    coherence = coherence_gate(source, valid_fov)
    coherence[~valid_fov] = 0
    support_mask = triangle.astype(bool) & valid_fov
    return {
        **source_maps,
        "source": source,
        "preprocessing_fov": preprocessing_fov,
        "processing_fov": valid_fov,
        "analysis_fov": analysis_fov,
        "border_exclusion": outer_circle,
        "border_cleanup": outer_circle,
        "inverted": inverted,
        "vessel_base": vessel_base,
        "zscore": vessel_base,
        "bsgmf": bsgmf,
        "top_hat": top_hat,
        "matched": bsgmf,
        "matched_small": top_hat,
        "jerman": jerman,
        "jerman_small": jerman,
        "frangi": frangi_map,
        "response_raw": combined,
        "combined": combined,
        "vessel_probability": vessel_probability,
        "soft_mask": vessel_probability,
        "strong_seeds": strong_seeds,
        "weak_candidates": weak_candidates,
        "triangle": triangle,
        "area_clean": area_clean,
        "od_removed": od_clean,
        "od_marker": od_marker,
        "skeleton": skeleton,
        "support_raw": bsgmf,
        "support_connected": top_hat,
        "support_mask": support_mask,
        "support": top_hat,
        "coherence": coherence,
        "thin": top_hat,
        "response": combined,
        "soft_vessel": combined,
        "mask_clean": od_clean,
        "mask_final": od_clean,
    }


def active_response_values(response: np.ndarray, fov: np.ndarray) -> np.ndarray:
    values = response[fov]
    if values.size == 0:
        return values
    floor = max(0.02, float(np.percentile(values, 35.0)))
    active = values[values > floor]
    return active if active.size else values


def active_percentile(response: np.ndarray, fov: np.ndarray, pct: float) -> float:
    values = active_response_values(response, fov)
    if values.size == 0:
        return 1.0
    return float(np.percentile(values, float(pct)))


def percentile_mask(response: np.ndarray, fov: np.ndarray, pct: float) -> np.ndarray:
    values = active_response_values(response, fov)
    if values.size == 0:
        return np.zeros(response.shape, dtype=bool)
    threshold = float(np.percentile(values, float(pct)))
    return ((response >= threshold) & fov).astype(bool)


def shape_aware_component_clean(
    mask: np.ndarray,
    min_area: int = 14,
    min_elongated_area: int = 5,
    min_axis_ratio: float = 2.2,
    min_skeleton_length: int = 4,
) -> np.ndarray:
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    output = np.zeros(mask.shape, dtype=bool)
    for label in range(1, n_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        component = labels == label
        if area >= int(min_area):
            output[component] = True
            continue
        if area < int(min_elongated_area):
            continue
        ys, xs = np.nonzero(component)
        if xs.size < 2:
            continue
        coords = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
        covariance = np.cov(coords, rowvar=False)
        eigvals = np.linalg.eigvalsh(covariance)
        axis_ratio = float(np.sqrt((eigvals.max() + 1e-6) / (eigvals.min() + 1e-6)))
        skeleton_length = int(skeletonize(component).sum())
        if axis_ratio >= float(min_axis_ratio) and skeleton_length >= int(min_skeleton_length):
            output[component] = True
    return output


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    radius = max(1, int(radius))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def component_is_vessel_like(
    component: np.ndarray,
    min_area: int,
    min_skeleton_length: int,
    min_axis_ratio: float,
) -> bool:
    area = int(component.sum())
    if area >= int(min_area):
        return True
    if area < 3:
        return False
    ys, xs = np.nonzero(component)
    if xs.size < 2:
        return False
    coords = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
    covariance = np.cov(coords, rowvar=False)
    eigvals = np.linalg.eigvalsh(covariance)
    axis_ratio = float(np.sqrt((eigvals.max() + 1e-6) / (eigvals.min() + 1e-6)))
    major_axis = float(4.0 * np.sqrt(max(float(eigvals.max()), 0.0)))
    return axis_ratio >= float(min_axis_ratio) and major_axis >= int(min_skeleton_length)


def prune_tree_growth(
    grown: np.ndarray,
    seed: np.ndarray,
    response: np.ndarray,
    support: np.ndarray,
    fov: np.ndarray,
) -> np.ndarray:
    output = seed.astype(bool).copy() & fov
    proposed = grown.astype(bool) & ~output & fov
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(proposed.astype(np.uint8), 8)
    seed_contact = dilate_mask(output, 2)
    for label in range(1, labels_count):
        component = labels == label
        area = int(stats[label, cv2.CC_STAT_AREA])
        if not np.any(component & seed_contact):
            continue
        component_response = float(response[component].mean()) if area else 0.0
        component_support = float(support[component].mean()) if area else 0.0
        vessel_like = component_is_vessel_like(component, min_area=6, min_skeleton_length=3, min_axis_ratio=1.6)
        high_confidence = component_response >= 0.18 and component_support >= 0.14
        if vessel_like or high_confidence:
            output[component] = True
    return output & fov


def connected_hysteresis_mask(
    response: np.ndarray,
    fov: np.ndarray,
    high_pct: float,
    low_pct: float,
    min_area: int,
    support: np.ndarray | None = None,
    support_floor: float = 0.0,
    support_pct_floor: float = 35.0,
) -> np.ndarray:
    values = active_response_values(response, fov)
    if values.size == 0:
        return np.zeros(response.shape, dtype=bool)
    high = float(np.percentile(values, float(high_pct)))
    low = float(np.percentile(values, float(low_pct)))
    gate = fov.copy()
    if support is not None and support_floor > 0:
        adaptive_support_floor = active_percentile(support, fov, float(support_pct_floor))
        gate &= support >= max(float(support_floor), adaptive_support_floor)
    strong = (response >= high) & gate
    weak = (response >= low) & gate
    labels, count = ndi.label(weak)
    if count == 0:
        return strong.astype(bool)
    strong_labels = np.unique(labels[strong])
    strong_labels = strong_labels[strong_labels > 0]
    connected = np.isin(labels, strong_labels) & fov
    return shape_aware_component_clean(connected, min_area=int(min_area))


def clean_balanced_noise(
    mask: np.ndarray,
    response: np.ndarray,
    support: np.ndarray,
    coherence: np.ndarray,
    fov: np.ndarray,
) -> np.ndarray:
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n_labels <= 1:
        return mask.astype(bool) & fov

    areas = stats[:, cv2.CC_STAT_AREA]
    large = np.zeros(mask.shape, dtype=bool)
    for label in range(1, n_labels):
        if int(areas[label]) >= 35:
            large[labels == label] = True
    near_large = dilate_mask(large, 8) if np.any(large) else np.zeros(mask.shape, dtype=bool)
    response_floor = active_percentile(response, fov, 48.0)
    support_floor = active_percentile(support, fov, 42.0)
    coherence_floor = max(0.05, float(np.percentile(coherence[fov], 35.0))) if np.any(fov) else 0.05

    cleaned = np.zeros(mask.shape, dtype=bool)
    for label in range(1, n_labels):
        component = labels == label
        area = int(areas[label])
        response_mean = float(response[component].mean()) if area else 0.0
        support_mean = float(support[component].mean()) if area else 0.0
        coherence_mean = float(coherence[component].mean()) if area else 0.0
        if area >= 35:
            cleaned[component] = True
            continue
        elongated = component_is_vessel_like(component, min_area=12, min_skeleton_length=5, min_axis_ratio=1.9)
        near_tree = bool(np.any(component & near_large))
        has_signal = response_mean >= response_floor and support_mean >= support_floor
        has_line_signal = support_mean >= support_floor and coherence_mean >= coherence_floor
        if elongated and (has_signal or (near_tree and has_line_signal)):
            cleaned[component] = True
    return cleaned & fov


def line_mask_between(shape: tuple[int, int], start: tuple[int, int], end: tuple[int, int]) -> np.ndarray:
    line = np.zeros(shape, dtype=np.uint8)
    cv2.line(line, (int(start[1]), int(start[0])), (int(end[1]), int(end[0])), 1, 1)
    return line.astype(bool)


def endpoint_connection_candidates(mask: np.ndarray) -> list[dict[str, object]]:
    skeleton = skeletonize(mask.astype(bool))
    endpoints = skeleton_endpoints(mask)
    labels_count, labels, _, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    coords = np.column_stack(np.nonzero(endpoints))
    candidates: list[dict[str, object]] = []
    for index, (y, x) in enumerate(coords):
        label = int(labels[int(y), int(x)])
        if label <= 0:
            continue
        direction = endpoint_outward_direction(skeleton, int(y), int(x))
        if direction is None:
            continue
        candidates.append({"index": index, "y": int(y), "x": int(x), "label": label, "direction": direction})
    return candidates


def connect_clean_endpoints(
    mask: np.ndarray,
    response: np.ndarray,
    support: np.ndarray,
    fov: np.ndarray,
    max_gap: int = 13,
) -> np.ndarray:
    endpoints = endpoint_connection_candidates(mask)
    if len(endpoints) < 2:
        return mask.astype(bool) & fov
    response_floor = active_percentile(response, fov, 38.0)
    support_floor = active_percentile(support, fov, 30.0)
    pairs: list[tuple[float, int, int, np.ndarray]] = []
    for i, first in enumerate(endpoints):
        y0, x0 = int(first["y"]), int(first["x"])
        dy0, dx0 = first["direction"]  # type: ignore[misc]
        for j in range(i + 1, len(endpoints)):
            second = endpoints[j]
            if int(first["label"]) == int(second["label"]):
                continue
            y1, x1 = int(second["y"]), int(second["x"])
            gap = float(np.hypot(y1 - y0, x1 - x0))
            if gap < 3.0 or gap > float(max_gap):
                continue
            dy1, dx1 = second["direction"]  # type: ignore[misc]
            vec_y = (y1 - y0) / gap
            vec_x = (x1 - x0) / gap
            align_first = dy0 * vec_y + dx0 * vec_x
            align_second = dy1 * (-vec_y) + dx1 * (-vec_x)
            if align_first < 0.45 or align_second < 0.25:
                continue
            bridge = line_mask_between(mask.shape, (y0, x0), (y1, x1)) & fov
            gap_pixels = bridge & ~mask
            if not np.any(gap_pixels):
                continue
            response_mean = float(response[gap_pixels].mean())
            support_mean = float(support[gap_pixels].mean())
            if response_mean < response_floor and support_mean < support_floor:
                continue
            score = gap - 2.0 * min(align_first, align_second) - response_mean - support_mean
            pairs.append((score, i, j, bridge))

    connected = mask.astype(bool).copy()
    used_endpoints: set[int] = set()
    for _, i, j, bridge in sorted(pairs, key=lambda item: item[0]):
        if i in used_endpoints or j in used_endpoints:
            continue
        connected |= bridge
        used_endpoints.add(i)
        used_endpoints.add(j)
    return connected & fov


def select_endpoint_aligned_fragments(
    seed: np.ndarray,
    candidate_mask: np.ndarray,
    response: np.ndarray,
    support: np.ndarray,
    fov: np.ndarray,
    max_distance: int = 14,
) -> np.ndarray:
    endpoints = endpoint_connection_candidates(seed)
    if not endpoints:
        return np.zeros(seed.shape, dtype=bool)

    candidate = candidate_mask.astype(bool) & ~dilate_mask(seed, 1) & fov
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate.astype(np.uint8), 8)
    selected = np.zeros(seed.shape, dtype=bool)
    response_floor = active_percentile(response, fov, 45.0)
    support_floor = active_percentile(support, fov, 36.0)
    for label in range(1, n_labels):
        component = labels == label
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 4:
            continue
        if not component_is_vessel_like(component, min_area=10, min_skeleton_length=5, min_axis_ratio=1.8):
            continue
        response_mean = float(response[component].mean())
        support_mean = float(support[component].mean())
        if response_mean < response_floor and support_mean < support_floor:
            continue
        ys, xs = np.nonzero(component)
        aligned = False
        for endpoint in endpoints:
            ey, ex = int(endpoint["y"]), int(endpoint["x"])
            dy, dx = endpoint["direction"]  # type: ignore[misc]
            distances = np.hypot(ys - ey, xs - ex)
            nearest_idx = int(np.argmin(distances))
            if float(distances[nearest_idx]) > float(max_distance):
                continue
            vec_y = (float(ys[nearest_idx]) - ey) / max(float(distances[nearest_idx]), 1e-6)
            vec_x = (float(xs[nearest_idx]) - ex) / max(float(distances[nearest_idx]), 1e-6)
            if dy * vec_y + dx * vec_x >= 0.35:
                aligned = True
                break
        if aligned:
            selected[component] = True
    return selected & fov


def select_thin_components_near_tree(
    seed: np.ndarray,
    thin_response: np.ndarray,
    support: np.ndarray,
    coherence: np.ndarray,
    fov: np.ndarray,
    response_pct: float = 84.0,
    support_pct: float = 38.0,
    near_radius: int = 10,
) -> np.ndarray:
    response_floor = active_percentile(thin_response, fov, response_pct)
    support_floor = active_percentile(support, fov, support_pct)
    coherence_floor = max(0.06, float(np.percentile(coherence[fov], 35.0))) if np.any(fov) else 0.06
    candidate = (
        (thin_response >= response_floor)
        & (support >= support_floor)
        & ((coherence >= coherence_floor) | (support >= active_percentile(support, fov, 55.0)))
        & fov
    )
    near_tree = dilate_mask(seed, near_radius)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate.astype(np.uint8), 8)
    selected = np.zeros(seed.shape, dtype=bool)
    for label in range(1, n_labels):
        component = labels == label
        if not np.any(component & near_tree):
            continue
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 4:
            continue
        thin_mean = float(thin_response[component].mean())
        support_mean = float(support[component].mean())
        coherence_mean = float(coherence[component].mean())
        vessel_like = component_is_vessel_like(component, min_area=12, min_skeleton_length=5, min_axis_ratio=1.8)
        high_confidence = thin_mean >= response_floor and support_mean >= max(0.08, support_floor)
        line_confidence = support_mean >= max(0.10, support_floor) and coherence_mean >= coherence_floor
        if vessel_like and (high_confidence or line_confidence):
            selected[component] = True
    return selected & fov


def balanced_plus_thin_variants(maps: dict[str, np.ndarray], seed: np.ndarray, fov: np.ndarray) -> list[tuple[str, np.ndarray]]:
    thin = maps["thin"]
    support = maps["support"]
    coherence = maps["coherence"]
    thin_strict = (
        (thin >= active_percentile(thin, fov, 93.0))
        & (support >= active_percentile(support, fov, 50.0))
        & fov
    )
    thin_strict = shape_aware_component_clean(thin_strict, min_area=8, min_elongated_area=4, min_axis_ratio=1.7)
    thin_near = select_thin_components_near_tree(
        seed, thin, support, coherence, fov, response_pct=89.0, support_pct=45.0, near_radius=9
    )
    combined = (seed | thin_near) & fov
    thin_near_strict = select_thin_components_near_tree(
        seed,
        thin,
        support,
        coherence,
        fov,
        response_pct=92.0,
        support_pct=52.0,
        near_radius=8,
    )
    combined_strict = (seed | thin_near_strict) & fov
    return [
        ("Thin strict", thin_strict),
        ("Thin near-tree", thin_near),
        ("Balanced+thin", combined),
        ("Thin near strict", thin_near_strict),
        ("Bal+thin strict", combined_strict),
    ]


def skeleton_endpoints(mask: np.ndarray) -> np.ndarray:
    skeleton = skeletonize(mask.astype(bool))
    neighbor_kernel = np.ones((3, 3), dtype=np.uint8)
    neighbor_count = cv2.filter2D(skeleton.astype(np.uint8), -1, neighbor_kernel, borderType=cv2.BORDER_CONSTANT)
    return skeleton & (neighbor_count == 2)


def endpoint_outward_direction(skeleton: np.ndarray, y: int, x: int, radius: int = 7) -> tuple[float, float] | None:
    y0, y1 = max(0, y - radius), min(skeleton.shape[0], y + radius + 1)
    x0, x1 = max(0, x - radius), min(skeleton.shape[1], x + radius + 1)
    yy, xx = np.nonzero(skeleton[y0:y1, x0:x1])
    if yy.size < 2:
        return None
    coords = np.column_stack([xx + x0, yy + y0]).astype(np.float32)
    distances = np.hypot(coords[:, 0] - x, coords[:, 1] - y)
    coords = coords[distances > 0]
    if coords.size == 0:
        return None
    inner = coords[np.argsort(distances[distances > 0])[: min(6, coords.shape[0])]]
    inward = inner.mean(axis=0) - np.array([x, y], dtype=np.float32)
    norm = float(np.hypot(inward[0], inward[1]))
    if norm <= 1e-6:
        return None
    outward = -inward / norm
    return float(outward[1]), float(outward[0])


def bridge_endpoint_gaps(seed: np.ndarray, candidate: np.ndarray, fov: np.ndarray, max_gap: int = 10) -> np.ndarray:
    bridged = seed.astype(bool).copy()
    skeleton = skeletonize(seed.astype(bool))
    endpoints = skeleton_endpoints(seed)
    endpoint_coords = np.column_stack(np.nonzero(endpoints))
    for y, x in endpoint_coords:
        direction = endpoint_outward_direction(skeleton, int(y), int(x))
        if direction is None:
            continue
        dy, dx = direction
        best: tuple[int, int] | None = None
        for step in range(3, int(max_gap) + 1):
            cy = int(round(float(y) + dy * step))
            cx = int(round(float(x) + dx * step))
            if cy < 0 or cy >= seed.shape[0] or cx < 0 or cx >= seed.shape[1]:
                break
            y0, y1 = max(0, cy - 2), min(seed.shape[0], cy + 3)
            x0, x1 = max(0, cx - 2), min(seed.shape[1], cx + 3)
            hits = np.column_stack(np.nonzero(candidate[y0:y1, x0:x1] & ~bridged[y0:y1, x0:x1]))
            if hits.size:
                hit = hits[0]
                best = (int(hit[0] + y0), int(hit[1] + x0))
                break
        if best is None:
            continue
        line = np.zeros(seed.shape, dtype=np.uint8)
        cv2.line(line, (int(x), int(y)), (best[1], best[0]), 1, 1)
        bridged |= (line.astype(bool) & fov)
    return bridged & fov


def iterative_tree_growth_mask(
    seed: np.ndarray,
    response: np.ndarray,
    support: np.ndarray,
    coherence: np.ndarray,
    fov: np.ndarray,
    bridge_gaps: bool = False,
    iterations: int = 7,
) -> np.ndarray:
    weak_response = active_percentile(response, fov, 76.0)
    weak_support = active_percentile(support, fov, 38.0)
    coherence_floor = max(0.08, float(np.percentile(coherence[fov], 40.0))) if np.any(fov) else 0.08
    support_strong = active_percentile(support, fov, 60.0)
    candidate = (
        (response >= weak_response)
        & (support >= weak_support)
        & ((coherence >= coherence_floor) | (support >= support_strong))
        & fov
    )
    accepted = seed.astype(bool).copy() & fov
    if bridge_gaps:
        accepted = bridge_endpoint_gaps(accepted, candidate, fov, max_gap=10)
    for _ in range(int(iterations)):
        frontier = dilate_mask(accepted, 1) & ~accepted
        proposed = candidate & frontier
        if not np.any(proposed):
            break
        grown = accepted | proposed
        pruned = prune_tree_growth(grown, seed=seed, response=response, support=support, fov=fov)
        new_pixels = pruned & ~accepted
        if int(new_pixels.sum()) < 3:
            break
        accepted = pruned
    return prune_tree_growth(accepted, seed=seed, response=response, support=support, fov=fov)


def local_adaptive_mask(response: np.ndarray, support: np.ndarray, fov: np.ndarray) -> np.ndarray:
    sauvola = threshold_sauvola(response.astype(np.float32), window_size=41, k=0.05, r=0.5)
    global_floor = active_percentile(response, fov, 78.0)
    support_floor = active_percentile(support, fov, 48.0)
    mask = (response >= np.maximum(sauvola + 0.03, global_floor)) & (support >= support_floor) & fov
    return shape_aware_component_clean(mask, min_area=14)


def build_threshold_variants(maps: dict[str, np.ndarray], fov: np.ndarray) -> list[tuple[str, np.ndarray]]:
    response = maps["response"]
    support = maps["support"]
    p94 = shape_aware_component_clean(percentile_mask(response, fov, 94.0), min_area=18)
    p97 = shape_aware_component_clean(percentile_mask(response, fov, 97.0), min_area=10)
    hyst_loose = connected_hysteresis_mask(
        response, fov, high_pct=96.0, low_pct=82.0, min_area=10, support=support, support_floor=0.08
    )
    hyst_balanced = connected_hysteresis_mask(
        response, fov, high_pct=97.0, low_pct=87.0, min_area=12, support=support, support_floor=0.12
    )
    hyst_strict = connected_hysteresis_mask(
        response, fov, high_pct=98.0, low_pct=91.0, min_area=18, support=support, support_floor=0.16
    )
    hyst_clean = clean_balanced_noise(hyst_balanced, response, support, maps["coherence"], fov)
    clean_connected = connect_clean_endpoints(hyst_clean, response, support, fov)
    loose_clean = clean_balanced_noise(hyst_loose, response, support, maps["coherence"], fov)
    weak_fragments = select_endpoint_aligned_fragments(hyst_clean, loose_clean, response, support, fov)
    weak_connected = connect_clean_endpoints(hyst_clean | weak_fragments, response, support, fov)
    adaptive = local_adaptive_mask(response, support, fov)
    vessel_first_response = normalize01(response * (0.40 + 0.60 * support), fov)
    vessel_first = connected_hysteresis_mask(
        vessel_first_response, fov, 96.0, 84.0, 10, support=support, support_floor=0.10
    )
    variants = [
        ("P94+shape", p94),
        ("P97+shape", p97),
        ("Hyst loose", hyst_loose),
        ("Hyst balanced", hyst_balanced),
        ("Hyst strict", hyst_strict),
        ("Balanced clean", hyst_clean),
        ("Clean connect", clean_connected),
        ("Weak fragments", weak_fragments),
        ("Weak connect", weak_connected),
    ]
    variants.extend(balanced_plus_thin_variants(maps, hyst_balanced, fov))
    variants.extend([
        ("Adaptive", adaptive),
        ("Resp+support", vessel_first),
    ])
    return variants


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, fov: np.ndarray) -> np.ndarray:
    base = rgb.copy()
    base[~fov] = 0
    overlay = base.copy()
    overlay[mask] = (0, 255, 120)
    return cv2.addWeighted(base, 0.62, overlay, 0.38, 0)


def zhao_rows(root: Path, limit: int) -> list[tuple[str, Path, Path | None]]:
    if limit <= 0 or not root.is_dir():
        return []
    classes = [
        path
        for path in sorted(root.iterdir())
        if path.is_dir() and path.name.lower() != "laser scars"
    ]
    class_files = {
        klass.name: sorted(path for path in klass.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
        for klass in classes
    }
    rows: list[tuple[str, Path, Path | None]] = []
    round_idx = 0
    while len(rows) < limit:
        added = False
        for _, files in class_files.items():
            if round_idx < len(files):
                path = files[round_idx]
                rows.append((f"Zhao {path.name}", path, None))
                added = True
                if len(rows) >= limit:
                    break
        if not added:
            break
        round_idx += 1
    return rows


def build_sheet(rows: list[tuple[str, Path, Path | None]], output_path: Path, tile_size: int, max_side: int) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet_rows = []
    for index, (name, image_path, mask_path) in enumerate(rows, start=1):
        rgb = resize_max_side(read_rgb(image_path), max_side)
        fov = estimate_fov_mask(rgb)
        gt = read_binary_mask(mask_path, fov.shape)
        maps = threshold_response_maps(rgb, fov)
        fov = maps["processing_fov"]
        variants = build_threshold_variants(maps, fov)
        display_maps: list[tuple[str, np.ndarray]] = [
            (f"{index}. {name}", rgb),
            ("FOV", fov),
            ("GT", gt),
            ("Almeida norm src", maps["source"]),
            ("Z response", maps["zscore"]),
            ("Support raw", maps["support_raw"]),
            ("Support conn", maps["support_connected"]),
            ("Support mask", maps["support_mask"]),
            ("Coherence", maps["coherence"]),
            ("Final resp", maps["response"]),
            ("Thin resp", maps["thin"]),
        ]
        display_maps.extend(variants)
        display_maps.append(("Overlay weak", overlay_mask(rgb, variants[8][1], fov)))
        row_tiles = [add_label(image, label, tile_size) for label, image in display_maps]
        sheet_rows.append(np.concatenate(row_tiles, axis=1))
    sheet = np.concatenate(sheet_rows, axis=0)
    cv2.imwrite(str(output_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate visual thresholding variants for retinal vessel maps.")
    parser.add_argument("--output", type=Path, default=DEFAULT_THRESHOLD_OUTPUT)
    parser.add_argument("--tile-size", type=int, default=150)
    parser.add_argument("--max-side", type=int, default=768)
    parser.add_argument("--retcam-count", type=int, default=5)
    parser.add_argument("--neo-count", type=int, default=5)
    parser.add_argument("--zhao-count", type=int, default=5)
    args = parser.parse_args()

    rows = agrawal_rows(args.retcam_count, args.neo_count)
    rows.extend(zhao_rows(ZHAO_ROOT, args.zhao_count))
    if not rows:
        raise RuntimeError(f"No debug images found under {AGRAWAL_ROOT} or {ZHAO_ROOT}")
    print(build_sheet(rows, args.output, tile_size=args.tile_size, max_side=args.max_side))


if __name__ == "__main__":
    main()
