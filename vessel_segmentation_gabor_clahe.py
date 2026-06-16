from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize


@dataclass(frozen=True)
class GaborClaheConfig:
    target_density: float = 0.115
    main_low_mult: float = 1.45
    main_high_mult: float = 0.60
    residual_enabled: bool = True
    residual_low_mult: float = 1.10
    recovery_axis_ratio: float = 2.6
    recovery_skeleton_length: int | None = 18
    recovery_branch_density: float = 0.10


BEST_GABOR_CLAHE_CONFIG = GaborClaheConfig()


def resize_max_side(image: np.ndarray, max_side: int, interpolation: int = cv2.INTER_AREA) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(1.0, float(max_side) / float(max(height, width)))
    if scale >= 1.0:
        return image.copy()
    new_size = (int(round(width * scale)), int(round(height * scale)))
    return cv2.resize(image, new_size, interpolation=interpolation)


def normalize01(image: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    values = image[mask] if mask is not None and np.any(mask) else image.reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.zeros(image.shape, dtype=np.float32)
    low = float(np.percentile(values, 1.0))
    high = float(np.percentile(values, 99.0))
    if high <= low:
        low = float(values.min())
        high = float(values.max())
    if high <= low:
        return np.zeros(image.shape, dtype=np.float32)
    output = np.clip((image.astype(np.float32) - low) / float(high - low), 0.0, 1.0)
    if mask is not None:
        output = output.copy()
        output[~mask.astype(bool)] = 0
    return output.astype(np.float32)


def uint8_image(image: np.ndarray) -> np.ndarray:
    if image.dtype == bool:
        image = image.astype(np.uint8) * 255
    elif np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0) * 255.0
    return np.clip(image, 0, 255).astype(np.uint8)


def estimate_fov_mask(rgb: np.ndarray) -> np.ndarray:
    max_channel = rgb.max(axis=2)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    visible = (max_channel > 8) | (gray > 6)

    min_side = max(1, min(rgb.shape[:2]))
    close_radius = max(5, int(round(min_side * 0.015)))
    open_radius = max(2, int(round(min_side * 0.004)))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_radius + 1, 2 * close_radius + 1))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * open_radius + 1, 2 * open_radius + 1))
    mask = cv2.morphologyEx(visible.astype(np.uint8), cv2.MORPH_CLOSE, close_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel).astype(bool)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n_labels > 1:
        largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = labels == largest_label
    mask = ndi.binary_fill_holes(mask)

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        smoothed = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(smoothed, [largest], -1, 1, thickness=cv2.FILLED)
        mask = ndi.binary_fill_holes(smoothed.astype(bool))
    return mask.astype(bool)


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


def fov_outline_subtraction_mask(fov: np.ndarray) -> np.ndarray:
    if not np.any(fov):
        return np.zeros(fov.shape, dtype=bool)

    fov = fov.astype(bool)
    min_side = max(1, min(fov.shape))
    thickness = int(np.clip(round(min_side * 0.034), 12, 26))
    outline = np.zeros(fov.shape, dtype=np.uint8)
    contours, _ = cv2.findContours(fov.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros(fov.shape, dtype=bool)
    cv2.drawContours(outline, contours, contourIdx=-1, color=1, thickness=thickness, lineType=cv2.LINE_AA)
    return outline.astype(bool) & fov


def clahe_channel(channel: np.ndarray, clip_limit: float, tile_grid_size: tuple[int, int]) -> np.ndarray:
    source = np.clip(channel, 0, 255).astype(np.uint8)
    return cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tile_grid_size).apply(source)


def percentile_stretch_uint8(
    channel: np.ndarray,
    fov: np.ndarray,
    low_pct: float = 0.5,
    high_pct: float = 99.5,
) -> np.ndarray:
    output = np.zeros(channel.shape, dtype=np.uint8)
    if not np.any(fov):
        return output
    values = channel[fov]
    low, high = np.percentile(values, [float(low_pct), float(high_pct)])
    if high <= low:
        low = float(values.min())
        high = float(values.max())
    if high <= low:
        return output
    stretched = np.clip((channel.astype(np.float32) - float(low)) / float(high - low), 0.0, 1.0)
    output[fov] = np.round(stretched[fov] * 255.0).astype(np.uint8)
    return output


def boost_small_dark_vessels(green: np.ndarray, fov: np.ndarray) -> np.ndarray:
    if not np.any(fov):
        return np.zeros(green.shape, dtype=np.uint8)

    source = green.astype(np.float32)
    median_value = float(np.median(source[fov]))
    source[~fov] = median_value

    min_side = max(1, min(green.shape))
    background_sigma = float(np.clip(round(min_side * 0.035), 12, 32))
    background = cv2.GaussianBlur(source, (0, 0), sigmaX=background_sigma, sigmaY=background_sigma)
    background = np.maximum(background, 1.0)
    flattened = source * median_value / background

    local_background = cv2.GaussianBlur(flattened, (0, 0), sigmaX=2.0, sigmaY=2.0)
    local_dark = np.maximum(local_background - flattened, 0.0)
    local_dark = cv2.GaussianBlur(local_dark, (0, 0), sigmaX=0.35, sigmaY=0.35)
    local_dark_norm = normalize01(local_dark, fov)

    scale_maps = []
    for sigma in (1.4, 2.2, 3.6, 5.5, 8.0):
        scale_background = cv2.GaussianBlur(flattened, (0, 0), sigmaX=sigma, sigmaY=sigma)
        scale_dark = np.maximum(scale_background - flattened, 0.0)
        scale_dark = cv2.GaussianBlur(scale_dark, (0, 0), sigmaX=0.35, sigmaY=0.35)
        scale_maps.append(normalize01(scale_dark, fov))
    multiscale_dark = np.max(np.stack(scale_maps, axis=0), axis=0)
    multiscale_dark = cv2.medianBlur(uint8_image(multiscale_dark), 3).astype(np.float32) / 255.0
    multiscale_dark[~fov] = 0

    vessel_boost = normalize01(0.35 * local_dark_norm + 0.65 * multiscale_dark, fov)
    boosted = flattened - 42.0 * vessel_boost
    boosted = cv2.GaussianBlur(boosted, (0, 0), sigmaX=0.25, sigmaY=0.25)
    boosted_uint8 = percentile_stretch_uint8(boosted, fov, low_pct=0.7, high_pct=99.3)
    boosted_uint8[~fov] = 0
    return boosted_uint8


def gabor_response(
    vessel_input: np.ndarray,
    fov: np.ndarray,
    wavelengths: tuple[float, ...] = (8.0, 12.0, 16.0),
    sigma: float = 4.0,
    gamma: float = 0.50,
) -> np.ndarray:
    source = vessel_input.astype(np.float32) / 255.0
    response = np.zeros(source.shape, dtype=np.float32)

    for wavelength in wavelengths:
        ksize = int(max(17, round(wavelength * 2.6))) | 1
        for angle in range(0, 180, 15):
            kernel = cv2.getGaborKernel(
                (ksize, ksize),
                sigma=float(sigma),
                theta=np.deg2rad(float(angle)),
                lambd=float(wavelength),
                gamma=float(gamma),
                psi=0.0,
                ktype=cv2.CV_32F,
            )
            kernel -= float(kernel.mean())
            norm = float(np.sum(np.abs(kernel)))
            if norm > 0:
                kernel /= norm
            filtered = cv2.filter2D(source, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT)
            response = np.maximum(response, filtered)

    response = np.maximum(response, 0.0)
    response[~fov] = 0
    return normalize01(response, fov)


def hysteresis_density_threshold(
    response: np.ndarray,
    fov: np.ndarray,
    target_density: float,
    low_mult: float = 1.28,
    high_mult: float = 0.50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = response[fov]
    if values.size == 0:
        empty = np.zeros(response.shape, dtype=bool)
        return empty, empty, empty

    low_density = float(np.clip(target_density * float(low_mult), 0.015, 0.20))
    high_density = float(np.clip(target_density * float(high_mult), 0.008, low_density * 0.85))
    low_threshold = float(np.percentile(values, 100.0 * (1.0 - low_density)))
    high_threshold = float(np.percentile(values, 100.0 * (1.0 - high_density)))

    low_mask = ((response >= low_threshold) & fov).astype(bool)
    high_mask = ((response >= high_threshold) & fov).astype(bool)
    if not np.any(low_mask) or not np.any(high_mask):
        return high_mask, high_mask, low_mask

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(low_mask.astype(np.uint8), 8)
    if n_labels <= 1:
        return low_mask, high_mask, low_mask

    min_side = max(1, min(response.shape))
    min_area = int(np.clip(round(min_side * min_side * 0.000035), 8, 28))
    seed_labels = set(np.unique(labels[high_mask]).tolist())
    seed_labels.discard(0)
    keep_labels = {label for label in seed_labels if int(stats[label, cv2.CC_STAT_AREA]) >= min_area}
    if not keep_labels:
        return high_mask, high_mask, low_mask

    mask = np.isin(labels, list(keep_labels)) & fov
    return mask.astype(bool), high_mask, low_mask


def keep_largest_components(mask: np.ndarray, count: int = 2) -> np.ndarray:
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n_labels <= 1:
        return mask.astype(bool)
    areas = [(int(stats[label, cv2.CC_STAT_AREA]), label) for label in range(1, n_labels)]
    keep_labels = {label for _, label in sorted(areas, reverse=True)[: int(count)]}
    return np.isin(labels, list(keep_labels))


def component_axis_ratio(component: np.ndarray) -> float:
    ys, xs = np.nonzero(component)
    if len(xs) < 3:
        return 1.0
    coords = np.column_stack((xs.astype(np.float32), ys.astype(np.float32)))
    coords -= coords.mean(axis=0, keepdims=True)
    covariance = np.cov(coords, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(covariance)
    minor = max(float(eigenvalues[0]), 1e-3)
    major = max(float(eigenvalues[-1]), minor)
    return float(np.sqrt(major / minor))


def recover_vessel_like_components(
    candidates: np.ndarray,
    response: np.ndarray,
    fov: np.ndarray,
    axis_ratio_min: float = 2.3,
    skeleton_length_min: int | None = None,
    branch_density_max: float = 0.12,
) -> np.ndarray:
    candidates = candidates.astype(bool) & fov
    if not np.any(candidates):
        return np.zeros(candidates.shape, dtype=bool)

    values = response[fov]
    if values.size == 0:
        return np.zeros(candidates.shape, dtype=bool)
    p45 = float(np.percentile(values, 45.0))
    p82 = float(np.percentile(values, 82.0))
    p93 = float(np.percentile(values, 93.0))

    min_side = max(1, min(candidates.shape))
    min_area = int(np.clip(round(min_side * min_side * 0.000025), 8, 24))
    if skeleton_length_min is None:
        skeleton_length_min = int(np.clip(round(min_side * 0.026), 14, 28))
    neighbor_kernel = np.ones((3, 3), dtype=np.uint8)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidates.astype(np.uint8), 8)
    recovered = np.zeros(candidates.shape, dtype=bool)
    for label in range(1, n_labels):
        component = labels == label
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        component_values = response[component]
        if component_values.size == 0:
            continue
        mean_response = float(component_values.mean())
        max_response = float(component_values.max())
        if component_axis_ratio(component) < float(axis_ratio_min):
            continue

        skeleton = skeletonize(component)
        skeleton_length = int(skeleton.sum())
        if skeleton_length < int(skeleton_length_min):
            continue
        neighbor_count = cv2.filter2D(skeleton.astype(np.uint8), -1, neighbor_kernel, borderType=cv2.BORDER_CONSTANT)
        branch_points = int(np.count_nonzero(skeleton & (neighbor_count >= 5)))
        branch_density = branch_points / float(max(1, skeleton_length))
        if branch_density > float(branch_density_max):
            continue

        strong_enough = (mean_response >= p45 and max_response >= p82) or max_response >= p93
        if strong_enough:
            recovered |= component

    return recovered & fov


def segment_soft_response(
    soft_response: np.ndarray,
    valid_fov: np.ndarray,
    fov_outline_subtraction: np.ndarray,
    config: GaborClaheConfig,
) -> dict[str, np.ndarray]:
    valid_fov = valid_fov.astype(bool)
    fov_outline_subtraction = fov_outline_subtraction.astype(bool) & valid_fov

    raw_mask, _, _ = hysteresis_density_threshold(
        soft_response,
        valid_fov,
        target_density=config.target_density,
        low_mult=config.main_low_mult,
        high_mult=config.main_high_mult,
    )
    raw_mask &= ~fov_outline_subtraction
    first_largest = keep_largest_components(raw_mask, count=2) & valid_fov
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    threshold1_final = cv2.morphologyEx(first_largest.astype(np.uint8), cv2.MORPH_CLOSE, close_kernel, iterations=1).astype(bool)
    threshold1_final &= valid_fov
    threshold1_final &= ~fov_outline_subtraction

    if config.residual_enabled:
        residual_block = cv2.dilate(
            threshold1_final.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        ).astype(bool)
        residual_fov = valid_fov & ~residual_block & ~fov_outline_subtraction
        residual_soft = soft_response.copy()
        residual_soft[~residual_fov] = 0
        residual_soft = normalize01(residual_soft, residual_fov)
        residual_candidates, _, _ = hysteresis_density_threshold(
            residual_soft,
            residual_fov,
            target_density=config.target_density * float(config.residual_low_mult),
            low_mult=config.main_low_mult,
            high_mult=config.main_high_mult,
        )
        recovered = recover_vessel_like_components(
            residual_candidates & residual_fov,
            residual_soft,
            residual_fov,
            axis_ratio_min=config.recovery_axis_ratio,
            skeleton_length_min=config.recovery_skeleton_length,
            branch_density_max=config.recovery_branch_density,
        )
    else:
        residual_soft = np.zeros(raw_mask.shape, dtype=np.float32)
        residual_candidates = np.zeros(raw_mask.shape, dtype=bool)
        recovered = np.zeros(raw_mask.shape, dtype=bool)

    threshold2_largest = keep_largest_components(recovered, count=2) & valid_fov & ~fov_outline_subtraction
    merged = (threshold1_final | threshold2_largest) & valid_fov & ~fov_outline_subtraction
    mask_final = cv2.morphologyEx(merged.astype(np.uint8), cv2.MORPH_CLOSE, close_kernel, iterations=1).astype(bool)
    mask_final &= valid_fov
    mask_final &= ~fov_outline_subtraction

    return {
        "raw_mask": raw_mask,
        "threshold1_final": threshold1_final,
        "residual_soft": residual_soft,
        "residual_candidates": residual_candidates,
        "recovered_vessels": recovered,
        "threshold2_largest2": threshold2_largest,
        "merged_mask": merged,
        "mask_final": mask_final,
    }


def gabor_clahe_maps(
    rgb: np.ndarray,
    config: GaborClaheConfig = BEST_GABOR_CLAHE_CONFIG,
) -> dict[str, np.ndarray]:
    fov = estimate_fov_mask(rgb)
    outline_subtraction = fov_outline_subtraction_mask(fov)

    green = rgb[:, :, 1].copy()
    green[~fov] = 0
    green_filled = fill_outside_fov_nearest(green, fov)
    boosted_green = boost_small_dark_vessels(green_filled, fov)

    clahe1 = clahe_channel(boosted_green, clip_limit=6.0, tile_grid_size=(16, 16))
    clahe1[~fov] = 0
    inverted = 255 - clahe1
    if np.any(fov):
        inverted[~fov] = int(np.median(inverted[fov]))

    gabor = gabor_response(inverted, fov)
    median = cv2.medianBlur(uint8_image(gabor), 7)
    median[~fov] = 0
    clahe2 = clahe_channel(median, clip_limit=12.0, tile_grid_size=(12, 12))
    clahe2[~fov] = 0

    clahe2_norm = normalize01(clahe2.astype(np.float32), fov)
    median_norm = normalize01(median.astype(np.float32), fov)
    soft_response = normalize01(0.65 * median_norm + 0.35 * clahe2_norm, fov)
    soft_response[~fov] = 0
    soft_response[outline_subtraction] = 0

    maps = {
        "fov": fov,
        "fov_outline_subtraction": outline_subtraction,
        "green": green,
        "green_filled": green_filled,
        "boosted_green": boosted_green,
        "clahe1": clahe1,
        "inverted": inverted,
        "gabor_response": gabor,
        "median7": median,
        "clahe2": clahe2,
        "soft_response": soft_response,
    }
    maps.update(segment_soft_response(soft_response, fov, outline_subtraction, config))
    return maps


def segment_vessels_gabor_clahe(
    rgb: np.ndarray,
    config: GaborClaheConfig = BEST_GABOR_CLAHE_CONFIG,
    max_side: int = 768,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    working_rgb = resize_max_side(rgb, max_side)
    maps = gabor_clahe_maps(working_rgb, config=config)
    return maps["soft_response"], maps["mask_final"], maps["fov"]
