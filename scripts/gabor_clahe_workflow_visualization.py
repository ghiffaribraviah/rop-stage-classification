from __future__ import annotations

import argparse
from dataclasses import dataclass
import heapq
from pathlib import Path

import cv2
import numpy as np
from skimage.filters import frangi
from skimage.morphology import skeletonize

from almeida_workflow_visualization import (
    DEFAULT_OUTPUT,
    add_label,
    agrawal_rows,
    estimate_fov_mask,
    normalize01,
    read_binary_mask,
    read_rgb,
    resize_max_side,
    uint8_image,
)
from astar_reconnection_visualization import fov_valid_overlay, overlay_final
from threshold_workflow_visualization import (
    ZHAO_ROOT,
    endpoint_connection_candidates,
    fill_outside_fov_nearest,
    zhao_rows,
)


DEFAULT_GABOR_OUTPUT = DEFAULT_OUTPUT.with_name("debug_gabor_clahe_p10_workflow_15_samples.jpg")


@dataclass(frozen=True)
class GaborClaheConfig:
    target_density: float = 0.09
    main_low_mult: float = 1.28
    main_high_mult: float = 0.50
    residual_enabled: bool = True
    residual_low_mult: float = 1.22
    recovery_axis_ratio: float = 2.3
    recovery_skeleton_length: int | None = None
    recovery_branch_density: float = 0.12


def fov_outline_subtraction_mask(fov: np.ndarray) -> np.ndarray:
    if not np.any(fov):
        return np.zeros(fov.shape, dtype=bool)

    fov = fov.astype(bool)
    min_side = max(1, min(fov.shape))
    thickness = int(np.clip(round(min_side * 0.034), 12, 26))
    outline = np.zeros(fov.shape, dtype=np.uint8)
    contours, _ = cv2.findContours(
        fov.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
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
        low, high = float(values.min()), float(values.max())
    if high <= low:
        return output
    stretched = np.clip((channel.astype(np.float32) - float(low)) / float(high - low), 0.0, 1.0)
    output[fov] = np.round(stretched[fov] * 255.0).astype(np.uint8)
    return output


def boost_small_dark_vessels(green: np.ndarray, fov: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not np.any(fov):
        empty = np.zeros(green.shape, dtype=np.uint8)
        return empty, empty, empty

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
    local_dark_uint8 = uint8_image(local_dark_norm)
    multiscale_uint8 = uint8_image(multiscale_dark)
    boosted_uint8[~fov] = 0
    local_dark_uint8[~fov] = 0
    multiscale_uint8[~fov] = 0
    return boosted_uint8, local_dark_uint8, multiscale_uint8


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


def density_threshold(response: np.ndarray, fov: np.ndarray, target_density: float) -> np.ndarray:
    values = response[fov]
    if values.size == 0:
        return np.zeros(response.shape, dtype=bool)
    density = float(np.clip(target_density, 0.001, 0.95))
    threshold = float(np.percentile(values, 100.0 * (1.0 - density)))
    return ((response >= threshold) & fov).astype(bool)


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
    keep_labels = {
        label
        for label in seed_labels
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_area
    }
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
        axis_ratio = component_axis_ratio(component)
        if axis_ratio < float(axis_ratio_min):
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


def astar_path(
    cost: np.ndarray,
    fov: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    margin: int = 18,
) -> list[tuple[int, int]] | None:
    height, width = cost.shape
    sy, sx = start
    gy, gx = goal
    y0 = max(0, min(sy, gy) - int(margin))
    y1 = min(height, max(sy, gy) + int(margin) + 1)
    x0 = max(0, min(sx, gx) - int(margin))
    x1 = min(width, max(sx, gx) + int(margin) + 1)

    local_cost = cost[y0:y1, x0:x1]
    local_fov = fov[y0:y1, x0:x1]
    start_l = (sy - y0, sx - x0)
    goal_l = (gy - y0, gx - x0)
    if not local_fov[start_l] or not local_fov[goal_l]:
        return None

    neighbors = (
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, np.sqrt(2.0)),
        (-1, 1, np.sqrt(2.0)),
        (1, -1, np.sqrt(2.0)),
        (1, 1, np.sqrt(2.0)),
    )
    open_heap: list[tuple[float, float, tuple[int, int]]] = [(0.0, 0.0, start_l)]
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    best_cost: dict[tuple[int, int], float] = {start_l: 0.0}

    while open_heap:
        _, current_cost, current = heapq.heappop(open_heap)
        if current == goal_l:
            path = [current]
            while path[-1] != start_l:
                path.append(came_from[path[-1]])
            path.reverse()
            return [(y + y0, x + x0) for y, x in path]
        if current_cost > best_cost.get(current, np.inf):
            continue

        cy, cx = current
        for dy, dx, step_len in neighbors:
            ny, nx = cy + dy, cx + dx
            if ny < 0 or ny >= local_cost.shape[0] or nx < 0 or nx >= local_cost.shape[1]:
                continue
            if not local_fov[ny, nx]:
                continue
            new_cost = current_cost + float(local_cost[ny, nx]) * float(step_len)
            neighbor = (ny, nx)
            if new_cost >= best_cost.get(neighbor, np.inf):
                continue
            best_cost[neighbor] = new_cost
            came_from[neighbor] = current
            heuristic = float(np.hypot(goal_l[0] - ny, goal_l[1] - nx)) * 0.05
            heapq.heappush(open_heap, (new_cost + heuristic, new_cost, neighbor))
    return None


def reconnect_density_mask(
    mask: np.ndarray,
    soft_response: np.ndarray,
    fov: np.ndarray,
    max_gap: int = 34,
    max_pairs: int = 90,
) -> tuple[np.ndarray, np.ndarray]:
    endpoints = endpoint_connection_candidates(mask)
    if len(endpoints) < 2:
        empty = np.zeros(mask.shape, dtype=bool)
        return mask.astype(bool) & fov, empty

    values = soft_response[fov]
    if values.size == 0:
        empty = np.zeros(mask.shape, dtype=bool)
        return mask.astype(bool) & fov, empty
    response_floor = max(0.08, float(np.percentile(values, 48.0)))
    high_response = max(response_floor, float(np.percentile(values, 68.0)))

    cost = 1.0 - np.clip(soft_response.astype(np.float32), 0.0, 1.0)
    cost[mask.astype(bool)] = 0.05
    cost[~fov] = 8.0

    pairs: list[tuple[float, dict[str, object], dict[str, object]]] = []
    for i, first in enumerate(endpoints):
        y0, x0 = int(first["y"]), int(first["x"])
        dy0, dx0 = first["direction"]  # type: ignore[misc]
        for second in endpoints[i + 1 :]:
            if int(first["label"]) == int(second["label"]):
                continue
            y1, x1 = int(second["y"]), int(second["x"])
            gap = float(np.hypot(y1 - y0, x1 - x0))
            if gap < 4.0 or gap > float(max_gap):
                continue
            dy1, dx1 = second["direction"]  # type: ignore[misc]
            vec_y = (y1 - y0) / gap
            vec_x = (x1 - x0) / gap
            align_first = dy0 * vec_y + dx0 * vec_x
            align_second = dy1 * (-vec_y) + dx1 * (-vec_x)
            if align_first < 0.20 or align_second < 0.15:
                continue
            pairs.append((gap - 7.0 * min(float(align_first), float(align_second)), first, second))

    connected = mask.astype(bool).copy() & fov
    accepted_paths = np.zeros(mask.shape, dtype=bool)
    used_endpoints: set[int] = set()
    for _, first, second in sorted(pairs, key=lambda item: item[0])[: int(max_pairs)]:
        first_index = int(first["index"])
        second_index = int(second["index"])
        if first_index in used_endpoints or second_index in used_endpoints:
            continue
        start = (int(first["y"]), int(first["x"]))
        goal = (int(second["y"]), int(second["x"]))
        straight = float(np.hypot(goal[0] - start[0], goal[1] - start[1]))
        path = astar_path(cost, fov, start, goal)
        if path is None or len(path) < 3:
            continue

        ys = np.array([point[0] for point in path], dtype=np.intp)
        xs = np.array([point[1] for point in path], dtype=np.intp)
        path_len = float(np.sum(np.hypot(np.diff(ys), np.diff(xs))))
        if path_len > 1.75 * straight + 6.0:
            continue
        response_values = soft_response[ys, xs]
        if float(response_values.mean()) < response_floor and float(np.mean(response_values >= high_response)) < 0.28:
            continue

        accepted_paths[ys, xs] = True
        connected[ys, xs] = True
        used_endpoints.add(first_index)
        used_endpoints.add(second_index)
    return connected & fov, accepted_paths & fov


def line_bridge_mask(
    shape: tuple[int, int],
    start: tuple[int, int],
    goal: tuple[int, int],
    thickness: int = 2,
) -> np.ndarray:
    bridge = np.zeros(shape, dtype=np.uint8)
    cv2.line(
        bridge,
        (int(start[1]), int(start[0])),
        (int(goal[1]), int(goal[0])),
        1,
        max(1, int(thickness)),
    )
    return bridge.astype(bool)


def bridge_density_mask(
    mask: np.ndarray,
    soft_response: np.ndarray,
    fov: np.ndarray,
    forbidden: np.ndarray | None = None,
    max_gap: int | None = None,
    max_pairs: int = 180,
    response_floor_pct: float = 28.0,
    high_response_pct: float = 50.0,
    min_high_fraction: float = 0.16,
) -> tuple[np.ndarray, np.ndarray]:
    endpoints = endpoint_connection_candidates(mask)
    if len(endpoints) < 2:
        empty = np.zeros(mask.shape, dtype=bool)
        return mask.astype(bool) & fov, empty

    values = soft_response[fov]
    if values.size == 0:
        empty = np.zeros(mask.shape, dtype=bool)
        return mask.astype(bool) & fov, empty

    min_side = max(1, min(mask.shape))
    if max_gap is None:
        max_gap = int(np.clip(round(min_side * 0.090), 40, 70))
    response_floor = max(0.03, float(np.percentile(values, float(response_floor_pct))))
    high_response = max(response_floor, float(np.percentile(values, float(high_response_pct))))
    forbidden_mask = np.zeros(mask.shape, dtype=bool) if forbidden is None else forbidden.astype(bool)
    corridor_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    pairs: list[tuple[float, dict[str, object], dict[str, object], np.ndarray]] = []
    for i, first in enumerate(endpoints):
        y0, x0 = int(first["y"]), int(first["x"])
        dy0, dx0 = first["direction"]  # type: ignore[misc]
        for second in endpoints[i + 1 :]:
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
            if align_first < 0.05 or align_second < -0.05:
                continue

            bridge = line_bridge_mask(mask.shape, (y0, x0), (y1, x1), thickness=2) & fov
            if np.any(bridge & forbidden_mask):
                continue

            bridge_corridor = cv2.dilate(bridge.astype(np.uint8), corridor_kernel, iterations=1).astype(bool) & fov
            if np.any(bridge_corridor & forbidden_mask):
                continue

            gap_pixels = bridge_corridor & ~mask.astype(bool)
            if not np.any(gap_pixels):
                continue
            response_values = soft_response[gap_pixels]
            response_mean = float(response_values.mean())
            high_fraction = float(np.mean(response_values >= high_response))
            if response_mean < response_floor and high_fraction < float(min_high_fraction):
                continue

            score = gap - 6.0 * min(float(align_first), float(align_second)) - 3.0 * response_mean
            pairs.append((score, first, second, bridge))

    connected = mask.astype(bool).copy() & fov
    bridge_paths = np.zeros(mask.shape, dtype=bool)
    used_endpoints: set[int] = set()
    for _, first, second, bridge in sorted(pairs, key=lambda item: item[0])[: int(max_pairs)]:
        first_index = int(first["index"])
        second_index = int(second["index"])
        if first_index in used_endpoints or second_index in used_endpoints:
            continue
        bridge_paths |= bridge
        connected |= bridge
        used_endpoints.add(first_index)
        used_endpoints.add(second_index)

    return connected & fov, bridge_paths & fov


def segment_soft_response(
    soft_response: np.ndarray,
    valid_fov: np.ndarray,
    fov_outline_subtraction: np.ndarray,
    config: GaborClaheConfig,
) -> dict[str, np.ndarray]:
    valid_fov = valid_fov.astype(bool)
    fov_outline_subtraction = fov_outline_subtraction.astype(bool) & valid_fov
    target_density = float(config.target_density)

    raw_mask, high_seed_mask, low_support_mask = hysteresis_density_threshold(
        soft_response,
        valid_fov,
        target_density=target_density,
        low_mult=config.main_low_mult,
        high_mult=config.main_high_mult,
    )
    raw_mask &= ~fov_outline_subtraction
    bridge_paths = np.zeros(raw_mask.shape, dtype=bool)
    bridge_display = bridge_paths
    reconnected = raw_mask.astype(bool) & valid_fov
    first_largest = keep_largest_components(reconnected, count=2) & valid_fov
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    first_mask = cv2.morphologyEx(first_largest.astype(np.uint8), cv2.MORPH_CLOSE, close_kernel, iterations=1).astype(bool)
    first_mask &= valid_fov
    first_mask &= ~fov_outline_subtraction

    if config.residual_enabled:
        residual_block = cv2.dilate(
            first_mask.astype(np.uint8),
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
            target_density=target_density * float(config.residual_low_mult),
            low_mult=config.main_low_mult,
            high_mult=config.main_high_mult,
        )
        residual_candidates &= residual_fov
        recovered_vessels = recover_vessel_like_components(
            residual_candidates,
            residual_soft,
            residual_fov,
            axis_ratio_min=config.recovery_axis_ratio,
            skeleton_length_min=config.recovery_skeleton_length,
            branch_density_max=config.recovery_branch_density,
        )
    else:
        residual_soft = np.zeros(raw_mask.shape, dtype=np.float32)
        residual_candidates = np.zeros(raw_mask.shape, dtype=bool)
        recovered_vessels = np.zeros(raw_mask.shape, dtype=bool)

    threshold2_largest = keep_largest_components(recovered_vessels, count=2) & valid_fov & ~fov_outline_subtraction
    merged_mask = (first_mask | threshold2_largest) & valid_fov & ~fov_outline_subtraction
    final_candidates = merged_mask
    mask_final = cv2.morphologyEx(final_candidates.astype(np.uint8), cv2.MORPH_CLOSE, close_kernel, iterations=1).astype(bool)
    mask_final &= valid_fov
    mask_final &= ~fov_outline_subtraction

    return {
        "high_seed_mask": high_seed_mask,
        "low_support_mask": low_support_mask,
        "raw_mask": raw_mask,
        "first_largest2": first_largest,
        "threshold1_final": first_mask,
        "residual_soft": residual_soft,
        "residual_candidates": residual_candidates,
        "recovered_vessels": recovered_vessels,
        "threshold2_largest2": threshold2_largest,
        "merged_mask": merged_mask,
        "connection_paths": bridge_paths,
        "connection_paths_display": bridge_display,
        "reconnected": reconnected,
        "largest2": final_candidates,
        "mask_final": mask_final,
    }


def gabor_clahe_maps(
    rgb: np.ndarray,
    fov: np.ndarray,
    target_density: float = 0.09,
    config: GaborClaheConfig | None = None,
) -> dict[str, np.ndarray]:
    if config is None:
        config = GaborClaheConfig(target_density=target_density)

    object_fov = fov.astype(bool)
    valid_fov = object_fov
    fov_outline_subtraction = fov_outline_subtraction_mask(object_fov)

    green = rgb[:, :, 1].copy()
    green[~object_fov] = 0
    green_filter = fill_outside_fov_nearest(green, object_fov)
    boosted_green, local_dark, multiscale_dark = boost_small_dark_vessels(green_filter, object_fov)

    clahe1 = clahe_channel(boosted_green, clip_limit=6.0, tile_grid_size=(16, 16))
    clahe1_display = clahe1.copy()
    clahe1_display[~object_fov] = 0

    inverted = 255 - clahe1
    if np.any(object_fov):
        inverted[~object_fov] = int(np.median(inverted[object_fov]))
    inverted_display = inverted.copy()
    inverted_display[~object_fov] = 0

    gabor = gabor_response(inverted, object_fov)
    gabor[~valid_fov] = 0
    gabor_norm = normalize01(gabor, valid_fov)
    gabor_norm[~valid_fov] = 0

    frangi_map = normalize01(
        frangi(
            inverted.astype(np.float32) / 255.0,
            sigmas=(1, 3, 5, 7),
            alpha=0.5,
            beta=15.0,
            black_ridges=False,
        ),
        valid_fov,
    )
    frangi_map[~valid_fov] = 0

    median = cv2.medianBlur(uint8_image(gabor_norm), 7)
    median[~valid_fov] = 0

    clahe2 = clahe_channel(median, clip_limit=12.0, tile_grid_size=(12, 12))
    clahe2[~valid_fov] = 0
    clahe2_norm = normalize01(clahe2.astype(np.float32), valid_fov)
    median_norm = normalize01(median.astype(np.float32), valid_fov)
    sharpened = normalize01(0.65 * median_norm + 0.35 * clahe2_norm, valid_fov)
    sharpened[~valid_fov] = 0
    sharpened[fov_outline_subtraction] = 0

    segmentation = segment_soft_response(sharpened, valid_fov, fov_outline_subtraction, config)

    maps = {
        "preprocessing_fov": object_fov,
        "processing_fov": valid_fov,
        "border_exclusion": np.zeros(object_fov.shape, dtype=bool),
        "final_subtraction": fov_outline_subtraction,
        "green": green,
        "green_filled": green_filter,
        "green_boosted": boosted_green,
        "local_dark": local_dark,
        "multiscale_dark": multiscale_dark,
        "clahe1": clahe1_display,
        "inverted": inverted_display,
        "gabor_response": gabor,
        "gabor_norm": gabor_norm,
        "frangi": frangi_map,
        "median7": median,
        "clahe2": clahe2,
        "soft_response": sharpened,
    }
    maps.update(segmentation)
    return maps


def workflow_tiles(
    rgb: np.ndarray,
    fov: np.ndarray,
    gt: np.ndarray,
    target_density: float,
    config: GaborClaheConfig | None = None,
) -> list[tuple[str, np.ndarray]]:
    maps = gabor_clahe_maps(rgb, fov, target_density=target_density, config=config)
    valid_fov = maps["processing_fov"]
    mask_final = maps["mask_final"]
    return [
        ("RGB", rgb),
        ("FOV / valid", fov_valid_overlay(rgb, fov, valid_fov, maps["preprocessing_fov"], maps["border_exclusion"])),
        ("GT", gt),
        ("Green", maps["green"]),
        ("Green filled", maps["green_filled"]),
        ("Local dark", maps["local_dark"]),
        ("Multi dark", maps["multiscale_dark"]),
        ("Boosted green", maps["green_boosted"]),
        ("CLAHE 6 16x16", maps["clahe1"]),
        ("Inverted", maps["inverted"]),
        ("Gabor max", maps["gabor_response"]),
        ("Gabor norm", maps["gabor_norm"]),
        ("Frangi", maps["frangi"]),
        ("Median 7", maps["median7"]),
        ("CLAHE 12 12x12", maps["clahe2"]),
        ("Soft response", maps["soft_response"]),
        ("Hyst mask", maps["raw_mask"]),
        ("No connection", maps["reconnected"]),
        ("Initial 2 CC", maps["first_largest2"]),
        ("T1 final", maps["threshold1_final"]),
        ("Residual soft", maps["residual_soft"]),
        ("Residual cand", maps["residual_candidates"]),
        ("Recovered", maps["recovered_vessels"]),
        ("T2 2 CC", maps["threshold2_largest2"]),
        ("T1+T2 union", maps["merged_mask"]),
        ("Final cand", maps["largest2"]),
        ("mask_final", mask_final),
        ("Overlay", overlay_final(rgb, mask_final, maps["connection_paths_display"], valid_fov)),
    ]


def build_sheet(
    rows: list[tuple[str, Path, Path | None]],
    output_path: Path,
    tile_size: int,
    max_side: int,
    target_density: float,
    config: GaborClaheConfig | None = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet_rows = []
    for index, (name, image_path, mask_path) in enumerate(rows, start=1):
        rgb = resize_max_side(read_rgb(image_path), max_side)
        fov = estimate_fov_mask(rgb)
        gt = read_binary_mask(mask_path, fov.shape)
        tiles = [(f"{index}. {name}", rgb)]
        tiles.extend(workflow_tiles(rgb, fov, gt, target_density=target_density, config=config)[1:])
        row_tiles = [add_label(image, label, tile_size) for label, image in tiles]
        sheet_rows.append(np.concatenate(row_tiles, axis=1))
    sheet = np.concatenate(sheet_rows, axis=0)
    cv2.imwrite(str(output_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize Gabor + CLAHE + hysteresis-threshold vessel workflow.")
    parser.add_argument("--output", type=Path, default=DEFAULT_GABOR_OUTPUT)
    parser.add_argument("--tile-size", type=int, default=150)
    parser.add_argument("--max-side", type=int, default=768)
    parser.add_argument("--target-density", type=float, default=0.09)
    parser.add_argument("--retcam-count", type=int, default=5)
    parser.add_argument("--neo-count", type=int, default=5)
    parser.add_argument("--zhao-count", type=int, default=5)
    args = parser.parse_args()

    rows = agrawal_rows(args.retcam_count, args.neo_count)
    rows.extend(zhao_rows(ZHAO_ROOT, args.zhao_count))
    if not rows:
        raise RuntimeError("No debug images found.")
    print(
        build_sheet(
            rows,
            args.output,
            tile_size=args.tile_size,
            max_side=args.max_side,
            target_density=args.target_density,
        )
    )


if __name__ == "__main__":
    main()
