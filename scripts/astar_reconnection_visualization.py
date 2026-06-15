from __future__ import annotations

import argparse
import heapq
from pathlib import Path

import cv2
import numpy as np
from skimage.morphology import skeletonize

from almeida_workflow_visualization import (
    DEFAULT_OUTPUT,
    add_label,
    agrawal_rows,
    estimate_fov_mask,
    read_binary_mask,
    read_rgb,
    resize_max_side,
)
from threshold_workflow_visualization import (
    ZHAO_ROOT,
    active_percentile,
    clean_balanced_noise,
    connected_hysteresis_mask,
    endpoint_connection_candidates,
    shape_aware_component_clean,
    threshold_response_maps,
    zhao_rows,
)


DEFAULT_ASTAR_OUTPUT = DEFAULT_OUTPUT.with_name("debug_full_vessel_pipeline_15_samples.jpg")


def clean_seed_mask(maps: dict[str, np.ndarray], fov: np.ndarray) -> np.ndarray:
    response = maps["response"]
    support = maps["support"]
    mask = connected_hysteresis_mask(
        response,
        fov,
        high_pct=95.5,
        low_pct=82.0,
        min_area=10,
        support=support,
        support_floor=0.14,
        support_pct_floor=32.0,
    )
    return clean_balanced_noise(mask, response, support, maps["coherence"], fov)


def path_cost_image(maps: dict[str, np.ndarray], fov: np.ndarray) -> np.ndarray:
    support = maps["support"]
    response = maps["response"]
    coherence = maps["coherence"]
    affinity = 0.46 * support + 0.36 * response + 0.18 * coherence
    cost = 1.0 - np.clip(affinity, 0.0, 1.0)
    cost = np.clip(cost, 0.05, 1.0).astype(np.float32)
    cost[~fov] = 8.0
    return cost


def endpoint_pair_candidates(
    mask: np.ndarray,
    max_gap: int = 34,
    max_pairs: int = 90,
) -> list[tuple[float, dict[str, object], dict[str, object]]]:
    endpoints = endpoint_connection_candidates(mask)
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
            if align_first < 0.20 or align_second < 0.20:
                continue
            score = gap - 8.0 * min(float(align_first), float(align_second))
            pairs.append((score, first, second))
    return sorted(pairs, key=lambda item: item[0])[:max_pairs]


def astar_path(
    cost: np.ndarray,
    fov: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    margin: int = 12,
) -> tuple[list[tuple[int, int]], float] | None:
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

    min_step = 0.05
    open_heap: list[tuple[float, float, tuple[int, int]]] = []
    heapq.heappush(open_heap, (0.0, 0.0, start_l))
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    best_cost: dict[tuple[int, int], float] = {start_l: 0.0}
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

    while open_heap:
        _, current_cost, current = heapq.heappop(open_heap)
        if current == goal_l:
            path: list[tuple[int, int]] = []
            node = current
            while node != start_l:
                path.append((node[0] + y0, node[1] + x0))
                node = came_from[node]
            path.append(start)
            path.reverse()
            return path, current_cost
        if current_cost > best_cost.get(current, np.inf):
            continue
        cy, cx = current
        for dy, dx, step_len in neighbors:
            ny, nx = cy + dy, cx + dx
            if ny < 0 or ny >= local_cost.shape[0] or nx < 0 or nx >= local_cost.shape[1]:
                continue
            if not local_fov[ny, nx]:
                continue
            step_cost = float(local_cost[ny, nx]) * float(step_len)
            new_cost = current_cost + step_cost
            neighbor = (ny, nx)
            if new_cost >= best_cost.get(neighbor, np.inf):
                continue
            best_cost[neighbor] = new_cost
            came_from[neighbor] = current
            heuristic = float(np.hypot(goal_l[0] - ny, goal_l[1] - nx)) * min_step
            heapq.heappush(open_heap, (new_cost + heuristic, new_cost, neighbor))
    return None


def accept_path(
    path: list[tuple[int, int]],
    cost: np.ndarray,
    maps: dict[str, np.ndarray],
    fov: np.ndarray,
    straight_distance: float,
) -> bool:
    if len(path) < 3:
        return False
    ys = np.array([point[0] for point in path], dtype=np.intp)
    xs = np.array([point[1] for point in path], dtype=np.intp)
    if not np.all(fov[ys, xs]):
        return False
    path_len = float(np.sum(np.hypot(np.diff(ys), np.diff(xs))))
    if path_len > 1.85 * straight_distance + 6.0:
        return False
    mean_cost = float(cost[ys, xs].mean())
    max_cost = float(cost[ys, xs].max())
    support_mean = float(maps["support_raw"][ys, xs].mean())
    response_mean = float(maps["response"][ys, xs].mean())
    support_floor = active_percentile(maps["support_raw"], fov, 38.0)
    response_floor = active_percentile(maps["response"], fov, 32.0)
    return (
        mean_cost <= 0.70
        and max_cost <= 0.98
        and (support_mean >= support_floor or response_mean >= response_floor)
    )


def astar_reconnect(
    mask: np.ndarray,
    maps: dict[str, np.ndarray],
    fov: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cost = path_cost_image(maps, fov)
    candidate_overlay = np.zeros(mask.shape, dtype=bool)
    accepted_paths = np.zeros(mask.shape, dtype=bool)
    used_endpoints: set[int] = set()

    for _, first, second in endpoint_pair_candidates(mask):
        first_index = int(first["index"])
        second_index = int(second["index"])
        if first_index in used_endpoints or second_index in used_endpoints:
            continue
        start = (int(first["y"]), int(first["x"]))
        goal = (int(second["y"]), int(second["x"]))
        straight = float(np.hypot(goal[0] - start[0], goal[1] - start[1]))
        line = np.zeros(mask.shape, dtype=np.uint8)
        cv2.line(line, (start[1], start[0]), (goal[1], goal[0]), 1, 1)
        candidate_overlay |= line.astype(bool) & fov
        result = astar_path(cost, fov, start, goal)
        if result is None:
            continue
        path, _ = result
        if not accept_path(path, cost, maps, fov, straight):
            continue
        for y, x in path:
            accepted_paths[y, x] = True
        used_endpoints.add(first_index)
        used_endpoints.add(second_index)

    final = shape_aware_component_clean(mask | accepted_paths, min_area=14, min_elongated_area=5, min_axis_ratio=1.7)
    return final & fov, candidate_overlay & fov, accepted_paths & fov, cost


def endpoints_overlay(mask: np.ndarray) -> np.ndarray:
    image = np.zeros((*mask.shape, 3), dtype=np.uint8)
    skeleton = skeletonize(mask.astype(bool))
    endpoints = endpoint_connection_candidates(mask)
    image[mask] = (80, 80, 80)
    image[skeleton] = (180, 180, 180)
    for endpoint in endpoints:
        cv2.circle(image, (int(endpoint["x"]), int(endpoint["y"])), 2, (255, 80, 80), -1, cv2.LINE_AA)
    return image


def color_mask_overlay(base_mask: np.ndarray, extra_mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    image = np.zeros((*base_mask.shape, 3), dtype=np.uint8)
    image[base_mask] = (115, 115, 115)
    image[extra_mask] = color
    return image


def overlay_final(rgb: np.ndarray, mask: np.ndarray, paths: np.ndarray, fov: np.ndarray) -> np.ndarray:
    base = rgb.copy()
    base[~fov] = 0
    overlay = base.copy()
    overlay[mask] = (0, 220, 120)
    overlay[paths] = (255, 80, 80)
    return cv2.addWeighted(base, 0.62, overlay, 0.38, 0)


def overlay_soft_vessels(
    rgb: np.ndarray,
    soft: np.ndarray,
    mask: np.ndarray,
    paths: np.ndarray,
    fov: np.ndarray,
) -> np.ndarray:
    base = rgb.copy()
    base[~fov] = 0

    values = soft[fov]
    if values.size:
        floor = max(0.14, float(np.percentile(values, 45.0)))
        high = max(floor + 1e-3, float(np.percentile(values, 99.0)))
        soft_alpha = np.clip((soft - floor) / (high - floor), 0.0, 1.0)
    else:
        soft_alpha = np.zeros_like(soft, dtype=np.float32)
    soft_alpha = (soft_alpha**1.25) * 0.48
    soft_alpha[~fov] = 0.0

    vessel_color = np.zeros_like(base)
    vessel_color[..., 0] = 30
    vessel_color[..., 1] = 235
    vessel_color[..., 2] = 185
    blended = (base.astype(np.float32) * (1.0 - soft_alpha[..., None])) + (
        vessel_color.astype(np.float32) * soft_alpha[..., None]
    )

    overlay = np.clip(blended, 0, 255).astype(np.uint8)
    overlay[mask] = (0, 235, 120)
    overlay[paths] = (255, 80, 80)
    return cv2.addWeighted(base, 0.42, overlay, 0.58, 0)


def fov_valid_overlay(
    rgb: np.ndarray,
    fov: np.ndarray,
    valid_fov: np.ndarray,
    preprocessing_fov: np.ndarray | None = None,
    border_exclusion: np.ndarray | None = None,
    border_cleanup: np.ndarray | None = None,
) -> np.ndarray:
    image = (0.58 * rgb).astype(np.uint8)
    image[~fov.astype(bool)] = 0
    border_only = fov.astype(bool) & ~valid_fov.astype(bool)
    tint = image.copy()
    tint[border_only] = (165, 165, 165)
    tint[valid_fov.astype(bool)] = (70, 220, 120)
    if border_cleanup is not None:
        tint[border_cleanup.astype(bool)] = (115, 115, 115)
    if border_exclusion is not None:
        tint[border_exclusion.astype(bool)] = (170, 170, 170)
    image = cv2.addWeighted(image, 0.62, tint, 0.38, 0)
    fov_contours, _ = cv2.findContours(fov.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    preprocessing_contours: list[np.ndarray] = []
    if preprocessing_fov is not None:
        preprocessing_contours, _ = cv2.findContours(
            preprocessing_fov.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
    valid_contours, _ = cv2.findContours(valid_fov.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, fov_contours, -1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.drawContours(image, preprocessing_contours, -1, (255, 210, 60), 1, cv2.LINE_AA)
    cv2.drawContours(image, valid_contours, -1, (0, 255, 120), 1, cv2.LINE_AA)
    return image


def workflow_tiles(rgb: np.ndarray, fov: np.ndarray, gt: np.ndarray) -> list[tuple[str, np.ndarray]]:
    maps = threshold_response_maps(rgb, fov)
    valid_fov = maps["processing_fov"]
    mask_final = maps["mask_final"]
    empty_paths = np.zeros(mask_final.shape, dtype=bool)
    return [
        ("RGB", rgb),
        ("FOV / valid", fov_valid_overlay(rgb, fov, valid_fov, maps["preprocessing_fov"], maps["border_exclusion"])),
        ("GT", gt),
        ("RGB masked", maps["bilateral_rgb"]),
        ("LAB L", maps["lab_l"]),
        ("LAB L CLAHE", maps["lab_l_clahe"]),
        ("LAB RGB", maps["lab_enhanced_rgb"]),
        ("Green", maps["green"]),
        ("Almeida bg", maps["green_background"]),
        ("Almeida norm", maps["green_flattened"]),
        ("Inverted", maps["inverted"]),
        ("Vessel input", maps["vessel_base"]),
        ("BSGMF", maps["bsgmf"]),
        ("Mod top-hat", maps["top_hat"]),
        ("Frangi", maps["frangi"]),
        ("Jerman", maps["jerman"]),
        ("70F 30J", maps["combined"]),
        ("Soft prob", maps["vessel_probability"]),
        ("Strong seeds", maps["strong_seeds"]),
        ("Weak candidates", maps["weak_candidates"]),
        ("Hysteresis", maps["triangle"]),
        ("Shape clean", maps["area_clean"]),
        ("OD removed", maps["od_removed"]),
        ("Skeleton", maps["skeleton"]),
        ("mask_final", mask_final),
        ("Overlay", overlay_final(rgb, mask_final, empty_paths, valid_fov)),
    ]


def build_sheet(rows: list[tuple[str, Path, Path | None]], output_path: Path, tile_size: int, max_side: int) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet_rows = []
    for index, (name, image_path, mask_path) in enumerate(rows, start=1):
        rgb = resize_max_side(read_rgb(image_path), max_side)
        fov = estimate_fov_mask(rgb)
        gt = read_binary_mask(mask_path, fov.shape)
        tiles = [(f"{index}. {name}", rgb)]
        tiles.extend(workflow_tiles(rgb, fov, gt)[1:])
        row_tiles = [add_label(image, label, tile_size) for label, image in tiles]
        sheet_rows.append(np.concatenate(row_tiles, axis=1))
    sheet = np.concatenate(sheet_rows, axis=0)
    cv2.imwrite(str(output_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize the Almeida-style retinal vessel workflow.")
    parser.add_argument("--output", type=Path, default=DEFAULT_ASTAR_OUTPUT)
    parser.add_argument("--tile-size", type=int, default=150)
    parser.add_argument("--max-side", type=int, default=768)
    parser.add_argument("--retcam-count", type=int, default=5)
    parser.add_argument("--neo-count", type=int, default=5)
    parser.add_argument("--zhao-count", type=int, default=5)
    args = parser.parse_args()

    rows = agrawal_rows(args.retcam_count, args.neo_count)
    rows.extend(zhao_rows(ZHAO_ROOT, args.zhao_count))
    if not rows:
        raise RuntimeError("No debug images found.")
    print(build_sheet(rows, args.output, tile_size=args.tile_size, max_side=args.max_side))


if __name__ == "__main__":
    main()
