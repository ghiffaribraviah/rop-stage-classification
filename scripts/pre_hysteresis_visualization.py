from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from almeida_workflow_visualization import (
    DEFAULT_OUTPUT,
    add_label,
    agrawal_rows,
    estimate_fov_mask,
    normalize01,
    read_binary_mask,
    read_rgb,
    resize_max_side,
)
from threshold_workflow_visualization import (
    ZHAO_ROOT,
    active_percentile,
    connected_hysteresis_mask,
    threshold_response_maps,
    zhao_rows,
)


DEFAULT_PRE_HYSTERESIS_OUTPUT = DEFAULT_OUTPUT.with_name("debug_pre_hysteresis_workflow_15_samples.jpg")


def balanced_hysteresis(response: np.ndarray, support: np.ndarray, fov: np.ndarray) -> np.ndarray:
    return connected_hysteresis_mask(
        response,
        fov,
        high_pct=97.0,
        low_pct=87.0,
        min_area=12,
        support=support,
        support_floor=0.12,
    )


def conditioned_response(maps: dict[str, np.ndarray], fov: np.ndarray) -> np.ndarray:
    response = maps["response"]
    support = maps["support"]
    coherence = maps["coherence"]
    conditioned = response * (0.45 + 0.55 * support) * (0.65 + 0.35 * coherence)
    return normalize01(conditioned, fov)


def dense_noise_score(response: np.ndarray, support: np.ndarray, coherence: np.ndarray, fov: np.ndarray) -> float:
    if not np.any(fov):
        return 0.0
    active = response >= active_percentile(response, fov, 65.0)
    weak_line = (support < active_percentile(support, fov, 48.0)) | (
        coherence < max(0.05, float(np.percentile(coherence[fov], 45.0)))
    )
    return float(np.mean(active[fov] & weak_line[fov]))


def adaptive_conditioned_response(maps: dict[str, np.ndarray], fov: np.ndarray) -> np.ndarray:
    response = maps["response"]
    support = maps["support"]
    coherence = maps["coherence"]
    noise = dense_noise_score(response, support, coherence, fov)
    strength = float(np.clip((noise - 0.10) / 0.18, 0.0, 1.0))
    support_floor = 0.45 - 0.20 * strength
    support_gain = 0.55 + 0.20 * strength
    coherence_floor = 0.72 - 0.27 * strength
    coherence_gain = 0.28 + 0.27 * strength
    floor_pct = 42.0 + 12.0 * strength
    gated = response * (support_floor + support_gain * support) * (coherence_floor + coherence_gain * coherence)
    return normalize01(response_percentile_condition(gated, fov, floor_pct=floor_pct), fov)


def response_percentile_condition(response: np.ndarray, fov: np.ndarray, floor_pct: float) -> np.ndarray:
    if not np.any(fov):
        return np.zeros(response.shape, dtype=np.float32)
    low = float(np.percentile(response[fov], float(floor_pct)))
    high = float(np.percentile(response[fov], 99.2))
    if high <= low:
        return normalize01(response, fov)
    conditioned = np.clip((response - low) / (high - low), 0.0, 1.0)
    conditioned[~fov] = 0
    return conditioned.astype(np.float32)


def component_axis_ratio(component: np.ndarray) -> float:
    ys, xs = np.nonzero(component)
    if xs.size < 3:
        return 1.0
    coords = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
    eigvals = np.linalg.eigvalsh(np.cov(coords, rowvar=False))
    return float(np.sqrt((eigvals.max() + 1e-6) / (eigvals.min() + 1e-6)))


def blob_suppressed_response(maps: dict[str, np.ndarray], fov: np.ndarray) -> np.ndarray:
    response = maps["response"].copy()
    support = maps["support"]
    coherence = maps["coherence"]
    pre_mask = (response >= active_percentile(response, fov, 78.0)) & fov
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(pre_mask.astype(np.uint8), 8)
    support_floor = active_percentile(support, fov, 45.0)
    coherence_floor = max(0.05, float(np.percentile(coherence[fov], 40.0))) if np.any(fov) else 0.05
    for label in range(1, n_labels):
        component = labels == label
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 45:
            continue
        axis_ratio = component_axis_ratio(component)
        support_mean = float(support[component].mean())
        coherence_mean = float(coherence[component].mean())
        blob_like = axis_ratio < 1.9
        weakly_supported = support_mean < support_floor or coherence_mean < coherence_floor
        if blob_like or (area >= 90 and weakly_supported):
            response[component] *= 0.45
    response[~fov] = 0
    return normalize01(response, fov)


def agreement_gated_response(maps: dict[str, np.ndarray], fov: np.ndarray) -> np.ndarray:
    response = maps["response"]
    signals = [
        response >= active_percentile(response, fov, 72.0),
        maps["support"] >= active_percentile(maps["support"], fov, 55.0),
        maps["thin"] >= active_percentile(maps["thin"], fov, 70.0),
        maps["matched"] >= active_percentile(maps["matched"], fov, 70.0),
        maps["zscore"] >= active_percentile(maps["zscore"], fov, 70.0),
    ]
    agreement = np.mean(np.stack(signals, axis=0).astype(np.float32), axis=0)
    gated = response * (0.25 + 0.75 * agreement)
    gated[~fov] = 0
    return normalize01(gated, fov)


def line_consensus_response(maps: dict[str, np.ndarray], fov: np.ndarray) -> np.ndarray:
    response = maps["response"]
    support = maps["support"]
    coherence = maps["coherence"]
    line_gate = normalize01(support * coherence, fov)
    gated = response * (0.18 + 0.82 * line_gate)
    return normalize01(response_percentile_condition(gated, fov, floor_pct=52.0), fov)


def pre_hysteresis_maps(maps: dict[str, np.ndarray], fov: np.ndarray) -> list[tuple[str, np.ndarray]]:
    variants = [
        ("Base resp", maps["response"]),
        ("Cond resp", conditioned_response(maps, fov)),
        ("Adaptive resp", adaptive_conditioned_response(maps, fov)),
        ("Blob supp resp", blob_suppressed_response(maps, fov)),
        ("Agree resp", agreement_gated_response(maps, fov)),
        ("Line resp", line_consensus_response(maps, fov)),
    ]
    outputs: list[tuple[str, np.ndarray]] = []
    for label, response in variants:
        outputs.append((label, response))
        outputs.append((label.replace("resp", "hyst"), balanced_hysteresis(response, maps["support"], fov)))
    return outputs


def build_sheet(rows: list[tuple[str, Path, Path | None]], output_path: Path, tile_size: int, max_side: int) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet_rows = []
    for index, (name, image_path, mask_path) in enumerate(rows, start=1):
        rgb = resize_max_side(read_rgb(image_path), max_side)
        fov = estimate_fov_mask(rgb)
        gt = read_binary_mask(mask_path, fov.shape)
        maps = threshold_response_maps(rgb, fov)
        fov = maps["processing_fov"]
        display_maps: list[tuple[str, np.ndarray]] = [
            (f"{index}. {name}", rgb),
            ("FOV", fov),
            ("GT", gt),
            ("Support raw", maps["support_raw"]),
            ("Support conn", maps["support_connected"]),
            ("Support mask", maps["support_mask"]),
            ("Coherence", maps["coherence"]),
        ]
        display_maps.extend(pre_hysteresis_maps(maps, fov))
        row_tiles = [add_label(image, label, tile_size) for label, image in display_maps]
        sheet_rows.append(np.concatenate(row_tiles, axis=1))
    sheet = np.concatenate(sheet_rows, axis=0)
    cv2.imwrite(str(output_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare response conditioning before balanced hysteresis.")
    parser.add_argument("--output", type=Path, default=DEFAULT_PRE_HYSTERESIS_OUTPUT)
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
