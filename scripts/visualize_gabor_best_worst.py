from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from skimage.morphology import skeletonize

from almeida_workflow_visualization import (
    add_label,
    agrawal_rows,
    estimate_fov_mask,
    read_binary_mask,
    read_rgb,
    resize_max_side,
)
from gabor_clahe_workflow_visualization import (
    GaborClaheConfig,
    workflow_tiles,
)
from tune_gabor_clahe_thresholds import cldice_row, metric_row, row_to_config


DEFAULT_OUTPUT_DIR = Path("output/00_debug_baseline")


def load_best_config(path: Path) -> GaborClaheConfig:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No configs found in {path}")
    return row_to_config(rows[0])


def image_metrics(pred: np.ndarray, gt: np.ndarray, fov: np.ndarray) -> dict[str, float]:
    pred = pred.astype(bool) & fov
    gt = gt.astype(bool) & fov
    tp = int(np.count_nonzero(pred & gt & fov))
    fp = int(np.count_nonzero(pred & ~gt & fov))
    fn = int(np.count_nonzero(~pred & gt & fov))
    tn = int(np.count_nonzero(~pred & ~gt & fov))
    metrics = metric_row(tp, fp, fn, tn)
    metrics.update(cldice_row(pred, gt, skeletonize(gt).astype(bool)))
    return metrics


def format_label(prefix: str, name: str, metrics: dict[str, float]) -> str:
    return (
        f"{prefix} {name} "
        f"clD {metrics['cldice']:.3f} D {metrics['dice']:.3f} "
        f"P {metrics['precision']:.3f} R {metrics['recall']:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize best and worst GT images for the tuned Gabor+CLAHE config.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config-csv", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--scores-csv", type=Path, default=None)
    parser.add_argument("--tile-size", type=int, default=150)
    parser.add_argument("--max-side", type=int, default=768)
    parser.add_argument("--retcam-count", type=int, default=5)
    parser.add_argument("--neo-count", type=int, default=5)
    parser.add_argument("--count", type=int, default=3)
    args = parser.parse_args()
    if args.config_csv is None:
        args.config_csv = args.output_dir / "top_by_cldice.csv"
    if args.output is None:
        args.output = args.output_dir / "best_config_best_worst_cldice.jpg"
    if args.scores_csv is None:
        args.scores_csv = args.output_dir / "best_config_per_image_scores.csv"

    config = load_best_config(args.config_csv)
    rows = agrawal_rows(args.retcam_count, args.neo_count)
    scored_rows = []
    for name, image_path, mask_path in rows:
        if mask_path is None:
            continue
        rgb = resize_max_side(read_rgb(image_path), args.max_side)
        fov = estimate_fov_mask(rgb)
        gt = read_binary_mask(mask_path, fov.shape) & fov
        tiles = workflow_tiles(rgb, fov, gt, target_density=config.target_density, config=config)
        pred = tiles[-2][1].astype(bool)
        metrics = image_metrics(pred, gt, fov)
        scored_rows.append(
            {
                "name": name,
                "image_path": image_path,
                "mask_path": mask_path,
                "rgb": rgb,
                "fov": fov,
                "gt": gt,
                "metrics": metrics,
            }
        )

    if not scored_rows:
        raise RuntimeError("No GT rows found.")

    best = sorted(scored_rows, key=lambda row: row["metrics"]["cldice"], reverse=True)[: args.count]
    worst = sorted(scored_rows, key=lambda row: row["metrics"]["cldice"])[: args.count]
    selected = [("BEST", row) for row in best]
    selected.extend(("WORST", row) for row in worst)

    sheet_rows = []
    for index, (prefix, row) in enumerate(selected, start=1):
        rgb = row["rgb"]
        fov = row["fov"]
        gt = row["gt"]
        label = format_label(f"{index}. {prefix}", row["name"], row["metrics"])
        tiles = [(label, rgb)]
        tiles.extend(workflow_tiles(rgb, fov, gt, target_density=config.target_density, config=config)[1:])
        row_tiles = [add_label(image, tile_label, args.tile_size) for tile_label, image in tiles]
        sheet_rows.append(np.concatenate(row_tiles, axis=1))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet = np.concatenate(sheet_rows, axis=0)
    cv2.imwrite(str(args.output), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))

    with args.scores_csv.open("w", newline="") as handle:
        fieldnames = ["rank_group", "name", "cldice", "dice", "precision", "recall", "accuracy", "image_path", "mask_path"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for prefix, row in selected:
            metrics = row["metrics"]
            writer.writerow(
                {
                    "rank_group": prefix,
                    "name": row["name"],
                    "cldice": metrics["cldice"],
                    "dice": metrics["dice"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "accuracy": metrics["accuracy"],
                    "image_path": row["image_path"],
                    "mask_path": row["mask_path"],
                }
            )

    print(f"config={config}")
    print(args.output)
    print(args.scores_csv)


if __name__ == "__main__":
    main()
