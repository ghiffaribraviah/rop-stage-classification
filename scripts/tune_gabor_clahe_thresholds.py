from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from multiprocessing import get_context
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
    gabor_clahe_maps,
    segment_soft_response,
    workflow_tiles,
)


DEFAULT_TUNING_DIR = Path("output/00_debug_baseline")
_WORKER_SAMPLES: list[dict[str, object]] = []


def init_worker(samples: list[dict[str, object]]) -> None:
    global _WORKER_SAMPLES
    _WORKER_SAMPLES = samples
    cv2.setNumThreads(1)


def evaluate_config_job(job: tuple[GaborClaheConfig, str, bool]) -> dict[str, object]:
    config, phase, include_cldice = job
    row = evaluate_config(config, _WORKER_SAMPLES, include_cldice=include_cldice)
    row["phase"] = phase
    return row


def config_key(config: GaborClaheConfig) -> tuple[object, ...]:
    return (
        round(float(config.target_density), 5),
        round(float(config.main_low_mult), 5),
        round(float(config.main_high_mult), 5),
        bool(config.residual_enabled),
        round(float(config.residual_low_mult), 5),
        round(float(config.recovery_axis_ratio), 5),
        config.recovery_skeleton_length,
        round(float(config.recovery_branch_density), 5),
    )


def metric_row(tp: int, fp: int, fn: int, tn: int) -> dict[str, float]:
    precision = tp / float(tp + fp) if tp + fp else 0.0
    recall = tp / float(tp + fn) if tp + fn else 0.0
    accuracy = (tp + tn) / float(tp + fp + fn + tn) if tp + fp + fn + tn else 0.0
    dice = 2.0 * precision * recall / float(precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "dice": dice,
        "f1": dice,
    }


def cldice_row(pred: np.ndarray, gt: np.ndarray, gt_skeleton: np.ndarray) -> dict[str, float]:
    pred_skeleton = skeletonize(pred.astype(bool)).astype(bool)
    pred_skeleton_count = int(np.count_nonzero(pred_skeleton))
    gt_skeleton_count = int(np.count_nonzero(gt_skeleton))
    cl_precision = (
        int(np.count_nonzero(pred_skeleton & gt.astype(bool))) / float(pred_skeleton_count)
        if pred_skeleton_count
        else 0.0
    )
    cl_recall = (
        int(np.count_nonzero(gt_skeleton.astype(bool) & pred.astype(bool))) / float(gt_skeleton_count)
        if gt_skeleton_count
        else 0.0
    )
    cldice = (
        2.0 * cl_precision * cl_recall / float(cl_precision + cl_recall)
        if cl_precision + cl_recall
        else 0.0
    )
    return {
        "cl_precision": cl_precision,
        "cl_recall": cl_recall,
        "cldice": cldice,
    }


def evaluate_config(
    config: GaborClaheConfig,
    samples: list[dict[str, object]],
    include_cldice: bool = False,
) -> dict[str, object]:
    total_tp = total_fp = total_fn = total_tn = 0
    image_metrics = []
    for sample in samples:
        maps = segment_soft_response(
            sample["soft_response"],
            sample["valid_fov"],
            sample["final_subtraction"],
            config,
        )
        pred = maps["mask_final"].astype(bool) & sample["valid_fov"].astype(bool)
        gt = sample["gt"].astype(bool) & sample["valid_fov"].astype(bool)
        region = sample["valid_fov"].astype(bool)

        tp = int(np.count_nonzero(pred & gt & region))
        fp = int(np.count_nonzero(pred & ~gt & region))
        fn = int(np.count_nonzero(~pred & gt & region))
        tn = int(np.count_nonzero(~pred & ~gt & region))
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn
        row = metric_row(tp, fp, fn, tn)
        if include_cldice:
            row.update(cldice_row(pred, gt, sample["gt_skeleton"]))
        image_metrics.append(row)

    metrics = metric_row(total_tp, total_fp, total_fn, total_tn)
    cl_precision_values = [row["cl_precision"] for row in image_metrics if "cl_precision" in row]
    cl_recall_values = [row["cl_recall"] for row in image_metrics if "cl_recall" in row]
    cldice_values = [row["cldice"] for row in image_metrics if "cldice" in row]
    result: dict[str, object] = {
        **asdict(config),
        **metrics,
        "mean_precision": float(np.mean([row["precision"] for row in image_metrics])) if image_metrics else 0.0,
        "mean_recall": float(np.mean([row["recall"] for row in image_metrics])) if image_metrics else 0.0,
        "mean_accuracy": float(np.mean([row["accuracy"] for row in image_metrics])) if image_metrics else 0.0,
        "mean_f1": float(np.mean([row["f1"] for row in image_metrics])) if image_metrics else 0.0,
        "mean_dice": float(np.mean([row["dice"] for row in image_metrics])) if image_metrics else 0.0,
        "cl_precision": float(np.mean(cl_precision_values)) if cl_precision_values else "",
        "cl_recall": float(np.mean(cl_recall_values)) if cl_recall_values else "",
        "cldice": float(np.mean(cldice_values)) if cldice_values else "",
        "cldice_evaluated": bool(include_cldice),
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "tn": total_tn,
        "sample_count": len(samples),
    }
    return result


def evaluate_jobs(
    jobs: list[tuple[GaborClaheConfig, str, bool]],
    samples: list[dict[str, object]],
    workers: int,
    label: str,
    progress_every: int,
) -> list[dict[str, object]]:
    if not jobs:
        return []

    worker_count = max(1, int(workers))
    if worker_count == 1:
        rows = []
        for index, job in enumerate(jobs, start=1):
            rows.append(evaluate_config_job_with_samples(job, samples))
            if index % progress_every == 0 or index == len(jobs):
                print(f"{label} {index}/{len(jobs)}", flush=True)
        return rows

    rows = []
    ctx = get_context("fork")
    with ctx.Pool(processes=worker_count, initializer=init_worker, initargs=(samples,)) as pool:
        for index, row in enumerate(pool.imap_unordered(evaluate_config_job, jobs, chunksize=1), start=1):
            rows.append(row)
            if index % progress_every == 0 or index == len(jobs):
                print(f"{label} {index}/{len(jobs)}", flush=True)
    return rows


def evaluate_config_job_with_samples(
    job: tuple[GaborClaheConfig, str, bool],
    samples: list[dict[str, object]],
) -> dict[str, object]:
    config, phase, include_cldice = job
    row = evaluate_config(config, samples, include_cldice=include_cldice)
    row["phase"] = phase
    return row


def threshold_grid() -> list[GaborClaheConfig]:
    configs = []
    for target_density in (0.085, 0.095, 0.105, 0.115):
        for low_mult in (1.25, 1.35, 1.45):
            for high_mult in (0.50, 0.55, 0.60):
                configs.append(
                    GaborClaheConfig(
                        target_density=target_density,
                        main_low_mult=low_mult,
                        main_high_mult=high_mult,
                        residual_enabled=False,
                    )
                )
    return configs


def recovery_grid(base_configs: list[GaborClaheConfig]) -> list[GaborClaheConfig]:
    configs = []
    for base in base_configs:
        for residual_low_mult in (1.10, 1.22, 1.35, 1.50):
            for axis_ratio in (2.0, 2.3, 2.6):
                for skeleton_length in (8, 12, 18):
                    for branch_density in (0.10, 0.15, 0.20):
                        configs.append(
                            GaborClaheConfig(
                                target_density=base.target_density,
                                main_low_mult=base.main_low_mult,
                                main_high_mult=base.main_high_mult,
                                residual_enabled=True,
                                residual_low_mult=residual_low_mult,
                                recovery_axis_ratio=axis_ratio,
                                recovery_skeleton_length=skeleton_length,
                                recovery_branch_density=branch_density,
                            )
                        )
    return configs


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def precompute_samples(rows: list[tuple[str, Path, Path | None]], max_side: int) -> list[dict[str, object]]:
    samples = []
    for index, (name, image_path, mask_path) in enumerate(rows, start=1):
        if mask_path is None:
            continue
        rgb = resize_max_side(read_rgb(image_path), max_side)
        fov = estimate_fov_mask(rgb)
        gt = read_binary_mask(mask_path, fov.shape) & fov
        maps = gabor_clahe_maps(rgb, fov, config=GaborClaheConfig(residual_enabled=False))
        samples.append(
            {
                "name": name,
                "image_path": image_path,
                "mask_path": mask_path,
                "soft_response": maps["soft_response"],
                "valid_fov": maps["processing_fov"],
                "final_subtraction": maps["final_subtraction"],
                "gt": gt,
                "gt_skeleton": skeletonize(gt).astype(bool),
            }
        )
        if index % 10 == 0 or index == len(rows):
            print(f"precompute {index}/{len(rows)}", flush=True)
    return samples


def row_to_config(row: dict[str, object]) -> GaborClaheConfig:
    skeleton = row["recovery_skeleton_length"]
    residual_enabled = row["residual_enabled"]
    if isinstance(residual_enabled, str):
        residual_enabled = residual_enabled.lower() in {"1", "true", "yes"}
    return GaborClaheConfig(
        target_density=float(row["target_density"]),
        main_low_mult=float(row["main_low_mult"]),
        main_high_mult=float(row["main_high_mult"]),
        residual_enabled=bool(residual_enabled),
        residual_low_mult=float(row["residual_low_mult"]),
        recovery_axis_ratio=float(row["recovery_axis_ratio"]),
        recovery_skeleton_length=None if skeleton in ("", None) else int(skeleton),
        recovery_branch_density=float(row["recovery_branch_density"]),
    )


def per_image_metrics(config: GaborClaheConfig, samples: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for sample in samples:
        maps = segment_soft_response(
            sample["soft_response"],
            sample["valid_fov"],
            sample["final_subtraction"],
            config,
        )
        fov = sample["valid_fov"].astype(bool)
        pred = maps["mask_final"].astype(bool) & fov
        gt = sample["gt"].astype(bool) & fov
        tp = int(np.count_nonzero(pred & gt & fov))
        fp = int(np.count_nonzero(pred & ~gt & fov))
        fn = int(np.count_nonzero(~pred & gt & fov))
        tn = int(np.count_nonzero(~pred & ~gt & fov))
        metrics = metric_row(tp, fp, fn, tn)
        metrics.update(cldice_row(pred, gt, sample["gt_skeleton"]))
        rows.append(
            {
                "name": sample["name"],
                "image_path": sample["image_path"],
                "mask_path": sample["mask_path"],
                **metrics,
            }
        )
    return rows


def best_worst_rows(
    config: GaborClaheConfig,
    samples: list[dict[str, object]],
    count: int,
) -> list[dict[str, object]]:
    scored = per_image_metrics(config, samples)
    best = sorted(scored, key=lambda row: float(row["cldice"]), reverse=True)[:count]
    worst = sorted(scored, key=lambda row: float(row["cldice"]))[:count]
    selected = []
    for row in best:
        selected.append({"rank_group": "BEST", **row})
    for row in worst:
        selected.append({"rank_group": "WORST", **row})
    return selected


def best_worst_label(index: int, row: dict[str, object]) -> str:
    return (
        f"{index}. {row['rank_group']} {row['name']} "
        f"clD {float(row['cldice']):.3f} D {float(row['dice']):.3f} "
        f"P {float(row['precision']):.3f} R {float(row['recall']):.3f}"
    )


def write_best_worst_outputs(
    rows: list[dict[str, object]],
    config: GaborClaheConfig,
    output_path: Path,
    scores_path: Path,
    tile_size: int,
    max_side: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet_rows = []
    for index, row in enumerate(rows, start=1):
        rgb = resize_max_side(read_rgb(row["image_path"]), max_side)
        fov = estimate_fov_mask(rgb)
        gt = read_binary_mask(row["mask_path"], fov.shape) & fov
        tiles = [(best_worst_label(index, row), rgb)]
        tiles.extend(workflow_tiles(rgb, fov, gt, target_density=config.target_density, config=config)[1:])
        row_tiles = [add_label(image, label, tile_size) for label, image in tiles]
        sheet_rows.append(np.concatenate(row_tiles, axis=1))

    sheet = np.concatenate(sheet_rows, axis=0)
    cv2.imwrite(str(output_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))

    fieldnames = [
        "rank_group",
        "name",
        "cldice",
        "dice",
        "precision",
        "recall",
        "accuracy",
        "image_path",
        "mask_path",
    ]
    with scores_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fieldnames})


def metric_display(row: dict[str, object], key: str) -> str:
    value = row.get(key, "")
    if value == "":
        return "n/a"
    return f"{float(value):.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune Gabor+CLAHE P10 threshold and recovery parameters against GT masks.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_TUNING_DIR)
    parser.add_argument("--max-side", type=int, default=768)
    parser.add_argument("--tile-size", type=int, default=150)
    parser.add_argument("--retcam-count", type=int, default=5)
    parser.add_argument("--neo-count", type=int, default=5)
    parser.add_argument("--top-threshold-count", type=int, default=1)
    parser.add_argument("--precision-floor", type=float, default=0.45)
    parser.add_argument("--cldice-shortlist-size", type=int, default=30)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--visual-count", type=int, default=3)
    parser.add_argument("--best-worst-output", type=Path, default=None)
    parser.add_argument("--best-worst-scores", type=Path, default=None)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    if args.best_worst_output is None:
        args.best_worst_output = args.output_dir / "best_config_best_worst_cldice.jpg"
    if args.best_worst_scores is None:
        args.best_worst_scores = args.output_dir / "best_config_per_image_scores.csv"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    gt_rows = agrawal_rows(args.retcam_count, args.neo_count)
    samples = precompute_samples(gt_rows, args.max_side)
    if not samples:
        raise RuntimeError("No GT masks found for tuning.")
    print(f"precomputed samples={len(samples)}", flush=True)

    results: list[dict[str, object]] = []
    seen: set[tuple[object, ...]] = set()

    phase1_configs = threshold_grid()
    print(f"phase1 threshold configs={len(phase1_configs)} workers={max(1, int(args.workers))}", flush=True)
    phase1_jobs = [(config, "phase1_threshold_base", False) for config in phase1_configs]
    phase1_results = evaluate_jobs(
        phase1_jobs,
        samples,
        workers=args.workers,
        label="phase1",
        progress_every=12,
    )

    top_n = max(1, int(args.top_threshold_count))
    selected_rows = sorted(phase1_results, key=lambda row: float(row["dice"]), reverse=True)[:top_n]
    selected_rows.extend(sorted(phase1_results, key=lambda row: float(row["recall"]), reverse=True)[:top_n])
    selected_base_configs: list[GaborClaheConfig] = []
    selected_keys: set[tuple[object, ...]] = set()
    for row in selected_rows:
        config = row_to_config(row)
        key = config_key(config)
        if key not in selected_keys:
            selected_keys.add(key)
            selected_base_configs.append(config)

    baseline = GaborClaheConfig()
    all_phase2 = recovery_grid(selected_base_configs)
    all_phase2.append(baseline)
    print(f"selected threshold bases={len(selected_base_configs)} full configs={len(all_phase2)}", flush=True)
    full_jobs = []
    for config in all_phase2:
        key = config_key(config)
        if key in seen:
            continue
        phase = "baseline_full_t1_t2" if key == config_key(baseline) else "full_t1_t2"
        full_jobs.append((config, phase, False))
        seen.add(key)
    results = evaluate_jobs(
        full_jobs,
        samples,
        workers=args.workers,
        label="full grid",
        progress_every=25,
    )

    eligible_for_cldice = [
        row for row in results if float(row["precision"]) >= float(args.precision_floor)
    ] or results
    shortlist_size = max(1, int(args.cldice_shortlist_size))
    shortlist_rows = []
    shortlist_rows.extend(sorted(eligible_for_cldice, key=lambda row: float(row["dice"]), reverse=True)[:shortlist_size])
    shortlist_rows.extend(sorted(eligible_for_cldice, key=lambda row: float(row["recall"]), reverse=True)[:shortlist_size])
    shortlist_rows.extend(sorted(eligible_for_cldice, key=lambda row: float(row["mean_dice"]), reverse=True)[:shortlist_size])
    result_by_key = {config_key(row_to_config(row)): row for row in results}
    cldice_keys: set[tuple[object, ...]] = set()
    print(f"clDice shortlist candidates={len(shortlist_rows)}", flush=True)
    cldice_jobs = []
    for row in shortlist_rows:
        key = config_key(row_to_config(row))
        if key in cldice_keys:
            continue
        cldice_keys.add(key)
        cldice_jobs.append((row_to_config(row), str(row["phase"]), True))
    cldice_results = evaluate_jobs(
        cldice_jobs,
        samples,
        workers=args.workers,
        label="clDice",
        progress_every=1,
    )
    for updated in cldice_results:
        key = config_key(row_to_config(updated))
        result_by_key[key].update(updated)

    sort_fields = [
        "phase",
        "target_density",
        "main_low_mult",
        "main_high_mult",
        "residual_enabled",
        "residual_low_mult",
        "recovery_axis_ratio",
        "recovery_skeleton_length",
        "recovery_branch_density",
        "precision",
        "recall",
        "accuracy",
        "dice",
        "f1",
        "cl_precision",
        "cl_recall",
        "cldice",
        "cldice_evaluated",
        "mean_precision",
        "mean_recall",
        "mean_accuracy",
        "mean_dice",
        "mean_f1",
        "tp",
        "fp",
        "fn",
        "tn",
        "sample_count",
    ]
    phase1_ordered = [{field: row.get(field, "") for field in sort_fields} for row in phase1_results]
    ordered_results = [{field: row.get(field, "") for field in sort_fields} for row in results]
    write_csv(args.output_dir / "phase1_threshold_base_results.csv", phase1_ordered)
    write_csv(args.output_dir / "tuning_results_t1_t2.csv", ordered_results)

    eligible_results = [
        row for row in ordered_results if float(row["precision"]) >= float(args.precision_floor)
    ] or ordered_results
    cldice_results = [
        row for row in eligible_results if row["cldice_evaluated"] and row["cldice"] != ""
    ]
    top_by_cldice = sorted(cldice_results, key=lambda row: float(row["cldice"]), reverse=True)[:20]
    top_by_dice = sorted(eligible_results, key=lambda row: float(row["dice"]), reverse=True)[:20]
    top_by_recall = sorted(eligible_results, key=lambda row: float(row["recall"]), reverse=True)[:20]
    write_csv(args.output_dir / "top_by_cldice.csv", top_by_cldice)
    write_csv(args.output_dir / "top_by_dice.csv", top_by_dice)
    write_csv(args.output_dir / "top_by_recall.csv", top_by_recall)

    best_cldice_config = row_to_config(top_by_cldice[0])
    best_dice_config = row_to_config(top_by_dice[0])
    best_recall_config = row_to_config(top_by_recall[0])
    best_worst = best_worst_rows(best_cldice_config, samples, count=max(1, int(args.visual_count)))
    write_best_worst_outputs(
        best_worst,
        best_cldice_config,
        args.best_worst_output,
        args.best_worst_scores,
        tile_size=args.tile_size,
        max_side=args.max_side,
    )

    print(f"samples={len(samples)} configs={len(ordered_results)}")
    print(
        f"best_cldice={top_by_cldice[0]['cldice']:.4f} "
        f"dice={top_by_cldice[0]['dice']:.4f} precision={top_by_cldice[0]['precision']:.4f} "
        f"recall={top_by_cldice[0]['recall']:.4f}"
    )
    print(f"best_cldice_config={best_cldice_config}")
    print(
        f"best_dice={metric_display(top_by_dice[0], 'dice')} "
        f"cldice={metric_display(top_by_dice[0], 'cldice')} "
        f"precision={metric_display(top_by_dice[0], 'precision')} "
        f"recall={metric_display(top_by_dice[0], 'recall')}"
    )
    print(f"best_dice_config={best_dice_config}")
    print(
        f"best_recall={metric_display(top_by_recall[0], 'recall')} "
        f"precision={metric_display(top_by_recall[0], 'precision')} "
        f"dice={metric_display(top_by_recall[0], 'dice')} "
        f"cldice={metric_display(top_by_recall[0], 'cldice')}"
    )
    print(f"best_recall_config={best_recall_config}")
    print(args.best_worst_output)
    print(args.best_worst_scores)


if __name__ == "__main__":
    main()
