"""
Experiment runner: compare channel sources and fusion strategies
for retinal vessel segmentation against Agrawal2021 ground truth.

Usage:
    uv run python experiments/run_experiments.py

Output goes to experiments/output/
"""

import sys
import csv
import math
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from vessel_pipeline import *

# ── Paths ──────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / 'data'
AGRAWAL_ROOT = DATA_ROOT / 'Agrawal2021'
OUTPUT_DIR = PROJECT_ROOT / 'experiments' / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIG = VesselPipelineConfig(
    process_max_side=768,
    vesselness_mode="almeida",
    clahe_clip=2.5,
    background_sigma=30.0,
    threshold_method="triangle",
    target_density=0.14,
    fov_erode_px=12,
    min_component_area=4,
    bilateral_d=9,
    bilateral_sigma_color=40.0,
    bilateral_sigma_space=9.0,
)

# ── Source types to test ───────────────────────────────────────────────

SOURCE_TYPES = [
    'C4G4',         # CLAHE L*4 + Green4
    'C6G6',         # CLAHE L*6 + Green6
    'BH isolated',  # Blackhat isolated dark structures
    'BHboost+G6',   # Blackhat boost + CLAHE G6
    'FlatSub+G6',   # Gaussian subtract flatten + CLAHE G6
    'FlatDiv+G6',   # Gaussian divide flatten + CLAHE G6
]

THRESHOLD_LABELS = ['P08', 'P10', 'P12', 'P14', 'Triangle', 'Triangle_clean']

# ── Fusion variants ────────────────────────────────────────────────────

FUSION_VARIANTS = [
    # name, [(source_type, weight), ...]
    ('C6G6_only', [('C6G6', 1.0)]),
    ('BH_only', [('BH isolated', 1.0)]),
    ('C6G6+BH_0.9_0.1', [('C6G6', 0.9), ('BH isolated', 0.1)]),
    ('C6G6+BH_0.8_0.2', [('C6G6', 0.8), ('BH isolated', 0.2)]),
    ('C6G6+BH_0.7_0.3', [('C6G6', 0.7), ('BH isolated', 0.3)]),
    ('C6G6+BH_0.6_0.4', [('C6G6', 0.6), ('BH isolated', 0.4)]),
    ('C6G6+BH_0.5_0.5', [('C6G6', 0.5), ('BH isolated', 0.5)]),
    ('C6G6+BHboost_0.8_0.2', [('C6G6', 0.8), ('BHboost+G6', 0.2)]),
    ('C6G6+FlatSub_0.8_0.2', [('C6G6', 0.8), ('FlatSub+G6', 0.2)]),
]


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def make_composite_visualization(
    rgb: np.ndarray, fov: np.ndarray, gt: np.ndarray | None,
    results: dict[str, dict], tile_size: int = 150,
) -> np.ndarray:
    """
    Create a visual comparison sheet showing source images, combined maps,
    and threshold variants side by side.
    """
    rows_tiles = []

    # Row 1: original, FOV, GT
    header_row = [
        add_tile_label(rgb, 'Original', tile_size),
        add_tile_label(fov.astype(np.uint8) * 255, 'FOV', tile_size),
    ]
    if gt is not None:
        header_row.append(add_tile_label(gt.astype(np.uint8) * 255, 'GT', tile_size))
        overlay = overlay_prediction(results.get('p10', np.zeros(fov.shape)) > 0, gt)
        header_row.append(add_tile_label(overlay, 'Overlay (P10)', tile_size))
    else:
        header_row.append(empty_tile((tile_size, tile_size)))
        header_row.append(empty_tile((tile_size, tile_size)))
    rows_tiles.append(header_row)

    # Per-source rows
    for name, res in results.items():
        row = [
        add_tile_label(res.get('source_channel', np.zeros(tuple(fov.shape) + (3,), dtype=np.uint8)),
                          f'{name} source', tile_size),
        add_tile_label(res.get('combined', np.zeros(fov.shape, dtype=np.float32)),
                          f'{name} combined', tile_size),
        ]
        for thresh in ['P08', 'P10', 'P12', 'Triangle']:
            img = res.get(thresh, np.zeros(fov.shape, dtype=np.float32))
            row.append(add_tile_label(img, f'{name} {thresh}', tile_size))
        rows_tiles.append(row)

    sheet = np.concatenate([np.concatenate(r, axis=1) for r in rows_tiles], axis=0)
    return sheet


# ═══════════════════════════════════════════════════════════════════════
#  EXPERIMENT 1: Channel Source Comparison
# ═══════════════════════════════════════════════════════════════════════

def experiment_channel_sources(pairs: list[dict]) -> Path:
    """Compare each channel source type across all Agrawal samples."""
    print("=" * 70)
    print("EXPERIMENT 1: Channel Source Comparison")
    print("=" * 70)

    all_metrics = []
    image_results = {}

    for idx, pair in enumerate(pairs):
        rgb = read_rgb(pair['image_path'])
        working_rgb = resize_max_side(rgb, CONFIG.process_max_side)
        fov = estimate_fov_mask(working_rgb)
        gt = read_binary_mask(pair['mask_path'], fov.shape)

        name = f"{pair['source']}_{pair['name']}"
        image_results[name] = {}
        sample_metrics = []

        for source_type in SOURCE_TYPES:
            start = time.time()
            res = process_channel_source(working_rgb, fov, CONFIG, source_type)
            elapsed = time.time() - start

            image_results[name][source_type] = res

            for thresh in THRESHOLD_LABELS:
                if thresh not in res:
                    continue
                pred = res[thresh] > 0
                metrics = segmentation_metrics(pred, gt)
                metrics.update({
                    'image': pair['name'],
                    'source': pair['source'],
                    'channel_source': source_type,
                    'threshold': thresh,
                    'time_sec': round(elapsed, 3),
                })
                all_metrics.append(metrics)
                sample_metrics.append(metrics)

            sys.stdout.write(f'    {source_type:15s} done in {elapsed:.1f}s\n')
            sys.stdout.flush()

            # Also report the combined (continuous) soft map AUPRC
            if gt is not None:
                from sklearn.metrics import average_precision_score, roc_auc_score
                y_true = gt.reshape(-1).astype(np.uint8)
                y_score = res['combined'].reshape(-1).astype(np.float32)
                auprc = float(average_precision_score(y_true, y_score))
                roc = float(roc_auc_score(y_true, y_score)) if np.unique(y_true).size == 2 else float('nan')
                all_metrics.append({
                    'image': pair['name'],
                    'source': pair['source'],
                    'channel_source': source_type,
                    'threshold': 'SOFT_AUPRC',
                    'dice': auprc,
                    'iou': roc,
                    'precision': 0, 'sensitivity': 0,
                    'specificity': 0, 'accuracy': 0,
                    'pred_density': 0, 'gt_density': 0,
                    'time_sec': 0,
                })

        sys.stdout.write(f"  [{idx+1}/{len(pairs)}] {pair['source']} {pair['name']} — {len(sample_metrics)} metrics\n")
        sys.stdout.flush()

    # Save all metrics
    metrics_path = OUTPUT_DIR / 'exp1_channel_source_metrics.csv'
    write_csv(metrics_path, all_metrics)
    print(f"\nSaved metrics: {metrics_path}")

    # Compute summary per (channel_source, threshold)
    summary = {}
    for source_type in SOURCE_TYPES:
        for thresh in THRESHOLD_LABELS + ['SOFT_AUPRC']:
            key = (source_type, thresh)
            rows = [m for m in all_metrics
                    if m['channel_source'] == source_type and m['threshold'] == thresh]
            if not rows:
                continue
            numeric = {k: [] for k in ['dice', 'iou', 'precision', 'sensitivity', 'specificity',
                                        'pred_density', 'gt_density']}
            for r in rows:
                for k in numeric:
                    numeric[k].append(r[k])
            summary[key] = {k: float(np.mean(v)) for k, v in numeric.items()}

    summary_rows = []
    for (source_type, thresh), metrics in sorted(summary.items()):
        summary_rows.append({'channel_source': source_type, 'threshold': thresh, **metrics})

    summary_path = OUTPUT_DIR / 'exp1_channel_source_summary.csv'
    write_csv(summary_path, summary_rows)
    print(f"Saved summary: {summary_path}")

    # Print best per source type
    print("\n── Best Dice per channel source (P10 threshold) ──")
    for source_type in SOURCE_TYPES:
        rows = [m for m in all_metrics
                if m['channel_source'] == source_type and m['threshold'] == 'P10']
        if rows:
            dice = float(np.mean([r['dice'] for r in rows]))
            print(f"  {source_type:20s}  Dice={dice:.4f}")

    return metrics_path


# ═══════════════════════════════════════════════════════════════════════
#  EXPERIMENT 2: C6G6 + BH Isolated Fusion (inverted-level)
# ═══════════════════════════════════════════════════════════════════════

def experiment_fusion(pairs: list[dict]) -> Path:
    """Test fusion of multiple channel sources at the inverted-float level."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: C6G6 + BH Isolated Fusion")
    print("=" * 70)

    all_metrics = []

    for idx, pair in enumerate(pairs):
        rgb = read_rgb(pair['image_path'])
        working_rgb = resize_max_side(rgb, CONFIG.process_max_side)
        fov = estimate_fov_mask(working_rgb)
        gt = read_binary_mask(pair['mask_path'], fov.shape)

        # First, get individual results for C6G6 and BH (for reference)
        ref_c6g6 = process_channel_source(working_rgb, fov, CONFIG, 'C6G6')
        ref_bh = process_channel_source(working_rgb, fov, CONFIG, 'BH isolated')

        for variant_name, sources in FUSION_VARIANTS:
            try:
                fused = process_fused_sources(working_rgb, fov, CONFIG, sources)
            except Exception as e:
                print(f"  ERROR {pair['name']} {variant_name}: {e}")
                continue

            for thresh in ['P08', 'P10', 'P12', 'P14', 'Triangle']:
                if thresh not in fused:
                    continue
                pred = fused[thresh] > 0
                metrics = segmentation_metrics(pred, gt)
                metrics.update({
                    'image': pair['name'],
                    'source': pair['source'],
                    'variant': variant_name,
                    'threshold': thresh,
                })
                all_metrics.append(metrics)

        sys.stdout.write(f"  [{idx+1}/{len(pairs)}] {pair['source']} {pair['name']} — {len(FUSION_VARIANTS)} variants\n")
        sys.stdout.flush()

    metrics_path = OUTPUT_DIR / 'exp2_fusion_metrics.csv'
    write_csv(metrics_path, all_metrics)
    print(f"\nSaved metrics: {metrics_path}")

    # Summary per variant
    summary = {}
    for variant_name, _ in FUSION_VARIANTS:
        for thresh in ['P08', 'P10', 'P12', 'P14', 'Triangle']:
            key = (variant_name, thresh)
            rows = [m for m in all_metrics
                    if m['variant'] == variant_name and m['threshold'] == thresh]
            if not rows:
                continue
            dice = float(np.mean([r['dice'] for r in rows]))
            if key not in summary:
                summary[key] = {}
            summary[key]['dice'] = dice
            summary[key]['iou'] = float(np.mean([r['iou'] for r in rows]))
            summary[key]['precision'] = float(np.mean([r['precision'] for r in rows]))
            summary[key]['sensitivity'] = float(np.mean([r['sensitivity'] for r in rows]))

    summary_rows = []
    for (variant, thresh), metrics in sorted(summary.items()):
        summary_rows.append({'variant': variant, 'threshold': thresh, **metrics})

    summary_path = OUTPUT_DIR / 'exp2_fusion_summary.csv'
    write_csv(summary_path, summary_rows)
    print(f"Saved summary: {summary_path}")

    # Best Dice per variant at P10
    print("\n── Best Dice per fusion variant (P10) ──")
    for variant_name, _ in FUSION_VARIANTS:
        rows = [m for m in all_metrics
                if m['variant'] == variant_name and m['threshold'] == 'P10']
        if rows:
            dice = float(np.mean([r['dice'] for r in rows]))
            sens = float(np.mean([r['sensitivity'] for r in rows]))
            prec = float(np.mean([r['precision'] for r in rows]))
            print(f"  {variant_name:30s}  Dice={dice:.4f}  Sens={sens:.4f}  Prec={prec:.4f}")

    return metrics_path


# ═══════════════════════════════════════════════════════════════════════
#  EXPERIMENT 3: Full filter fusion + C6G6+BH at combined level
# ═══════════════════════════════════════════════════════════════════════

def experiment_combined_fusion(pairs: list[dict]) -> Path:
    """
    Fuse C6G6 and BH isolated at the COMBINED (after full filter stack) level.
    This is different from experiment 2 which fuses at the inverted-float level.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Combined-map-level Fusion (C6G6 + BH)")
    print("=" * 70)

    FUSION_WEIGHTS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    all_metrics = []

    for idx, pair in enumerate(pairs):
        rgb = read_rgb(pair['image_path'])
        working_rgb = resize_max_side(rgb, CONFIG.process_max_side)
        fov = estimate_fov_mask(working_rgb)
        gt = read_binary_mask(pair['mask_path'], fov.shape)

        # Get C6G6 and BH combined maps
        res_c6g6 = process_channel_source(working_rgb, fov, CONFIG, 'C6G6')
        res_bh = process_channel_source(working_rgb, fov, CONFIG, 'BH isolated')

        c6_combined = res_c6g6['combined']
        bh_combined = res_bh['combined']

        for w in FUSION_WEIGHTS:
            w_c6 = 1.0 - w
            w_bh = w
            combined = normalize01(w_c6 * c6_combined + w_bh * bh_combined, fov)

            for density in [8, 10, 12]:
                label = f'P{density:02d}'
                thresholded = threshold_response_map(
                    combined, fov, method='percentile', target_density=density / 100.0,
                    fov_erode_px=max(0, int(CONFIG.fov_erode_px)))
                pred = thresholded > 0
                metrics = segmentation_metrics(pred, gt)
                metrics.update({
                    'image': pair['name'],
                    'source': pair['source'],
                    'variant': f'C6G6_wc6={w_c6:.1f}_wbh={w_bh:.1f}',
                    'threshold': label,
                })
                all_metrics.append(metrics)

        sys.stdout.write(f"  [{idx+1}/{len(pairs)}] {pair['source']} {pair['name']} — {len(FUSION_WEIGHTS)} weights\n")
        sys.stdout.flush()

    metrics_path = OUTPUT_DIR / 'exp3_combined_fusion_metrics.csv'
    write_csv(metrics_path, all_metrics)
    print(f"\nSaved metrics: {metrics_path}")

    # Summary
    summary_rows = []
    for w in FUSION_WEIGHTS:
        w_c6 = 1.0 - w
        w_bh = w
        variant = f'C6G6_wc6={w_c6:.1f}_wbh={w_bh:.1f}'
        for density in [8, 10, 12]:
            label = f'P{density:02d}'
            rows = [m for m in all_metrics if m['variant'] == variant and m['threshold'] == label]
            if rows:
                summary_rows.append({
                    'variant': variant,
                    'w_c6': round(w_c6, 1),
                    'w_bh': round(w_bh, 1),
                    'threshold': label,
                    'dice': float(np.mean([r['dice'] for r in rows])),
                    'iou': float(np.mean([r['iou'] for r in rows])),
                    'precision': float(np.mean([r['precision'] for r in rows])),
                    'sensitivity': float(np.mean([r['sensitivity'] for r in rows])),
                })

    summary_path = OUTPUT_DIR / 'exp3_combined_fusion_summary.csv'
    write_csv(summary_path, summary_rows)
    print(f"Saved summary: {summary_path}")

    # Print best combined weights
    print("\n── Best Dice per fusion weight (P10) ──")
    for w in FUSION_WEIGHTS:
        w_c6 = 1.0 - w
        w_bh = w
        variant = f'C6G6_wc6={w_c6:.1f}_wbh={w_bh:.1f}'
        rows = [m for m in all_metrics if m['variant'] == variant and m['threshold'] == 'P10']
        if rows:
            dice = float(np.mean([r['dice'] for r in rows]))
            sens = float(np.mean([r['sensitivity'] for r in rows]))
            prec = float(np.mean([r['precision'] for r in rows]))
            print(f"  C6G6={w_c6:.1f} BH={w_bh:.1f}  Dice={dice:.4f}  Sens={sens:.4f}  Prec={prec:.4f}")

    return metrics_path


# ═══════════════════════════════════════════════════════════════════════
#  EXPERIMENT 4: Threshold sweep on best configuration
# ═══════════════════════════════════════════════════════════════════════

def experiment_threshold_sweep(pairs: list[dict]) -> Path:
    """Fine-grained threshold sweep on the most promising source."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Threshold Sweep on Best Config")
    print("=" * 70)

    # We'll test the best C6G6 and best C6G6+BH fusion from exp3
    # On each with a range of percentile densities from 5% to 20%
    best_configs = [
        ('C6G6_only', lambda rgb, fov: process_channel_source(rgb, fov, CONFIG, 'C6G6')),
        ('C6G6+BH_0.8_0.2_combined',
         lambda rgb, fov: process_fused_sources(rgb, fov, CONFIG, [('C6G6', 0.8), ('BH isolated', 0.2)])),
    ]

    DENSITIES = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20]
    all_metrics = []

    for idx, pair in enumerate(pairs):
        rgb = read_rgb(pair['image_path'])
        working_rgb = resize_max_side(rgb, CONFIG.process_max_side)
        fov = estimate_fov_mask(working_rgb)
        gt = read_binary_mask(pair['mask_path'], fov.shape)

        for cfg_name, cfg_fn in best_configs:
            res = cfg_fn(working_rgb, fov)
            combined = res.get('combined') if isinstance(res, dict) else res[0]

            for density in DENSITIES:
                thresholded = threshold_response_map(
                    combined, fov, method='percentile',
                    target_density=density / 100.0,
                    fov_erode_px=max(0, int(CONFIG.fov_erode_px)))
                pred = thresholded > 0
                metrics = segmentation_metrics(pred, gt)
                metrics.update({
                    'image': pair['name'],
                    'source': pair['source'],
                    'config': cfg_name,
                    'density_pct': density,
                })
                all_metrics.append(metrics)

        if (idx + 1) % 5 == 0 or idx == len(pairs) - 1:
            print(f"  [{idx+1}/{len(pairs)}] done")

    metrics_path = OUTPUT_DIR / 'exp4_threshold_sweep_metrics.csv'
    write_csv(metrics_path, all_metrics)
    print(f"\nSaved metrics: {metrics_path}")

    # Summary
    summary_rows = []
    for cfg_name, _ in best_configs:
        for density in DENSITIES:
            rows = [m for m in all_metrics
                    if m['config'] == cfg_name and m['density_pct'] == density]
            if rows:
                summary_rows.append({
                    'config': cfg_name,
                    'density_pct': density,
                    'dice': float(np.mean([r['dice'] for r in rows])),
                    'iou': float(np.mean([r['iou'] for r in rows])),
                    'precision': float(np.mean([r['precision'] for r in rows])),
                    'sensitivity': float(np.mean([r['sensitivity'] for r in rows])),
                    'pred_density': float(np.mean([r['pred_density'] for r in rows])),
                })

    summary_path = OUTPUT_DIR / 'exp4_threshold_sweep_summary.csv'
    write_csv(summary_path, summary_rows)
    print(f"Saved summary: {summary_path}")

    # Best per config
    print("\n── Best density per config (by Dice) ──")
    for cfg_name, _ in best_configs:
        rows = [r for r in summary_rows if r['config'] == cfg_name]
        if rows:
            best = max(rows, key=lambda r: r['dice'])
            print(f"  {cfg_name:30s}  density={best['density_pct']}%  Dice={best['dice']:.4f}  "
                  f"Sens={best['sensitivity']:.4f}  Prec={best['precision']:.4f}")

    return metrics_path


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("ROP Vessel Pipeline Experiments")
    print(f"Agrawal root: {AGRAWAL_ROOT}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    # Find Agrawal paired images
    pairs = find_agrawal_pairs(AGRAWAL_ROOT)
    if not pairs:
        print("ERROR: No Agrawal2021 pairs found!")
        print(f"Checked under: {AGRAWAL_ROOT}")
        sys.exit(1)

    print(f"Found {len(pairs)} image/mask pairs")
    print(f"  RetCam: {len([p for p in pairs if p['source'] == 'RetCam'])}")
    print(f"  Neo:    {len([p for p in pairs if p['source'] == 'Neo'])}")
    print()

    # Use a subset for faster iteration during development
    # Full set: 100 pairs. Subset: first 10 RetCam + 10 Neo
    import random
    random.seed(42)
    retcam = [p for p in pairs if p['source'] == 'RetCam']
    neo = [p for p in pairs if p['source'] == 'Neo']
    sample_pairs = retcam[:10] + neo[:10]  # 20 total
    print(f"Using subset: {len(sample_pairs)} images ({len(retcam[:10])} RetCam + {len(neo[:10])} Neo)")
    print()

    # Run experiments
    experiment_channel_sources(sample_pairs)
    experiment_fusion(sample_pairs)
    experiment_combined_fusion(sample_pairs)
    experiment_threshold_sweep(sample_pairs)

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print(f"Results in: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    main()
