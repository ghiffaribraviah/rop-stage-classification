"""
Visual threshold viewer: see how different thresholds affect vessel detection.
Run with:
  source experiments/.venv/bin/activate
  python experiments/show_thresholds.py
"""

import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from vessel_pipeline import *

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'experiments' / 'output' / 'threshold_views'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIG = VesselPipelineConfig()

# Pick a few representative images - one RetCam with GT, one Neo with GT
AGRAWAL_ROOT = PROJECT_ROOT / 'data' / 'Agrawal2021'

pairs = find_agrawal_pairs(AGRAWAL_ROOT)
# Pick 3 RetCam and 3 Neo with variety
selected = []
retcam = [p for p in pairs if p['source'] == 'RetCam']
neo = [p for p in pairs if p['source'] == 'Neo']
selected.extend(retcam[::10][:3])   # every 10th to get variety
selected.extend(neo[::10][:3])
print(f"Visualizing {len(selected)} images")

TILE = 200
THRESHOLDS = {
    'P05': ('percentile', 0.05),
    'P08': ('percentile', 0.08),
    'P10': ('percentile', 0.10),
    'P12': ('percentile', 0.12),
    'P14': ('percentile', 0.14),
    'P20': ('percentile', 0.20),
    'Triangle': ('triangle', 0.14),
    'Hybrid10': ('hybrid', 0.10),
}


def make_composite(img_idx, pair, source_type='C6G6'):
    rgb = read_rgb(pair['image_path'])
    working = resize_max_side(rgb, CONFIG.process_max_side)
    fov = estimate_fov_mask(working)
    gt = read_binary_mask(pair['mask_path'], fov.shape)

    res = process_channel_source(working, fov, CONFIG, source_type)
    combined = res['combined']

    rows = []

    # Row 0: original, FOV, GT, overlay of best threshold
    overlay_best = overlay_prediction(res['Triangle_clean'] > 0, gt)
    row0 = [
        add_tile_label(working, f'{img_idx}. Orig', TILE),
        add_tile_label(fov.astype(np.uint8)*255, 'FOV', TILE),
        add_tile_label(gt.astype(np.uint8)*255, 'Ground Truth', TILE),
        add_tile_label(overlay_best, 'TriClean overlay\n(TP=white FP=red FN=cyan)', TILE),
        add_tile_label(res['source_channel'], f'{source_type} source', TILE),
        add_tile_label(combined, f'{source_type} combined', TILE),
    ]
    while len(row0) < 8:
        row0.append(add_tile_label(empty_tile((TILE, TILE)), '', TILE))
    rows.append(row0)

    # Row 1: density thresholds
    thresh_row = []
    for name, (method, density) in THRESHOLDS.items():
        thresholded = threshold_response_map(combined, fov, method=method, target_density=density,
                                             fov_erode_px=max(0, int(CONFIG.fov_erode_px)))
        pred = thresholded > 0
        metrics = segmentation_metrics(pred, gt)
        label = f'{name}\nDice={metrics["dice"]:.3f}\nSens={metrics["sensitivity"]:.3f}\nPrec={metrics["precision"]:.3f}'
        overlay = overlay_prediction(pred, gt)
        thresh_row.append(add_tile_label(overlay, label, TILE))

    # Pad row if needed
    while len(thresh_row) < 8:
        thresh_row.append(add_tile_label(empty_tile((TILE, TILE)), '', TILE))
    rows.append(thresh_row)

    # Row 2: C4G4 for comparison
    res_c4 = process_channel_source(working, fov, CONFIG, 'C4G4')
    combined_c4 = res_c4['combined']
    c4_row = []
    for name, (method, density) in THRESHOLDS.items():
        thresholded = threshold_response_map(combined_c4, fov, method=method, target_density=density,
                                             fov_erode_px=max(0, int(CONFIG.fov_erode_px)))
        pred = thresholded > 0
        metrics = segmentation_metrics(pred, gt)
        label = f'C4 {name}\nDice={metrics["dice"]:.3f}'
        overlay = overlay_prediction(pred, gt)
        c4_row.append(add_tile_label(overlay, label, TILE))
    while len(c4_row) < 8:
        c4_row.append(add_tile_label(empty_tile((TILE, TILE)), '', TILE))
    rows.append(c4_row)

    # Row 3: FlatSub+G6 comparison
    res_flat = process_channel_source(working, fov, CONFIG, 'FlatSub+G6')
    combined_flat = res_flat['combined']
    flat_row = []
    for name, (method, density) in THRESHOLDS.items():
        thresholded = threshold_response_map(combined_flat, fov, method=method, target_density=density,
                                             fov_erode_px=max(0, int(CONFIG.fov_erode_px)))
        pred = thresholded > 0
        metrics = segmentation_metrics(pred, gt)
        label = f'FlatSub {name}\nDice={metrics["dice"]:.3f}'
        overlay = overlay_prediction(pred, gt)
        flat_row.append(add_tile_label(overlay, label, TILE))
    while len(flat_row) < 8:
        flat_row.append(add_tile_label(empty_tile((TILE, TILE)), '', TILE))
    rows.append(flat_row)

    sheet = np.concatenate([np.concatenate(r, axis=1) for r in rows], axis=0)
    return sheet


if __name__ == '__main__':
    sheets = []
    for i, pair in enumerate(selected):
        print(f"  Processing {pair['source']} {pair['name']}...")
        sheet = make_composite(i+1, pair, 'C6G6')
        sheets.append(sheet)

    # Stack all images vertically
    final = np.concatenate(sheets, axis=0)
    out_path = OUTPUT_DIR / 'threshold_comparison.jpg'
    cv2.imwrite(str(out_path), cv2.cvtColor(final, cv2.COLOR_RGB2BGR))
    print(f"\nSaved: {out_path}")
    print(f"Image size: {final.shape[1]}x{final.shape[0]}")
