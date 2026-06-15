"""
Efficient test: precompute vesselness maps, then vary thresholding/postprocessing.
"""
import sys, csv, time
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent))
from vessel_pipeline import *
from advanced_pipeline import *

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'experiments' / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
AGRAWAL_ROOT = PROJECT_ROOT / 'data' / 'Agrawal2021'
CONFIG = VesselPipelineConfig()
N_RETCAM = 10
N_NEO = 10


def compute_all_soft_maps(rgb, fov, cfg):
    maps = {}
    for src in ['C6G6', 'C4G4', 'FlatSub+G6']:
        res = process_channel_source(rgb, fov, cfg, src)
        maps[f'{src}_combined'] = res['combined']
        maps[f'{src}_P10'] = res['P10']
        maps[f'{src}_TriClean'] = res['Triangle_clean']
    c6_ch = cielab_green_clahe_source(rgb, fov, l_clip=6.0, green_clip=6.0)
    c6_inv = normalize01((255 - c6_ch).astype(np.float32), fov)
    maps['C6_Gabor'] = gabor_filter_response(c6_inv, fov)
    maps['C6_DoG'] = cosfire_filter_response(c6_inv, fov)
    c6_comb = maps['C6G6_combined']
    maps['C6_Gabor_Fused'] = normalize01(0.7 * c6_comb + 0.3 * maps['C6_Gabor'], fov)
    maps['C6_Triple'] = normalize01(0.5 * c6_comb + 0.25 * maps['C6_Gabor'] + 0.25 * maps['C6_DoG'], fov)
    maps['MultiFused'] = fuse_multiple_sources(rgb, fov, cfg)
    return maps


def apply_variants(soft_maps, fov):
    results = []
    for key, soft in soft_maps.items():
        p10 = threshold_response_map(soft, fov, method='percentile', target_density=0.10)
        results.append((f'{key}_P10', (p10 > 0)))
        hyst = adaptive_hysteresis(soft, fov)
        results.append((f'{key}_Hyst', hyst))
        hyst_clean = morphological_cleanup(hyst, fov, min_vessel_area=4, close_radius=2)
        results.append((f'{key}_HystCln', hyst_clean))
        hyst_conn = vessel_connectivity_refinement(hyst, fov)
        results.append((f'{key}_HystConn', hyst_conn))
        elbow = threshold_with_elbow(soft, fov)
        results.append((f'{key}_Elbow', (elbow > 0)))
        elbow_clean = morphological_cleanup(elbow > 0, fov, min_vessel_area=4, close_radius=2)
        results.append((f'{key}_ElbowCln', elbow_clean))
        p10_clean = morphological_cleanup(p10 > 0, fov, min_vessel_area=4, close_radius=2)
        results.append((f'{key}_P10Cln', p10_clean))
    return results


def main():
    print("=" * 80)
    print("EFFICIENT ENHANCEMENT TEST")
    print("=" * 80)
    pairs = find_agrawal_pairs(AGRAWAL_ROOT)
    retcam = [p for p in pairs if p['source'] == 'RetCam']
    neo = [p for p in pairs if p['source'] == 'Neo']
    sample = retcam[:N_RETCAM] + neo[:N_NEO]
    print(f"Images: {len(sample)} ({N_RETCAM} RetCam + {N_NEO} Neo)")
    all_metrics = []
    for idx, pair in enumerate(sample):
        start_img = time.time()
        rgb = read_rgb(pair['image_path'])
        working = resize_max_side(rgb, CONFIG.process_max_side)
        fov = estimate_fov_mask(working)
        gt = read_binary_mask(pair['mask_path'], fov.shape)
        soft_maps = compute_all_soft_maps(working, fov, CONFIG)
        variants = apply_variants(soft_maps, fov)
        for name, binary in variants:
            metrics = segmentation_metrics(binary, gt)
            metrics.update({'image': pair['name'], 'source': pair['source'], 'method': name})
            all_metrics.append(metrics)
        elapsed = time.time() - start_img
        sys.stdout.write(f"\r  [{idx+1}/{len(sample)}] {pair['name']} - {len(variants)} variants in {elapsed:.1f}s")
        sys.stdout.flush()
    print("\n\nSaving metrics...")
    metrics_path = OUTPUT_DIR / 'enhancement_metrics.csv'
    write_csv(metrics_path, all_metrics)
    method_names = []
    seen = set()
    for m in all_metrics:
        if m['method'] not in seen:
            method_names.append(m['method'])
            seen.add(m['method'])
    summary = []
    for method in method_names:
        rows = [m for m in all_metrics if m['method'] == method]
        summary.append({
            'method': method,
            'dice': float(np.mean([r['dice'] for r in rows])),
            'sensitivity': float(np.mean([r['sensitivity'] for r in rows])),
            'precision': float(np.mean([r['precision'] for r in rows])),
            'iou': float(np.mean([r['iou'] for r in rows])),
        })
    summary.sort(key=lambda r: r['dice'], reverse=True)
    summary_path = OUTPUT_DIR / 'enhancement_summary.csv'
    write_csv(summary_path, summary)
    print(f"\n{'Method':45s} {'Dice':>8s} {'Sens':>8s} {'Prec':>8s} {'IoU':>8s}")
    print("-" * 80)
    for r in summary:
        print(f"{r['method']:45s} {r['dice']:>8.4f} {r['sensitivity']:>8.4f} {r['precision']:>8.4f} {r['iou']:>8.4f}")
    print(f"\n═══ TOP 10 ═══")
    for r in summary[:10]:
        print(f"{r['method']:45s} Dice={r['dice']:.4f}")


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == '__main__':
    main()
