"""
Round 4: Entropy seeds → region growing. Combines best precision (entropy)
with best recall (region growing) for a two-stage approach.
Also: postprocessing refinements on the Gabor map.
"""
import sys, csv, time
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent))
from vessel_pipeline import *
from advanced_pipeline import gabor_filter_response, adaptive_hysteresis, morphological_cleanup
from round3 import entropy_threshold, region_growing_vessels, gaussian_matched_filter_response

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'experiments' / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
AGRAWAL_ROOT = PROJECT_ROOT / 'data' / 'Agrawal2021'
CONFIG = VesselPipelineConfig()


def entropy_seeded_region_growing(soft, fov, entropy_pct=99, grow_pct=40):
    """
    Two-stage: entropy threshold for clean seeds, then permissive growing.
    """
    inner = erode_mask(fov, 8)
    vals = soft[inner]
    vals = vals[vals > 0]
    if len(vals) < 50:
        return (soft > np.percentile(vals, 80)) if len(vals) > 10 else (soft > 0.5)

    # Stage 1: Entropy threshold for clean seeds
    ent_th = entropy_threshold(vals)
    # If entropy threshold is too high, use percentile
    if ent_th >= float(np.percentile(vals, 99)):
        ent_th = float(np.percentile(vals, entropy_pct))

    seeds = (soft >= ent_th) & inner

    # Stage 2: Grow into much weaker regions
    grow_th = float(np.percentile(vals, grow_pct))
    candidates = (soft >= grow_th) & inner

    # Iterative growing, up to 100 iterations
    result = seeds.copy()
    kernel = np.ones((3, 3), dtype=np.uint8)
    last_count = np.count_nonzero(result)
    for _ in range(100):
        dilated = cv2.dilate(result.astype(np.uint8), kernel, iterations=1).astype(bool)
        new_pixels = dilated & candidates & ~result
        result |= new_pixels
        current = np.count_nonzero(result)
        if current - last_count < 10:
            break
        last_count = current

    result = keep_components_at_least(result, 4)
    return result & fov


def entropy_only(soft, fov):
    """Pure entropy thresholding, best precision baseline."""
    inner = erode_mask(fov, 8)
    vals = soft[inner]
    vals = vals[vals > 0]
    if len(vals) < 50:
        return (soft > 0.5)
    th = entropy_threshold(vals)
    return (soft >= th) & inner


def iterative_cleanup(binary, fov, min_area=4, close_r=2, open_r=1, iterations=2):
    """
    Iterative morphological cleanup: open → close → remove small → repeat.
    """
    result = binary.copy()
    result[~fov] = 0
    for _ in range(iterations):
        if open_r > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*open_r+1, 2*open_r+1))
            result = cv2.morphologyEx(result.astype(np.uint8), cv2.MORPH_OPEN, k).astype(bool)
        result = keep_components_at_least(result, min_area)
        if close_r > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*close_r+1, 2*close_r+1))
            result = cv2.morphologyEx(result.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool)
        result[~fov] = 0
    result = remove_border_components(result, border_px=5)
    return result & fov


def fill_vessel_gaps(binary, fov, max_gap=10):
    """
    Fill small gaps/holes in vessel structures.
    Uses morphological reconstruction to fill holes without over-filling.
    """
    # Close small gaps
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool)
    # But only keep pixels that are connected to original binary
    n_labels, labels, _, _ = cv2.connectedComponentsWithStats(closed.astype(np.uint8), 8)
    output = np.zeros(binary.shape, dtype=bool)
    for label in range(1, n_labels):
        comp = labels == label
        if np.any(binary[comp]):
            output[comp] = True
    return output & fov


def main():
    print("=" * 70)
    print("ROUND 4: Entropy-seeded growing + iterative cleanup")
    print("=" * 70)

    pairs = find_agrawal_pairs(AGRAWAL_ROOT)
    retcam = [p for p in pairs if p['source'] == 'RetCam'][:8]
    neo = [p for p in pairs if p['source'] == 'Neo'][:8]
    sample = retcam + neo
    print(f"Images: {len(sample)}")

    all_metrics = []

    for idx, pair in enumerate(sample):
        rgb = read_rgb(pair['image_path'])
        working = resize_max_side(rgb, CONFIG.process_max_side)
        fov = estimate_fov_mask(working)
        gt = read_binary_mask(pair['mask_path'], fov.shape)

        # Get the best soft map: Gabor on C6G6
        c6_ch = cielab_green_clahe_source(working, fov, l_clip=6.0, green_clip=6.0)
        c6_inv_f = normalize01((255 - c6_ch).astype(np.float32), fov)
        gabor = gabor_filter_response(c6_inv_f, fov)

        # Also get the standard Almeida combined
        res = process_channel_source(working, fov, CONFIG, 'C6G6')
        almeida = res['combined']

        # Fuse them
        fused = normalize01(0.5 * almeida + 0.5 * gabor, fov)

        soft_maps = {
            'gabor': gabor,
            'almeida': almeida,
            'fused': fused,
        }

        for sname, soft in soft_maps.items():
            inner = erode_mask(fov, 8)
            vals = soft[inner]
            vals = vals[vals > 0]

            # P10 baseline
            p10 = threshold_response_map(soft, fov, method='percentile', target_density=0.10)
            m = segmentation_metrics(p10 > 0, gt)
            m['method'] = f'{sname}_P10'; all_metrics.append(m)

            # P10 + iterative cleanup
            p10_cln = iterative_cleanup(p10 > 0, fov, min_area=4, close_r=2, open_r=1, iterations=2)
            m = segmentation_metrics(p10_cln, gt)
            m['method'] = f'{sname}_P10_iClean'; all_metrics.append(m)

            # P10 + iterative cleanup + fill gaps
            p10_gap = fill_vessel_gaps(p10_cln, fov)
            m = segmentation_metrics(p10_gap, gt)
            m['method'] = f'{sname}_P10_FillGap'; all_metrics.append(m)

            # P08 (tighter)
            p08 = threshold_response_map(soft, fov, method='percentile', target_density=0.08)
            m = segmentation_metrics(p08 > 0, gt)
            m['method'] = f'{sname}_P08'; all_metrics.append(m)
            p08_cln = iterative_cleanup(p08 > 0, fov, min_area=4, close_r=1, open_r=1, iterations=1)
            m = segmentation_metrics(p08_cln, gt)
            m['method'] = f'{sname}_P08_iClean'; all_metrics.append(m)

            # Entropy only
            if len(vals) > 50:
                ent = entropy_only(soft, fov)
                m = segmentation_metrics(ent, gt)
                m['method'] = f'{sname}_Entropy'; all_metrics.append(m)

            # Hysteresis
            if len(vals) > 50:
                hyst = adaptive_hysteresis(soft, fov)
                m = segmentation_metrics(hyst, gt)
                m['method'] = f'{sname}_Hyst'; all_metrics.append(m)

                hyst_cln = iterative_cleanup(hyst, fov, min_area=4, close_r=2, open_r=1, iterations=1)
                m = segmentation_metrics(hyst_cln, gt)
                m['method'] = f'{sname}_Hyst_iClean'; all_metrics.append(m)

            # Entropy-seeded region growing (the new combo)
            if len(vals) > 50:
                esrg = entropy_seeded_region_growing(soft, fov)
                m = segmentation_metrics(esrg, gt)
                m['method'] = f'{sname}_EntSeedGrow'; all_metrics.append(m)

                esrg_cln = iterative_cleanup(esrg, fov, min_area=4, close_r=2, open_r=1, iterations=2)
                m = segmentation_metrics(esrg_cln, gt)
                m['method'] = f'{sname}_EntSeedGrow_iClean'; all_metrics.append(m)

                esrg_gap = fill_vessel_gaps(esrg_cln, fov)
                m = segmentation_metrics(esrg_gap, gt)
                m['method'] = f'{sname}_EntSeedGrow_FillGap'; all_metrics.append(m)

            # Region growing only (for reference)
            if len(vals) > 50:
                rg = region_growing_vessels(soft, fov)
                m = segmentation_metrics(rg, gt)
                m['method'] = f'{sname}_RegionGrow'; all_metrics.append(m)

        sys.stdout.write(f"\r  [{idx+1}/{len(sample)}] {pair['name']}")
        sys.stdout.flush()

    print(f"\n\nTotal: {len(all_metrics)} metrics")

    # Summary
    methods = list(dict.fromkeys([m['method'] for m in all_metrics]))
    summary = []
    for m in methods:
        rows = [r for r in all_metrics if r['method'] == m]
        summary.append({
            'method': m,
            'dice': float(np.mean([r['dice'] for r in rows])),
            'sensitivity': float(np.mean([r['sensitivity'] for r in rows])),
            'precision': float(np.mean([r['precision'] for r in rows])),
            'iou': float(np.mean([r['iou'] for r in rows])),
        })
    summary.sort(key=lambda r: r['dice'], reverse=True)

    write_csv(OUTPUT_DIR / 'round4_summary.csv', summary)
    write_csv(OUTPUT_DIR / 'round4_metrics.csv', all_metrics)

    print(f"\n{'Method':45s} {'Dice':>8s} {'Sens':>8s} {'Prec':>8s}")
    print("-" * 72)
    for r in summary:
        print(f"{r['method']:45s} {r['dice']:>8.4f} {r['sensitivity']:>8.4f} {r['precision']:>8.4f}")

    print("\n═══ BEST 15 ═══")
    for r in summary[:15]:
        print(f"  {r['method']:40s} Dice={r['dice']:.4f}  Sens={r['sensitivity']:.4f}  Prec={r['precision']:.4f}")


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
