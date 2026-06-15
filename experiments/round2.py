"""
Round 2 (focused): Color channel fusion + tuned hysteresis + preprocessing.
Only the most promising combos, fewer images, saves per-image.
"""
import sys, csv, time, json
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


def channel_process(channel, fov, clahe_clip=2.5):
    ch = np.clip(channel, 0, 255).astype(np.uint8)
    ch[~fov] = 0
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    enh = clahe.apply(ch)
    enh[~fov] = 0
    inv = 255 - enh
    inv_f = normalize01(inv.astype(np.float32), fov)
    matched = matched_filter_response(inv_f, fov)
    gabor = gabor_filter_response(inv_f, fov)
    dog = cosfire_filter_response(inv_f, fov)
    top_hat = modified_tophat(inv, fov)
    frangi_map = normalize01(safe_filter_call('frangi', inv_f), fov)
    jerman_map = jerman_vesselness(inv_f, fov)
    combined = fuse_vessel_responses(top_hat, matched, frangi_map, jerman_map, fov)
    combined_gabor = normalize01(0.7 * combined + 0.3 * gabor, fov)
    combined_dog = normalize01(0.7 * combined + 0.3 * dog, fov)
    final = normalize01(0.5 * combined + 0.25 * gabor + 0.25 * dog, fov)
    return {'combined': combined, 'gabor': gabor, 'dog': dog, 'matched': matched,
            'combined_gabor': combined_gabor, 'combined_dog': combined_dog, 'final': final}


def threshold_variants(soft, fov, name_prefix, gt):
    results = []
    for dens in [8, 10, 12]:
        th = threshold_response_map(soft, fov, method='percentile', target_density=dens/100.0)
        m = segmentation_metrics(th > 0, gt)
        m['method'] = f'{name_prefix}_P{dens:02d}'
        results.append(m)
    # Hysteresis with tuned percentiles
    inner = erode_mask(fov, 8)
    vals = soft[inner]
    vals = vals[vals > 0]
    if len(vals) > 50:
        high_th = float(np.percentile(vals, 88))
        low_th = float(np.percentile(vals, 55))
        if high_th > low_th:
            hyst = hysteresis_threshold(soft, fov, high_threshold=high_th, low_threshold=low_th)
            m_h = segmentation_metrics(hyst, gt)
            m_h['method'] = f'{name_prefix}_Hyst'
            results.append(m_h)
            # Hysteresis + cleanup
            cln = morphological_cleanup(hyst, fov, min_vessel_area=4, close_radius=2)
            m_c = segmentation_metrics(cln, gt)
            m_c['method'] = f'{name_prefix}_HystCln'
            results.append(m_c)
    return results


def main():
    print("=" * 70)
    print("ROUND 2: Color channels + fusion + hysteresis")
    print("=" * 70)
    pairs = find_agrawal_pairs(AGRAWAL_ROOT)
    retcam = [p for p in pairs if p['source'] == 'RetCam'][:5]
    neo = [p for p in pairs if p['source'] == 'Neo'][:5]
    sample = retcam + neo
    print(f"Images: {len(sample)} (5 RetCam + 5 Neo)")
    all_metrics = []

    for idx, pair in enumerate(sample):
        rgb = read_rgb(pair['image_path'])
        working = resize_max_side(rgb, CONFIG.process_max_side)
        fov = estimate_fov_mask(working)
        gt = read_binary_mask(pair['mask_path'], fov.shape)

        maps = {}

        # Individual channels
        for name, ch in [('Red', working[:,:,0]), ('Green', working[:,:,1]),
                         ('Blue', working[:,:,2]),
                         ('Gray', cv2.cvtColor(working, cv2.COLOR_RGB2GRAY))]:
            maps[name] = channel_process(ch, fov)

        # HSV/LAB
        hsv = cv2.cvtColor(working, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(working, cv2.COLOR_RGB2LAB)
        for name, ch in [('HSV_V', hsv[:,:,2]), ('LAB_L', lab[:,:,0]), ('LAB_A', lab[:,:,1])]:
            maps[name] = channel_process(ch, fov)

        # C6G6
        c6 = cielab_green_clahe_source(working, fov, l_clip=6.0, green_clip=6.0)
        maps['C6G6'] = channel_process(c6, fov)

        # Fusions
        fusions = [
            ('Green_Red', 'Green', 0.6, 'Red', 0.4),
            ('Green_Blue', 'Green', 0.7, 'Blue', 0.3),
            ('Green_LA', 'Green', 0.7, 'LAB_A', 0.3),
            ('Green_HSVV', 'Green', 0.7, 'HSV_V', 0.3),
            ('C6G6_Red', 'C6G6', 0.7, 'Red', 0.3),
            ('C6G6_Gray', 'C6G6', 0.7, 'Gray', 0.3),
        ]
        for fn_name, s1, w1, s2, w2 in fusions:
            c_combined = normalize01(w1 * maps[s1]['combined'] + w2 * maps[s2]['combined'], fov)
            c_final = normalize01(0.5 * c_combined + 0.25 * maps[s1]['gabor'] + 0.25 * maps[s2]['gabor'], fov)
            maps[fn_name] = {'combined': c_combined, 'final': c_final}

        # Test all
        for map_name, map_data in maps.items():
            for key in ['combined', 'gabor', 'dog', 'combined_gabor', 'final']:
                if key in map_data:
                    all_metrics.extend(threshold_variants(map_data[key], fov, f'{map_name}_{key}', gt))

        n_m = len([m for m in all_metrics if m.get('image') == pair['name']])
        sys.stdout.write(f"\r  [{idx+1}/{len(sample)}] {pair['name']} - {n_m} metrics")
        sys.stdout.flush()

    print(f"\n\nTotal metrics: {len(all_metrics)}")
    write_csv(OUTPUT_DIR / 'round2_metrics.csv', all_metrics)

    # Summary
    methods = []
    seen = set()
    for m in all_metrics:
        if m['method'] not in seen:
            methods.append(m['method'])
            seen.add(m['method'])

    summary = []
    for m in methods:
        rows = [r for r in all_metrics if r['method'] == m]
        summary.append({
            'method': m, 'dice': float(np.mean([r['dice'] for r in rows])),
            'sensitivity': float(np.mean([r['sensitivity'] for r in rows])),
            'precision': float(np.mean([r['precision'] for r in rows])),
        })
    summary.sort(key=lambda r: r['dice'], reverse=True)
    write_csv(OUTPUT_DIR / 'round2_summary.csv', summary)

    print(f"\n{'Method':45s} {'Dice':>8s} {'Sens':>8s} {'Prec':>8s}")
    print("-" * 72)
    for r in summary[:25]:
        print(f"{r['method']:45s} {r['dice']:>8.4f} {r['sensitivity']:>8.4f} {r['precision']:>8.4f}")

    print("\n═══ BEST PER CHANNEL (combined P10) ═══")
    for ch in ['Red', 'Green', 'Blue', 'Gray', 'HSV_V', 'LAB_L', 'LAB_A', 'C6G6']:
        rows = [r for r in summary if r['method'] == f'{ch}_combined_P10']
        if rows:
            print(f"  {ch:12s} Dice={rows[0]['dice']:.4f}  Sens={rows[0]['sensitivity']:.4f}  Prec={rows[0]['precision']:.4f}")

    print("\n═══ BEST FUSIONS (final P10) ═══")
    for f in ['Green_Red', 'Green_Blue', 'Green_LA', 'Green_HSVV', 'C6G6_Red', 'C6G6_Gray']:
        rows = [r for r in summary if r['method'] == f'{f}_final_P10']
        if rows:
            print(f"  {f:15s} Dice={rows[0]['dice']:.4f}  Sens={rows[0]['sensitivity']:.4f}  Prec={rows[0]['precision']:.4f}")


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
