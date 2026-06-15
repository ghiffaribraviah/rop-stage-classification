"""
Z-Fused-Coherence: coherence-enhancing anisotropic diffusion preprocessing +
Z-score normalized fusion of multiple vesselness filters.
"""
import sys, csv, time
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent))
from vessel_pipeline import *
from advanced_pipeline import gabor_filter_response, adaptive_hysteresis

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'experiments' / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
AGRAWAL_ROOT = PROJECT_ROOT / 'data' / 'Agrawal2021'
CONFIG = VesselPipelineConfig()


def coherence_diffusion(channel, fov, iterations=10, kappa=15, dt=0.1):
    """
    Coherence-enhancing anisotropic diffusion (Weickert 1999).
    Diffuses along vessel direction, preserves edges perpendicular to vessel.
    Simplified: edge-stopping Perona-Malik style anisotropic diffusion.
    """
    img = channel.astype(np.float32) / 255.0
    img[~fov] = 0
    filled = img.copy()
    if np.any(fov):
        filled[~fov] = float(np.median(img[fov]))
    
    for _ in range(iterations):
        grad_x = cv2.Sobel(filled, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(filled, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(grad_x**2 + grad_y**2) + 1e-10
        # Edge-stopping function (Perona-Malik): c = exp(-(mag/kappa)^2)
        c = np.exp(-(mag / kappa)**2)
        # Diffusivity along edges: more diffusion in flat areas, less at edges
        div_x = cv2.Sobel(c * grad_x, cv2.CV_32F, 1, 0, ksize=3)
        div_y = cv2.Sobel(c * grad_y, cv2.CV_32F, 0, 1, ksize=3)
        filled = filled + dt * (div_x + div_y)
        filled[~fov] = 0 if not np.any(fov) else float(np.median(img[fov]))
    
    result = np.clip(filled * 255, 0, 255).astype(np.uint8)
    result[~fov] = 0
    return result


def zscore_normalize(response, fov):
    """Z-score normalize a vesselness response within FOV."""
    r = response.astype(np.float32)
    r[~fov] = 0
    vals = r[fov]
    if vals.size < 10:
        return r
    mean = float(vals.mean())
    std = float(max(vals.std(), 1e-8))
    r = (r - mean) / std
    r[~fov] = 0
    return r


def zscore_fusion(maps, fov, weights=None):
    """
    Z-score normalize each map, then weighted average.
    """
    normalized = []
    for i, m in enumerate(maps):
        z = zscore_normalize(m, fov)
        normalized.append(z)
    
    if weights is None:
        weights = [1.0 / len(maps)] * len(maps)
    
    fused = np.zeros_like(normalized[0])
    for z, w in zip(normalized, weights):
        fused += w * z
    fused[~fov] = 0
    # Re-normalize to [0, 1]
    return normalize01(fused, fov)


def double_threshold_with_connectivity(soft, fov, low_th=0.3, high_th=0.6):
    """
    Dual-threshold decision (DTD) from MDF-Net paper.
    Pixels above high_th are sure vessel.
    Pixels between low_th and high_th are vessel only if connected to sure vessel.
    """
    inner = erode_mask(fov, 8)
    working = soft.astype(np.float32).copy()
    working[~inner] = 0
    
    # Normalize to [0, 1]
    vals = working[inner]
    vals = vals[vals > 0]
    if len(vals) > 10:
        low_v = np.percentile(vals, 1)
        high_v = np.percentile(vals, 99)
        if high_v > low_v:
            working = np.clip((working - low_v) / (high_v - low_v), 0, 1)
    
    sure_vessel = (working >= high_th) & inner
    uncertain = (working >= low_th) & (working < high_th) & inner
    
    # Connected component analysis: keep uncertain pixels connected to sure vessel
    combined = np.zeros(working.shape, dtype=bool)
    combined[sure_vessel | uncertain] = True
    
    n_labels, labels, _, _ = cv2.connectedComponentsWithStats(combined.astype(np.uint8), 8)
    output = np.zeros(working.shape, dtype=bool)
    for label in range(1, n_labels):
        comp = labels == label
        if np.any(sure_vessel[comp]):
            output[comp] = True
    
    return output & inner


def main():
    print("=" * 70)
    print("Z-FUSED-COHERENCE: Coherence diffusion + Z-score fusion + DTD")
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

        # ── Coherence-enhanced preprocessing ──
        green = working[:,:,1].copy()
        green_coh = coherence_diffusion(green, fov, iterations=10, kappa=15, dt=0.1)
        
        # Also coherence on CLAHE-enhanced green
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        green_clahe = clahe.apply(green)
        green_clahe[~fov] = 0
        green_coh_clahe = coherence_diffusion(green_clahe, fov, iterations=8, kappa=12, dt=0.08)
        
        # C6G6 source (with and without coherence)
        c6_raw = cielab_green_clahe_source(working, fov, l_clip=6.0, green_clip=6.0)
        c6_coh = cielab_green_clahe_source_from_channel(green_coh_clahe, fov)
        
        # ── Compute vesselness responses ──
        def get_gabor(ch, fov):
            inv = 255 - np.clip(ch, 0, 255).astype(np.uint8)
            inv_f = normalize01(inv.astype(np.float32), fov)
            return gabor_filter_response(inv_f, fov)
        
        def get_almeida(ch, fov):
            inv = 255 - np.clip(ch, 0, 255).astype(np.uint8)
            inv_f = normalize01(inv.astype(np.float32), fov)
            top_hat = modified_tophat(inv, fov)
            matched = matched_filter_response(inv_f, fov)
            frangi_map = normalize01(safe_filter_call('frangi', inv_f), fov)
            jerman_map = jerman_vesselness(inv_f, fov)
            return fuse_vessel_responses(top_hat, matched, frangi_map, jerman_map, fov)
        
        # Responses from different sources
        maps_raw = {
            'gabor_raw_green': get_gabor(green, fov),
            'gabor_raw_clahe': get_gabor(green_clahe, fov),
            'gabor_coh_green': get_gabor(green_coh, fov),
            'gabor_coh_clahe': get_gabor(green_coh_clahe, fov),
            'gabor_c6': get_gabor(c6_raw, fov),
            'almeida_c6': get_almeida(c6_raw, fov),
        }
        
        # Try coherence on the C6G6 channel
        c6_inv = 255 - np.clip(c6_raw, 0, 255).astype(np.uint8)
        c6_inv_f = normalize01(c6_inv.astype(np.float32), fov)
        maps_raw['gabor_c6_inv'] = gabor_filter_response(c6_inv_f, fov)
        
        # Also try: gabor on inverted coherence-enhanced
        inv_coh = 255 - green_coh
        inv_coh_f = normalize01(inv_coh.astype(np.float32), fov)
        maps_raw['gabor_coh_inv'] = gabor_filter_response(inv_coh_f, fov)
        
        # ── Median filter on the best maps (from round6 findings) ──
        for key in ['gabor_c6', 'gabor_c6_inv']:
            if key in maps_raw:
                r = (maps_raw[key] * 255).astype(np.uint8)
                r = cv2.medianBlur(r, 7)
                r[~fov] = 0
                maps_raw[f'{key}_median7'] = normalize01(r.astype(np.float32), fov)
        
        # ── Z-score Fusions ──
        # Fusion 1: green + clahe + c6 gabor responses
        fusion_sets = [
            ('z_fuse_gabor3', ['gabor_raw_green', 'gabor_raw_clahe', 'gabor_c6_inv']),
            ('z_fuse_gabor2', ['gabor_raw_green', 'gabor_c6_inv']),
            ('z_fuse_coh', ['gabor_coh_green', 'gabor_coh_clave', 'gabor_c6_inv']),
            ('z_fuse_all', ['gabor_raw_green', 'gabor_raw_clahe', 'gabor_coh_green', 'gabor_c6_inv']),
            ('z_fuse_c6_best', ['gabor_c6_inv', 'almeida_c6']),
        ]
        
        for fname, keys in fusion_sets:
            available = [k for k in keys if k in maps_raw]
            if len(available) >= 2:
                maps_to_fuse = [maps_raw[k] for k in available]
                maps_raw[fname] = zscore_fusion(maps_to_fuse, fov)
        
        # ── Test all variants ──
        for sname, soft in maps_raw.items():
            inner = erode_mask(fov, 8)
            vals = soft[inner]
            vals = vals[vals > 0]
            
            for d in [8, 10, 12]:
                th = threshold_response_map(soft, fov, method='percentile', target_density=d/100.0)
                m = segmentation_metrics(th > 0, gt)
                m['method'] = f'{sname}_P{d:02d}'; all_metrics.append(m)
            
            # DTD (dual-threshold decision)
            if len(vals) > 50:
                for low_pct in [40, 50, 60]:
                    for high_pct in [85, 90, 92]:
                        low_th = float(np.percentile(vals, low_pct))
                        high_th = float(np.percentile(vals, high_pct))
                        if high_th > low_th:
                            dtd = double_threshold_with_connectivity(soft, fov, low_th=low_th, high_th=high_th)
                            m = segmentation_metrics(dtd, gt)
                            m['method'] = f'{sname}_DTD{low_pct}h{high_pct}'; all_metrics.append(m)
            
            # Median filter + P10
            r7 = cv2.medianBlur((soft * 255).astype(np.uint8), 7)
            r7[~fov] = 0
            soft_m7 = normalize01(r7.astype(np.float32), fov)
            for d in [10]:
                th = threshold_response_map(soft_m7, fov, method='percentile', target_density=d/100.0)
                m = segmentation_metrics(th > 0, gt)
                m['method'] = f'{sname}_median7_P{d:02d}'; all_metrics.append(m)
            
            # Hysteresis
            if len(vals) > 50:
                hyst = adaptive_hysteresis(soft, fov)
                m = segmentation_metrics(hyst, gt)
                m['method'] = f'{sname}_Hyst'; all_metrics.append(m)
        
        sys.stdout.write(f"\r  [{idx+1}/{len(sample)}] {pair['name']}")
        sys.stdout.flush()
    
    print(f"\n\nTotal: {len(all_metrics)} metrics")
    
    methods = list(dict.fromkeys([m['method'] for m in all_metrics]))
    summary = []
    for m in methods:
        rows = [r for r in all_metrics if r['method'] == m]
        summary.append({
            'method': m,
            'dice': float(np.mean([r['dice'] for r in rows])),
            'sensitivity': float(np.mean([r['sensitivity'] for r in rows])),
            'precision': float(np.mean([r['precision'] for r in rows])),
        })
    summary.sort(key=lambda r: r['dice'], reverse=True)
    write_csv(OUTPUT_DIR / 'round7_zfc_summary.csv', summary)
    write_csv(OUTPUT_DIR / 'round7_zfc_metrics.csv', all_metrics)
    
    print(f"\n{'Method':50s} {'Dice':>8s} {'Sens':>8s} {'Prec':>8s}")
    print("-" * 78)
    for r in summary[:30]:
        print(f"{r['method']:50s} {r['dice']:>8.4f} {r['sensitivity']:>8.4f} {r['precision']:>8.4f}")
    
    print("\n═══ TOP 15 ═══")
    for r in summary[:15]:
        print(f"  {r['method']:50s} Dice={r['dice']:.4f}  Sens={r['sensitivity']:.4f}  Prec={r['precision']:.4f}")


def cielab_green_clahe_source_from_channel(channel, fov):
    """Apply C6G6-style processing on an already-processed channel."""
    ch = np.clip(channel, 0, 255).astype(np.uint8)
    ch[~fov] = 0
    clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8, 8))
    return clahe.apply(ch)


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
