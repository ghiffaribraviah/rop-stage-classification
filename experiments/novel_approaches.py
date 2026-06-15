"""
NOVEL VESSEL SEGMENTATION APPROACHES
Out-of-the-box thinking beyond standard thresholding.
"""
import sys, numpy as np, cv2
from pathlib import Path
from skimage.morphology import skeletonize
from scipy import ndimage as ndi
sys.path.insert(0, str(Path(__file__).parent))
from vessel_pipeline import *
from advanced_pipeline import gabor_filter_response

ROOT = Path('data/Agrawal2021')
CONFIG = VesselPipelineConfig()
OUTPUT = Path('experiments/output')
OUTPUT.mkdir(exist_ok=True)

def best_soft(rgb, fov):
    green = rgb[:,:,1].copy()
    clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(16, 16))
    enh = clahe.apply(green); enh[~fov] = 0
    inv = 255 - enh; inv_f = normalize01(inv.astype(np.float32), fov)
    gab = gabor_filter_response(inv_f, fov)
    r7 = cv2.medianBlur((gab*255).astype(np.uint8), 7); r7[~fov] = 0
    return normalize01(r7.astype(np.float32), fov)

def top_n(binary, n):
    nl, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
    if nl <= 1: return binary
    areas = [(stats[i, cv2.CC_STAT_AREA], i) for i in range(1, nl)]
    areas.sort(reverse=True)
    keep = {idx for _, idx in areas[:min(n, len(areas))]}
    return np.isin(labels, list(keep))

# ── IDEA 1: Iterative Seed Growth ─────────────────────────────────────
def iterative_seed_growth(soft, fov, start_density=0.05, end_density=0.20, step=20):
    """
    Start with a high threshold (only sure vessels).
    Iteratively lower the threshold but ONLY add pixels connected to existing vessels.
    This is like hysteresis but with many steps instead of just two.
    """
    inner_fov = erode_mask(fov, 8)
    densities = np.linspace(start_density, end_density, step)
    result = None
    
    for d in densities:
        th = threshold_response_map(soft, fov, method='percentile', target_density=d) > 0
        if result is None:
            result = th
        else:
            # Only add th pixels that are connected to current result
            new_pixels = th & ~result
            # Check connectivity: dilate result and see what new pixels it touches
            kernel = np.ones((3,3), dtype=np.uint8)
            dilated = cv2.dilate(result.astype(np.uint8), kernel, iterations=3).astype(bool)
            connected = new_pixels & dilated
            result |= connected
            # If very few new pixels, stop early (converged)
            if np.count_nonzero(connected) < 20:
                break
    
    return result & inner_fov

# ── IDEA 2: Vessel Path Tracing ───────────────────────────────────────
def trace_vessel_paths(soft, fov, seed_density=0.04, min_path_length=30):
    """
    Treat vessel detection as path tracing.
    1. Find seed points (very high confidence vessel pixels)
    2. From each seed, trace along the vessel path
    3. Follow the ridge of the soft map
    4. Stop when the ridge drops below threshold
    """
    from skimage.feature import hessian_matrix, hessian_matrix_eigvals
    
    inner = erode_mask(fov, 8)
    
    # Find seeds at very high threshold
    seeds = threshold_response_map(soft, fov, method='percentile', target_density=seed_density) > 0
    
    # Compute vessel orientation from Hessian
    hes = hessian_matrix(soft.astype(np.float32), sigma=2.0, order='rc', use_gaussian_derivatives=True)
    _, eig_large = hessian_matrix_eigvals(hes)
    # Orientation from the eigenvector of the largest eigenvalue
    # (perpendicular to vessel direction)
    
    # Actually, let's use a simpler approach: trace using gradient direction
    result = np.zeros(soft.shape, dtype=bool)
    
    # Get all seed points
    seed_pts = np.argwhere(seeds & inner)
    np.random.shuffle(seed_pts)  # avoid bias
    
    for pt in seed_pts[:200]:  # limit to 200 seeds
        y, x = pt[0], pt[1]
        path = [(y, x)]
        
        # Trace forward
        for direction in [1, -1]:
            cy, cx = y, x
            for _ in range(200):  # max trace length
                # Compute local gradient
                if cy < 3 or cy >= soft.shape[0]-3 or cx < 3 or cx >= soft.shape[1]-3:
                    break
                patch = soft[cy-1:cy+2, cx-1:cx+2]
                gy, gx = np.gradient(patch)
                gy, gx = gy[1,1], gx[1,1]
                
                # Move perpendicular to gradient (along vessel)
                if abs(gy) > abs(gx):
                    step_x = int(np.sign(gx)) if abs(gx) > 0.01 else 0
                    step_y = 0
                else:
                    step_y = int(np.sign(gy)) if abs(gy) > 0.01 else 0
                    step_x = 0
                
                # If no clear direction, use direction from previous step
                if step_x == 0 and step_y == 0:
                    step_x = direction  # continue in same direction
                
                ny, nx = cy + step_y * direction, cx + step_x * direction
                
                # Check bounds
                if ny < 0 or ny >= soft.shape[0] or nx < 0 or nx >= soft.shape[1]:
                    break
                
                # Check if still on vessel (soft map is high enough)
                if soft[ny, nx] < 0.15:  # lower threshold
                    break
                
                # Check if we've been here
                if (ny, nx) in path:
                    break
                    
                path.append((ny, nx))
                cy, cx = ny, nx
        
        # Mark the path if it's long enough
        if len(path) >= min_path_length:
            for py, px in path:
                if inner[py, px] and soft[py, px] > 0.1:
                    result[py, px] = True
    
    return result & inner

# ── IDEA 3: Endpoint Connection (Road Network Style) ──────────────────
def connect_vessel_endpoints(binary, fov, max_gap=30, angle_threshold=30):
    """
    Detect endpoints of vessel segments, then connect nearby endpoints
    that are approximately aligned. Like road network completion.
    """
    from skimage.morphology import skeletonize
    
    # Skeletonize
    skel = skeletonize(binary).astype(bool)
    
    # Find endpoints (pixels with exactly 1 skeleton neighbor)
    kernel = np.ones((3,3), dtype=np.uint8)
    n_count = ndi.convolve(skel.astype(np.uint8), kernel, mode='constant', cval=0)
    endpoints = (n_count == 2) & skel  # 2 = self + 1 neighbor
    
    end_pts = np.argwhere(endpoints)
    
    # Compute orientation at each endpoint from the skeleton
    orientations = []
    for pt in end_pts:
        y, x = pt[0], pt[1]
        # Look at neighbors to determine direction
        neighbors = []
        for dy in [-1,0,1]:
            for dx in [-1,0,1]:
                ny, nx = y+dy, x+dx
                if 0 <= ny < skel.shape[0] and 0 <= nx < skel.shape[1] and skel[ny, nx] and (ny != y or nx != x):
                    neighbors.append((ny, nx))
        if len(neighbors) >= 1:
            # Direction is from endpoint to the average of its neighbors
            avg_y = float(np.mean([n[0] for n in neighbors]))
            avg_x = float(np.mean([n[1] for n in neighbors]))
            angle = np.degrees(np.arctan2(avg_y - y, avg_x - x))
            orientations.append(angle)
        else:
            orientations.append(0)
    
    # Connect nearby endpoints
    result = binary.copy()
    connected_pairs = set()
    
    for i in range(len(end_pts)):
        if orientations[i] is None:
            continue
        for j in range(i+1, len(end_pts)):
            if orientations[j] is None:
                continue
            
            yi, xi = end_pts[i][0], end_pts[i][1]
            yj, xj = end_pts[j][0], end_pts[j][1]
            
            # Distance
            dist = np.sqrt((yi-yj)**2 + (xi-xj)**2)
            if dist > max_gap or dist < 5:
                continue
            
            # Angle between the two endpoint orientations
            angle_diff = abs(orientations[i] - orientations[j])
            angle_diff = min(angle_diff, 180 - angle_diff)
            
            # Angle from endpoint i to endpoint j vs orientation i
            to_j_angle = np.degrees(np.arctan2(yj-yi, xj-xi))
            diff_i = abs(orientations[i] - to_j_angle)
            diff_i = min(diff_i, 180 - diff_i)
            
            # They should be roughly pointing toward each other
            if diff_i < angle_threshold and angle_diff < 60:
                # Draw a line connecting them
                pair_id = tuple(sorted([i, j]))
                if pair_id not in connected_pairs:
                    connected_pairs.add(pair_id)
                    # Bresenham line
                    yy, xx = np.linspace(yi, yj, int(dist)+1, dtype=int), np.linspace(xi, xj, int(dist)+1, dtype=int)
                    for py, px in zip(yy, xx):
                        if 0 <= py < result.shape[0] and 0 <= px < result.shape[1]:
                            result[py, px] = True
    
    return result & fov

# ── IDEA 4: Tensor Voting for vessel completion ───────────────────────
def tensor_voting_completion(soft, fov, votes_per_pixel=5):
    """
    Each high-confidence pixel "votes" for its neighbors along the vessel direction.
    Pixels with many votes from different directions are likely real vessels.
    This propagates confidence along vessel paths.
    """
    from skimage.feature import hessian_matrix, hessian_matrix_eigvals
    
    inner = erode_mask(fov, 8)
    
    # Get orientation at each pixel using Hessian
    hes = hessian_matrix(soft.astype(np.float32), sigma=2.0, order='rc', use_gaussian_derivatives=True)
    eig_small, eig_large = hessian_matrix_eigvals(hes)
    
    # Vessel direction is along the eigenvector of the smaller eigenvalue
    # (the direction of least curvature = along the vessel)
    # For now, use gradient direction
    result = np.zeros(soft.shape, dtype=np.float32)
    
    # Only vote from high-confidence pixels
    high_conf = soft > np.percentile(soft[inner], 85)
    
    high_pts = np.argwhere(high_conf & inner)
    
    for pt in high_pts[::3]:  # subsample for speed
        y, x = pt[0], pt[1]
        
        # Local gradient
        if y < 2 or y >= soft.shape[0]-2 or x < 2 or x >= soft.shape[1]-2:
            continue
        patch = soft[y-1:y+2, x-1:x+2].astype(np.float32)
        gy, gx = np.gradient(patch)
        gy, gx = gy[1,1], gx[1,1]
        mag = np.sqrt(gy**2 + gx**2) + 1e-10
        
        # Perpendicular to gradient = along vessel
        vy, vx = -gx/mag, gy/mag
        
        # Vote along vessel direction (both forward and backward)
        for dist in range(1, votes_per_pixel+1):
            for s in [-1, 1]:
                ny = int(round(y + s * vy * dist * 5))
                nx = int(round(x + s * vx * dist * 5))
                if 0 <= ny < soft.shape[0] and 0 <= nx < soft.shape[1]:
                    # Vote decays with distance
                    result[ny, nx] += soft[y, x] * (1.0 - dist / (votes_per_pixel + 1))
    
    # Normalize
    if np.max(result) > 0:
        result = result / np.max(result)
    result[~inner] = 0
    return result

# ── IDEA 5: Adaptive vessel width normalization ───────────────────────
def adaptive_width_vesselness(soft, fov):
    """
    Normalize vesselness by local vessel width.
    Thin vessels get a boost because they tend to have weaker responses.
    Thick vessels get slightly suppressed.
    """
    inner = erode_mask(fov, 8)
    
    # Estimate local width from response: wider vessels have broader response
    # Use distance transform on a rough binarization
    rough = (soft > np.percentile(soft[inner], 80)) & inner
    
    from scipy.ndimage import distance_transform_edt
    dist = distance_transform_edt(~rough)
    
    # Normalize: boost pixels where vessel width is small (thin vessels)
    # penalize where width is large
    width = np.maximum(dist, 1.0)
    width_norm = np.clip(10.0 / width, 0.5, 2.0)  # boost thin, cap at 2x
    
    result = soft * width_norm
    result[~fov] = 0
    return normalize01(result, fov)

# ── IDEA 6: Phase congruency vesselness ───────────────────────────────
def phase_congruency_vesselness(green, fov):
    """
    Phase congruency: illumination-invariant feature detection.
    Detects vessels regardless of local contrast.
    """
    from skimage.filters import difference_of_gaussians
    
    green_f = green.astype(np.float32)
    green_f[~fov] = 0
    
    # Multi-scale DoG (bandpass filtering)
    result = np.zeros_like(green_f)
    for sigma in [1.0, 1.5, 2.5, 4.0, 6.0]:
        k = 1.6
        inner_dog = cv2.GaussianBlur(green_f, (0, 0), sigmaX=sigma)
        outer_dog = cv2.GaussianBlur(green_f, (0, 0), sigmaX=sigma * k)
        dog = inner_dog - outer_dog
        # Phase congruency approximation: use local energy
        # Local energy = sqrt(convolve^2 + hilbert^2)
        # Simplified: just use absolute value of DoG
        energy = np.abs(dog)
        result = np.maximum(result, energy)
    
    result[~fov] = 0
    return normalize01(result, fov)

# ── Test everything ──────────────────────────────────────────────────

pairs = find_agrawal_pairs(ROOT)
sample = [p for p in pairs if p['source'] == 'RetCam'][:10] + [p for p in pairs if p['source'] == 'Neo'][:10]
results = []

print("=" * 70)
print("NOVEL VESSEL SEGMENTATION APPROACHES")
print("=" * 70)

for idx, pair in enumerate(sample):
    rgb = read_rgb(pair['image_path'])
    working = resize_max_side(rgb, CONFIG.process_max_side)
    fov = estimate_fov_mask(working)
    gt = read_binary_mask(pair['mask_path'], fov.shape)
    green = working[:,:,1].copy()
    soft = best_soft(working, fov)
    
    # ── BASELINES ──
    b10 = threshold_response_map(soft, fov, method='percentile', target_density=0.10) > 0
    b12 = threshold_response_map(soft, fov, method='percentile', target_density=0.12) > 0
    b15 = threshold_response_map(soft, fov, method='percentile', target_density=0.15) > 0
    
    m = segmentation_metrics(b10, gt); m['method'] = 'P10'; results.append(m)
    m = segmentation_metrics(b12, gt); m['method'] = 'P12'; results.append(m)
    
    for density in [10, 12]:
        b = threshold_response_map(soft, fov, method='percentile', target_density=density/100.0) > 0
        t2 = top_n(b, 2)
        m = segmentation_metrics(t2, gt); m['method'] = f'P{density}_t2'; results.append(m)
    
    # ── IDEA 1: Iterative seed growth ──
    for sd, ed in [(0.04, 0.20), (0.06, 0.18), (0.05, 0.15)]:
        isg = iterative_seed_growth(soft, fov, start_density=sd, end_density=ed, step=15)
        m = segmentation_metrics(isg, gt); m['method'] = f'isg_{sd:.2f}_{ed:.2f}'; results.append(m)
    
    # ── IDEA 2: Vessel path tracing ──
    vpt = trace_vessel_paths(soft, fov, seed_density=0.04, min_path_length=20)
    m = segmentation_metrics(vpt, gt); m['method'] = 'path_trace'; results.append(m)
    
    # ── IDEA 3: Endpoint connection ──
    for density in [10, 12]:
        b = threshold_response_map(soft, fov, method='percentile', target_density=density/100.0) > 0
        t2 = top_n(b, 2)
        ec = connect_vessel_endpoints(t2, fov, max_gap=20, angle_threshold=30)
        m = segmentation_metrics(ec, gt); m['method'] = f'P{density}_t2_ec'; results.append(m)
        
        # Also try without top2
        ec_full = connect_vessel_endpoints(b, fov, max_gap=20, angle_threshold=30)
        m = segmentation_metrics(ec_full, gt); m['method'] = f'P{density}_ec'; results.append(m)
    
    # ── IDEA 4: Tensor voting ──
    tv = tensor_voting_completion(soft, fov, votes_per_pixel=5)
    for d in [10, 12]:
        th = threshold_response_map(tv, fov, method='percentile', target_density=d/100.0) > 0
        m = segmentation_metrics(th, gt); m['method'] = f'tensor_vote_P{d:02d}'; results.append(m)
    
    # ── IDEA 5: Adaptive width ──
    aw = adaptive_width_vesselness(soft, fov)
    for d in [10, 12]:
        th = threshold_response_map(aw, fov, method='percentile', target_density=d/100.0) > 0
        m = segmentation_metrics(th, gt); m['method'] = f'adap_width_P{d:02d}'; results.append(m)
        t2 = top_n(th, 2)
        m = segmentation_metrics(t2, gt); m['method'] = f'adap_width_P{d:02d}_t2'; results.append(m)
    
    # ── IDEA 6: Phase congruency ──
    pc = phase_congruency_vesselness(green, fov)
    for d in [10, 12, 15, 20]:
        th = threshold_response_map(pc, fov, method='percentile', target_density=d/100.0) > 0
        m = segmentation_metrics(th, gt); m['method'] = f'phase_P{d:02d}'; results.append(m)
    
    sys.stdout.write(f"\r  [{idx+1}/{len(sample)}]")
    sys.stdout.flush()

print(f"\n\nTotal: {len(results)} metrics")

methods = sorted(set(r['method'] for r in results))
summary = []
for m in methods:
    rows = [r for r in results if r['method'] == m]
    summary.append({'m': m, 'd': float(np.mean([r['dice'] for r in rows])),
        'se': float(np.mean([r['sensitivity'] for r in rows])),
        'pr': float(np.mean([r['precision'] for r in rows]))})
summary.sort(key=lambda r: r['d'], reverse=True)

print(f"\n{'Method':35s} {'Dice':>8s} {'Sens':>8s} {'Prec':>8s}")
print('-' * 62)
for r in summary[:30]:
    print(f'{r["m"]:35s} {r["d"]:>8.4f} {r["se"]:>8.4f} {r["pr"]:>8.4f}')

print("\n═══ IDEA CATEGORY BREAKDOWN ═══")
for category, prefix in [('Baselines', 'P1'), ('Top2', '_t2'), ('Seed Growth', 'isg'),
                          ('Path Trace', 'path_trace'), ('Endpoint Connect', '_ec'),
                          ('Tensor Vote', 'tensor_vote'), ('Adaptive Width', 'adap_width'),
                          ('Phase', 'phase')]:
    rows = [r for r in summary if r['m'].startswith(prefix) or prefix in r['m']]
    if rows:
        best = max(rows, key=lambda r: r['d'])
        print(f"  {category:20s} Best: {best['m']:35s} Dice={best['d']:.4f}")
