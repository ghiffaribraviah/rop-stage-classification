"""Meijering/Gabor fusion-weight sweep for the masked-CNN ROP stage classifier.

Question this answers: the vessel-segmentation champion (0.40*gabor + 0.60*meijering)
won on binary Dice (0.4739) but LOST on classification macro-F1 (0.7332 vs 0.7425
baseline). Does a lower meijering weight recover macro-F1 above the baseline?

Strategy (cost control): the expensive part is building the per-image component maps
(gabor_tophat, meijering_fine, ridge, green). Those do NOT depend on the fusion weight,
so we build them ONCE and cache four uint8 channels per image at 224x224:
  {key}_comp.png  = BGR-packed [gabor, meijering, ridge]
  {key}_grn.png   = grayscale  green
Then for each weight w_m in the sweep we fuse  (1-w_m)*gabor + w_m*meijering  at
dataset-load time, renormalize, and run the IDENTICAL 5-fold CV protocol. One channel
build, N cheap CV runs.

Anchor check: w_m=0.60 MUST reproduce the champion's ~0.7332 macro-F1 (within CV noise).
If it does not, the load-time-fusion approximation is rejected and the result is void.
"""
import modal

try:
    import cv2, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
    import pandas as pd, time, random, warnings, json
    from pathlib import Path
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
    from torch.utils.data import Dataset, DataLoader
    from scipy import ndimage as ndi
    from skimage.filters import meijering
    warnings.filterwarnings('ignore')
except ModuleNotFoundError:
    class _Stub:
        Module = object
        def __getattr__(self, _): return object
    nn = _Stub(); Dataset = object

app = modal.App("rop-masked-cnn-weight-sweep")
cache_vol = modal.Volume.from_name("rop-masked-cv-sweep-cache", create_if_missing=True)

image = (modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install("torch", "torchvision", "opencv-python", "numpy", "scikit-image",
                 "scikit-learn", "pandas", "scipy")
    .add_local_dir("data/Zhao2024", remote_path="/root/data/Zhao2024"))

# Sweep grid: meijering weight; gabor weight = 1 - w_m. 0.60 is the champion anchor.
SWEEP_WM = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60)

def norm01(i, m=None):
    v = i[m] if m is not None else i.ravel(); v = v[np.isfinite(v)]
    if v.size == 0: return np.zeros(i.shape, np.float32)
    lo, hi = np.percentile(v, [1, 99]); hi = max(hi - lo, 1e-8)
    return np.clip((i.astype(np.float32) - float(lo)) / hi, 0, 1).astype(np.float32)

def est_fov(rgb):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    m = (g > max(3, int(np.percentile(g, 1)))).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, 2); m = ndi.binary_fill_holes(m > 0).astype(np.uint8)
    nl, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if nl > 1: m = (labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    return m.astype(bool)

MEIJERING_FINE_SCALES = (0.8, 1.4, 2.0, 2.8, 3.6, 4.5)
FULL_KS = (5, 7, 9, 13, 17, 23, 31)

def tophat_custom(inv, fov, kernel_sizes=FULL_KS):
    maps = []
    for ks in kernel_sizes:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
        opened = cv2.morphologyEx(inv, cv2.MORPH_OPEN, k)
        maps.append(cv2.subtract(inv, opened).astype(np.float32))
    resp = np.max(np.stack(maps, 0), 0)
    resp = cv2.GaussianBlur(resp, (0, 0), 0.8)
    return norm01(resp, fov)

def gabor_resp(inv_f, fov):
    r = np.zeros(inv_f.shape, np.float32)
    for sg, lm in [(1.5, 3), (2.5, 5), (3.5, 7), (5, 10)]:
        sz = max(7, int(6 * sg) + (1 - int(6 * sg) % 2))
        for a in range(0, 180, 15):
            th = np.deg2rad(a); c = sz // 2; y, x = np.ogrid[-c:sz - c, -c:sz - c]
            xt = x * np.cos(th) + y * np.sin(th); yt = -x * np.sin(th) + y * np.cos(th)
            gk = np.exp(-0.5 * (xt ** 2 / sg ** 2 + yt ** 2 * 0.25 / sg ** 2)) * np.cos(2 * np.pi * xt / lm)
            gk = (gk - gk.mean()).astype(np.float32)
            r = np.maximum(r, cv2.filter2D(inv_f, cv2.CV_32F, gk, borderType=cv2.BORDER_REFLECT))
    r[~fov] = 0; return norm01(r, fov)

# ── COMPONENT 1: gabor_tophat (champion term, gabor weight = 1 - w_m) ──
def gabor_tophat_softmap(rgb, fov):
    g = rgb[:, :, 1].copy(); g[~fov] = 0
    enh = cv2.createCLAHE(clipLimit=6, tileGridSize=(16, 16)).apply(g); enh[~fov] = 0
    inv = 255 - enh
    th = tophat_custom(inv, fov, FULL_KS)
    inv_f = norm01(0.5 * norm01(inv.astype(np.float32), fov) + 0.5 * th, fov)
    gab = gabor_resp(inv_f, fov)
    r7 = cv2.medianBlur((gab * 255).astype(np.uint8), 7); r7[~fov] = 0
    soft = norm01(r7.astype(np.float32), fov)
    u8 = np.clip(soft * 255, 0, 255).astype(np.uint8); u8[~fov] = 0
    enh2 = cv2.createCLAHE(clipLimit=12, tileGridSize=(12, 12)).apply(u8); enh2[~fov] = 0
    return norm01(enh2.astype(np.float32), fov)

def meijering_fine_softmap(rgb, fov):
    g = rgb[:, :, 1].copy(); g[~fov] = 0
    enh = cv2.createCLAHE(clipLimit=6, tileGridSize=(16, 16)).apply(g); enh[~fov] = 0
    inv_f = (255 - enh).astype(np.float32) / 255.0; inv_f[~fov] = 0.0
    resp = meijering(inv_f.astype(np.float64), sigmas=list(MEIJERING_FINE_SCALES), black_ridges=False)
    resp = np.nan_to_num(resp, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32); resp[~fov] = 0.0
    return norm01(resp, fov)

# ── ridge softmap (channel 2; weight-independent) ──
RIDGE_SCALES = (3, 5, 7, 9, 11)
def ridge_source_channel(rgb, fov):
    green = rgb[:, :, 1].copy(); green[~fov] = 0
    enh = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16)).apply(green); enh[~fov] = 0
    return norm01(enh.astype(np.float32), fov)

def hessian_ridge(channel, fov):
    resp = meijering(channel.astype(np.float64), sigmas=list(RIDGE_SCALES), black_ridges=False)
    resp = np.nan_to_num(resp, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32); resp[~fov] = 0.0
    return norm01(resp, fov)

def oriented_tophat(channel, fov, length=21, n_angles=12):
    u8 = np.clip(channel * 255, 0, 255).astype(np.uint8); best = np.zeros_like(channel, np.float32)
    for k in range(n_angles):
        base = np.zeros((length, length), np.uint8)
        cv2.line(base, (0, length // 2), (length - 1, length // 2), 255, 1)
        M = cv2.getRotationMatrix2D((length / 2, length / 2), 180.0 * k / n_angles, 1.0)
        se = (cv2.warpAffine(base, M, (length, length)) > 0).astype(np.uint8)
        best = np.maximum(best, cv2.morphologyEx(u8, cv2.MORPH_TOPHAT, se).astype(np.float32))
    best[~fov] = 0.0; return norm01(best, fov)

def ridge_softmap(rgb, fov):
    ch = ridge_source_channel(rgb, fov)
    resp = 0.6 * hessian_ridge(ch, fov) + 0.4 * oriented_tophat(ch, fov)
    return norm01(resp.astype(np.float32), fov)

# ────────────────────────── component cache builder ──────────────────────────
# We cache FOUR weight-independent maps per image, all at 224x224 uint8, all FOV-zeroed:
#   comp.png (BGR) = [B=gabor, G=meijering, R=ridge];  grn.png = green (CLAHE).
# Splitting gabor and meijering into separate cached channels is the whole point: the
# fusion (1-w_m)*gab + w_m*mei is then a cheap load-time op, so one build serves every w_m.
def build_components(path):
    rgb = cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]; s = min(1.0, 768 / max(h, w))
    wrk = cv2.resize(rgb, (int(w * s), int(h * s)), cv2.INTER_AREA) if s < 1 else rgb.copy()
    fov = est_fov(wrk)
    gab = gabor_tophat_softmap(wrk, fov)      # full-res, FOV-zeroed
    mei = meijering_fine_softmap(wrk, fov)
    rid = ridge_softmap(wrk, fov)
    grn = ridge_source_channel(wrk, fov)
    comp = np.stack([gab, mei, rid], axis=-1)  # HxWx3 float [0,1]
    comp = cv2.resize(comp, (224, 224), interpolation=cv2.INTER_AREA)
    grn = cv2.resize(grn, (224, 224), interpolation=cv2.INTER_AREA)
    comp_u8 = np.clip(comp * 255, 0, 255).astype(np.uint8)  # H W 3 = [gab, mei, rid]
    grn_u8 = np.clip(grn * 255, 0, 255).astype(np.uint8)
    return comp_u8, grn_u8

class BasicBlock(nn.Module):
    def __init__(self, ic, oc, s=1):
        super().__init__()
        self.c1 = nn.Conv2d(ic, oc, 3, s, 1, bias=False); self.b1 = nn.BatchNorm2d(oc)
        self.c2 = nn.Conv2d(oc, oc, 3, 1, 1, bias=False); self.b2 = nn.BatchNorm2d(oc)
        self.sk = nn.Identity() if s == 1 and ic == oc else nn.Sequential(
            nn.Conv2d(ic, oc, 1, s, bias=False), nn.BatchNorm2d(oc))
    def forward(self, x):
        o = F.relu(self.b1(self.c1(x))); o = self.b2(self.c2(o)); return F.relu(o + self.sk(x))

class TinyResNetV2(nn.Module):
    def __init__(self, nc=4, wd=(48, 96, 192)):
        super().__init__()
        def ml(ic, oc, b, s): return nn.Sequential(*[BasicBlock(ic, oc, s)] + [BasicBlock(oc, oc, 1) for _ in range(1, b)])
        self.st = nn.Sequential(nn.Conv2d(3, wd[0], 3, 1, 1, bias=False), nn.BatchNorm2d(wd[0]), nn.ReLU(True), nn.MaxPool2d(2))
        self.l1 = ml(wd[0], wd[0], 2, 1); self.l2 = ml(wd[0], wd[1], 2, 2); self.l3 = ml(wd[1], wd[2], 2, 2)
        self.p = nn.AdaptiveAvgPool2d((1, 1)); self.do = nn.Dropout(0.3); self.f = nn.Linear(wd[2], nc)
    def forward(self, x):
        x = self.st(x); x = self.l1(x); x = self.l2(x); x = self.l3(x)
        x = self.p(x).flatten(1); x = self.do(x); return self.f(x)

class FocalLoss(nn.Module):
    def __init__(self, gamma=1.0, weight=None):
        super().__init__(); self.gamma = gamma; self.weight = weight
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        return ((1 - torch.exp(-ce)) ** self.gamma * ce).mean()

# Reads the cached weight-independent channels and fuses the vessel map per w_m:
#   vessel = norm01( (1-w_m)*gabor + w_m*meijering ), then stacks [vessel, ridge, green].
# Byte-equivalent to champion.build_channels EXCEPT the fusion weight is swept instead of
# fixed at 0.60; renorm, channel order, [0,1] scaling and augmentation stay identical so
# folds remain comparable to the champion run.
class FusionDataset(Dataset):
    def __init__(self, cache, df, w_m, augment=False):
        self.cache = cache; self.df = df.reset_index(drop=True)
        self.w_m = float(w_m); self.augment = augment
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        comp = cv2.imread(str(self.cache / f"{r['key']}_comp.png"), cv2.IMREAD_COLOR)  # BGR=[rid,mei,gab]
        grn = cv2.imread(str(self.cache / f"{r['key']}_grn.png"), cv2.IMREAD_GRAYSCALE)
        # cv2 reads BGR; we wrote RGB=[gab,mei,rid] so BGR channels are [rid,mei,gab].
        rid = comp[:, :, 0].astype(np.float32) / 255.0
        mei = comp[:, :, 1].astype(np.float32) / 255.0
        gab = comp[:, :, 2].astype(np.float32) / 255.0
        grn = grn.astype(np.float32) / 255.0
        fused = (1.0 - self.w_m) * gab + self.w_m * mei
        ves = norm01(fused.astype(np.float32))  # renorm exactly like vessel_softmap
        img = np.stack([ves, rid, grn], axis=-1)  # H W 3 = [vessel, ridge, green]
        if self.augment:
            if random.random() > 0.5: img = np.fliplr(img).copy()
            if random.random() > 0.5: img = np.flipud(img).copy()
            ang = random.uniform(-15, 15); h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return torch.from_numpy(img.transpose(2, 0, 1)).float(), torch.tensor(int(r['label_id']), dtype=torch.long)

def run_cv_for_weight(cache, df, w_m, device):
    """Run the IDENTICAL 5-fold CV protocol (seed=42, 80 epochs) for one fusion weight."""
    X = np.arange(len(df)); y = df['label_id'].values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_pred = np.full(len(df), -1, dtype=int)
    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        tr_df, te_df = df.iloc[tr_idx], df.iloc[te_idx]
        tl = DataLoader(FusionDataset(cache, tr_df, w_m, augment=True), 32, shuffle=True, num_workers=4, drop_last=True)
        el = DataLoader(FusionDataset(cache, te_df, w_m, augment=False), 32, shuffle=False, num_workers=4)
        model = TinyResNetV2().to(device)
        cnt = np.array([int((tr_df['label_id'] == i).sum()) for i in range(4)], np.float32)
        w = torch.tensor(cnt.sum() / (4 * np.maximum(cnt, 1)), dtype=torch.float32).to(device)
        crit = FocalLoss(gamma=1.0, weight=w)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=80)
        for ep in range(1, 81):
            model.train()
            for imgs, lbls in tl:
                imgs, lbls = imgs.to(device), lbls.to(device)
                opt.zero_grad(); crit(model(imgs), lbls).backward(); opt.step()
            sched.step()
        model.eval(); preds = []
        with torch.no_grad():
            for imgs, _ in el:
                preds.extend(torch.argmax(model(imgs.to(device)), 1).cpu().tolist())
        oof_pred[te_idx] = preds
        ff = precision_recall_fscore_support(te_df['label_id'].values, preds, average='macro', zero_division=0)[2]
        print(f"  [w_m={w_m:.2f}] fold {fold}: macro-F1={ff:.4f} (n={len(te_idx)})")
    acc = accuracy_score(y, oof_pred)
    p, r, f, _ = precision_recall_fscore_support(y, oof_pred, average='macro', zero_division=0)
    return {'w_m': float(w_m), 'accuracy': float(acc), 'f1': float(f),
            'precision': float(p), 'recall': float(r)}

@app.function(image=image, volumes={"/cache": cache_vol}, gpu="A100-80GB", timeout=43200)
def run_sweep():
    classes = ('Normal', 'Stage1', 'Stage2', 'Stage3'); cls2id = {n: i for i, n in enumerate(classes)}
    root = Path("/root/data/Zhao2024"); cache = Path("/cache/components")
    exts = {'.jpg', '.jpeg', '.png'}; rows = []
    for c in classes:
        d = root / c
        if not d.exists(): continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in exts:
                rows.append({'path': str(p), 'label': c, 'label_id': cls2id[c], 'key': f"{c}_{p.stem}"})
    df = pd.DataFrame(rows); print(f"Loaded {len(df)} images: {df['label'].value_counts().to_dict()}")

    # Build weight-independent component channels ONCE (cached on volume).
    cache.mkdir(parents=True, exist_ok=True)
    todo = [r for _, r in df.iterrows() if not (cache / f"{r['key']}_comp.png").exists()]
    if todo:
        print(f"Building {len(todo)} component sets...")
        t0 = time.time()
        for i, r in enumerate(todo):
            comp_u8, grn_u8 = build_components(r['path'])
            # comp_u8 is RGB=[gab,mei,rid]; cv2.imwrite expects BGR, so convert to keep it round-trippable.
            cv2.imwrite(str(cache / f"{r['key']}_comp.png"), cv2.cvtColor(comp_u8, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(cache / f"{r['key']}_grn.png"), grn_u8)
            if (i + 1) % 100 == 0: print(f"  [{i + 1}/{len(todo)}] {time.time() - t0:.0f}s")
        cache_vol.commit(); print(f"Component build done in {time.time() - t0:.0f}s")
    else:
        print("All component channels cached.")

    device = torch.device('cuda')
    results = []
    for w_m in SWEEP_WM:
        print(f"\n{'=' * 60}\nFUSION WEIGHT SWEEP  w_m={w_m:.2f}  (gabor={1 - w_m:.2f})\n{'=' * 60}")
        res = run_cv_for_weight(cache, df, w_m, device)
        results.append(res)
        print(f"  -> w_m={w_m:.2f}: OOF Acc={res['accuracy']:.4f} macro-F1={res['f1']:.4f}")

    # Anchor check: w_m=0.60 must reproduce champion ~0.7332 within CV noise (~0.01).
    anchor = next((r for r in results if abs(r['w_m'] - 0.60) < 1e-6), None)
    CHAMPION_F1 = 0.7332; BASELINE_F1 = 0.7425
    anchor_ok = anchor is not None and abs(anchor['f1'] - CHAMPION_F1) <= 0.015
    best = max(results, key=lambda r: r['f1'])

    print("\n" + "=" * 60)
    print("WEIGHT SWEEP SUMMARY  |  5-fold OOF (StratifiedKFold seed=42)")
    print("=" * 60)
    print(f"{'w_m':>6} {'gabor':>6} {'Acc':>8} {'macro-F1':>10}")
    for r in results:
        flag = ''
        if abs(r['w_m'] - 0.60) < 1e-6: flag += ' <-anchor'
        if r is best: flag += ' <-BEST'
        print(f"{r['w_m']:>6.2f} {1 - r['w_m']:>6.2f} {r['accuracy']:>8.4f} {r['f1']:>10.4f}{flag}")
    print("-" * 60)
    print(f"Anchor (w_m=0.60) macro-F1 = {anchor['f1'] if anchor else float('nan'):.4f} "
          f"vs champion {CHAMPION_F1:.4f} -> {'OK' if anchor_ok else 'REJECTED (approximation void)'}")
    print(f"Baseline macro-F1 = {BASELINE_F1:.4f}")
    print(f"Best: w_m={best['w_m']:.2f} macro-F1={best['f1']:.4f} "
          f"-> {'BEATS baseline' if best['f1'] > BASELINE_F1 else 'no improvement over baseline'}")
    return {'results': results, 'anchor_ok': bool(anchor_ok), 'best': best,
            'champion_f1': CHAMPION_F1, 'baseline_f1': BASELINE_F1}

@app.local_entrypoint()
def main():
    # .spawn() (not .remote()): a spawned FunctionCall is owned by Modal and survives the
    # local launcher disconnecting. Print the call ID so the run can be re-attached via
    # modal.FunctionCall.from_id(<id>).get() from any process.
    fc = run_sweep.spawn()
    print(f"SPAWNED_CALL_ID={fc.object_id}", flush=True)
    with open("/tmp/weight_sweep_call_id.txt", "w") as f:
        f.write(fc.object_id)
    r = fc.get()
    best = r['best']
    print(f"\nSweep done. anchor_ok={r['anchor_ok']} "
          f"best w_m={best['w_m']:.2f} macro-F1={best['f1']:.4f} "
          f"vs baseline {r['baseline_f1']:.4f} "
          f"-> {'WIN' if best['f1'] > r['baseline_f1'] else 'no improvement'}")
    print("RESULT_JSON=" + json.dumps(r), flush=True)
