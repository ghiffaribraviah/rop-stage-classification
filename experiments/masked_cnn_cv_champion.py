"""Masked-TinyResNet under the SAME 5-fold CV protocol as the 0.5147 classical baseline.

Honest head-to-head: StratifiedKFold(5, shuffle=True, random_state=42) over all 756
images, out-of-fold predictions pooled, macro-F1 reported. The CNN input is NOT raw
RGB - it is the byte-identical 3-channel masked construction used by the classical
pipeline: [vessel_softmap, ridge_softmap, masked_CLAHE_green], all FOV-masked, 224x224.

This isolates the question: given the same hand-engineered vessel/ridge evidence the
classical model saw, does a small CNN learn a better decision boundary than the
classical head?  Same folds, same images, same features -> comparable F1.
"""
import modal

# Heavy deps exist only inside the Modal image; importing them locally fails.
# The local process only resolves the entrypoint and calls run_cv.remote(),
# which executes inside the image - it never touches these names. The stub lets
# `class X(nn.Module)` resolve its base class at import time without the deps.
try:
    import cv2, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
    import pandas as pd, time, random, warnings
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

app = modal.App("rop-masked-cnn-cv-champion")
cache_vol = modal.Volume.from_name("rop-masked-cv-champion-cache", create_if_missing=True)

image = (modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install("torch", "torchvision", "opencv-python", "numpy", "scikit-image",
                 "scikit-learn", "pandas", "scipy")
    .add_local_dir("data/Zhao2024", remote_path="/root/data/Zhao2024"))

# ────────────────────────── shared primitives ──────────────────────────
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

# ── vessel softmap: CHAMPION recipe (g40_m60_fine) ─────────────────────────────
# Source of truth: experiments/vessel_champion_config.csv. The champion that won the
# vessel-segmentation sweep (Dice 0.4739 vs 0.4469 baseline) is the fusion
#   0.40 * gabor_tophat  +  0.60 * meijering_fine
# where gabor_tophat is the original Gabor->median->CLAHE soft map (kept byte-identical
# below) and meijering_fine is a Hessian ridge detector at FINE sigmas tuned for thin
# peripheral vessels.
#
# NOTE ON BINARIZATION: the champion's Dice score also involved a threshold + connected-
# component cleanup (P0.16, top-3 CC, 3x3 close). That stage exists only to emit a BINARY
# mask for Dice scoring. Here the map feeds a CNN channel, which benefits from the
# continuous soft signal, so we deliberately fuse and renormalize but DO NOT binarize.
MEIJERING_FINE_SCALES = (0.8, 1.4, 2.0, 2.8, 3.6, 4.5)

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

def gabor_tophat_softmap(rgb, fov):
    """The original masked_cnn_cv vessel_softmap, unchanged. Champion term, weight 0.40."""
    g = rgb[:, :, 1].copy(); g[~fov] = 0
    enh = cv2.createCLAHE(clipLimit=6, tileGridSize=(16, 16)).apply(g); enh[~fov] = 0
    inv = 255 - enh; inv_f = norm01(inv.astype(np.float32), fov)
    gab = gabor_resp(inv_f, fov)
    r7 = cv2.medianBlur((gab * 255).astype(np.uint8), 7); r7[~fov] = 0
    soft = norm01(r7.astype(np.float32), fov)
    u8 = np.clip(soft * 255, 0, 255).astype(np.uint8); u8[~fov] = 0
    enh2 = cv2.createCLAHE(clipLimit=12, tileGridSize=(12, 12)).apply(u8); enh2[~fov] = 0
    return norm01(enh2.astype(np.float32), fov)

def meijering_fine_softmap(rgb, fov):
    """Hessian ridge response at fine sigmas (thin peripheral vessels). Champion term, weight 0.60."""
    g = rgb[:, :, 1].copy(); g[~fov] = 0
    enh = cv2.createCLAHE(clipLimit=6, tileGridSize=(16, 16)).apply(g); enh[~fov] = 0
    inv_f = norm01((255 - enh).astype(np.float32), fov)
    resp = meijering(inv_f.astype(np.float64), sigmas=list(MEIJERING_FINE_SCALES), black_ridges=False)
    resp = np.nan_to_num(resp, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32); resp[~fov] = 0.0
    return norm01(resp, fov)

def vessel_softmap(rgb, fov):
    """Champion fusion: 0.40*gabor_tophat + 0.60*meijering_fine, FOV-masked, renormalized."""
    gab = gabor_tophat_softmap(rgb, fov)
    mei = meijering_fine_softmap(rgb, fov)
    fused = 0.40 * gab + 0.60 * mei
    fused[~fov] = 0.0
    return norm01(fused.astype(np.float32), fov)

# ── ridge softmap (meijering + oriented tophat, weight 0.4; identical to ridge_response_map) ──
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

def build_channels(path):
    """3-channel masked input: [vessel, ridge, masked_green] at 224x224, uint8."""
    rgb = cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]; s = min(1.0, 768 / max(h, w))
    wrk = cv2.resize(rgb, (int(w * s), int(h * s)), cv2.INTER_AREA) if s < 1 else rgb.copy()
    fov = est_fov(wrk)
    ves = vessel_softmap(wrk, fov)
    rid = ridge_softmap(wrk, fov)
    grn = ridge_source_channel(wrk, fov)  # CLAHE green, FOV-masked
    stack = np.stack([ves, rid, grn], axis=-1)  # HxWx3 float [0,1]
    stack = cv2.resize(stack, (224, 224), interpolation=cv2.INTER_AREA)
    return np.clip(stack * 255, 0, 255).astype(np.uint8)

# ────────────────────────── model ──────────────────────────
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

class CacheDataset(Dataset):
    def __init__(self, cache, df, augment=False):
        self.cache = cache; self.df = df.reset_index(drop=True); self.augment = augment
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        img = cv2.imread(str(self.cache / f"{r['key']}.png"), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.augment:
            if random.random() > 0.5: img = np.fliplr(img).copy()
            if random.random() > 0.5: img = np.flipud(img).copy()
            ang = random.uniform(-15, 15); h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        # No ImageNet normalization: channels are [0,1] softmaps, not natural RGB.
        img = img.astype(np.float32) / 255.0
        return torch.from_numpy(img.transpose(2, 0, 1)).float(), torch.tensor(int(r['label_id']), dtype=torch.long)

@app.function(image=image, volumes={"/cache": cache_vol}, gpu="L40S", timeout=10800)
def run_cv():
    classes = ('Normal', 'Stage1', 'Stage2', 'Stage3'); cls2id = {n: i for i, n in enumerate(classes)}
    root = Path("/root/data/Zhao2024"); cache = Path("/cache/masked3ch")
    exts = {'.jpg', '.jpeg', '.png'}; rows = []
    for c in classes:
        d = root / c
        if not d.exists(): continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in exts:
                rows.append({'path': str(p), 'label': c, 'label_id': cls2id[c], 'key': f"{c}_{p.stem}"})
    df = pd.DataFrame(rows); print(f"Loaded {len(df)} images: {df['label'].value_counts().to_dict()}")

    # Pre-generate masked 3-channel inputs once (cached on volume)
    cache.mkdir(parents=True, exist_ok=True)
    todo = [r for _, r in df.iterrows() if not (cache / f"{r['key']}.png").exists()]
    if todo:
        print(f"Generating {len(todo)} masked inputs...")
        t0 = time.time()
        for i, r in enumerate(todo):
            ch = build_channels(r['path'])
            cv2.imwrite(str(cache / f"{r['key']}.png"), cv2.cvtColor(ch, cv2.COLOR_RGB2BGR))
            if (i + 1) % 100 == 0: print(f"  [{i + 1}/{len(todo)}] {time.time() - t0:.0f}s")
        cache_vol.commit(); print(f"Generation done in {time.time() - t0:.0f}s")
    else:
        print("All masked inputs cached.")

    DEVICE = torch.device('cuda')
    X = np.arange(len(df)); y = df['label_id'].values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_pred = np.full(len(df), -1, dtype=int)

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        tr_df, te_df = df.iloc[tr_idx], df.iloc[te_idx]
        tl = DataLoader(CacheDataset(cache, tr_df, augment=True), 32, shuffle=True, num_workers=4, drop_last=True)
        el = DataLoader(CacheDataset(cache, te_df, augment=False), 32, shuffle=False, num_workers=4)

        model = TinyResNetV2().to(DEVICE)
        cnt = np.array([int((tr_df['label_id'] == i).sum()) for i in range(4)], np.float32)
        w = torch.tensor(cnt.sum() / (4 * np.maximum(cnt, 1)), dtype=torch.float32).to(DEVICE)
        crit = FocalLoss(gamma=1.0, weight=w)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=80)

        for ep in range(1, 81):
            model.train()
            for imgs, lbls in tl:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                opt.zero_grad(); crit(model(imgs), lbls).backward(); opt.step()
            sched.step()
            if ep % 20 == 0 or ep == 1: print(f"  fold {fold} epoch {ep}/80")

        model.eval(); preds = []
        with torch.no_grad():
            for imgs, _ in el:
                preds.extend(torch.argmax(model(imgs.to(DEVICE)), 1).cpu().tolist())
        oof_pred[te_idx] = preds
        f = precision_recall_fscore_support(te_df['label_id'].values, preds, average='macro', zero_division=0)[2]
        print(f"Fold {fold}: macro-F1={f:.4f} (n={len(te_idx)})")

    acc = accuracy_score(y, oof_pred)
    p, r, f, _ = precision_recall_fscore_support(y, oof_pred, average='macro', zero_division=0)
    print("\n" + "=" * 60)
    print("MASKED-TinyResNet  |  5-fold OOF (StratifiedKFold seed=42)")
    print("=" * 60)
    print(classification_report(y, oof_pred, target_names=classes, zero_division=0))
    print(f"OOF Acc={acc:.4f}  macro-F1={f:.4f}  P={p:.4f}  R={r:.4f}")
    print(f"\nClassical baseline macro-F1 = 0.5147")
    print(f"Masked-CNN     macro-F1 = {f:.4f}  ({'WIN' if f > 0.5147 else 'no improvement'})")
    return {'accuracy': float(acc), 'f1': float(f), 'precision': float(p), 'recall': float(r),
            'baseline_f1': 0.5147, 'improved': bool(f > 0.5147)}

@app.local_entrypoint()
def main():
    call = run_cv.spawn()
    print(f"Spawned (detached) run_cv: {call.object_id}")
    print("Survives launcher death; reattach with: modal app logs rop-masked-cnn-cv")
    r = call.get()
    print(f"\nMasked-CNN OOF: Acc={r['accuracy']:.4f} F1={r['f1']:.4f} "
          f"vs baseline 0.5147 -> {'WIN' if r['improved'] else 'no improvement'}")
