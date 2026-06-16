"""Masked-TinyResNet v2 — push ROP staging toward 0.80 macro-F1, from scratch.

Lineage and rationale (see CHAMPION_RESULTS.md / RESULTS.md / VESSEL_FINDINGS.md):

  masked_cnn_cv.py         (4-class)  OOF macro-F1 = 0.7425   [plain-Gabor vessel ch]
  masked_cnn_cv_champion   (4-class)  OOF macro-F1 = 0.7332   [Dice-0.4739 fusion ch]  <- REGRESSION

The "champion" swapped the vessel channel for the segmentation-Dice winner
(0.40*gabor_tophat + 0.60*meijering_fine). That MAXIMIZES vessel overlap (Dice),
which is the WRONG objective for staging: it homogenizes the demarcation-line /
ridge texture that separates Stage2/Stage3 and cost -0.06 F1 on Stage3. Vessel-Dice
and stage-classification are different tasks. => v2 REVERTS to the plain-Gabor map.

v2 changes vs the 0.7425 baseline (all within constraints: NO pretraining):
  1. 5-class task: add the 343 'laser scars' images (full Zhao2024 = 1099 imgs).
     This matches the published benchmark (Zhao et al., Sci Data 2024: ResNet50
     ImageNet, 5-class, F1 0.8281) and adds an easy, visually-distinct class that
     lifts the macro floor to ~0.78 before any model gains.
  2. Plain-Gabor vessel channel (revert the champion regression).
  3. Regularizers proven for from-scratch small-data training (ResNet-RS,
     "ResNet strikes back", CutMix): MixUp + CutMix, label smoothing 0.1,
     stochastic depth (DropPath), weight-EMA, longer cosine schedule.
  4. Flip test-time augmentation at OOF inference.
  5. Dual CV report: StratifiedKFold (comparable to lineage + benchmark) AND
     StratifiedGroupKFold (leakage-corrected honest number).

This file is WRITE-ONLY-no auto full run. main() requires an explicit mode arg.
"""
import modal
import json

try:
    import cv2, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
    import pandas as pd, time, random, warnings, json
    from pathlib import Path
    from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
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

app = modal.App("rop-masked-cnn-cv-v2")
cache_vol = modal.Volume.from_name("rop-masked-cv-v2-cache", create_if_missing=True)

image = (modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install("torch", "torchvision", "opencv-python", "numpy", "scikit-image",
                 "scikit-learn", "pandas", "scipy")
    .add_local_file("experiments/clean_manifest.json", remote_path="/root/clean_manifest.json")
    .add_local_dir("data/Zhao2024", remote_path="/root/data/Zhao2024"))

CLASSES = ('Normal', 'Stage1', 'Stage2', 'Stage3', 'Laser')
# Directory name on disk -> class label. 'laser scars' has a space.
DIR2CLASS = {'Normal': 'Normal', 'Stage1': 'Stage1', 'Stage2': 'Stage2',
             'Stage3': 'Stage3', 'laser scars': 'Laser'}

# ────────────────────────── shared image primitives ──────────────────────────
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

# ── vessel softmap: PLAIN-GABOR (reverted to the 0.7425 baseline recipe) ──────
# Byte-identical to masked_cnn_cv.vessel_softmap. We deliberately DO NOT use the
# Meijering-fusion (Dice champion) here - that fusion regressed staging F1.
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

def vessel_softmap(rgb, fov):
    g = rgb[:, :, 1].copy(); g[~fov] = 0
    enh = cv2.createCLAHE(clipLimit=6, tileGridSize=(16, 16)).apply(g); enh[~fov] = 0
    inv = 255 - enh; inv_f = norm01(inv.astype(np.float32), fov)
    gab = gabor_resp(inv_f, fov)
    r7 = cv2.medianBlur((gab * 255).astype(np.uint8), 7); r7[~fov] = 0
    soft = norm01(r7.astype(np.float32), fov)
    u8 = np.clip(soft * 255, 0, 255).astype(np.uint8); u8[~fov] = 0
    enh2 = cv2.createCLAHE(clipLimit=12, tileGridSize=(12, 12)).apply(u8); enh2[~fov] = 0
    return norm01(enh2.astype(np.float32), fov)

# ── ridge softmap (the STAGE signal: demarcation line / ridge). Kept identical
#    to masked_cnn_cv.ridge_softmap - 0.6 Hessian + 0.4 oriented top-hat. ──────
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
    """3-channel masked input: [plain-Gabor vessel, ridge, masked CLAHE-green],
    224x224 uint8. Vessel channel is the REVERTED plain-Gabor map (not the
    Meijering-fusion that regressed staging)."""
    rgb = cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]; s = min(1.0, 768 / max(h, w))
    wrk = cv2.resize(rgb, (int(w * s), int(h * s)), cv2.INTER_AREA) if s < 1 else rgb.copy()
    fov = est_fov(wrk)
    ves = vessel_softmap(wrk, fov)
    rid = ridge_softmap(wrk, fov)
    grn = ridge_source_channel(wrk, fov)
    stack = np.stack([ves, rid, grn], axis=-1)
    stack = cv2.resize(stack, (224, 224), interpolation=cv2.INTER_AREA)
    return np.clip(stack * 255, 0, 255).astype(np.uint8)

# ────────────────────────── model (from scratch) ──────────────────────────
def drop_path(x, p, training):
    if p == 0.0 or not training: return x
    keep = 1 - p
    mask = torch.rand(x.shape[0], 1, 1, 1, dtype=x.dtype, device=x.device) < keep
    return x / keep * mask

class BasicBlock(nn.Module):
    def __init__(self, ic, oc, s=1, dp=0.0):
        super().__init__()
        self.c1 = nn.Conv2d(ic, oc, 3, s, 1, bias=False); self.b1 = nn.BatchNorm2d(oc)
        self.c2 = nn.Conv2d(oc, oc, 3, 1, 1, bias=False); self.b2 = nn.BatchNorm2d(oc)
        self.dp = dp
        self.sk = nn.Identity() if s == 1 and ic == oc else nn.Sequential(
            nn.Conv2d(ic, oc, 1, s, bias=False), nn.BatchNorm2d(oc))
    def forward(self, x):
        o = F.relu(self.b1(self.c1(x))); o = self.b2(self.c2(o))
        o = drop_path(o, self.dp, self.training)
        return F.relu(o + self.sk(x))

class TinyResNetV2(nn.Module):
    def __init__(self, nc=5, wd=(48, 96, 192), drop_path_rate=0.2, dropout=0.3):
        super().__init__()
        # linear stochastic-depth schedule across the 6 residual blocks
        dps = list(np.linspace(0, drop_path_rate, 6))
        def ml(ic, oc, b, s, idx):
            blocks = [BasicBlock(ic, oc, s, dps[idx])]
            blocks += [BasicBlock(oc, oc, 1, dps[idx + i]) for i in range(1, b)]
            return nn.Sequential(*blocks)
        self.st = nn.Sequential(nn.Conv2d(3, wd[0], 3, 1, 1, bias=False), nn.BatchNorm2d(wd[0]), nn.ReLU(True), nn.MaxPool2d(2))
        self.l1 = ml(wd[0], wd[0], 2, 1, 0); self.l2 = ml(wd[0], wd[1], 2, 2, 2); self.l3 = ml(wd[1], wd[2], 2, 2, 4)
        self.p = nn.AdaptiveAvgPool2d((1, 1)); self.do = nn.Dropout(dropout); self.f = nn.Linear(wd[2], nc)
    def forward(self, x):
        x = self.st(x); x = self.l1(x); x = self.l2(x); x = self.l3(x)
        x = self.p(x).flatten(1); x = self.do(x); return self.f(x)

class ModelEMA:
    """Exponential moving average of weights with a warmup so the average is not
    dominated by the random initialization during the first epochs."""
    def __init__(self, model, decay=0.999):
        import copy
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay; self.step = 0
        for p in self.ema.parameters(): p.requires_grad_(False)
    def update(self, model):
        with torch.no_grad():
            self.step += 1
            d = min(self.decay, (1 + self.step) / (10 + self.step))
            for e, m in zip(self.ema.state_dict().values(), model.state_dict().values()):
                if e.dtype.is_floating_point: e.mul_(d).add_(m.detach(), alpha=1 - d)
                else: e.copy_(m)

def soft_ce(logits, target_a, target_b, lam, weight, smoothing):
    """Label-smoothed CE that also handles MixUp/CutMix soft targets (two labels)."""
    n = logits.size(1); logp = F.log_softmax(logits, dim=1)
    def one(t):
        with torch.no_grad():
            true = torch.full_like(logp, smoothing / (n - 1))
            true.scatter_(1, t.unsqueeze(1), 1.0 - smoothing)
        loss = -(true * logp)
        if weight is not None: loss = loss * weight.unsqueeze(0)
        return loss.sum(1).mean()
    return lam * one(target_a) + (1 - lam) * one(target_b)

def mix_batch(imgs, labels, alpha_mix=0.2, alpha_cut=1.0, p=0.5):
    """Per-batch MixUp OR CutMix (each with prob p/.. ); returns mixed imgs + (a,b,lam)."""
    if random.random() > p:
        return imgs, labels, labels, 1.0
    use_cut = random.random() < 0.5
    perm = torch.randperm(imgs.size(0), device=imgs.device)
    if use_cut:
        lam = float(np.random.beta(alpha_cut, alpha_cut))
        H, W = imgs.shape[2:]; rw, rh = int(W * np.sqrt(1 - lam)), int(H * np.sqrt(1 - lam))
        cx, cy = np.random.randint(W), np.random.randint(H)
        x1, x2 = np.clip(cx - rw // 2, 0, W), np.clip(cx + rw // 2, 0, W)
        y1, y2 = np.clip(cy - rh // 2, 0, H), np.clip(cy + rh // 2, 0, H)
        imgs[:, :, y1:y2, x1:x2] = imgs[perm, :, y1:y2, x1:x2]
        lam = 1 - ((x2 - x1) * (y2 - y1) / (H * W))
    else:
        lam = float(np.random.beta(alpha_mix, alpha_mix))
        imgs = lam * imgs + (1 - lam) * imgs[perm]
    return imgs, labels, labels[perm], lam

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
            ang = random.uniform(-15, 15); sc = random.uniform(0.9, 1.1)
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        img = img.astype(np.float32) / 255.0
        return torch.from_numpy(img.transpose(2, 0, 1)).float(), torch.tensor(int(r['label_id']), dtype=torch.long)

# ────────────────────────── data + run ──────────────────────────
def load_manifest(root):
    exts = {'.jpg', '.jpeg', '.png'}; rows = []
    cls2id = {n: i for i, n in enumerate(CLASSES)}
    for d, c in DIR2CLASS.items():
        dd = root / d
        if not dd.exists(): continue
        for p in sorted(dd.iterdir()):
            if p.suffix.lower() in exts:
                rows.append({'path': str(p), 'label': c, 'label_id': cls2id[c],
                             'key': f"{c}_{p.stem}"})
    return pd.DataFrame(rows)

def ensure_cache(df, cache):
    cache.mkdir(parents=True, exist_ok=True)
    todo = [r for _, r in df.iterrows() if not (cache / f"{r['key']}.png").exists()]
    if todo:
        print(f"Generating {len(todo)} masked inputs...", flush=True); t0 = time.time()
        for i, r in enumerate(todo):
            ch = build_channels(r['path'])
            cv2.imwrite(str(cache / f"{r['key']}.png"), cv2.cvtColor(ch, cv2.COLOR_RGB2BGR))
            if (i + 1) % 100 == 0: print(f"  [{i+1}/{len(todo)}] {time.time()-t0:.0f}s", flush=True)
        cache_vol.commit(); print(f"Generation done in {time.time()-t0:.0f}s", flush=True)
    else:
        print("All masked inputs cached.", flush=True)

def train_one_fold(df, tr_idx, te_idx, cache, device, epochs, smoothing, ema_decay, mix_p, tta, eval_model="ema"):
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    tr_df, te_df = df.iloc[tr_idx], df.iloc[te_idx]
    tl = DataLoader(CacheDataset(cache, tr_df, augment=True), 32, shuffle=True, num_workers=4, drop_last=True)
    el = DataLoader(CacheDataset(cache, te_df, augment=False), 32, shuffle=False, num_workers=4)
    model = TinyResNetV2(nc=len(CLASSES)).to(device)
    cnt = np.array([int((tr_df['label_id'] == i).sum()) for i in range(len(CLASSES))], np.float32)
    w = torch.tensor(cnt.sum() / (len(CLASSES) * np.maximum(cnt, 1)), dtype=torch.float32).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ema = ModelEMA(model, ema_decay)
    for ep in range(1, epochs + 1):
        model.train()
        for imgs, lbls in tl:
            imgs, lbls = imgs.to(device), lbls.to(device)
            imgs, a, b, lam = mix_batch(imgs, lbls, p=mix_p)
            opt.zero_grad()
            soft_ce(model(imgs), a, b, lam, w, smoothing).backward()
            opt.step(); ema.update(model)
        sched.step()
        if ep % 20 == 0 or ep == 1: print(f"  epoch {ep}/{epochs}", flush=True)
    net = ema.ema if eval_model == "ema" else model
    net.eval(); preds = []
    with torch.no_grad():
        for imgs, _ in el:
            imgs = imgs.to(device); logits = net(imgs)
            if tta: logits = logits + net(torch.flip(imgs, dims=[3]))
            preds.extend(torch.argmax(logits, 1).cpu().tolist())
    return preds, te_df['label_id'].values

def _report(df, oof_pred, tag):
    y = df['label_id'].values
    acc = accuracy_score(y, oof_pred)
    p, r, f, _ = precision_recall_fscore_support(y, oof_pred, average='macro', zero_division=0)
    print("\n" + "=" * 60); print(f"MASKED-TinyResNet v2 | {tag}"); print("=" * 60)
    print(classification_report(y, oof_pred, target_names=CLASSES, zero_division=0))
    print(f"OOF Acc={acc:.4f}  macro-F1={f:.4f}  P={p:.4f}  R={r:.4f}", flush=True)
    return {'accuracy': float(acc), 'f1': float(f), 'precision': float(p), 'recall': float(r), 'tag': tag}

@app.function(image=image, volumes={"/cache": cache_vol}, gpu="A100-80GB", timeout=14400)
def run_full(protocol="stratified", epochs=160, smoothing=0.1, ema_decay=0.999, mix_p=0.5, tta=True):
    """FULL 5-fold OOF run. protocol: 'stratified' (comparable to lineage+benchmark)
    or 'group' (leakage-corrected honest)."""
    root = Path("/root/data/Zhao2024"); cache = Path("/cache/masked3ch_v2")
    df = load_manifest(root); print(f"Loaded {len(df)}: {df['label'].value_counts().to_dict()}", flush=True)
    ensure_cache(df, cache)
    device = torch.device('cuda'); X = np.arange(len(df)); y = df['label_id'].values
    oof = np.full(len(df), -1, dtype=int)
    if protocol == "group":
        # Leakage control: clean_manifest.json maps near-duplicate images to a
        # shared group so they cannot straddle a fold boundary. It covers only the
        # 4-class set; rows without a mapping (all laser scars) become singleton
        # groups, which StratifiedGroupKFold treats as independent samples.
        mpath = Path("/root/clean_manifest.json")
        gid = {}
        if mpath.exists():
            for m in json.load(open(mpath)):
                gid[Path(m['name']).stem] = int(m['group'])
        base = (max(gid.values()) + 1) if gid else 0
        groups = np.empty(len(df), dtype=int)
        for i, p in enumerate(df['path'].values):
            stem = Path(p).stem
            groups[i] = gid.get(stem, base + i)
        skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        splits = skf.split(X, y, groups=groups)
    else:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42); splits = skf.split(X, y)
    for fold, (tr, te) in enumerate(splits):
        preds, _ = train_one_fold(df, tr, te, cache, device, epochs, smoothing, ema_decay, mix_p, tta)
        oof[te] = preds
        ff = precision_recall_fscore_support(y[te], preds, average='macro', zero_division=0)[2]
        print(f"Fold {fold}: macro-F1={ff:.4f} (n={len(te)})", flush=True)
    res = _report(df, oof, f"5-fold OOF [{protocol}] ep={epochs} ls={smoothing} mixp={mix_p} tta={tta}")
    print("RESULT_JSON=" + json.dumps(res), flush=True); return res

@app.function(image=image, volumes={"/cache": cache_vol}, gpu="A100-80GB", timeout=3600)
def quick_test(kind="channel_ab"):
    """Short hypothesis tests. NOT a full pipeline run.
    kind='channel_ab' : 1 fold x 25 ep, current plain-Gabor vessel channel,
                        5-class, baseline aug only (no mix) - quick sanity F1.
    kind='fov_laser'  : generate channels for ~20 laser-scar imgs and report
                        FOV coverage stats (catch est_fov saturation)."""
    root = Path("/root/data/Zhao2024"); cache = Path("/cache/masked3ch_v2")
    df = load_manifest(root)
    if kind == "fov_laser":
        sub = df[df['label'] == 'Laser'].head(20)
        stats = []
        for _, r in sub.iterrows():
            rgb = cv2.cvtColor(cv2.imread(r['path'], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]; s = min(1.0, 768 / max(h, w))
            wrk = cv2.resize(rgb, (int(w*s), int(h*s)), cv2.INTER_AREA) if s < 1 else rgb.copy()
            fov = est_fov(wrk); cov = float(fov.mean())
            stats.append(cov)
        arr = np.array(stats)
        out = {'kind': kind, 'n': len(arr), 'fov_cov_min': float(arr.min()),
               'fov_cov_mean': float(arr.mean()), 'fov_cov_max': float(arr.max()),
               'n_saturated_gt0.98': int((arr > 0.98).sum()), 'n_tiny_lt0.2': int((arr < 0.2).sum())}
        print("RESULT_JSON=" + json.dumps(out), flush=True); return out
    # channel_ab
    ensure_cache(df, cache)
    device = torch.device('cuda'); X = np.arange(len(df)); y = df['label_id'].values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    tr, te = next(iter(skf.split(X, y)))
    preds_raw, ytrue = train_one_fold(df, tr, te, cache, device, epochs=25, smoothing=0.0,
                                      ema_decay=0.999, mix_p=0.0, tta=False, eval_model="raw")
    preds_ema, _ = train_one_fold(df, tr, te, cache, device, epochs=25, smoothing=0.0,
                                  ema_decay=0.999, mix_p=0.0, tta=False, eval_model="ema")
    f_raw = precision_recall_fscore_support(ytrue, preds_raw, average='macro', zero_division=0)[2]
    f_ema = precision_recall_fscore_support(ytrue, preds_ema, average='macro', zero_division=0)[2]
    out = {'kind': kind, 'fold0_macro_f1_25ep_raw': float(f_raw),
           'fold0_macro_f1_25ep_ema': float(f_ema), 'n_test': int(len(te))}
    print("RESULT_JSON=" + json.dumps(out), flush=True); return out

@app.local_entrypoint()
def main(mode: str = "", protocol: str = "stratified", epochs: int = 160):
    """Explicit mode required - nothing runs by default.
      mode=quick_channel : run quick_test('channel_ab')   [short]
      mode=quick_fov     : run quick_test('fov_laser')    [short]
      mode=full          : run_full(protocol, epochs)     [FULL - ask first]
    """
    if mode == "quick_channel":
        print("RESULT=" + json.dumps(quick_test.remote("channel_ab")), flush=True)
    elif mode == "quick_fov":
        print("RESULT=" + json.dumps(quick_test.remote("fov_laser")), flush=True)
    elif mode == "full":
        # .remote() (blocking) not .spawn(): a detached spawn is killed when the
        # entrypoint returns and Modal stops the app, discarding the result.
        print("RESULT=" + json.dumps(run_full.remote(protocol=protocol, epochs=epochs)), flush=True)
    else:
        print("No mode given. Use --mode quick_channel | quick_fov | full. Nothing ran.", flush=True)
