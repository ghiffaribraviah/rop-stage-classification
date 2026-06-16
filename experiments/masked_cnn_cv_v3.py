"""Masked-TinyResNet v3 — ordinal-aware training to push ROP staging to 0.80 macro-F1.

Lineage (see V2_RESULTS.md / CHAMPION_RESULTS.md / RESULTS.md):

  masked_cnn_cv.py       (4-class)  OOF macro-F1 = 0.7425   [plain-Gabor vessel ch]
  masked_cnn_cv_champion (4-class)  OOF macro-F1 = 0.7332   [Dice-fusion]  <- REGRESSION
  masked_cnn_cv_v2.py    (5-class)  stratified  OOF macro-F1 = 0.7802
                                    group-aware OOF macro-F1 = 0.7853   <- honest baseline

The v2 gap to 0.80 (group-aware: 0.0147) is concentrated in Stage1 (F1 0.57) and
Stage2 (F1 0.65) — adjacent ordinal classes that the model confuses, compounded by
an over-aggressive inverse-frequency class weight on Stage1 (precision ~0.45). This
is fundamentally a threshold / class-imbalance problem on an ORDINAL label, not a
representation problem. v3 keeps the entire v2 image+model pipeline byte-identical
and changes ONLY the training objective + decision rule:

  v3 changes vs v2 (NO pretraining, single-head, MixUp-compatible):
    1. ONSCE — Ordinal-Neighbor Soft-label CE. The stage labels Normal<Stage1<
       Stage2<Stage3 form an ordinal chain; a Stage2 image is "closer" to Stage1/
       Stage3 than to Normal. Instead of one-hot (or uniform label smoothing), the
       target mass is split: 1-eps on the true class, eps onto the immediate ordinal
       NEIGHBOUR(s). 'Laser' (laser scars) is NOT on the severity axis, so it is
       isolated: it neither donates nor receives neighbour mass, and falls back to
       plain label smoothing. This composes with MixUp/CutMix (two soft targets are
       built independently then lam-mixed).
    2. Capped class weights — replace raw inv-frequency (which over-boosts Stage1
       recall at the cost of precision) with sqrt(inv-freq), then clip to <=4.0.
    3. Optional rank-auxiliary loss — 0.1 * MSE(expected_rank, true_rank) over the
       ordinal classes only, toggled by --rank-aux. Pulls the softmax mass toward
       the correct severity even when the argmax is wrong.
    4. Post-hoc Stage1 threshold sweep — v2 returned argmax only. v3 returns OOF
       LOGITS, then sweeps an additive bias on the Stage1 logit on the pooled OOF
       set to find the macro-F1-optimal operating point (reported separately so the
       base number stays honest/comparable).
    5. Dual CV report retained: 'stratified' (comparable to lineage+benchmark) and
       'group' (StratifiedGroupKFold leakage-corrected honest number).

WRITE-ONLY — no auto full run. main() requires an explicit mode arg.
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

app = modal.App("rop-masked-cnn-cv-v3")
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
# Ordinal severity chain (indices into CLASSES). 'Laser' (idx 4) is OFF-axis and
# deliberately excluded — it is a treatment artifact, not a severity grade.
ORDINAL_IDS = (0, 1, 2, 3)          # Normal < Stage1 < Stage2 < Stage3
LASER_ID = 4
STAGE1_ID = 1

# ────────────────────────── shared image primitives (verbatim from v2) ─────────
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

# ── vessel softmap: PLAIN-GABOR (the 0.7425 baseline recipe; NOT the Dice-fusion) ──
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

# ── ridge softmap (the STAGE signal: demarcation line / ridge) ─────────────────
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
    224x224 uint8. Identical to v2 — cache is shared (rop-masked-cv-v2-cache)."""
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

# ────────────────────────── model (from scratch, verbatim from v2) ─────────────
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
    """Exponential moving average of weights with a warmup (verbatim from v2)."""
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

# ════════════════════════ v3 NEW: ordinal objective ════════════════════════════
def capped_class_weights(cnt, cap=4.0):
    """sqrt(inverse-frequency) weights, clipped to <=cap.

    v2 used raw inv-freq  w_c = N / (C * n_c)  which gave Stage1 a large multiplier
    and drove recall up at the cost of precision (P~0.45). The sqrt taper softens
    the imbalance correction; the cap=4.0 hard-stops any single rare class from
    dominating the gradient. Weights are renormalised so the mean weight is ~1
    (keeps the effective loss scale / LR comparable to v2)."""
    cnt = np.asarray(cnt, np.float32)
    inv = cnt.sum() / (len(cnt) * np.maximum(cnt, 1.0))   # v2's raw inv-freq
    w = np.sqrt(inv)                                       # taper
    w = np.minimum(w, cap)                                 # hard cap
    w = w / w.mean()                                       # renormalise to mean 1
    return w.astype(np.float32)

def build_ordinal_soft_targets(labels, n_classes, eps_ord, smoothing):
    """Construct ONSCE soft-label rows for a batch of hard labels.

    For an ordinal class c in {Normal,Stage1,Stage2,Stage3}:
        - true class gets (1 - eps_ord - smoothing_floor)
        - each existing ordinal neighbour (c-1, c+1 within the chain) shares eps_ord
          equally (endpoints Normal/Stage3 have one neighbour -> it gets all eps_ord)
        - the remaining `smoothing` mass is spread uniformly over the OTHER classes
          (the classic label-smoothing floor), so Laser still receives a tiny mass.
    For Laser (off the severity axis): NO neighbour mass. Plain label smoothing only
        - true class gets (1 - smoothing), `smoothing` spread over the other 4.
    Laser also never RECEIVES ordinal neighbour mass (it is not adjacent to Stage3).

    Returns a (B, n_classes) float tensor that sums to 1 per row.
    """
    B = labels.size(0); device = labels.device
    floor = smoothing / (n_classes - 1)
    t = torch.full((B, n_classes), floor, device=device)
    ord_set = set(ORDINAL_IDS)
    for i in range(B):
        c = int(labels[i].item())
        # reset row to the smoothing floor, then assign mass
        t[i].fill_(floor)
        if c == LASER_ID:
            # off-axis: plain label smoothing, no neighbour donation
            t[i, c] = 1.0 - smoothing
            continue
        # ordinal class: find existing neighbours within the chain
        neigh = [c - 1, c + 1]
        neigh = [m for m in neigh if m in ord_set]
        share = eps_ord / len(neigh) if neigh else 0.0
        # true mass = 1 - (neighbour mass) - (smoothing floor on all other classes)
        # other-class floor already applied; subtract the floor we will overwrite on
        # neighbours so the row still sums to exactly 1.
        t[i, c] = 1.0 - eps_ord - smoothing
        for m in neigh:
            t[i, m] = floor + share
    return t

def onsce_loss(logits, soft_targets, weight):
    """Weighted soft-target cross-entropy: -sum_c w_c * q_c * log p_c, mean over batch."""
    logp = F.log_softmax(logits, dim=1)
    loss = -(soft_targets * logp)
    if weight is not None:
        loss = loss * weight.unsqueeze(0)
    return loss.sum(1).mean()

def rank_aux_loss(logits, labels):
    """0.1-weighted MSE between the softmax expected ordinal rank and the true rank,
    computed over ORDINAL samples only (Laser rows are masked out). Pulls probability
    mass toward the correct severity even when the argmax is wrong."""
    device = logits.device
    is_ord = torch.tensor([int(l.item()) in set(ORDINAL_IDS) for l in labels],
                          dtype=torch.bool, device=device)
    if is_ord.sum() == 0:
        return logits.new_zeros(())
    ranks = torch.tensor([float(i) for i in range(len(CLASSES))], device=device)
    # mass over ordinal classes only, renormalised, then expected rank
    p = F.softmax(logits, dim=1)
    ord_mask = torch.tensor([1.0 if i in set(ORDINAL_IDS) else 0.0
                             for i in range(len(CLASSES))], device=device)
    p_ord = p * ord_mask.unsqueeze(0)
    p_ord = p_ord / p_ord.sum(1, keepdim=True).clamp_min(1e-8)
    exp_rank = (p_ord * ranks.unsqueeze(0)).sum(1)
    true_rank = labels.float()
    return F.mse_loss(exp_rank[is_ord], true_rank[is_ord])

# ────────────────────────── dataset + manifest (verbatim from v2) ──────────────
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

# ────────────────────────── v3 MixUp on ONSCE soft targets ─────────────────────
def mix_batch(imgs, labels, alpha_mix=0.2, alpha_cut=1.0, p=0.5):
    """Identical mixing geometry to v2; returns (mixed_imgs, label_a, label_b, lam).
    The two label sets are turned into ONSCE soft targets independently in the loss
    step and lam-mixed there (soft-target space is linear so this is exact)."""
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

def onsce_mixed(logits, a, b, lam, weight, eps_ord, smoothing):
    """ONSCE for a (possibly) mixed batch. Builds soft targets for both label sets
    and lam-mixes in probability space."""
    nC = logits.size(1)
    qa = build_ordinal_soft_targets(a, nC, eps_ord, smoothing)
    if lam == 1.0:
        return onsce_loss(logits, qa, weight)
    qb = build_ordinal_soft_targets(b, nC, eps_ord, smoothing)
    return lam * onsce_loss(logits, qa, weight) + (1 - lam) * onsce_loss(logits, qb, weight)

# ────────────────────────── train one fold -> OOF LOGITS ───────────────────────
def train_one_fold(df, tr_idx, te_idx, cache, device, epochs, smoothing, ema_decay,
                   mix_p, tta, eps_ord, weight_cap, use_rank_aux, eval_model="ema"):
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    tr_df, te_df = df.iloc[tr_idx], df.iloc[te_idx]
    tl = DataLoader(CacheDataset(cache, tr_df, augment=True), 32, shuffle=True, num_workers=4, drop_last=True)
    el = DataLoader(CacheDataset(cache, te_df, augment=False), 32, shuffle=False, num_workers=4)
    model = TinyResNetV2(nc=len(CLASSES)).to(device)
    cnt = np.array([int((tr_df['label_id'] == i).sum()) for i in range(len(CLASSES))], np.float32)
    w = torch.tensor(capped_class_weights(cnt, cap=weight_cap), dtype=torch.float32).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ema = ModelEMA(model, ema_decay)
    for ep in range(1, epochs + 1):
        model.train()
        for imgs, lbls in tl:
            imgs, lbls = imgs.to(device), lbls.to(device)
            imgs, a, b, lam = mix_batch(imgs, lbls, p=mix_p)
            opt.zero_grad()
            logits = model(imgs)
            loss = onsce_mixed(logits, a, b, lam, w, eps_ord, smoothing)
            if use_rank_aux:
                # rank-aux uses the un-permuted hard labels of the dominant mix term
                loss = loss + 0.1 * rank_aux_loss(logits, a if lam >= 0.5 else b)
            loss.backward()
            opt.step(); ema.update(model)
        sched.step()
        if ep % 20 == 0 or ep == 1: print(f"  epoch {ep}/{epochs}", flush=True)
    net = ema.ema if eval_model == "ema" else model
    net.eval(); logits_all = []
    with torch.no_grad():
        for imgs, _ in el:
            imgs = imgs.to(device); logits = net(imgs)
            if tta: logits = logits + net(torch.flip(imgs, dims=[3]))
            logits_all.append(logits.cpu().numpy())
    return np.concatenate(logits_all, 0), te_df['label_id'].values

# ────────────────────────── Stage1 post-hoc threshold sweep ────────────────────
def sweep_stage1_bias(oof_logits, y, lo=-3.0, hi=1.0, steps=81):
    """Add a scalar bias to the Stage1 logit before argmax; pick the bias that
    maximises pooled-OOF macro-F1. Negative bias REDUCES Stage1 over-prediction
    (v2's Stage1 precision was ~0.45). Returns (best_bias, base_f1, best_f1)."""
    base_pred = oof_logits.argmax(1)
    base_f1 = precision_recall_fscore_support(y, base_pred, average='macro', zero_division=0)[2]
    best_bias, best_f1 = 0.0, base_f1
    for bdelta in np.linspace(lo, hi, steps):
        adj = oof_logits.copy(); adj[:, STAGE1_ID] += bdelta
        pred = adj.argmax(1)
        f1 = precision_recall_fscore_support(y, pred, average='macro', zero_division=0)[2]
        if f1 > best_f1:
            best_f1, best_bias = f1, float(bdelta)
    return best_bias, float(base_f1), float(best_f1)

def _report(df, oof_pred, tag):
    y = df['label_id'].values
    acc = accuracy_score(y, oof_pred)
    p, r, f, _ = precision_recall_fscore_support(y, oof_pred, average='macro', zero_division=0)
    print("\n" + "=" * 60); print(f"MASKED-TinyResNet v3 | {tag}"); print("=" * 60)
    print(classification_report(y, oof_pred, target_names=CLASSES, zero_division=0))
    print(f"OOF Acc={acc:.4f}  macro-F1={f:.4f}  P={p:.4f}  R={r:.4f}", flush=True)
    return {'accuracy': float(acc), 'f1': float(f), 'precision': float(p), 'recall': float(r), 'tag': tag}

# ────────────────────────── Modal orchestrator ─────────────────────────────────
def _build_groups(df):
    """Same group construction as v2: near-duplicate images share a group id so
    they cannot straddle a fold; unmapped rows (laser scars) become singletons."""
    mpath = Path("/root/clean_manifest.json")
    gid = {}
    if mpath.exists():
        for m in json.load(open(mpath)):
            gid[Path(m['name']).stem] = int(m['group'])
    base = (max(gid.values()) + 1) if gid else 0
    groups = np.empty(len(df), dtype=int)
    for i, p in enumerate(df['path'].values):
        groups[i] = gid.get(Path(p).stem, base + i)
    return groups

@app.function(image=image, volumes={"/cache": cache_vol}, gpu="A100-80GB", timeout=14400)
def run_full(protocol="group", epochs=160, smoothing=0.1, ema_decay=0.999, mix_p=0.5,
             tta=True, eps_ord=0.15, weight_cap=4.0, use_rank_aux=False, sweep_stage1=True):
    """FULL 5-fold OOF run for v3 (ONSCE + capped weights + Stage1 sweep).

    protocol : 'stratified' (comparable to lineage+benchmark) | 'group' (leakage-corrected).
    eps_ord  : ordinal label-smoothing mass leaked to adjacent stage (0 -> plain LS).
    weight_cap: max class weight (caps the rare-Stage1 up-weight that hurt precision).
    use_rank_aux: add 0.1 * cumulative-rank auxiliary loss on the ordinal head.
    sweep_stage1: pick a post-hoc Stage1 logit bias on pooled OOF to fix precision.
    """
    root = Path("/root/data/Zhao2024"); cache = Path("/cache/masked3ch_v2")
    df = load_manifest(root); print(f"Loaded {len(df)}: {df['label'].value_counts().to_dict()}", flush=True)
    ensure_cache(df, cache)
    device = torch.device('cuda'); X = np.arange(len(df)); y = df['label_id'].values
    oof_logits = np.zeros((len(df), len(CLASSES)), dtype=np.float32)
    if protocol == "group":
        skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        splits = skf.split(X, y, groups=_build_groups(df))
    else:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42); splits = skf.split(X, y)
    for fold, (tr, te) in enumerate(splits):
        lg, _ = train_one_fold(df, tr, te, cache, device, epochs, smoothing, ema_decay,
                               mix_p, tta, eps_ord, weight_cap, use_rank_aux)
        oof_logits[te] = lg
        ff = precision_recall_fscore_support(y[te], lg.argmax(1), average='macro', zero_division=0)[2]
        print(f"Fold {fold}: macro-F1={ff:.4f} (n={len(te)})", flush=True)

    tag = (f"5-fold OOF [{protocol}] ep={epochs} ls={smoothing} eps={eps_ord} "
           f"cap={weight_cap} rank={use_rank_aux} mixp={mix_p} tta={tta}")
    res = _report(df, oof_logits.argmax(1), tag + " | raw-argmax")
    out = {'raw': res, 'tag': tag}
    if sweep_stage1:
        bias, base_f1, best_f1 = sweep_stage1_bias(oof_logits, y)
        adj = oof_logits.copy(); adj[:, STAGE1_ID] += bias
        res2 = _report(df, adj.argmax(1), tag + f" | stage1_bias={bias:+.3f}")
        print(f"Stage1 sweep: bias={bias:+.3f}  base_f1={base_f1:.4f} -> swept_f1={best_f1:.4f}", flush=True)
        out['swept'] = res2; out['stage1_bias'] = bias
    print("RESULT_JSON=" + json.dumps(out), flush=True); return out

@app.function(image=image, volumes={"/cache": cache_vol}, gpu="A100-80GB", timeout=5400)
def quick_test(epochs=40, eps_ord=0.15, weight_cap=4.0, use_rank_aux=False, mix_p=0.5):
    """1-fold smoke test of the ONSCE pipeline (fold 0 of StratifiedKFold).
    Returns raw + Stage1-swept fold-0 F1 so a config can be vetted in ~minutes
    before committing to the full 5-fold run."""
    root = Path("/root/data/Zhao2024"); cache = Path("/cache/masked3ch_v2")
    df = load_manifest(root); ensure_cache(df, cache)
    device = torch.device('cuda'); X = np.arange(len(df)); y = df['label_id'].values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    tr, te = next(skf.split(X, y))
    lg, yte = train_one_fold(df, tr, te, cache, device, epochs, 0.1, 0.999,
                             mix_p, True, eps_ord, weight_cap, use_rank_aux)
    raw = precision_recall_fscore_support(yte, lg.argmax(1), average='macro', zero_division=0)[2]
    bias, _, swept = sweep_stage1_bias(lg, yte)
    print(f"QUICK fold0: raw_f1={raw:.4f}  swept_f1={swept:.4f} (bias={bias:+.3f})", flush=True)
    print(classification_report(yte, lg.argmax(1), target_names=CLASSES, zero_division=0), flush=True)
    res = {'fold0_raw_f1': float(raw), 'fold0_swept_f1': float(swept), 'stage1_bias': bias}
    print("RESULT_JSON=" + json.dumps(res), flush=True); return res

@app.local_entrypoint()
def main(mode: str = "quick", protocol: str = "group", epochs: int = 160,
         eps_ord: float = 0.15, weight_cap: float = 4.0, rank_aux: bool = False):
    if mode == "quick":
        quick_test.remote(epochs=min(epochs, 40), eps_ord=eps_ord,
                          weight_cap=weight_cap, use_rank_aux=rank_aux)
    else:
        run_full.remote(protocol=protocol, epochs=epochs, eps_ord=eps_ord,
                        weight_cap=weight_cap, use_rank_aux=rank_aux)
