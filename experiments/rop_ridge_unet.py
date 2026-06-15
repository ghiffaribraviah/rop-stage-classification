"""From-scratch ridge/demarcation-line U-Net for ROP.

Trains ONLY on Agrawal2021 HVDROPDB-RIDGE pairs (100 images: 50 RetCam + 50 Neo).
No pretrained weights anywhere - all layers initialized randomly.

The ridge occupies ~1% of pixels, so we use Tversky loss (recall-weighted) plus
heavy augmentation to avoid the empty-mask collapse that BCE/Dice fall into on
extreme foreground imbalance.

Outputs:
  - ridge_unet.pt           trained weights
  - ridge_val_grid.png      val-set image|gt|pred panels (held-out Agrawal)
"""
from __future__ import annotations

import time
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import modal

app = modal.App("rop-ridge-unet")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install("torch", "torchvision", "opencv-python", "numpy")
    .add_local_dir(
        "data/Agrawal2021/HVDROPDB-RIDGE",
        remote_path="/root/ridge",
    )
)

IMG_SIZE = 384


class DoubleConv(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ic, oc, 3, 1, 1, bias=False), nn.BatchNorm2d(oc), nn.ReLU(True),
            nn.Conv2d(oc, oc, 3, 1, 1, bias=False), nn.BatchNorm2d(oc), nn.ReLU(True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        self.d1 = DoubleConv(3, base)
        self.d2 = DoubleConv(base, base * 2)
        self.d3 = DoubleConv(base * 2, base * 4)
        self.d4 = DoubleConv(base * 4, base * 8)
        self.bott = DoubleConv(base * 8, base * 16)
        self.pool = nn.MaxPool2d(2)
        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, 2)
        self.u4 = DoubleConv(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2)
        self.u3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.u2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.u1 = DoubleConv(base * 2, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        c1 = self.d1(x)
        c2 = self.d2(self.pool(c1))
        c3 = self.d3(self.pool(c2))
        c4 = self.d4(self.pool(c3))
        b = self.bott(self.pool(c4))
        x = self.u4(torch.cat([self.up4(b), c4], 1))
        x = self.u3(torch.cat([self.up3(x), c3], 1))
        x = self.u2(torch.cat([self.up2(x), c2], 1))
        x = self.u1(torch.cat([self.up1(x), c1], 1))
        return self.out(x)


def tversky_loss(logits, target, alpha=0.3, beta=0.7, smooth=1.0):
    prob = torch.sigmoid(logits)
    p = prob.reshape(prob.size(0), -1)
    g = target.reshape(target.size(0), -1)
    tp = (p * g).sum(1)
    fp = (p * (1 - g)).sum(1)
    fn = ((1 - p) * g).sum(1)
    tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return (1 - tversky).mean()


def dice_score(logits, target, thr=0.5, smooth=1.0):
    prob = (torch.sigmoid(logits) > thr).float()
    p = prob.reshape(prob.size(0), -1)
    g = target.reshape(target.size(0), -1)
    inter = (p * g).sum(1)
    return ((2 * inter + smooth) / (p.sum(1) + g.sum(1) + smooth)).mean().item()


def augment(img, mask):
    if random.random() > 0.5:
        img, mask = np.fliplr(img).copy(), np.fliplr(mask).copy()
    if random.random() > 0.5:
        img, mask = np.flipud(img).copy(), np.flipud(mask).copy()
    a = random.uniform(-25, 25)
    s = random.uniform(0.85, 1.15)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), a, s)
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    mask = cv2.warpAffine(mask, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if random.random() > 0.4:
        f = random.uniform(0.7, 1.3)
        img = np.clip(img.astype(np.float32) * f, 0, 255).astype(np.uint8)
    return img, mask


class RidgeDataset(Dataset):
    def __init__(self, pairs, train):
        self.pairs = pairs
        self.train = train

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ip, mp = self.pairs[idx]
        img = cv2.cvtColor(cv2.imread(ip, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        if self.train:
            img, mask = augment(img, mask)
        x = img.astype(np.float32) / 255.0
        x = (x - np.array([0.5, 0.5, 0.5])) / np.array([0.25, 0.25, 0.25])
        x = torch.from_numpy(x.transpose(2, 0, 1)).float()
        y = torch.from_numpy((mask > 127).astype(np.float32))[None]
        return x, y


def collect_pairs(root: Path):
    pairs = []
    for src in ("RetCam", "Neo"):
        idir = root / f"{src}_Ridge_images"
        mdir = root / f"{src}_Ridge_masks"
        for ip in sorted(idir.iterdir()):
            mp = mdir / ip.name
            if mp.exists():
                pairs.append((str(ip), str(mp)))
    return pairs


@app.function(image=image, gpu="L40S", timeout=3600)
def train():
    cv2.setNumThreads(0)
    root = Path("/root/ridge")
    pairs = collect_pairs(root)
    rng = random.Random(42)
    rng.shuffle(pairs)
    n_val = max(1, int(0.2 * len(pairs)))
    val_pairs, train_pairs = pairs[:n_val], pairs[n_val:]
    print(f"Ridge pairs: {len(pairs)} (train {len(train_pairs)} / val {len(val_pairs)})")

    dev = torch.device("cuda")
    net = UNet(base=32).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=150)

    tl = DataLoader(RidgeDataset(train_pairs, True), 8, shuffle=True, num_workers=2, drop_last=True)
    vl = DataLoader(RidgeDataset(val_pairs, False), 8, num_workers=2)

    best, best_state = 0.0, None
    t0 = time.time()
    for ep in range(1, 151):
        net.train()
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            tversky_loss(net(x), y).backward()
            opt.step()
        sched.step()
        net.eval()
        ds = []
        with torch.no_grad():
            for x, y in vl:
                ds.append(dice_score(net(x.to(dev)), y.to(dev)))
        d = float(np.mean(ds)) if ds else 0.0
        if d > best:
            best, best_state = d, {k: v.cpu().clone() for k, v in net.state_dict().items()}
        if ep % 15 == 0:
            print(f"  E{ep:3d}/150 val_dice={d:.4f} best={best:.4f} ({time.time()-t0:.0f}s)")

    net.load_state_dict(best_state)
    print(f"\nBest val Dice = {best:.4f}")

    net.eval()
    panels = []
    with torch.no_grad():
        for x, y in vl:
            pr = (torch.sigmoid(net(x.to(dev))) > 0.5).float().cpu().numpy()
            xb = x.numpy()
            yb = y.numpy()
            for i in range(min(2, xb.shape[0])):
                im = ((xb[i].transpose(1, 2, 0) * 0.25 + 0.5) * 255).clip(0, 255).astype(np.uint8)
                gt = (yb[i, 0] * 255).astype(np.uint8)
                pd = (pr[i, 0] * 255).astype(np.uint8)
                gt3 = cv2.cvtColor(gt, cv2.COLOR_GRAY2RGB)
                pd3 = cv2.cvtColor(pd, cv2.COLOR_GRAY2RGB)
                panels.append(np.concatenate([im, gt3, pd3], axis=1))
            if len(panels) >= 6:
                break
    grid = np.concatenate(panels[:6], axis=0)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    weights = {k: v for k, v in best_state.items()}
    return best, buf.tobytes(), {k: v.numpy() for k, v in weights.items()}


@app.local_entrypoint()
def main():
    best, png, _ = train.remote()
    out = Path(__file__).resolve().parent / "output"
    (out / "ridge_val_grid.png").write_bytes(png)
    print(f"Best val Dice={best:.4f}; wrote output/ridge_val_grid.png")
