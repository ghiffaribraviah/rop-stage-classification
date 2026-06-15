"""
UNet for retinal vessel segmentation on Agrawal2021 dataset.
Trains on image/mask pairs, evaluates Dice on held-out test set.
"""
import sys, os, time, csv, json, random
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent))
from vessel_pipeline import read_rgb, resize_max_side, estimate_fov_mask, IMAGE_EXTENSIONS, resize_binary_mask

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'experiments' / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
AGRAWAL_ROOT = PROJECT_ROOT / 'data' / 'Agrawal2021'
MODEL_DIR = OUTPUT_DIR / 'unet_models'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ──
IMG_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE = 15
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Device: {DEVICE}")


# ── Dataset ──

def find_pairs(root):
    if not root or not root.exists():
        return []
    bv = root / 'HVDROPDB-BV'
    subs = [
        ('RetCam', bv / 'RetCam_Vessels_images', bv / 'RetCam_Vessels_masks'),
        ('RetCam', bv / 'Retcam_Vessels_images', bv / 'Retcam_Vessels_masks'),
        ('Neo', bv / 'Neo_Vessels_images', bv / 'Neo_Vessels_masks'),
    ]
    rows = []
    for src, img_dir, mask_dir in subs:
        if not img_dir.is_dir() or not mask_dir.is_dir():
            continue
        imgs = {p.stem: p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS}
        masks = {p.stem: p for p in mask_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS}
        for stem in sorted(set(imgs) & set(masks)):
            rows.append({'source': src, 'img': str(imgs[stem]), 'mask': str(masks[stem]), 'name': imgs[stem].name})
    return rows


class VesselDataset(Dataset):
    def __init__(self, pairs, size=IMG_SIZE, augment=False):
        self.pairs = pairs
        self.size = size
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p = self.pairs[idx]
        img = cv2.imread(p['img'], cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(p['mask'], cv2.IMREAD_GRAYSCALE)

        # Resize
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
        if mask.max() > 1:
            mask = (mask >= 127).astype(np.float32)
        else:
            mask = mask.astype(np.float32)
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0.5).astype(np.float32)

        # Augmentation
        if self.augment:
            if random.random() > 0.5:
                img = np.fliplr(img).copy()
                mask = np.fliplr(mask).copy()
            if random.random() > 0.5:
                img = np.flipud(img).copy()
                mask = np.flipud(mask).copy()
            angle = random.uniform(-15, 15)
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            mask = cv2.warpAffine(mask, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

            # Brightness/contrast
            if random.random() > 0.5:
                alpha = random.uniform(0.85, 1.15)
                beta = random.uniform(-10, 10)
                img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        # Normalize
        img = img.astype(np.float32) / 255.0
        img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()
        mask = torch.from_numpy(mask).float().unsqueeze(0)
        return img, mask


# ── UNet ──

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[32, 64, 128, 256]):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        for f in features:
            self.encoders.append(DoubleConv(in_channels, f))
            in_channels = f

        for f in reversed(features):
            self.decoders.append(nn.ConvTranspose2d(f * 2, f, 2, stride=2))
            self.decoders.append(DoubleConv(f * 2, f))

        self.final = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for i in range(0, len(self.decoders), 2):
            x = self.decoders[i](x)
            skip = skips.pop()
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = self.decoders[i + 1](x)

        return torch.sigmoid(self.final(x))


# ── Loss ──

def dice_loss(pred, target, smooth=1.0):
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = (pred * target).sum()
    return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def combined_loss(pred, target):
    bce = F.binary_cross_entropy(pred, target)
    dice = dice_loss(pred, target)
    return bce + dice


# ── Training ──

def train_model(train_loader, val_loader):
    model = UNet().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    best_dice = 0.0
    best_epoch = -1
    stale = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = combined_loss(preds, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_dice = 0.0
        val_loss = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                preds = model(imgs)
                loss = combined_loss(preds, masks)
                val_loss += loss.item() * imgs.size(0)
                pred_bin = (preds > 0.5).float()
                intersection = (pred_bin * masks).sum((1,2,3))
                union = pred_bin.sum((1,2,3)) + masks.sum((1,2,3))
                dice = (2 * intersection + 1) / (union + 1)
                val_dice += dice.sum().item()
        val_loss /= len(val_loader.dataset)
        val_dice /= len(val_loader.dataset)

        scheduler.step()
        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, 'val_dice': val_dice})

        if val_dice > best_dice:
            best_dice = val_dice
            best_epoch = epoch
            stale = 0
            torch.save(model.state_dict(), MODEL_DIR / 'best_unet.pt')
        else:
            stale += 1

        if (epoch) % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_dice={val_dice:.4f}  best={best_dice:.4f}")

        if stale >= PATIENCE:
            print(f"  Early stopping at epoch {epoch}. Best at {best_epoch}: Dice={best_dice:.4f}")
            break

    # Save history
    with open(MODEL_DIR / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    # Load best model
    model.load_state_dict(torch.load(MODEL_DIR / 'best_unet.pt', map_location=DEVICE))
    return model, best_dice, best_epoch


def evaluate(model, loader):
    model.eval()
    all_dice, all_iou, all_sens, all_prec = [], [], [], []
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            preds = model(imgs)
            pred_bin = (preds > 0.5).float()

            for i in range(imgs.size(0)):
                p = pred_bin[i, 0].cpu().numpy().astype(bool)
                t = masks[i, 0].cpu().numpy().astype(bool)
                tp = np.logical_and(p, t).sum()
                fp = np.logical_and(p, ~t).sum()
                fn = np.logical_and(~p, t).sum()
                eps = 1e-8
                d = (2*tp) / (2*tp + fp + fn + eps)
                iou = tp / (tp + fp + fn + eps)
                sens = tp / (tp + fn + eps)
                prec = tp / (tp + fp + eps)
                all_dice.append(d)
                all_iou.append(iou)
                all_sens.append(sens)
                all_prec.append(prec)
    return {
        'dice': float(np.mean(all_dice)),
        'iou': float(np.mean(all_iou)),
        'sensitivity': float(np.mean(all_sens)),
        'precision': float(np.mean(all_prec)),
        'dice_std': float(np.std(all_dice)),
    }


def main():
    print("=" * 70)
    print("UNET VESSEL SEGMENTATION TRAINING")
    print("=" * 70)
    print(f"Output: {MODEL_DIR}")

    # Load data
    pairs = find_pairs(AGRAWAL_ROOT)
    print(f"Total pairs: {len(pairs)}")
    retcam = [p for p in pairs if p['source'] == 'RetCam']
    neo = [p for p in pairs if p['source'] == 'Neo']
    print(f"  RetCam: {len(retcam)}, Neo: {len(neo)}")

    # Stratified split: 70% train, 10% val, 20% test per source
    random.seed(42)
    random.shuffle(retcam)
    random.shuffle(neo)

    def split_data(data, train_r=0.7, val_r=0.1):
        n = len(data)
        n_train = int(n * train_r)
        n_val = int(n * val_r)
        return data[:n_train], data[n_train:n_train+n_val], data[n_train+n_val:]

    train_r = []; val_r = []; test_r = []; test_n = []
    test_r, test_n = [], []
    for data in [retcam, neo]:
        tr, va, te = split_data(data)
        train_r.extend(tr)
        val_r.extend(va)
        test_r.extend(te)
        test_n.extend(te)

    print(f"Train: {len(train_r)}, Val: {len(val_r)}, Test: {len(test_r)}")

    # Datasets
    train_ds = VesselDataset(train_r, augment=True)
    val_ds = VesselDataset(val_r, augment=False)
    test_ds = VesselDataset(test_r, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Train
    print(f"\nTraining UNet on {DEVICE}...")
    model, best_dice, best_epoch = train_model(train_loader, val_loader)
    print(f"\nBest validation Dice: {best_dice:.4f} at epoch {best_epoch}")

    # Evaluate on test
    print("\nEvaluating on test set...")
    test_metrics = evaluate(model, test_loader)
    print(f"\n═══ TEST RESULTS ═══")
    print(f"  Dice:       {test_metrics['dice']:.4f} +/- {test_metrics['dice_std']:.4f}")
    print(f"  IoU:        {test_metrics['iou']:.4f}")
    print(f"  Sensitivity:{test_metrics['sensitivity']:.4f}")
    print(f"  Precision:  {test_metrics['precision']:.4f}")

    # Save test metrics
    with open(MODEL_DIR / 'test_metrics.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)
    print(f"Saved to {MODEL_DIR}")

    # Generate some visual results
    print("\nGenerating visual results...")
    model.eval()
    vis_dir = MODEL_DIR / 'visualizations'
    vis_dir.mkdir(exist_ok=True)

    for i, (img, mask) in enumerate(test_loader):
        if i >= 4:
            break
        with torch.no_grad():
            pred = model(img.to(DEVICE))
        for j in range(min(2, img.size(0))):
            im = img[j].cpu().numpy().transpose(1,2,0)
            im = np.clip(im * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1)
            im = (im * 255).astype(np.uint8)
            gt = mask[j, 0].cpu().numpy()
            pr = (pred[j, 0].cpu().numpy() > 0.5).astype(np.float32)

            # Create overlay: white=TP, red=FP, cyan=FN
            overlay = np.zeros((*gt.shape, 3), dtype=np.uint8)
            overlay[(pr > 0.5) & (gt > 0.5)] = (255, 255, 255)
            overlay[(pr > 0.5) & (gt < 0.5)] = (255, 0, 0)
            overlay[(pr < 0.5) & (gt > 0.5)] = (0, 255, 255)

            # Stack: original, GT, prediction, overlay
            gt_rgb = np.repeat((gt * 255).astype(np.uint8)[:,:,None], 3, axis=2)
            pr_rgb = np.repeat((pr * 255).astype(np.uint8)[:,:,None], 3, axis=2)
            combined = np.concatenate([im, gt_rgb, pr_rgb, overlay], axis=1)
            cv2.imwrite(str(vis_dir / f'test_{i}_{j}.jpg'), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

    print(f"Visualizations saved to {vis_dir}")
    print(f"\n{'='*70}")
    print(f"TARGET ACHIEVED: Dice = {test_metrics['dice']:.4f}")
    target = 0.70
    status = "✓ MET" if test_metrics['dice'] >= target else f"{(target - test_metrics['dice']) * 100:.1f}% below target"
    print(f"Target 0.70: {status}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
