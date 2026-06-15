"""
Local ROP Classification Training
Generates scenarios, trains Tiny ResNet on all 4 scenarios, reports results.
Run: uv run python experiments/train_local.py
"""
import sys, time, json, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import cv2
from vessel_pipeline import *
from advanced_pipeline import gabor_filter_response
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / 'data' / 'Zhao2024'
OUTPUT_DIR = PROJECT_ROOT / 'experiments' / 'output' / 'training'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = OUTPUT_DIR / 'scenarios'
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 40
PATIENCE = 8
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")


# ── Scenario generation ──

def generate_scenario_images():
    """Pre-generate all S1-S4 scenario images."""
    classes = ('Normal', 'Stage1', 'Stage2', 'Stage3')
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    
    # Build index
    rows = []
    for cls in classes:
        d = DATA_ROOT / cls
        if not d.exists(): continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in exts:
                rows.append({'path': str(p), 'label': cls})
    df = pd.DataFrame(rows)
    print(f"Found {len(df)} images")
    
    train_df, temp = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    val_df, test_df = train_test_split(temp, test_size=0.5, stratify=temp['label'], random_state=42)
    
    def enhance(rgb, fov):
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        le = cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8)).apply(l)
        en = cv2.cvtColor(cv2.merge([le, a, b]), cv2.COLOR_LAB2RGB)
        en[~fov] = 0
        return en
    
    def seg_vessels(rgb, fov):
        """Our best vessel pipeline."""
        green = rgb[:,:,1].copy(); green[~fov] = 0
        clahe = cv2.createCLAHE(clipLimit=6, tileGridSize=(16,16))
        enh = clahe.apply(green); enh[~fov] = 0
        inv = 255 - enh; inv_f = normalize01(inv.astype(np.float32), fov)
        gabor = gabor_filter_response(inv_f, fov)
        r7 = cv2.medianBlur((gabor*255).astype(np.uint8), 7); r7[~fov] = 0
        soft = normalize01(r7.astype(np.float32), fov)
        u8 = np.clip(soft*255, 0, 255).astype(np.uint8); u8[~fov] = 0
        c = cv2.createCLAHE(clipLimit=12, tileGridSize=(12,12))
        enh_s = c.apply(u8); enh_s[~fov] = 0
        sharp = normalize01(enh_s.astype(np.float32), fov)
        inner = erode_mask(fov, 8)
        vals = sharp[inner]; vals = vals[vals>0]
        if len(vals) == 0: return np.zeros(fov.shape, bool), fov
        th = float(np.percentile(vals, 84))
        binary = (sharp >= th) & inner
        nl, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
        if nl > 1:
            areas = [(stats[i, cv2.CC_STAT_AREA], i) for i in range(1, nl)]
            areas.sort(reverse=True)
            keep = {idx for _, idx in areas[:2]}
            binary = np.isin(labels, list(keep))
        return binary, fov
    
    def vessel_guided(enh, binary, fov):
        g = binary.astype(np.float32)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        g = cv2.dilate(g, k, iterations=2)
        g = cv2.GaussianBlur(g, (0,0), sigmaX=3)
        g = 0.75 + 0.25 * normalize01(g, fov)
        guided = enh.astype(np.float32) * g[:,:,None]
        guided[~fov] = 0
        return np.clip(guided, 0, 255).astype(np.uint8)
    
    for split_name, sdf in [('train', train_df), ('val', val_df), ('test', test_df)]:
        for _, row in sdf.iterrows():
            for sc in ('S1_raw','S2_enhanced','S3_vessel_mask','S4_vessel_guided'):
                p = CACHE_DIR / sc / split_name / row['label']
                if not p.exists() or not any(p.iterdir()):
                    p.mkdir(parents=True, exist_ok=True)
    
    for split_name, sdf in [('train', train_df), ('val', val_df), ('test', test_df)]:
        for idx, (_, row) in enumerate(sdf.iterrows()):
            out_stem = CACHE_DIR
            stem = Path(row['path']).stem
            rgb = read_rgb(row['path'])
            working = resize_max_side(rgb, 768)
            fov = estimate_fov_mask(working)
            
            # S1
            s1 = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out_stem / 'S1_raw' / split_name / row['label'] / f'{stem}.jpg'),
                       cv2.cvtColor(s1, cv2.COLOR_RGB2BGR))
            
            # S2
            en = enhance(rgb, fov)
            s2 = cv2.resize(en, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out_stem / 'S2_enhanced' / split_name / row['label'] / f'{stem}.jpg'),
                       cv2.cvtColor(s2, cv2.COLOR_RGB2BGR))
            
            # S3 + S4
            binary, fv = seg_vessels(rgb, fov)
            br = cv2.resize(binary.astype(np.uint8), (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST).astype(bool)
            fr = cv2.resize(fv.astype(np.uint8), (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST).astype(bool)
            s3 = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            s3[br] = 255
            s3[~fr] = 0
            cv2.imwrite(str(out_stem / 'S3_vessel_mask' / split_name / row['label'] / f'{stem}.jpg'),
                       cv2.cvtColor(s3, cv2.COLOR_RGB2BGR))
            
            s2r = cv2.resize(en, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
            s4 = vessel_guided(s2r, br, fr)
            cv2.imwrite(str(out_stem / 'S4_vessel_guided' / split_name / row['label'] / f'{stem}.jpg'),
                       cv2.cvtColor(s4, cv2.COLOR_RGB2BGR))
            
            if (idx+1) % 50 == 0:
                print(f"  {split_name} [{idx+1}/{len(sdf)}]")
    
    # Save splits
    for name, sdf in [('train', train_df), ('val', val_df), ('test', test_df)]:
        sdf.to_csv(str(OUTPUT_DIR / f'split_{name}.csv'), index=False)
    print("Scenario generation complete!")


# ── Dataset and Model ──

class ROPDataset(Dataset):
    def __init__(self, df, scenario, split, augment=False):
        self.df = df; self.scenario = scenario; self.split = split; self.augment = augment
        import albumentations as A
        self.tfm = None
        if augment:
            self.tfm = A.Compose([
                A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.2),
                A.Rotate(limit=12, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.08, contrast_limit=0.08, p=0.35),
            ])
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        stem = Path(row['path']).stem
        img = cv2.imread(str(CACHE_DIR / self.scenario / self.split / row['label'] / f'{stem}.jpg'), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.tfm: img = self.tfm(image=img)['image']
        img = img.astype(np.float32) / 255.0
        img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        return torch.from_numpy(img.transpose(2,0,1)).float(), torch.tensor(int(row['label_id']), dtype=torch.long)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, ic, oc, s=1):
        super().__init__()
        self.c1 = nn.Conv2d(ic, oc, 3, s, 1, bias=False)
        self.b1 = nn.BatchNorm2d(oc)
        self.c2 = nn.Conv2d(oc, oc, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(oc)
        self.sk = nn.Identity() if s==1 and ic==oc else nn.Sequential(nn.Conv2d(ic, oc, 1, s, bias=False), nn.BatchNorm2d(oc))
    def forward(self, x):
        o = F.relu(self.b1(self.c1(x)))
        o = self.b2(self.c2(o))
        return F.relu(o + self.sk(x))

class TinyResNet(nn.Module):
    def __init__(self, nc=4, wd=(32, 64, 128)):
        super().__init__()
        def ml(ic, oc, b, s): return nn.Sequential(*[BasicBlock(ic, oc, s)] + [BasicBlock(oc, oc, 1) for _ in range(1, b)])
        self.stem = nn.Sequential(nn.Conv2d(3, wd[0], 3, 1, 1, bias=False), nn.BatchNorm2d(wd[0]), nn.ReLU(True), nn.MaxPool2d(2))
        self.l1 = ml(wd[0], wd[0], 2, 1)
        self.l2 = ml(wd[0], wd[1], 2, 2)
        self.l3 = ml(wd[1], wd[2], 2, 2)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.do = nn.Dropout(0.25)
        self.fc = nn.Linear(wd[2], nc)
    def forward(self, x):
        x = self.stem(x); x = self.l1(x); x = self.l2(x); x = self.l3(x)
        x = self.pool(x).flatten(1); x = self.do(x)
        return self.fc(x)


def train():
    classes = ('Normal', 'Stage1', 'Stage2', 'Stage3')
    cls2id = {n:i for i,n in enumerate(classes)}
    
    # Load splits
    train_df = pd.read_csv(str(OUTPUT_DIR / 'split_train.csv'))
    val_df = pd.read_csv(str(OUTPUT_DIR / 'split_val.csv'))
    test_df = pd.read_csv(str(OUTPUT_DIR / 'split_test.csv'))
    for df in [train_df, val_df, test_df]:
        df['label_id'] = df['label'].map(cls2id)
    
    SCENARIOS = ['S1_raw', 'S2_enhanced', 'S3_vessel_mask', 'S4_vessel_guided']
    all_results = []
    
    for scenario in SCENARIOS:
        print(f"\n{'='*50}\nTraining {scenario}\n{'='*50}")
        
        train_ds = ROPDataset(train_df, scenario, 'train', augment=True)
        val_ds = ROPDataset(val_df, scenario, 'val', augment=False)
        test_ds = ROPDataset(test_df, scenario, 'test', augment=False)
        
        tl = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=2)
        vl = DataLoader(val_ds, BATCH_SIZE, shuffle=False, num_workers=2)
        tel = DataLoader(test_ds, BATCH_SIZE, shuffle=False, num_workers=2)
        
        model = TinyResNet().to(DEVICE)
        cnt = np.array([len(train_df[train_df['label_id']==i]) for i in range(4)], dtype=np.float32)
        w = torch.tensor(cnt.sum() / (4 * np.maximum(cnt, 1)), dtype=torch.float32).to(DEVICE)
        crit = nn.CrossEntropyLoss(weight=w)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        
        bf1, bsd, stale = 0, None, 0
        for ep in range(1, EPOCHS+1):
            model.train()
            for imgs, lbls in tl:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                opt.zero_grad(); crit(model(imgs), lbls).backward(); opt.step()
            sched.step()
            
            model.eval()
            yt, yp = [], []
            with torch.no_grad():
                for imgs, lbls in vl:
                    lg = model(imgs.to(DEVICE))
                    yt.extend(lbls.tolist()); yp.extend(torch.argmax(lg, 1).cpu().tolist())
            p, r, f, _ = precision_recall_fscore_support(yt, yp, average='macro', zero_division=0)
            if f > bf1: bf1 = f; bsd = model.state_dict().copy(); stale = 0
            else: stale += 1
            if ep % 10 == 0 or ep == 1: print(f"  E{ep:3d} val_f1={f:.4f}")
            if stale >= PATIENCE: print(f"  Early stop at {ep}"); break
        
        model.load_state_dict(bsd)
        model.eval()
        yt, yp = [], []
        with torch.no_grad():
            for imgs, lbls in tel:
                lg = model(imgs.to(DEVICE))
                yt.extend(lbls.tolist()); yp.extend(torch.argmax(lg, 1).cpu().tolist())
        acc = accuracy_score(yt, yp)
        p, r, f, _ = precision_recall_fscore_support(yt, yp, average='macro', zero_division=0)
        all_results.append({'scenario': scenario, 'accuracy': float(acc), 'precision': float(p), 'recall': float(r), 'f1': float(f)})
        print(f"Test: Acc={acc:.4f} F1={f:.4f}")
        print(classification_report(yt, yp, target_names=classes, zero_division=0))
    
    print(f"\n{'='*50}\nFINAL COMPARISON\n{'='*50}")
    print(f"{'Scenario':20s} {'Accuracy':>8s} {'F1':>8s} {'Precision':>8s} {'Recall':>8s}")
    print('-' * 60)
    for r in all_results:
        print(f"{r['scenario']:20s} {r['accuracy']:>8.4f} {r['f1']:>8.4f} {r['precision']:>8.4f} {r['recall']:>8.4f}")
    
    with open(str(OUTPUT_DIR / 'results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_DIR / 'results.json'}")


if __name__ == '__main__':
    import sys
    if '--generate-only' in sys.argv:
        generate_scenario_images()
    elif '--train-only' in sys.argv:
        train()
    else:
        # Check if scenarios exist
        s1_path = CACHE_DIR / 'S1_raw' / 'train'
        scenarios_exist = s1_path.exists() and any(s1_path.iterdir()) if s1_path.exists() else False
        
        if not scenarios_exist:
            print("Step 1: Generating scenario images...")
            t0 = time.time()
            generate_scenario_images()
            print(f"  Took {time.time()-t0:.1f}s")
        
        print("\nStep 2: Training...")
        train()
