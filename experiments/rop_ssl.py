""" SimCLR self-supervised pretraining on all 1099 Zhao2024 images.
Step 1: Pretrain encoder on ALL images (including laser scars, no labels needed)
Step 2: Fine-tune encoder on labeled 756 images for ROP stage classification
"""
import cv2, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import pandas as pd, time, random, warnings
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from torch.utils.data import Dataset, DataLoader
warnings.filterwarnings('ignore')
import modal

app = modal.App("rop-ssl")
image = (modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install("torch","torchvision","opencv-python","numpy","pandas","scikit-learn","scipy","tqdm")
    .add_local_dir("data/Zhao2024", remote_path="/root/data/Zhao2024"))

# ── TinyResNet encoder (same architecture that was stable) ──
class BasicBlock(nn.Module):
    expansion=1
    def __init__(self,ic,oc,s=1):
        super().__init__()
        self.c1=nn.Conv2d(ic,oc,3,s,1,bias=False);self.b1=nn.BatchNorm2d(oc)
        self.c2=nn.Conv2d(oc,oc,3,1,1,bias=False);self.b2=nn.BatchNorm2d(oc)
        self.sk=nn.Identity() if s==1 and ic==oc else nn.Sequential(nn.Conv2d(ic,oc,1,s,bias=False),nn.BatchNorm2d(oc))
    def forward(self,x): o=F.relu(self.b1(self.c1(x)));o=self.b2(self.c2(o));return F.relu(o+self.sk(x))

class Encoder(nn.Module):
    """TinyResNet encoder - proven to work from scratch on this data."""
    def __init__(self, dim=128):
        super().__init__()
        def ml(ic,oc,b,s): return nn.Sequential(*[BasicBlock(ic,oc,s)]+[BasicBlock(oc,oc,1) for _ in range(1,b)])
        self.st=nn.Sequential(nn.Conv2d(3,32,3,1,1,bias=False),nn.BatchNorm2d(32),nn.ReLU(True),nn.MaxPool2d(2))
        self.l1=ml(32,32,2,1);self.l2=ml(32,64,2,2);self.l3=ml(64,128,2,2)
        self.p=nn.AdaptiveAvgPool2d((1,1))
        self.proj=nn.Sequential(nn.Linear(128,128),nn.BatchNorm1d(128),nn.ReLU(),nn.Linear(128,dim))
    def forward(self,x,return_embed=False):
        x=self.st(x);x=self.l1(x);x=self.l2(x);x=self.l3(x);x=self.p(x).flatten(1)
        if return_embed: return x
        return self.proj(x)

# ── Contrastive loss ──
def nt_xent_loss(z, temp=0.5):
    z = F.normalize(z, dim=1)
    two_n = z.shape[0]
    n = two_n // 2
    sim = z @ z.T / temp
    sim = sim - torch.eye(two_n, device=z.device) * 1e9
    idx = torch.arange(two_n, device=z.device)
    positive_idx = (idx + n) % two_n
    return F.cross_entropy(sim, positive_idx)

# ── Augmentation for SimCLR (stronger) ──
def ssl_augment(img):
    img = img.copy()
    if random.random()>0.5: img=np.fliplr(img)
    a=random.uniform(-30,30); h,w=img.shape[:2]
    M=cv2.getRotationMatrix2D((w/2,h/2),a,1.0)
    img=cv2.warpAffine(img,M,(w,h),borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    # Color jitter
    if random.random()>0.3:
        for _ in range(3):
            ch=random.randint(0,2)
            img[:,:,ch]=np.clip(img[:,:,ch].astype(np.float32)*random.uniform(0.6,1.4),0,255).astype(np.uint8)
    # Gaussian blur
    if random.random()>0.5:
        ks=random.choice([3,5,7])
        img=cv2.GaussianBlur(img,(ks,ks),0)
    img=cv2.resize(img,(224,224),cv2.INTER_AREA)
    img=img.astype(np.float32)/255.0
    img=(img-np.array([0.485,0.456,0.406]))/np.array([0.229,0.224,0.225])
    return torch.from_numpy(img.transpose(2,0,1)).float()

@app.function(image=image, gpu="L40S", timeout=14400)
def run_ssl():
    cv2.setNumThreads(0)
    root=Path("/root/data/Zhao2024")
    classes=('Normal','Stage1','Stage2','Stage3','laser scars')
    cls2id={n:i for i,n in enumerate(classes)}
    exts={'.jpg','.jpeg','.png'}; rows=[]
    for c in classes:
        d=root/c
        if not d.exists():continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in exts: rows.append({'path':str(p),'label':c,'label_id':cls2id[c]})
    all_df=pd.DataFrame(rows)
    # Exclude laser scars for fine-tuning
    finetune_df=all_df[all_df['label']!='laser scars'].reset_index(drop=True)
    print(f"SSL pretrain: {len(all_df)} images, Fine-tune: {len(finetune_df)} images")

    DEVICE=torch.device('cuda'); print(f"Device: {DEVICE}")

    # ═══ Step 1: SimCLR Pretraining on ALL 1099 images ═══
    print("\n=== STEP 1: SimCLR Pretraining ===")
    encoder=Encoder(dim=128).to(DEVICE)
    opt=torch.optim.AdamW(encoder.parameters(),lr=3e-4,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=200)

    class SSLDataset(Dataset):
        def __init__(self,df): self.df=df
        def __len__(self): return len(self.df)
        def __getitem__(self,idx):
            img=cv2.imread(self.df.iloc[idx]['path'],cv2.IMREAD_COLOR)
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            return ssl_augment(img), ssl_augment(img)

    ssl_loader=DataLoader(SSLDataset(all_df),128,shuffle=True,num_workers=2,drop_last=True,pin_memory=True)
    
    t0=time.time()
    for ep in range(1,201):
        encoder.train()
        loss_sum=0
        for imgs1,imgs2 in ssl_loader:
            x=torch.cat([imgs1,imgs2],dim=0).to(DEVICE)
            z=encoder(x)
            loss=nt_xent_loss(z)
            opt.zero_grad();loss.backward();opt.step()
            loss_sum+=loss.item()
        sched.step()
        if ep%10==0:print(f"  E{ep:3d}/200 loss={loss_sum/len(ssl_loader):.4f} ({time.time()-t0:.0f}s)")

    # ═══ Step 2: Fine-tune on labeled 756 images ═══
    print("\n=== STEP 2: Fine-tuning ===")

    class Classifier(nn.Module):
        def __init__(self, enc):
            super().__init__()
            self.enc = enc
            self.fc = nn.Linear(128, 4)
        def forward(self, x):
            f = self.enc(x, return_embed=True)
            return self.fc(f)
    
    # Split
    train_df,temp=train_test_split(finetune_df,test_size=0.2,stratify=finetune_df['label_id'],random_state=42)
    val_df,test_df=train_test_split(temp,test_size=0.5,stratify=temp['label_id'],random_state=42)
    
    class FinetuneDataset(Dataset):
        def __init__(self,df,augment=False):
            self.df=df; self.augment=augment
        def __len__(self): return len(self.df)
        def __getitem__(self,idx):
            r=self.df.iloc[idx]
            img=cv2.imread(r['path'],cv2.IMREAD_COLOR); img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img=cv2.resize(img,(224,224),cv2.INTER_AREA)
            if self.augment:
                if random.random()>0.5: img=np.fliplr(img)
                if random.random()>0.5: img=np.flipud(img)
                a=random.uniform(-15,15);h,w=img.shape[:2]
                M=cv2.getRotationMatrix2D((w/2,h/2),a,1.0)
                img=cv2.warpAffine(img,M,(w,h),borderMode=cv2.BORDER_CONSTANT,borderValue=0)
            img=img.astype(np.float32)/255.0
            img=(img-np.array([0.485,0.456,0.406]))/np.array([0.229,0.224,0.225])
            return torch.from_numpy(img.transpose(2,0,1)).float(), torch.tensor(int(r['label_id']),dtype=torch.long)
    
    results=[]
    for seed in [42, 123, 456]:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        print(f"\nFine-tuning seed {seed}")
        
        model = Classifier(Encoder(dim=128).to(DEVICE))
        model.enc.load_state_dict(encoder.state_dict())
        for p in model.enc.parameters(): p.requires_grad = True

        cnt=np.array([len(train_df[train_df['label_id']==i]) for i in range(4)],np.float32)
        w=torch.tensor(cnt.sum()/(4*np.maximum(cnt,1)),dtype=torch.float32).to(DEVICE)
        
        tl=DataLoader(FinetuneDataset(train_df,True),32,shuffle=True,num_workers=2)
        vl=DataLoader(FinetuneDataset(val_df,False),32,num_workers=2)
        tel=DataLoader(FinetuneDataset(test_df,False),32,num_workers=2)
        
        opt2=torch.optim.AdamW([
            {'params':model.enc.parameters(),'lr':1e-4},
            {'params':model.fc.parameters(),'lr':1e-3},
        ],weight_decay=1e-4)
        sched2=torch.optim.lr_scheduler.CosineAnnealingLR(opt2,T_max=80)
        crit=nn.CrossEntropyLoss(weight=w)
        
        bf1,bsd,stale=0,None,0
        for ep in range(1,81):
            model.train()
            for imgs,lbls in tl: imgs,lbls=imgs.to(DEVICE),lbls.to(DEVICE);opt2.zero_grad();crit(model(imgs),lbls).backward();opt2.step()
            sched2.step()
            model.eval();yt,yp=[],[]
            with torch.no_grad():
                for imgs,lbls in vl: lg=model(imgs.to(DEVICE));yt.extend(lbls.tolist());yp.extend(torch.argmax(lg,1).cpu().tolist())
            p,r,f,_=precision_recall_fscore_support(yt,yp,average='macro',zero_division=0)
            if f>bf1:bf1=f;bsd=model.state_dict().copy();stale=0
            else:stale+=1
            if ep%20==0:print(f"  E{ep:3d} val_f1={f:.4f}")
            if stale>=15:break
        
        model.load_state_dict(bsd);model.eval();yt,yp=[],[]
        with torch.no_grad():
            for imgs,lbls in tel: lg=model(imgs.to(DEVICE));yt.extend(lbls.tolist());yp.extend(torch.argmax(lg,1).cpu().tolist())
        acc=accuracy_score(yt,yp);p,r,f,_=precision_recall_fscore_support(yt,yp,average='macro',zero_division=0)
        print(f"Test: Acc={acc:.4f} F1={f:.4f}")
        print(classification_report(yt,yp,target_names=('Normal','Stage1','Stage2','Stage3'),zero_division=0))
        results.append({'seed':seed,'accuracy':float(acc),'f1':float(f)})
    
    print(f"\n{'='*50}\nSUMMARY")
    for r in results: print(f"  seed={r['seed']}: Acc={r['accuracy']:.4f} F1={r['f1']:.4f}")
    print(f"  Mean F1={np.mean([r['f1'] for r in results]):.4f} ± {np.std([r['f1'] for r in results]):.4f}")
    return results

@app.local_entrypoint()
def main():
    print("SimCLR pretraining on all Zhao2024 images, then fine-tune on 756 labeled")
    call = run_ssl.spawn()
    print(f"Spawned (detached) run_ssl: {call.object_id}")
    r = call.get()
    print(f"\nFinal: F1={np.mean([x['f1'] for x in r]):.4f} ± {np.std([x['f1'] for x in r]):.4f}")
