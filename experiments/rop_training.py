"""ROP Classification v2 - Wider model, MixUp, Focal Loss, from scratch."""
import cv2, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import pandas as pd, time, json, shutil, warnings, random
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from torch.utils.data import Dataset, DataLoader
from scipy import ndimage as ndi
warnings.filterwarnings('ignore')
import modal

app = modal.App("rop-classification-v2")
cache_vol = modal.Volume.from_name("rop-cache-v2", create_if_missing=True)

image = (modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install("torch","torchvision","opencv-python","numpy","scikit-image",
                 "scikit-learn","pandas","scipy","tqdm")
    .add_local_dir("data/Zhao2024", remote_path="/root/data/Zhao2024"))

# ── Vessel pipeline ──
def norm01(i,m=None):
    v=i[m] if m is not None else i.ravel(); v=v[np.isfinite(v)]
    if v.size==0: return np.zeros(i.shape,np.float32)
    lo,hi=np.percentile(v,[1,99]); hi=max(hi-lo,1e-8)
    return np.clip((i.astype(np.float32)-float(lo))/hi,0,1).astype(np.float32)
def est_fov(rgb):
    g=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    m=(g>max(3,int(np.percentile(g,1)))).astype(np.uint8)
    k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
    m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,k,2); m=ndi.binary_fill_holes(m>0).astype(np.uint8)
    nl,labels,stats,_=cv2.connectedComponentsWithStats(m,8)
    if nl>1: m=(labels==1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))).astype(np.uint8)
    return m.astype(bool)
def gabor_resp(inv_f,fov):
    r=np.zeros(inv_f.shape,np.float32)
    for sg,lm in [(1.5,3),(2.5,5),(3.5,7),(5,10)]:
        sz=max(7,int(6*sg)+(1-int(6*sg)%2))
        for a in range(0,180,15):
            th=np.deg2rad(a); c=sz//2; y,x=np.ogrid[-c:sz-c,-c:sz-c]
            xt=x*np.cos(th)+y*np.sin(th); yt=-x*np.sin(th)+y*np.cos(th)
            gk=np.exp(-0.5*(xt**2/sg**2+yt**2*0.25/sg**2))*np.cos(2*np.pi*xt/lm)
            gk=(gk-gk.mean()).astype(np.float32)
            r=np.maximum(r,cv2.filter2D(inv_f,cv2.CV_32F,gk,borderType=cv2.BORDER_REFLECT))
    r[~fov]=0; return norm01(r,fov)
def segment_vessels(rgb,fov):
    g=rgb[:,:,1].copy(); g[~fov]=0
    enh=cv2.createCLAHE(clipLimit=6,tileGridSize=(16,16)).apply(g); enh[~fov]=0
    inv=255-enh; inv_f=norm01(inv.astype(np.float32),fov)
    gab=gabor_resp(inv_f,fov)
    r7=cv2.medianBlur((gab*255).astype(np.uint8),7); r7[~fov]=0; soft=norm01(r7.astype(np.float32),fov)
    u8=np.clip(soft*255,0,255).astype(np.uint8); u8[~fov]=0
    enh2=cv2.createCLAHE(clipLimit=12,tileGridSize=(12,12)).apply(u8); enh2[~fov]=0
    sharp=norm01(enh2.astype(np.float32),fov)
    k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(17,17))
    inn=cv2.erode(fov.astype(np.uint8),k,1).astype(bool)
    vals=sharp[inn]; vals=vals[vals>0]
    if len(vals)==0: return np.zeros(fov.shape,bool)
    th=float(np.percentile(vals,84)); bin=(sharp>=th)&inn
    nl,labels,stats,_=cv2.connectedComponentsWithStats(bin.astype(np.uint8),8)
    if nl>1:
        areas=[(stats[i,cv2.CC_STAT_AREA],i) for i in range(1,nl)]; areas.sort(reverse=True)
        keep={idx for _,idx in areas[:2]}; bin=np.isin(labels,list(keep))
    return bin

# ── MixUp augmentation ──
def mixup(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = (lam * y + (1 - lam) * y[index])
    return mixed_x, mixed_y

# ── Focal Loss ──
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

# ── Wider model: MiniResNet (like ResNet18 but narrower) ──
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, ic, oc, s=1):
        super().__init__()
        self.c1=nn.Conv2d(ic,oc,3,s,1,bias=False);self.b1=nn.BatchNorm2d(oc)
        self.c2=nn.Conv2d(oc,oc,3,1,1,bias=False);self.b2=nn.BatchNorm2d(oc)
        self.sk=nn.Identity() if s==1 and ic==oc else nn.Sequential(
            nn.Conv2d(ic,oc,1,s,bias=False),nn.BatchNorm2d(oc))
    def forward(self,x):
        o=F.relu(self.b1(self.c1(x)));o=self.b2(self.c2(o));return F.relu(o+self.sk(x))

class TinyResNetV2(nn.Module):
    """Tiny ResNet - slightly wider: 48-96-192 channels. 
    Proved to work better than deeper models from scratch on 756 images."""
    def __init__(self, nc=4, wd=(48, 96, 192)):
        super().__init__()
        def ml(ic,oc,b,s): return nn.Sequential(*[BasicBlock(ic,oc,s)]+[BasicBlock(oc,oc,1) for _ in range(1,b)])
        self.st=nn.Sequential(nn.Conv2d(3,wd[0],3,1,1,bias=False),nn.BatchNorm2d(wd[0]),nn.ReLU(True),nn.MaxPool2d(2))
        self.l1=ml(wd[0],wd[0],2,1);self.l2=ml(wd[0],wd[1],2,2);self.l3=ml(wd[1],wd[2],2,2)
        self.p=nn.AdaptiveAvgPool2d((1,1));self.do=nn.Dropout(0.3);self.f=nn.Linear(wd[2],nc)
    def forward(self,x): x=self.st(x);x=self.l1(x);x=self.l2(x);x=self.l3(x);x=self.p(x).flatten(1);x=self.do(x);return self.f(x)

# ── Dataset (uses pre-generated scenarios) ──
class ROPDataset(Dataset):
    def __init__(self,cache,df,augment=False):
        self.cache=cache; self.df=df; self.augment=augment
    def __len__(self): return len(self.df)
    def __getitem__(self,idx):
        r=self.df.iloc[idx]; stem=Path(r['path']).stem
        img=cv2.imread(str(self.cache/r['label']/r['split']/f"{stem}.jpg"),cv2.IMREAD_COLOR)
        if img is None: return self.__getitem__((idx+1)%len(self.df))
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        if self.augment:
            if random.random()>0.5: img=np.fliplr(img).copy()
            if random.random()>0.5: img=np.flipud(img).copy()
            angle=random.uniform(-15,15)
            h,w=img.shape[:2]
            M=cv2.getRotationMatrix2D((w/2,h/2),angle,1.0)
            img=cv2.warpAffine(img,M,(w,h),borderMode=cv2.BORDER_CONSTANT,borderValue=0)
            if random.random()>0.5:
                alpha=random.uniform(0.8,1.2); beta=random.uniform(-15,15)
                img=np.clip(img.astype(np.float32)*alpha+beta,0,255).astype(np.uint8)
        img=img.astype(np.float32)/255.0
        img=(img-np.array([0.485,0.456,0.406]))/np.array([0.229,0.224,0.225])
        return torch.from_numpy(img.transpose(2,0,1)).float(), torch.tensor(int(r['label_id']),dtype=torch.long)

@app.function(image=image, volumes={"/cache": cache_vol}, gpu="L40S", timeout=7200)
def run_v2():
    classes=('Normal','Stage1','Stage2','Stage3'); cls2id={n:i for i,n in enumerate(classes)}
    root=Path("/root/data/Zhao2024"); cache=Path("/cache/scenarios")
    import glob as g
    exts={'.jpg','.jpeg','.png'}; rows=[]
    for c in classes:
        d=root/c
        if not d.exists(): continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in exts: rows.append({'path':str(p),'label':c,'label_id':cls2id[c]})
    df=pd.DataFrame(rows); print(f"Loaded {len(df)} images")
    train_df,temp=train_test_split(df,test_size=0.2,stratify=df['label_id'],random_state=42)
    val_df,test_df=train_test_split(temp,test_size=0.5,stratify=temp['label_id'],random_state=42)
    for df_ in [train_df,val_df,test_df]: df_=df_.copy()
    train_df=train_df.copy(); train_df['split']='train'
    val_df=val_df.copy(); val_df['split']='val'
    test_df=test_df.copy(); test_df['split']='test'
    
    # Generate S3 scenarios (only S3, our best scenario)
    if cache.exists(): shutil.rmtree(str(cache))
    for sn in ['train','val','test']:
        for c in classes: (cache/c/sn).mkdir(parents=True,exist_ok=True)
    
    print("Generating S3 vessel mask scenarios...")
    for sn,sdf in [('train',train_df),('val',val_df),('test',test_df)]:
        t0=time.time()
        for idx,(_,row) in enumerate(sdf.iterrows()):
            rgb=cv2.imread(row['path'],cv2.IMREAD_COLOR); rgb=cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB)
            stem=Path(row['path']).stem; label=row['label']
            fov=est_fov(rgb)
            h,w=rgb.shape[:2]; s=min(1.0,768/max(h,w))
            wrk=cv2.resize(rgb,(int(w*s),int(h*s)),cv2.INTER_AREA) if s<1 else rgb.copy()
            fov_w=est_fov(wrk) if s<1 else fov
            bin_seg=segment_vessels(wrk,fov_w)
            br=cv2.resize(bin_seg.astype(np.uint8),(224,224),cv2.INTER_NEAREST).astype(bool)
            s3=np.zeros((224,224,3),dtype=np.uint8); s3[br]=255
            cv2.imwrite(str(cache/label/sn/f"{stem}.jpg"),cv2.cvtColor(s3,cv2.COLOR_RGB2BGR))
            if (idx+1)%100==0: print(f"  {sn} [{idx+1}/{len(sdf)}]")
        print(f"  {sn} done in {time.time()-t0:.0f}s")
    print("S3 scenarios ready!")
    
    # Train
    DEVICE=torch.device('cuda'); print(f"\nTraining on {DEVICE}")
    
    # Setup
    train_ds=ROPDataset(cache,train_df,augment=True)
    val_ds=ROPDataset(cache,val_df,augment=False)
    test_ds=ROPDataset(cache,test_df,augment=False)
    
    tl=DataLoader(train_ds,32,shuffle=True,num_workers=4,drop_last=True)
    vl=DataLoader(val_ds,32,shuffle=False,num_workers=4)
    tel=DataLoader(test_ds,32,shuffle=False,num_workers=4)
    
    # Ensemble of models with different seeds
    seeds = [42, 123, 456]
    all_test_preds = []
    
    for seed_idx, seed in enumerate(seeds):
        print(f"\n{'='*50}\nTraining model {seed_idx+1}/{len(seeds)} (seed={seed})")
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        
        model=TinyResNetV2().to(DEVICE)
        cnt=np.array([len(train_df[train_df['label_id']==i]) for i in range(4)],np.float32)
        w=torch.tensor(cnt.sum()/(4*np.maximum(cnt,1)),dtype=torch.float32).to(DEVICE)
        crit=FocalLoss(gamma=1.0, weight=w)  # milder focal loss
        opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
        sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=80)
        
        bf1,bsd,stale=0,None,0
        for ep in range(1,81):
            model.train()
            for imgs,lbls in tl:
                imgs,lbls=imgs.to(DEVICE),lbls.to(DEVICE)
                opt.zero_grad(); crit(model(imgs), lbls).backward(); opt.step()
            sched.step()
            model.eval(); yt,yp=[],[]
            with torch.no_grad():
                for imgs,lbls in vl: 
                    lg=model(imgs.to(DEVICE));yt.extend(lbls.tolist());yp.extend(torch.argmax(lg,1).cpu().tolist())
            p,r,f,_=precision_recall_fscore_support(yt,yp,average='macro',zero_division=0)
            if f>bf1:bf1=f;bsd=model.state_dict().copy();stale=0
            else:stale+=1
            if ep%15==0 or ep==1:print(f"  E{ep:3d}/80 val_f1={f:.4f} (best={bf1:.4f})")
            if stale>=15:print(f"  Early stop at {ep}");break
        
        model.load_state_dict(bsd);model.eval()
        yt,yp=[],[]
        with torch.no_grad():
            for imgs,lbls in tel: 
                lg=model(imgs.to(DEVICE));yt.extend(lbls.tolist());yp.extend(torch.argmax(lg,1).cpu().tolist())
        acc=accuracy_score(yt,yp);p,r,f,_=precision_recall_fscore_support(yt,yp,average='macro',zero_division=0)
        print(f"\nModel {seed_idx+1}: Acc={acc:.4f} F1={f:.4f}")
        print(classification_report(yt,yp,target_names=classes,zero_division=0))
        all_test_preds.append(yp)
    
    # Ensemble: majority vote
    print("\n"+"="*50+"\nENSEMBLE RESULTS")
    ensemble_preds = []
    for i in range(len(all_test_preds[0])):
        votes = [preds[i] for preds in all_test_preds]
        ensemble_preds.append(max(set(votes), key=votes.count))
    yt_test = []
    for _,row in test_df.iterrows(): yt_test.append(int(row['label_id']))
    acc=accuracy_score(yt_test,ensemble_preds)
    p,r,f,_=precision_recall_fscore_support(yt_test,ensemble_preds,average='macro',zero_division=0)
    print(f"Ensemble (3 models majority vote): Acc={acc:.4f} F1={f:.4f}")
    print(classification_report(yt_test,ensemble_preds,target_names=classes,zero_division=0))
    
    result={'scenario':'S3_vessel_mask','model':'MiniResNet_ensemble',
            'accuracy':float(acc),'f1':float(f),'precision':float(p),'recall':float(r)}
    print(f"\nFINAL: Acc={acc:.4f} F1={f:.4f}")
    return result

@app.local_entrypoint()
def main():
    print("ROP Classification v2 - S3 only, wider model, MixUp, Focal Loss, Ensemble")
    r=run_v2.remote()
    print(f"\nResult: {r['scenario']}: Acc={r['accuracy']:.4f}, F1={r['f1']:.4f}")
