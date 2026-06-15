""" ResNet50 pretrained on ImageNet. S1_raw + WeightedRandomSampler. """
import cv2, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import pandas as pd, time, json, random, warnings
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models
warnings.filterwarnings('ignore')
import modal

app = modal.App("rop-pretrained")

image = (modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install("torch","torchvision","opencv-python","numpy","pandas","scikit-learn","scipy","tqdm")
    .add_local_dir("data/Zhao2024", remote_path="/root/data/Zhao2024"))

class ROPDataset(Dataset):
    def __init__(self,df,augment=False):
        self.df=df; self.augment=augment
    def __len__(self): return len(self.df)
    def __getitem__(self,idx):
        r=self.df.iloc[idx]
        img=cv2.imread(r['path'],cv2.IMREAD_COLOR)
        if img is None: return self.__getitem__((idx+1)%len(self.df))
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img=cv2.resize(img,(224,224),cv2.INTER_AREA)
        if self.augment:
            if random.random()>0.5: img=np.fliplr(img).copy()
            if random.random()>0.5: img=np.flipud(img).copy()
            a=random.uniform(-15,15); h,w=img.shape[:2]
            M=cv2.getRotationMatrix2D((w/2,h/2),a,1.0)
            img=cv2.warpAffine(img,M,(w,h),borderMode=cv2.BORDER_CONSTANT,borderValue=0)
            if random.random()>0.5:
                al=random.uniform(0.85,1.15); be=random.uniform(-10,10)
                img=np.clip(img.astype(np.float32)*al+be,0,255).astype(np.uint8)
        img=img.astype(np.float32)/255.0
        img=(img-np.array([0.485,0.456,0.406]))/np.array([0.229,0.224,0.225])
        return torch.from_numpy(img.transpose(2,0,1)).float(), torch.tensor(int(r['label_id']),dtype=torch.long)

@app.function(image=image, gpu="L40S", timeout=3600)
def run():
    classes=('Normal','Stage1','Stage2','Stage3'); cls2id={n:i for i,n in enumerate(classes)}
    root=Path("/root/data/Zhao2024"); exts={'.jpg','.jpeg','.png'}; rows=[]
    for c in classes:
        for p in sorted((root/c).iterdir()):
            if p.suffix.lower() in exts: rows.append({'path':str(p),'label':c,'label_id':cls2id[c]})
    df=pd.DataFrame(rows); print(f"{len(df)} images")
    train_df,temp=train_test_split(df,test_size=0.2,stratify=df['label_id'],random_state=42)
    val_df,test_df=train_test_split(temp,test_size=0.5,stratify=temp['label_id'],random_state=42)
    print(f"Train: {train_df['label'].value_counts().to_dict()}")
    
    # WeightedRandomSampler for balanced training
    labels=train_df['label_id'].values
    class_counts=np.bincount(labels)
    weights=1.0/class_counts[labels]
    sampler=WeightedRandomSampler(weights,len(weights),replacement=True)
    
    DEVICE=torch.device('cuda'); print(f"Device: {DEVICE}")
    crit=nn.CrossEntropyLoss()  # no class weights
    
    all_results=[]
    for seed in [42, 123, 456, 789, 1111]:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        print(f"\n--- Seed {seed} ---")
        
        tl=DataLoader(ROPDataset(train_df,True),32,sampler=sampler,num_workers=4)
        vl=DataLoader(ROPDataset(val_df,False),32,num_workers=4)
        tel=DataLoader(ROPDataset(test_df,False),32,num_workers=4)
        
        model=models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc=nn.Linear(model.fc.in_features,4)
        model=model.to(DEVICE)
        opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
        sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=80)
        
        bf1,bsd,stale=0,None,0
        for ep in range(1,81):
            model.train()
            for imgs,lbls in tl: imgs,lbls=imgs.to(DEVICE),lbls.to(DEVICE);opt.zero_grad();crit(model(imgs),lbls).backward();opt.step()
            sched.step()
            model.eval(); yt,yp=[],[]
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
        print(classification_report(yt,yp,target_names=classes,zero_division=0))
        all_results.append({'seed':seed,'accuracy':float(acc),'f1':float(f)})
    
    print(f"\n{'='*50}\nSUMMARY (5 seeds)")
    accs=[r['accuracy'] for r in all_results]; f1s=[r['f1'] for r in all_results]
    for r in all_results: print(f"  seed={r['seed']}: Acc={r['accuracy']:.4f} F1={r['f1']:.4f}")
    print(f"  Mean: Acc={np.mean(accs):.4f}±{np.std(accs):.4f}, F1={np.mean(f1s):.4f}±{np.std(f1s):.4f}")
    return all_results

@app.local_entrypoint()
def main():
    r=run.remote()
    print(f"\nMean F1: {np.mean([x['f1'] for x in r]):.4f} ± {np.std([x['f1'] for x in r]):.4f}")
