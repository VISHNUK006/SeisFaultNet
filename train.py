import argparse, random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupShuffleSplit
from src.data import SeismicDataset
from src.model import UNet2D
from src.losses import BCEDiceLoss
from src.metrics import binary_metrics

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def main():
    p=argparse.ArgumentParser(description="Train SeisFaultNet with group-held-out validation.")
    p.add_argument("--data",required=True)
    p.add_argument("--epochs",type=int,default=30)
    p.add_argument("--batch-size",type=int,default=4)
    p.add_argument("--lr",type=float,default=1e-3)
    p.add_argument("--patience",type=int,default=7)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--output",default="models/seisfaultnet.pt")
    a=p.parse_args()
    seed_everything(a.seed)

    z=np.load(a.data,allow_pickle=False)
    required={"seismic","fault_mask","groups"}
    if not required.issubset(z.files):
        raise ValueError(f"Dataset must contain {sorted(required)}")
    seismic,masks,groups=z["seismic"],z["fault_mask"],z["groups"]
    if seismic.ndim!=3 or masks.shape!=seismic.shape:
        raise ValueError("seismic and fault_mask must have shape (N,H,W)")
    if len(groups)!=len(seismic) or len(np.unique(groups))<2:
        raise ValueError("groups must have one ID per section and at least two unique groups.")

    splitter=GroupShuffleSplit(n_splits=1,test_size=.25,random_state=a.seed)
    tr,va=next(splitter.split(seismic,groups=groups))
    if set(groups[tr]) & set(groups[va]):
        raise RuntimeError("Group leakage detected.")

    train_ds=SeismicDataset(seismic,masks,augment=True)
    val_ds=SeismicDataset(seismic,masks,augment=False)
    tl=DataLoader(Subset(train_ds,tr),batch_size=a.batch_size,shuffle=True)
    vl=DataLoader(Subset(val_ds,va),batch_size=a.batch_size,shuffle=False)

    device="cuda" if torch.cuda.is_available() else "cpu"
    model=UNet2D().to(device)
    pos=max(int((masks[tr]>0).sum()),1)
    neg=max(int(masks[tr].size-pos),1)
    criterion=BCEDiceLoss(pos_weight=min(neg/pos,50.)).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="max",factor=.5,patience=2)
    best=-1.; stale=0
    history=[]

    for epoch in range(1,a.epochs+1):
        model.train(); losses=[]
        for x,y in tl:
            x,y=x.to(device),y.to(device)
            optimizer.zero_grad(); loss=criterion(model(x),y); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step(); losses.append(loss.item())

        model.eval(); ys=[]; ps=[]
        with torch.no_grad():
            for x,y in vl:
                ps.append(torch.sigmoid(model(x.to(device))).cpu().numpy()); ys.append(y.numpy())
        m=binary_metrics(np.concatenate(ys),np.concatenate(ps))
        scheduler.step(m["f1"])
        row={"epoch":epoch,"loss":float(np.mean(losses)),**m,"lr":optimizer.param_groups[0]["lr"]}
        history.append(row)
        print(f"Epoch {epoch:03d} | loss={row['loss']:.4f} | F1={m['f1']:.4f} | IoU={m['iou']:.4f}")
        if m["f1"]>best:
            best=m["f1"]; stale=0
            out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
            torch.save({"model_state_dict":model.state_dict(),"validation_metrics":m,
                        "train_groups":[str(v) for v in np.unique(groups[tr])],
                        "validation_groups":[str(v) for v in np.unique(groups[va])],
                        "seed":a.seed},out)
        else: stale+=1
        if stale>=a.patience:
            print("Early stopping.")
            break

    Path("outputs").mkdir(exist_ok=True)
    np.savetxt("outputs/training_history.csv",
               np.array([[r[k] for k in ["epoch","loss","precision","recall","f1","iou","lr"]] for r in history]),
               delimiter=",",header="epoch,loss,precision,recall,f1,iou,lr",comments="")
    print(f"Best held-out-group F1: {best:.4f}")
    print(f"Saved checkpoint: {a.output}")

if __name__=="__main__":
    main()
