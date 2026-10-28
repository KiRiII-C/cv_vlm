import os, json, time, tarfile, random
from pathlib import Path
import requests, cv2, numpy as np, torch
from torch import nn
from torch.nn import functional as F

torch.set_num_threads(3)
torch.manual_seed(20261028); np.random.seed(20261028); random.seed(20261028)
D=Path("rgbd_data"); D.mkdir(exist_ok=True)
ids=[1,3,5,8,10,12,15,18,22,26]

def fetch(url,p):
    p=Path(p)
    if p.is_file() and p.stat().st_size>100: return
    r=requests.get(url,stream=True,timeout=(25,120))
    r.raise_for_status()
    with p.open("wb") as f:
        for chunk in r.iter_content(1048576):
            if chunk: f.write(chunk)

def dataset():
    entries=[]
    base="https://raw.githubusercontent.com/tum-vision/rgbd_scribble_benchmark/master/LabeledImages/"
    for i in ids:
        stem=f"kitchen_{i}"
        rgb=D/f"{stem}_image.png"; dep=D/f"{stem}_depth.png"
        fetch(base+rgb.name,rgb);fetch(base+dep.name,dep)
        entries.append((rgb,dep,"scribble",stem,1000))
    archive=D/"rgbd_dataset_freiburg1_xyz.tgz"
    errors=[]
    if not archive.exists():
        for u in [
          "https://webshare.cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz",
          "https://vision.in.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz",
          "https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz"
        ]:
            try:
                print("Downloading TUM Freiburg archive:",u,flush=True)
                fetch(u,archive); break
            except Exception as exc:
                errors.append(repr(exc))
                archive.unlink(missing_ok=True)
        if not archive.exists(): raise RuntimeError("Cannot download second dataset: "+str(errors))
    with tarfile.open(archive,"r:gz") as tf:
        names={m.name:m for m in tf.getmembers() if m.isfile()}
        def locate(s):
            x=[n for n in names if n.endswith("/"+s) or n==s]
            if not x: raise ValueError("Not in TUM archive: "+s)
            return x[0]
        def parse(name):
            lines=tf.extractfile(names[locate(name)]).read().decode().splitlines()
            return [(float(q[0]),q[1]) for line in lines if line and not line.startswith("#") for q in [line.split()] if len(q)>1]
        rg=parse("rgb.txt"); dp=parse("depth.txt")
        tt=np.array([x[0] for x in dp])
        pairs=[]
        for t,r in rg:
            j=int(np.argmin(abs(tt-t)))
            if abs(tt[j]-t)<.035:pairs.append((r,dp[j][1],t))
        if len(pairs)<10:raise RuntimeError("Too few synchronized pairs")
        for j,index in enumerate(np.linspace(0,len(pairs)-1,10,dtype=int)):
            r,d,t=pairs[int(index)]
            a=D/f"freiburg_{j:02}_rgb.png"; b=D/f"freiburg_{j:02}_depth.png"
            for src,out in [(r,a),(d,b)]:
                with tf.extractfile(names[locate(src)]) as file:
                    out.write_bytes(file.read())
            entries.append((a,b,"freiburg1_xyz",f"{t:.6f}",5000))
    return entries

def read(entries):
    samples=[]
    for a,b,group,id,scale in entries:
        im=cv2.imread(str(a),cv2.IMREAD_COLOR)
        dep=cv2.imread(str(b),cv2.IMREAD_UNCHANGED)
        if im is None or dep is None or dep.ndim!=2 or im.shape[:2]!=dep.shape:
            raise RuntimeError(f"Non-aligned RGB/depth: {a} {b}")
        im=cv2.resize(cv2.cvtColor(im,cv2.COLOR_BGR2RGB),(80,64),interpolation=cv2.INTER_AREA)/255.
        dep=dep.astype("float32")/scale
        v=((dep>.3)&(dep<5)).astype("float32")
        numer=cv2.resize(dep*v,(80,64),interpolation=cv2.INTER_AREA)
        denom=cv2.resize(v,(80,64),interpolation=cv2.INTER_AREA)
        d=np.where(denom>.75,numer/np.maximum(denom,1e-5),0.).astype("float32")
        v=((denom>.75)&(d>.3)&(d<5)).astype("float32")
        samples.append(dict(rgb=torch.from_numpy(im.astype("float32")).permute(2,0,1)[None],
                            d=torch.from_numpy(d)[None,None],valid=torch.from_numpy(v)[None,None],
                            group=group,id=id))
    return samples

def sample(s,seed):
    gen=torch.Generator().manual_seed(seed)
    v=s["valid"]
    hole=(torch.rand(v.shape,generator=gen)<.18).float()
    for _ in range(2):
        h=int(torch.randint(7,17,(1,),generator=gen))
        w=int(torch.randint(9,20,(1,),generator=gen))
        y=int(torch.randint(0,64-h,(1,),generator=gen))
        x=int(torch.randint(0,80-w,(1,),generator=gen))
        hole[:,:,y:y+h,x:x+w]=1
    missing=hole*v
    observed=(v-missing).clamp_min(0)
    return s["rgb"],s["d"]*observed,observed,missing,s["d"]

def enc(i,c=16):
    return nn.Sequential(nn.Conv2d(i,c,3,padding=1),nn.GELU(),nn.Conv2d(c,c,3,padding=1),nn.GELU())

class Net(nn.Module):
    def __init__(self,mode):
        super().__init__()
        self.mode=mode;c=16
        if mode=="depth":self.depth=enc(2);n=c+2
        elif mode=="early":self.early=enc(5);n=c+2
        else:
            self.rgb=enc(3);self.depth=enc(2)
            n=(3 if mode in ("attn_no_geo","geo_attn") else 2)*c+2
            if mode in ("attn_no_geo","geo_attn"):
                self.q=nn.Conv2d(c,c,1)
                self.k=nn.Conv2d(c,c,1)
                self.v=nn.Conv2d(c,c,1)
                self.out=nn.Conv2d(c,c,1)
                self.geo_scale=nn.Parameter(torch.tensor(1.))
        self.head=nn.Sequential(nn.Conv2d(n,c,3,padding=1),nn.GELU(),
                                nn.Conv2d(c,c,3,padding=1),nn.GELU(),nn.Conv2d(c,1,1))
    def forward(self,rgb,z,obs):
        mode=self.mode
        if mode=="depth":features=self.depth(torch.cat([z/5,obs],dim=1))
        elif mode=="early":features=self.early(torch.cat([rgb,z/5,obs],dim=1))
        else:
            r=self.rgb(rgb);d=self.depth(torch.cat([z/5,obs],dim=1))
            features=torch.cat((r,d),dim=1)
            if mode in ("attn_no_geo","geo_attn"):
                b,c,h,w=r.shape
                q=self.q(r).permute(0,2,3,1).reshape(b,h*w,1,c)
                k=F.unfold(self.k(d),3,padding=1).reshape(b,c,9,h*w).permute(0,3,2,1)
                v=F.unfold(self.v(d),3,padding=1).reshape(b,c,9,h*w).permute(0,3,2,1)
                mask=F.unfold(obs,3,padding=1).transpose(1,2)
                logits=(q*k).sum(-1)/(c**.5)
                if mode=="geo_attn":
                    # Build a proxy query XYZ using ONLY observed neighboring depth.
                    # Thus geometric attention remains active when central depth is hidden.
                    yy,xx=torch.meshgrid(torch.arange(h,device=z.device),
                                         torch.arange(w,device=z.device),indexing="ij")
                    nx=(xx-w/2)/(525*w/640)
                    ny=(yy-h/2)/(525*h/480)
                    xyz=torch.cat([z*nx[None,None],z*ny[None,None],z],dim=1)
                    neigh=F.unfold(xyz,3,padding=1).reshape(b,3,9,h*w).permute(0,3,2,1)
                    neigh_z=F.unfold(z,3,padding=1).transpose(1,2)
                    # Fill missing query depth by the local observed mean (no GT leakage).
                    avg_z=(neigh_z*mask).sum(-1,keepdim=True)/mask.sum(-1,keepdim=True).clamp_min(1)
                    q_z=torch.where(obs.flatten(2).transpose(1,2)>.5,
                                    z.flatten(2).transpose(1,2),avg_z)
                    query_ray=torch.stack([nx,ny,torch.ones_like(nx)],dim=-1).reshape(1,h*w,1,3)
                    anchor=q_z.unsqueeze(-1)*query_ray
                    logits=logits-self.geo_scale.abs()*((neigh-anchor)**2).sum(-1)
                logits=logits.masked_fill(mask<.5,-1e4)
                prob=logits.softmax(-1)*mask
                prob=prob/prob.sum(-1,keepdim=True).clamp_min(1e-8)
                attend=(prob[...,None]*v).sum(2).transpose(1,2).reshape(b,c,h,w)
                features=torch.cat((features,self.out(attend)),dim=1)
        return 5*torch.sigmoid(self.head(torch.cat((features,z/5,obs),dim=1)))

def score(net,test):
    net.eval();acc={}
    for group in ["scribble","freiburg1_xyz","all"]:acc[group]=[0.,0.,0]
    with torch.no_grad():
        for i,s in enumerate(test):
            rgb,z,obs,miss,target=sample(s,9876+100*i)
            err=net(rgb,z,obs)-target
            for g in (s["group"],"all"):
                acc[g][0]+=float((err.square()*miss).sum())
                acc[g][1]+=float((err.abs()*miss).sum())
                acc[g][2]+=int(miss.sum())
    return {k:dict(rmse_m=(v[0]/v[2])**.5 if v[2] else None,
                   mae_m=v[1]/v[2] if v[2] else None,
                   valid_test_pixels=v[2]) for k,v in acc.items()}

def run():
    s=read(dataset())
    train=[];test=[]
    for g in ("scribble","freiburg1_xyz"):
        seq=[x for x in s if x["group"]==g]
        assert len(seq)==10
        train+=seq[:7];test+=seq[7:]
    results={"task":"depth inpainting of withheld valid pixels",
             "dataset_counts":{"scribble":10,"freiburg1_xyz":10},
             "train_frame_ids":[[x["group"],x["id"]] for x in train],
             "test_frame_ids":[[x["group"],x["id"]] for x in test],
             "train_frames":14,"test_frames":6,
             "epochs":25,"seed":20261028,"input_resolution":[80,64],"models":{},
             "notes":["Two published RGB-D datasets, no object supervision",
                      "Approximate pinhole calibration for geometric bias",
                      "Different parameter budgets; 20 images are insufficient for generalization claims"]}
    for mode in ("depth","early","late","attn_no_geo","geo_attn"):
        torch.manual_seed(20261028)
        model=Net(mode)
        opt=torch.optim.AdamW(model.parameters(),lr=.002)
        started=time.perf_counter()
        for ep in range(25):
            losses=[]
            order=np.random.default_rng(20261028+ep).permutation(len(train))
            model.train()
            for idx in order:
                rgb,z,obs,miss,target=sample(train[int(idx)],10000+ep*200+int(idx))
                pred=model(rgb,z,obs)
                loss=(abs(pred-target)*miss).sum()/miss.sum().clamp_min(1)
                opt.zero_grad();loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),3.)
                opt.step();losses.append(float(loss))
            m=score(model,test)["all"]
            print(mode,"epoch",ep+1,"train_mae",round(np.mean(losses),4),"test_rmse",round(m["rmse_m"],4),flush=True)
        results["models"][mode]={"metrics":score(model,test),
                                  "parameters":sum(p.numel() for p in model.parameters()),
                                  "train_seconds":round(time.perf_counter()-started,2)}
        # Classical inpainting baseline: no learned RGB-D fusion.
    telea={}
    for g in ("scribble","freiburg1_xyz","all"):telea[g]=[0.,0.,0]
    for i,x in enumerate(test):
        rgb,z,obs,miss,target=sample(x,9876+100*i)
        zin=z[0,0].numpy().astype("float32")
        missing=(miss[0,0].numpy()>.5).astype("uint8")
        pred=cv2.inpaint(zin,missing,3,cv2.INPAINT_TELEA)
        err=pred-target[0,0].numpy()
        for g in (x["group"],"all"):
            telea[g][0]+=float(((err**2)*missing).sum())
            telea[g][1]+=float((abs(err)*missing).sum())
            telea[g][2]+=int(missing.sum())
    results["models"]["opencv_telea"]={"metrics":{g:{
        "rmse_m":(v[0]/v[2])**.5,"mae_m":v[1]/v[2],
        "valid_test_pixels":v[2]} for g,v in telea.items()},
        "parameters":0,"train_seconds":0}
    Path("results.json").write_text(json.dumps(results,indent=2))
    print("FINAL_METRICS",json.dumps(results["models"]),flush=True)

if __name__=="__main__":
    run()
