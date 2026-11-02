"""EXP-20261102-03 cross-scene real RGB-D representation transfer (no Bonn tuning).

Load five TUM-trained scratch encoder weights; evaluate 5-way retrieval on held-out
TUM and independent measured Bonn sequence with physically distinct RGB-hard negatives.
"""
import argparse,pathlib,json
import numpy as np, torch
from real_rgbd import load
from hard_negative_retrieval import create_queries,corrected_frames
from real_neural_contrastive import Encoder,crop
SEED=20261102
NAMES=("mlp","rgb_cnn","rgbd_cnn","geometry_cnn","patchvit")

def gather(rows):
    samples=[];pairs=[]
    for i,A,B,(x,y),positions in rows:
        fA=(A[0],A[1],A[6],A[4],A[5])
        fB=(B[0],B[1],B[6],B[4],B[5])
        query=crop(fA,x,y)
        if query is None:continue
        c=[crop(fB,px,py) for px,py in positions]
        if any(z is None for z in c):continue
        samples.append(query);pairs.append((i,np.stack(c)))
    if not samples:raise RuntimeError("No real measured hard negative patches")
    return np.asarray(samples,dtype=np.float32),np.stack([p[1] for p in pairs]).astype('f'),np.array([p[0] for p in pairs])

def encode(net,arr):
    net.eval();res=[]
    with torch.no_grad():
        for sl in range(0,len(arr),256):
            z=net(torch.from_numpy(arr[sl:sl+256]))
            res.append(z.numpy())
    return np.concatenate(res)

def evaluate(net,rows,seed):
    X,Y,groups=gather(rows)
    if len(X)<70:raise RuntimeError("Insufficient real hard negative observations")
    z=encode(net,X)
    p=encode(net,Y.reshape(-1,7,9,9)).reshape(len(X),5,-1)
    sims=np.einsum('nd,nkd->nk',z,p)
    success=(np.argmax(sims,axis=1)==0)
    per_group={i:success[groups==i] for i in np.unique(groups)}
    rng=np.random.default_rng(seed);keys=list(per_group);boot=[]
    for _ in range(300):
        draw=rng.choice(keys,len(keys),replace=True)
        boot.append(float(np.mean(np.concatenate([per_group[i] for i in draw]))))
    return dict(n_queries=int(len(X)),n_view_pairs=int(len(keys)),rank1=float(np.mean(success)),
        ci95=[float(x) for x in np.percentile(boot,[2.5,97.5])],
        n_parameters=sum(x.numel() for x in net.parameters()))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--tum-root",required=True);p.add_argument("--bonn-root",required=True)
    p.add_argument("--checkpoints",default="results/real_rgbd/tum_xyz/checkpoints")
    p.add_argument("--out",default="results/real_rgbd/transfer");a=p.parse_args()
    gt_tum=[517.3,516.5,318.6,255.3];gt_bonn=[542.822841,542.576870,315.593520,237.756098]
    T,_=corrected_frames(a.tum_root,gt_tum,'tum_xyz',100)
    B,_=corrected_frames(a.bonn_root,gt_bonn,'bonn_box',84)
    # First 70% TUM training; retrieval comes from later 25%, with a 5%-frame buffer.
    rT=create_queries(T[int(len(T)*.75):])
    rB=create_queries(B)
    results={}
    for name in NAMES:
        net=Encoder(name)
        path=pathlib.Path(a.checkpoints)/f"{name}.pt"
        if not path.exists():raise RuntimeError("Missing trained checkpoint "+str(path))
        net.load_state_dict(torch.load(path,map_location="cpu",weights_only=True))
        results[name]={
            "TUM_heldout":evaluate(net,rT,SEED+2),
            "Bonn_unseen":evaluate(net,rB,SEED+3)
        }
    torch.manual_seed(SEED)
    random_model=Encoder('patchvit')
    results["patchvit_untrained"]={
        "TUM_heldout":evaluate(random_model,rT,SEED+2),
        "Bonn_unseen":evaluate(random_model,rB,SEED+3)
    }
    output={"experiment_id":"EXP-20261102-03","n_tum_frames":len(T),"n_bonn_frames":len(B),
      "train":"TUM first ~70%: supervised correspondences using measured RGB-D + provided pose; no Bonn finetuning",
      "evaluation":"TUM last 25% and Bonn independent full measured recording, 1 correct match among 4 RGB-similar measured 3D alternatives",
      "scores":results,"limitations":"One static TUM room and one dynamic Bonn room; unequal model parameter budgets; hard negatives selected by RGB; no object-level ground truth."}
    out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True)
    (out/"transfer_results.json").write_text(json.dumps(output,indent=2))
    print("CROSS_SCENE_RESULT",json.dumps(output,sort_keys=True))
if __name__=="__main__":main()
