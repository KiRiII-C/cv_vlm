"""EXP-20261102-01: Multi-view depth prediction and evidence-aware memory.

Actual measured RGB-D and external GT camera poses; tests a physical ray-consistency
hypothesis, not learned semantic tracking. Explicit abstention and overlap-risk curves.
No synthetic observations, no oracle target data in prediction.
"""
from __future__ import annotations
import argparse, json, pathlib, warnings
import numpy as np
from ego_motion_warp import load
from bonn_extrinsics import camera_pose_from_marker
SEED = 20261102
OFFSETS=(1,2,3,4,6,8)
METHODS=("recent","min_depth","median","robust_mean","evidence_gate","strict_gate")

def frames_with_poses(root, k, dataset, n):
    frames=load(root,k,n)
    if dataset=="bonn_box":
        out=[]
        for rgb,z,K,R,T in frames:
            A=np.eye(4);A[:3,:3]=R;A[:3,3]=T
            A=camera_pose_from_marker(A)
            out.append((rgb,z,K,A[:3,:3],A[:3,3]))
        frames=out
    return frames

def warp_depth(src,tgt):
    _,z,K,R,T=src
    _,_,Kt,Rt,Tt=tgt
    H,W=z.shape
    yy,xx=np.mgrid[:H,:W]
    valid=np.isfinite(z)&(z>.3)&(z<4.5)
    d=z[valid];x=xx[valid];y=yy[valid]
    pts=np.stack(((x-K[2])*d/K[0],(y-K[3])*d/K[1],d),axis=-1)
    world=pts@R.T+T
    p=(world-Tt)@Rt
    good=np.isfinite(p).all(-1)&(p[:,2]>.3)&(p[:,2]<4.5)
    p=p[good];depth=p[:,2]
    u=np.rint(Kt[0]*p[:,0]/depth+Kt[2]).astype(int)
    v=np.rint(Kt[1]*p[:,1]/depth+Kt[3]).astype(int)
    ok=(u>=0)&(u<W)&(v>=0)&(v<H)
    pix=v[ok]*W+u[ok];q=depth[ok]
    output=np.full(H*W,np.inf)
    np.minimum.at(output,pix,q)
    output[~np.isfinite(output)]=np.nan
    return output.reshape(H,W)

def predict(w):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",RuntimeWarning)
        with np.errstate(invalid="ignore"):
            count=np.sum(np.isfinite(w),axis=0)
            median=np.nanmedian(w,axis=0)
            mad=np.nanmedian(abs(w-median),axis=0)
            robust=abs(w-median)<=.075
            numerator=np.nansum(np.where(robust,w,np.nan),axis=0)
            denominator=np.sum(robust&np.isfinite(w),axis=0)
            rob=np.divide(numerator,denominator,out=np.full_like(median,np.nan),where=denominator>0)
            return {
                "recent": w[0],
                "min_depth": np.nanmin(w,axis=0),
                "median": median,
                "robust_mean": rob,
                "evidence_gate": np.where((count>=2)&(mad<.06),median,np.nan),
                "strict_gate": np.where((count>=3)&(mad<.035),median,np.nan),
            }

def aggregate(records,method):
    return {
        "coverage":sum(x["covered"] for x in records)/max(1,sum(x["valid_target"] for x in records)),
        "mae_m":sum(x["absolute_error"] for x in records)/max(1,sum(x["covered"] for x in records)),
        "bad_rate_10cm":sum(x["bad"] for x in records)/max(1,sum(x["covered"] for x in records)),
        "false_free_space_rate":sum(x["new_nearer"] for x in records)/max(1,sum(x["covered"] for x in records)),
        "apparent_clearing_rate":sum(x["old_nearer"] for x in records)/max(1,sum(x["covered"] for x in records)),
        "n_support_pixels":sum(x["covered"] for x in records),
        "n_target_pixels":sum(x["valid_target"] for x in records),
    }

def eval_frames(frames):
    results={m:[] for m in METHODS}
    paired=[]
    for j in range(16,len(frames),2):
        target=frames[j]
        gt=target[1]
        valid=np.isfinite(gt)&(gt>.3)&(gt<4.5)
        source=[warp_depth(frames[j-k],target) for k in OFFSETS]
        stack=np.stack(source)
        preds=predict(stack)
        baseline=frames[j-1][1]
        both=valid&np.isfinite(preds["recent"])&np.isfinite(preds["median"])
        if both.sum()>10:
            paired.append({
                "recent_mae":float(np.mean(abs(preds["recent"][both]-gt[both]))),
                "median_mae":float(np.mean(abs(preds["median"][both]-gt[both]))),
                "n":int(both.sum())})
        for name,pr in preds.items():
            mask=valid&np.isfinite(pr)
            err=pr[mask]-gt[mask]
            results[name].append({
                "frame":j,"covered":int(mask.sum()),"valid_target":int(valid.sum()),
                "absolute_error":float(abs(err).sum()),
                "bad":int(np.count_nonzero(abs(err)>.10)),
                "new_nearer":int(np.count_nonzero(err>.10)),
                "old_nearer":int(np.count_nonzero(err<-.10)),
                "same_pixel_baseline_mae":float(abs(baseline[mask]-gt[mask]).mean()) if mask.any() else None,
            })
    return results,paired

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",required=True);p.add_argument("--dataset",required=True)
    p.add_argument("--intrinsics",required=True);p.add_argument("--frames",type=int,default=100)
    p.add_argument("--out",default="results/real_rgbd")
    args=p.parse_args()
    frames=frames_with_poses(args.root,list(map(float,args.intrinsics.split(","))),args.dataset,args.frames)
    if len(frames)<35:raise RuntimeError("Not enough synced measured frames")
    observations,paired=eval_frames(frames)
    rng=np.random.default_rng(SEED)
    summary={}
    for m,rec in observations.items():
        if sum(x["covered"] for x in rec)<100:raise RuntimeError("Insufficient prediction overlap: "+m)
        agg=aggregate(rec,m)
        boot=[]
        for i in range(400):
            subset=[rec[k] for k in rng.integers(0,len(rec),len(rec))]
            boot.append(aggregate(subset,m))
        for metric in ("coverage","mae_m","bad_rate_10cm"):
            agg[metric+"_ci95"]=list(map(float,np.percentile([z[metric] for z in boot],[2.5,97.5])))
        summary[m]=agg
    common={
        "n_frames":len(paired),
        "n_pixels":sum(x["n"] for x in paired),
        "recent_mae":sum(x["recent_mae"]*x["n"] for x in paired)/max(1,sum(x["n"] for x in paired)),
        "median_mae":sum(x["median_mae"]*x["n"] for x in paired)/max(1,sum(x["n"] for x in paired)),
    }
    output={
        "experiment_id":"EXP-20261102-01","dataset":args.dataset,
        "real_frames":len(frames),"target_frames":len(observations["recent"]),
        "methods":summary,"common_support_comparison":common,
        "observation":"Target measured depth used for metric evaluation only; GT pose used in ray projection.",
        "interpretation":"Geometric inconsistency measures do not constitute semantic change labels.",
        "limitations":"GT pose oracle, projected pixels only; no inferred SLAM; frame-block bootstrap not independent scenes.",
    }
    dest=pathlib.Path(args.out)/args.dataset;dest.mkdir(parents=True,exist_ok=True)
    (dest/"ray_memory_evidence.json").write_text(json.dumps(output,indent=2))
    print("EXPERIMENT_RESULT",json.dumps(output,sort_keys=True))
    print("PASS real multiview memory",args.dataset,len(frames),len(observations["recent"]))
if __name__=="__main__":main()
