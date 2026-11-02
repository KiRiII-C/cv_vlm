"""EXP-20261102-06. Evidence-gated sparse memory and physical temporal forgetting.

Three real RGB-D recordings with externally provided GT poses, never synthetic.
Tests preserving most recent observations and filling *only* unobserved rays
with mutually agreeing historical projections. Reports error and coverage
separately, including newly filled pixel quality and risk-coverage frontier.
"""
from __future__ import annotations
import argparse,pathlib,json
import numpy as np
from multiview_evidence import frames_with_poses,warp_depth

SEED=20261102
OFFSETS=(1,2,3,4,6,8)
POLICIES=('last','union_median','decay_weighted','last_plus_unverified','last_plus_consistent','last_plus_strict')

def weighted_median(arr,weights):
    # first dim is temporal observation number; missing pixels have zero weight
    a=np.asarray(arr);v=np.isfinite(a)
    clean=np.where(v,a,np.inf)
    ordering=np.argsort(clean,axis=0)
    sorted_values=np.take_along_axis(clean,ordering,axis=0)
    w=np.broadcast_to(np.asarray(weights)[:,None,None],a.shape)
    ordered_weights=np.take_along_axis(np.where(v,w,0.),ordering,axis=0)
    total=ordered_weights.sum(axis=0)
    index=np.argmax(ordered_weights.cumsum(axis=0)>=total[None]/2,axis=0)
    estimated=np.take_along_axis(sorted_values,index[None],axis=0)[0]
    return np.where(total>0,estimated,np.nan)

def history_predict(frames,t):
    target=frames[t]
    old=np.stack([warp_depth(frames[t-k],target) for k in OFFSETS])
    finite=np.isfinite(old)
    count=finite.sum(axis=0)
    with np.errstate(invalid='ignore'):
        median=weighted_median(old,[1]*len(OFFSETS))
        decay=weighted_median(old,np.exp(-np.asarray(OFFSETS)/1.6))
        med_abs=np.abs(old-median)
        order=np.sort(np.where(np.isfinite(med_abs),med_abs,np.inf),axis=0)
        # Consensus of any two camera rays close to median: at least 2,
        # not counting multiple reused samples of the same frame.
        agreement=np.sum(np.isfinite(old)&(med_abs<=.06),axis=0)
        latest=old[0]
        missing=~np.isfinite(latest)
        safe=(count>=2)&(agreement>=2)
        strict=(count>=3)&(agreement>=3)
        return dict(last=latest,
            union_median=median,
            decay_weighted=decay,
            last_plus_unverified=np.where(missing,median,latest),
            last_plus_consistent=np.where(missing&safe,median,latest),
            last_plus_strict=np.where(missing&strict,median,latest))

def report(samples,name):
    valid=sum(s['total'] for s in samples)
    n=sum(s['observed'] for s in samples)
    errors=sum(s['ae_sum'] for s in samples)
    bad=sum(s['bad'] for s in samples)
    filled=sum(s['filled'] for s in samples)
    err_fill=sum(s['new_ae_sum'] for s in samples)
    bad_fill=sum(s['new_bad'] for s in samples)
    return dict(method=name,n_target=valid,n_observed=n,
        coverage=n/max(valid,1),mae_m=errors/max(n,1),
        bad_rate_10cm=bad/max(n,1),n_newly_filled=filled,
        new_fill_mae_m=err_fill/max(filled,1) if filled else None,
        new_fill_bad_rate=bad_fill/max(filled,1) if filled else None)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True);p.add_argument('--dataset',
                   choices=['tum_xyz','tum_desk','bonn_box'],required=True)
    p.add_argument('--out',default='results/real_rgbd/age_memory')
    p.add_argument('--n',type=int,default=100)
    args=p.parse_args()
    K=[542.822841,542.576870,315.593520,237.756098] if args.dataset=='bonn_box' else [517.3,516.5,318.6,255.3]
    frames=frames_with_poses(args.root,K,args.dataset,args.n)
    if len(frames)<35:raise RuntimeError('Not enough measured real RGB-D frames')
    records={k:[] for k in POLICIES}
    for t in range(16,len(frames),2):
        current=frames[t][1]
        valid=np.isfinite(current)&(current>.3)&(current<4.5)
        pred=history_predict(frames,t)
        only_latest=valid&np.isfinite(pred['last'])
        for key,estimate in pred.items():
            mask=valid&np.isfinite(estimate)
            new=mask&~only_latest
            diff=estimate-current
            records[key].append(dict(total=int(valid.sum()),observed=int(mask.sum()),
                ae_sum=float(np.abs(diff[mask]).sum()),
                bad=int(np.sum(np.abs(diff[mask])>.1)),filled=int(new.sum()),
                new_ae_sum=float(np.abs(diff[new]).sum()),
                new_bad=int(np.sum(np.abs(diff[new])>.1))))
    rng=np.random.default_rng(SEED)
    rows={}
    for key,rec in records.items():
        if sum(x['observed'] for x in rec)<100:raise RuntimeError('Insufficient predicted depth support')
        stats=report(rec,key);boot=[]
        for _ in range(250):
            indices=rng.integers(0,len(rec),size=len(rec))
            boot.append(report([rec[i] for i in indices],key))
        for col in ('coverage','mae_m','bad_rate_10cm'):
            stats[col+'_ci95']=list(map(float,np.quantile([x[col] for x in boot],[.025,.975])))
        rows[key]=stats
    output={'experiment':'EXP-20261102-06','dataset':args.dataset,
        'n_frames':len(frames),'targets':len(records['last']),'methods':rows,
        'equation':'Only previous physical depth observations and oracle camera poses projected to target view; target observed depth solely used after prediction for evaluation.',
        'hypothesis':'last_plus_consistent may improve coverage without replacing last reliable depth; compare false risk on added pixels and against unverified fill.',
        'limitations':'No object segmentation. Frame-block bootstrap does not include inter-room variation. Scene density, trajectory, dynamic content differ across datasets.'}
    dest=pathlib.Path(args.out);dest.mkdir(parents=True,exist_ok=True)
    (dest/(args.dataset+'.json')).write_text(json.dumps(output,indent=2))
    print('EXP06_RESULT',json.dumps({'dataset':args.dataset,'targets':output['targets'],
        'methods':{m:{k:round(v[k],6) if v[k] is not None else None for k in ('coverage','mae_m','bad_rate_10cm','n_newly_filled','new_fill_bad_rate')} for m,v in rows.items()}},sort_keys=True))
if __name__=='__main__':main()
