#!/usr/bin/env python3
"""B4 posthoc clustered bootstrap: paired mask association score ablation.

All input is immutable published YOLOE-26x B4 evaluation scores from real
100/100-frame Hypersim runs. No adjustment or retraining after seeing GT.
Resample SOURCE FRAME TRANSITIONS (not individual correlated candidate pairs)
in temporal 8-frame blocks. 95% CIs are descriptive, not independent-scene
generalization guarantees (only three scenes).
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
SCENES=("ai_002_001","ai_005_001","ai_017_001")

def load(root,scene,near):
    p=root/f"b4_{scene}"/"b4_pairwise_scores.jsonl"
    if not p.is_file():raise RuntimeError(f"B4 missing scene {scene}: {p}")
    items=[json.loads(s) for s in p.read_text().splitlines() if s.strip()]
    if near:items=[v for v in items if v["hard_negative_distance_le_0_8m"]]
    assert all(v["source_frame"]+1==v["target_frame"] for v in items)
    return items

def measure(rows):
    truth=np.array([int(r["same_physical_GT_scoring_only"]) for r in rows],dtype=int)
    if len(np.unique(truth))!=2:return None
    a=np.array([r["baseline_score"] for r in rows],dtype=float)
    b=np.array([r["joint_score"] for r in rows],dtype=float)
    return np.array([
      roc_auc_score(truth,a),roc_auc_score(truth,b),
      average_precision_score(truth,a),average_precision_score(truth,b)],dtype=float)

def sample_clusters(items,rng,block_length=8):
    # Keep all correlated pairs that share a source transition, rather than
    # treating 1000 pairs in one frame as 1000 independent observations.
    byframe={}
    for row in items:byframe.setdefault(int(row["source_frame"]),[]).append(row)
    keys=list(range(99))
    selected=[]
    while len(selected)<99:
        start=int(rng.integers(0,99))
        selected.extend(keys[(start+i)%99] for i in range(block_length))
    return [row for k in selected[:99] for row in byframe.get(k,[])]

def summarize(items,n_resamples,seed):
    base=measure(items)
    if base is None:
        return {"valid":False,"pairs":len(items),"reason":"single physical GT label"}
    rng=np.random.default_rng(seed)
    differences=[]
    for _ in range(n_resamples):
        output=measure(sample_clusters(items,rng))
        if output is not None:differences.append([output[1]-output[0],output[3]-output[2]])
    d=np.array(differences,dtype=float)
    report={"valid":True,"pairs":len(items),
        "positive_pairs":sum(int(x["same_physical_GT_scoring_only"]) for x in items),
        "bootstrap_block_length_frames":8,
        "bootstrap_valid_replicates":len(d),
        "baseline_roc_auc":float(base[0]),"joint_roc_auc":float(base[1]),
        "delta_roc_auc":float(base[1]-base[0]),
        "baseline_AP":float(base[2]),"joint_AP":float(base[3]),
        "delta_AP":float(base[3]-base[2])}
    if len(d)>=100:
        for idx,metric in enumerate(("delta_roc_auc","delta_AP")):
            report[metric+"_95pct_block_bootstrap_CI"]=[
                float(x) for x in np.quantile(d[:,idx],[.025,.975])]
            report[metric+"_bootstrap_fraction_positive"]=float(np.mean(d[:,idx]>0))
    else:report["warning"]="Too few valid paired block bootstrap replicates"
    return report

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,default=Path("results"))
    parser.add_argument("--out",type=Path,default=Path("results/b4_posthoc"))
    parser.add_argument("--n",type=int,default=750)
    args=parser.parse_args()
    if args.n<100:raise ValueError("At least 100 bootstrap samples required")
    output={}
    for near in (False,True):
        scope="near_0_8m" if near else "all_same_class"
        by_scene={s:load(args.root,s,near) for s in SCENES}
        output[scope]={s:summarize(v,args.n,2026+i) for i,(s,v) in enumerate(by_scene.items())}
        # For pooled bootstrap, resample each scene's transitions separately.
        pooled=sum(by_scene.values(),[])
        base=measure(pooled)
        if base is not None:
            rng=np.random.default_rng(4297)
            differences=[]
            for rep in range(args.n):
                drawn=[]
                for s in SCENES:drawn.extend(sample_clusters(by_scene[s],rng))
                got=measure(drawn)
                if got is not None:differences.append([got[1]-got[0],got[3]-got[2]])
            d=np.asarray(differences,dtype=float)
            output[scope]["pooled"]={
                "pairs":len(pooled),"positive_pairs":sum(int(i["same_physical_GT_scoring_only"]) for i in pooled),
                "delta_roc_auc":float(base[1]-base[0]),
                "delta_AP":float(base[3]-base[2]),
                "baseline_auc":float(base[0]),"joint_auc":float(base[1]),
                "baseline_AP":float(base[2]),"joint_AP":float(base[3]),
                "delta_auc_95pct_CI":np.quantile(d[:,0],[.025,.975]).tolist(),
                "delta_AP_95pct_CI":np.quantile(d[:,1],[.025,.975]).tolist(),
                "valid_bootstrap_runs":len(d)}
    result={"experiment":"B4_SAME_MASKS_PAIRED_BLOCK_BOOTSTRAP",
            "source":"three_real_all_100_Hypersim_YOLOE26x_physical_GT_pair_scores",
            "GT_used_for_ranking_or_training":False,
            "block_length_frames":8,"replicates_requested":args.n,"seed":2026,
            "bootstrap_clusters":"temporal transitions, not individual candidate pairs",
            "significance_scope":"ONLY these 3 scenes, dependent consecutive detections",
            "statistics":output}
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/"b4_bootstrap_report.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print("B4_BOOTSTRAP_POOLED",json.dumps({k:v["pooled"] for k,v in output.items()}),flush=True)

if __name__=="__main__":main()
