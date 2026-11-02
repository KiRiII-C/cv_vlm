#!/usr/bin/env python3
"""A8 independent query: assess SPARSE accepted map poses without a common world GT.

Evaluate metric relative pose errors between accepted localization events
within each TUM desk2 session. This is invariant to unknown fixed transform
between desk and desk2 GT worlds. Query GT is never read before SLAM finishes.

In addition, SE3-register first half of accepted poses to the query GT system
and evaluate second half as temporal holdout; gauges cannot be interpreted as
absolute map accuracy.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
def parse(path):
    rows=[]
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts=line.split()
        rows.append((float(parts[0]),parts[1:]))
    return rows

MODES=("first_node","saved_last","wrong_8m")

def matrix(p,q):
    T=np.eye(4)
    q=np.asarray(q,dtype=float)
    if not np.all(np.isfinite(q)) or not .95<np.linalg.norm(q)<1.05:
        raise ValueError("Invalid quaternion")
    T[:3,:3]=Rotation.from_quat(q/np.linalg.norm(q)).as_matrix()
    T[:3,3]=np.asarray(p,dtype=float)
    return T

def se3_alignment(X,Y):
    a=X.mean(0);b=Y.mean(0)
    U,_,Vt=np.linalg.svd((X-a).T@(Y-b))
    R=Vt.T@np.diag([1.,1.,np.sign(np.linalg.det(Vt.T@U.T))])@U.T
    t=b-R@a
    return R,t

def stats(values):
    a=np.asarray(values,dtype=float)
    if len(a)==0:return {"n":0}
    return {"n":len(a),
        "median":float(np.median(a)),"rmse":float(np.sqrt(np.mean(a*a))),
        "p95":float(np.percentile(a,95)),"max":float(a.max())}

def evaluate(root:Path,gt_path:Path):
    true=parse(gt_path)
    if not true or len(true)<500:
        raise RuntimeError("Missing REAL independent TUM groundtruth sequence")
    gt_t=np.array([x[0] for x in true],dtype=float)
    reports={}
    for mode in MODES:
        folder=root/mode
        p=json.loads((folder/"full_desk2.publish.json").read_text())
        stamp_to_t={int(x["stamp_ns"]):float(x["source_tum_timestamp"]) for x in p}
        frames_by_stamp={int(x["stamp_ns"]):int(x["idx"]) for x in p}
        observed=[json.loads(row) for row in (folder/"localization_pose.jsonl").read_text().splitlines()
            if row.strip()]
        values=[]
        for item in observed:
            stamp=int(item["stamp_ns"])
            if stamp not in stamp_to_t:
                raise RuntimeError(f"Accepted localization pose without query RGB-D frame: {mode}")
            t=stamp_to_t[stamp]
            idx=int(np.argmin(np.abs(gt_t-t)))
            if abs(float(gt_t[idx])-t)>.03:continue
            pose=true[idx][1]
            E=matrix(item["xyz"],item["quaternion"])
            G=matrix(pose[:3],pose[3:7])
            values.append((frames_by_stamp[stamp],t,E,G))
        values.sort(key=lambda x:x[0])
        if len(set(x[0] for x in values))!=len(values):
            raise RuntimeError("Duplicate accepted localization pose for same query frame")
        trans_err=[];rot_err=[];dist_err=[];gaps=[]
        for left,right in zip(values,values[1:]):
            frame_gap=right[0]-left[0]
            if frame_gap<=0:raise RuntimeError("Pose time order corrupt")
            a=np.linalg.inv(left[2])@right[2]
            g=np.linalg.inv(left[3])@right[3]
            error=np.linalg.inv(g)@a
            trans_err.append(float(np.linalg.norm(error[:3,3])))
            rot_err.append(float(np.degrees(
                np.linalg.norm(Rotation.from_matrix(error[:3,:3]).as_rotvec()))))
            dist_err.append(abs(float(np.linalg.norm(a[:3,3])-np.linalg.norm(g[:3,3]))))
            gaps.append(frame_gap)
        holdout=None
        if len(values)>=8:
            k=len(values)//2
            X=np.asarray([v[2][:3,3] for v in values])
            Y=np.asarray([v[3][:3,3] for v in values])
            R,t=se3_alignment(X[:k],Y[:k])
            errors=np.linalg.norm(X[k:]@R.T+t-Y[k:],axis=1)
            holdout={"train_accepted_poses":k,"test_accepted_poses":len(values)-k,
                     "translation_errors_m":stats(errors),
                     "interpretation":"SE3_GAUGE_FIT_TO_FIRST_HALF_NOT_ABSOLUTE_MAP_GT"}
        reports[mode]={
            "accepted_pose_messages":len(observed),
            "accepted_poses_with_close_GT":len(values),
            "accepted_pose_indices_with_GT":[x[0] for x in values],
            "successive_accepted_pose_relative_translation_error_m":stats(trans_err),
            "successive_accepted_pose_relative_rotation_error_deg":stats(rot_err),
            "successive_accepted_pose_translation_distance_error_m":stats(dist_err),
            "frame_gaps_between_accepted_poses":stats(gaps),
            "temporal_holdout_alignment":holdout,
            "no_common_GT_world_frame_assumed":True}
    result={"experiment":"A8_POST_HOC_SPARSE_LOCALIZATION_GEOMETRY",
            "source":"TUM_freiburg1_desk2_GROUNDTRUTH_EVALUATION_ONLY",
            "GT_used_in_SLAM":False,
            "cross_sequence_absolute_map_pose_error":"NOT_EVALUATED",
            "relative_error_invariant_to_world_gauge":True,
            "arms":reports}
    target=root/"a8_sparse_metric_report.json"
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print("A8_SPARSE_GEOMETRY",json.dumps({
        k:{"matched":v["accepted_poses_with_close_GT"],
            "rpe_translation_rmse_m":v["successive_accepted_pose_relative_translation_error_m"].get("rmse"),
            "rpe_rotation_rmse_deg":v["successive_accepted_pose_relative_rotation_error_deg"].get("rmse"),
            "holdout_rmse_m":(v["temporal_holdout_alignment"] or {}).get(
                "translation_errors_m",{}).get("rmse")}
        for k,v in reports.items()}),flush=True)
    return result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,default=Path("results/a8"))
    ap.add_argument("--gt",type=Path,default=Path(
        "data/tum_a7/rgbd_dataset_freiburg1_desk2/groundtruth.txt"))
    args=ap.parse_args()
    evaluate(args.root,args.gt)

if __name__=="__main__":main()
