#!/usr/bin/env python3
"""A5 full-sequence study: do RTAB-Map's OdomInfo features predict true pose errors?

Evaluation-only: groundtruth.txt is loaded AFTER A4 sensor processing. No pose
GT enters SLAM, frame selection, filtering, registration or map construction.

Report paired per-frame SE3-aligned ATE and successive-frame relative
SE3 RPE (translation and rotation), association coverage, rank correlation,
AUROC of fixed untrained scores, and temporally BLOCK-bootstrapped intervals.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
from collections import Counter

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import spearmanr, rankdata
from semantic_map.tum_yolo_pipeline import parse


def match_trajectories(root: Path, gt_path: Path, max_gt_dt=.03):
    src=json.loads((root/"full_sequence.json").read_text())
    sent=json.loads((root/"full_sequence.publish.json").read_text())
    odom=json.loads((root/"full_sequence.odom.json").read_text())
    info=json.loads((root/"full_sequence.odominfo.json").read_text())
    gt=parse(gt_path)
    gt_t=np.asarray([p[0] for p in gt],dtype=float)
    if src["published_frames"]!=len(sent) or len(sent)!=src["synchronized_rgbd_pairs"]:
        raise RuntimeError("Frame coverage mismatch; do not infer a complete trajectory")
    if src["odominfo_ack_frames"]!=len(sent) or src["odominfo_ack_rate"]<1.:
        raise RuntimeError("Not every full-dataset frame has OdomInfo confirmation")
    if not gt or not np.all(np.diff(gt_t)>0):
        raise ValueError("Invalid GT trajectory")
    info_by={x["stamp_ns"]:x for x in info}
    odom_by={x["stamp_ns"]:x for x in odom}
    rows=[]
    no_gt=0
    for pub in sent:
        k=pub["stamp_ns"]
        if k not in info_by:
            raise RuntimeError(f"Missing OdomInfo for frame {pub['idx']}")
        i=int(np.argmin(np.abs(gt_t-pub["source_tum_timestamp"])))
        if abs(float(gt_t[i])-pub["source_tum_timestamp"])>max_gt_dt:
            no_gt+=1
            continue
        d=gt[i][1]
        G=np.eye(4,dtype=float)
        G[:3,:3]=Rotation.from_quat(np.asarray(d[3:7],dtype=float)).as_matrix()
        G[:3,3]=[float(a) for a in d[:3]]
        ii=info_by[k]
        oo=odom_by.get(k)
        if oo is None or not oo.get("numerically_valid",False):
            rows.append({"idx":pub["idx"],"t":pub["source_tum_timestamp"],
                 "gt_pose":G,"est_pose":None,"odominfo":ii,"odom":oo})
            continue
        E=np.eye(4,dtype=float)
        E[:3,:3]=Rotation.from_quat(oo["quaternion_xyzw"]).as_matrix()
        E[:3,3]=np.asarray(oo["xyz"],dtype=float)
        rows.append({"idx":pub["idx"],"t":pub["source_tum_timestamp"],
             "gt_pose":G,"est_pose":E,"odominfo":ii,"odom":oo})
    return rows,{"source_dataset":src["source_dataset"],
        "published_full_dataset":src["published_frames"],
        "diagnosed_frames":src["odominfo_ack_frames"],
        "gt_missing_by_30ms":no_gt,
        "ground_truth_used_by_SLAM":False,
        "ground_truth_source":str(gt_path)}


def rigid_alignment(X,Y):
    x0=X.mean(axis=0); y0=Y.mean(axis=0)
    U,_,Vt=np.linalg.svd((X-x0).T @ (Y-y0))
    R=Vt.T @ np.diag([1.,1.,np.sign(np.linalg.det(Vt.T@U.T))]) @ U.T
    t=y0-R@x0
    return R,t


def auc(labels,score):
    labels=np.asarray(labels,dtype=bool)
    scores=np.asarray(score,dtype=float)
    ok=np.isfinite(scores)
    labels=labels[ok];scores=scores[ok]
    npos=int(labels.sum()); nneg=int((~labels).sum())
    if not npos or not nneg or len(np.unique(scores))<2:return None
    ranks=rankdata(scores,method="average")
    return float((ranks[labels].sum()-npos*(npos+1)/2)/(npos*nneg))


def correlation(x,y):
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float)
    valid=np.isfinite(x)&np.isfinite(y)
    if sum(valid)<10 or len(np.unique(x[valid]))<2 or len(np.unique(y[valid]))<2:
        return None
    return float(spearmanr(x[valid],y[valid]).statistic)


def numeric_summary(values):
    a=np.asarray(values,dtype=float)
    a=a[np.isfinite(a)]
    if not len(a):return {"n":0}
    return {"n":int(len(a)),"mean":float(a.mean()),"median":float(np.median(a)),
            "p90":float(np.percentile(a,90)),"p95":float(np.percentile(a,95)),
            "max":float(a.max()),"rmse":float(np.sqrt(np.mean(a*a)))}


def block_bootstrap_auc(labels,scores,block=30,reps=300,seed=20261102):
    """Time-dependent samples are NOT iid. Block resampling gives an exploratory CI."""
    y=np.asarray(labels,dtype=bool)
    s=np.asarray(scores,dtype=float)
    if len(y)<2*block or y.sum()<5 or (~y).sum()<5:
        return None
    rng=np.random.default_rng(seed)
    values=[]
    count=int(np.ceil(len(y)/block))
    for _ in range(reps):
        starts=rng.integers(0,max(1,len(y)-block+1),size=count)
        ids=np.concatenate([np.arange(k,min(k+block,len(y))) for k in starts])[:len(y)]
        z=auc(y[ids],s[ids])
        if z is not None:
            values.append(z)
    if len(values)<max(40,reps//4):return None
    return {"p2_5":float(np.percentile(values,2.5)),
            "p97_5":float(np.percentile(values,97.5)),
            "replicates_valid":len(values),"block_frames":block,
            "interpretation":"exploratory_temporal_block_bootstrap"}


def analyse(rows,meta):
    usable=[r for r in rows if r["est_pose"] is not None]
    if len(usable)<50:
        raise RuntimeError(f"Too few estimated poses for A5: {len(usable)}")
    X=np.array([r["est_pose"][:3,3] for r in usable])
    Y=np.array([r["gt_pose"][:3,3] for r in usable])
    align_R,align_t=rigid_alignment(X,Y)
    ate=np.linalg.norm(X@align_R.T+align_t-Y,axis=1)
    metrics={}
    for j,r in enumerate(usable):
        info=r["odominfo"]
        cov=np.asarray(info["cov_xyz_diagonal"],dtype=float)
        metrics[r["idx"]]={
            "frame":int(r["idx"]),"time_s":float(r["t"]-rows[0]["t"]),
            "ate_m":float(ate[j]),"rpe_translation_m":None,
            "rpe_rotation_deg":None,
            "lost":bool(info["lost"]),
            "inliers":int(info["inliers"]),"matches":int(info["matches"]),
            "inlier_fraction":float(info["inliers"]/info["matches"]) if info["matches"]>0 else None,
            "features":int(info["features"]),
            "key_frame_added":bool(info["key_frame_added"]),
            "time_estimation_s":float(info["time_estimation_s"]),
            "sigma_xyz_m":float(np.sqrt(np.max(cov))) if np.isfinite(cov).all() and np.min(cov)>=0 else None,
            "numerically_valid":bool(r["odom"]["numerically_valid"]),
        }
    for r0,r1 in zip(usable,usable[1:]):
        if r1["idx"]!=r0["idx"]+1:
            continue
        G0=r0["gt_pose"];G1=r1["gt_pose"]
        E0=r0["est_pose"];E1=r1["est_pose"]
        delta_gt=np.linalg.inv(G0)@G1
        delta_est=np.linalg.inv(E0)@E1
        relative=np.linalg.inv(delta_gt)@delta_est
        v=metrics[r1["idx"]]
        v["rpe_translation_m"]=float(np.linalg.norm(relative[:3,3]))
        v["rpe_rotation_deg"]=float(np.degrees(
            np.linalg.norm(Rotation.from_matrix(relative[:3,:3]).as_rotvec())))
    values=list(metrics.values())
    fields={
        "low_inliers":lambda m:-m["inliers"],
        "low_matches":lambda m:-m["matches"],
        "low_inlier_fraction":lambda m:-m["inlier_fraction"] if m["inlier_fraction"] is not None else float("nan"),
        "large_odom_covariance":lambda m:m["sigma_xyz_m"] if m["sigma_xyz_m"] is not None else float("nan"),
        "slow_registration":lambda m:m["time_estimation_s"],
        "low_feature_count":lambda m:-m["features"],
        "rtabmap_lost_flag":lambda m:int(m["lost"]),
    }
    correlations={}
    for target,cutoff in [("ate_m",.10),("rpe_translation_m",.025),
                          ("rpe_rotation_deg",3.)]:
        subset=[m for m in values if m[target] is not None]
        y=np.array([m[target] for m in subset])
        labels=y>cutoff
        row={"error_threshold":cutoff,"positives":int(labels.sum()),
             "negatives":int((~labels).sum()),"total":len(labels),"predictors":{}}
        for name,fun in fields.items():
            score=np.asarray([fun(m) for m in subset],dtype=float)
            a=auc(labels,score)
            row["predictors"][name]={
                "spearman_rank_correlation_with_error":correlation(score,y),
                "bad_pose_auroc":a,
                "block_bootstrap_auc_ci_95":block_bootstrap_auc(labels,score) if a is not None else None,
                "auroc_undefined_reason":"constant_label_or_score" if a is None else None,
            }
        correlations[target]=row
    q80=float(np.percentile([m["inliers"] for m in values],80))
    confident_bad=[m for m in values if m["inliers"]>=q80 and m["ate_m"]>.10]
    top=sorted(values,key=lambda x:x["ate_m"],reverse=True)[:15]
    lost_frames=[r for r in rows if r["odominfo"]["lost"]]
    report={**meta,
        "experiment":"A5_ODEMINFO_GT_ERROR_PREDICTIVE_POWER",
        "number_of_groundtruth_matched_poses":len(values),
        "total_observed_frames_including_initialization":len(rows),
        "global_SE3_alignment":True,"scale_fixed":True,
        "ate_all":numeric_summary([m["ate_m"] for m in values]),
        "relative_step_translation_rpe":numeric_summary([m["rpe_translation_m"] for m in values
                                  if m["rpe_translation_m"] is not None]),
        "relative_step_rotation_rpe_deg":numeric_summary([m["rpe_rotation_deg"] for m in values
                                  if m["rpe_rotation_deg"] is not None]),
        "internal_lost_true":len(lost_frames),
        "rtabmap_lost_detected_bad_poses":int(sum(m["lost"] and m["ate_m"]>.10 for m in values)),
        "high_confidence_inlier_top_quintile_threshold":q80,
        "high_inliers_but_large_ATE_count":len(confident_bad),
        "high_inliers_but_large_ATE_fraction":len(confident_bad)/len(values),
        "fixed_untrained_predictive_scores":correlations,
        "worst_ATE_frames":[{k:m[k] for k in ("frame","time_s","ate_m","inliers","matches",
                                             "inlier_fraction","sigma_xyz_m")} for m in top],
        "warning":"One scene, one tracked trajectory; temporal dependence and GT alignment; "
                  "do not generalize predictive ability or interpret sigma as global map uncertainty",
        "training_occurred":False,
        "all_dataset_frames_included_in_analysis":True,
    }
    return report,values


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,default=Path("results/a4"))
    ap.add_argument("--gt",type=Path,
        default=Path("data/tum/rgbd_dataset_freiburg1_desk/groundtruth.txt"))
    ap.add_argument("--out",type=Path,default=Path("results/a5"))
    args=ap.parse_args()
    rows,meta=match_trajectories(args.root,args.gt)
    report,values=analyse(rows,meta)
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/"odometry_predictive_report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    cols=["frame","time_s","ate_m","rpe_translation_m","rpe_rotation_deg","lost",
          "inliers","matches","inlier_fraction","features","key_frame_added",
          "time_estimation_s","sigma_xyz_m","numerically_valid"]
    with (args.out/"per_frame_pose_error.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=cols)
        writer.writeheader();writer.writerows(values)
    mini={"frames":report["total_observed_frames_including_initialization"],
          "analyzed_poses":report["number_of_groundtruth_matched_poses"],
          "ate_rmse_m":report["ate_all"]["rmse"],
          "RPE_translation_rmse_m":report["relative_step_translation_rpe"]["rmse"],
          "RPE_rotation_rmse_deg":report["relative_step_rotation_rpe_deg"]["rmse"],
          "high_inlier_bad_ATE":report["high_inliers_but_large_ATE_count"],
          "ranked_predictors_ate":{k:v["bad_pose_auroc"] for k,v in
                            report["fixed_untrained_predictive_scores"]["ate_m"]["predictors"].items()}}
    print("A5_RESULTS",json.dumps(mini,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
