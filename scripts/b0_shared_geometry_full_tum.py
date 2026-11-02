#!/usr/bin/env python3
"""B0 real full-TUM benchmark: one shared organized cloud per RGB-D frame.

Comparison mask is a fixed DEPTH-DEFINED TEST REGION (not YOLO / physical GT).
The purpose here is only memory/geometry parity and latency, not segmentation.
All 595 original paired sensor observations are processed; pose-invalid frames
remain counted but are not mapped or compared in global coordinates.
"""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import cv2
import numpy as np
from semantic_map.geometry import Intrinsics,masked_pointcloud
from semantic_map.shared_rgbd_geometry import SharedRGBDFrame,reproject_object
from tum_full_replay_a4 import full_rgbd_pairs

K=Intrinsics(517.3,516.5,318.6,255.3,640,480,1/5000)

def summary(seq):
    if not seq:return {"n":0}
    a=np.asarray(seq,dtype=float)
    return {"n":len(a),"mean_ms":float(np.mean(a)),"p50_ms":float(np.percentile(a,50)),
            "p95_ms":float(np.percentile(a,95)),"max_ms":float(np.max(a))}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",default="data/tum",type=Path)
    p.add_argument("--poses",default="results/a6/optimized_full_poses.json",type=Path)
    p.add_argument("--output",default="results/b0",type=Path)
    args=p.parse_args()
    pairs,manifest=full_rgbd_pairs(args.data)
    d=json.loads(args.poses.read_text())
    assert d["GT_used_as_input"] is False
    poses={int(v["index"]):np.asarray(v["T_map_cam"],dtype=float) for v in d["poses"]}
    args.output.mkdir(parents=True,exist_ok=True)
    counter={"frames_total":0,"frames_with_measured_pose":0,
             "frames_without_pose":0,"observations_probe":0,"no_valid_depth_probe":0,
             "parity_failures":0,"pairs_reprojected":0}
    timings={"shared_backproject_ms":[],"legacy_mask_repeat_ms":[],
             "shared_mask_ms":[],"reproject_ms":[]}
    ratios=[];ious=[];probes=[]
    previous=None
    for i, (ts,rgbpath,depthpath,depthts) in enumerate(pairs):
        depth=cv2.imread(str(depthpath),cv2.IMREAD_UNCHANGED)
        if depth is None or depth.shape!=(480,640):raise RuntimeError(f"Depth read mismatch at {i}")
        # Fixed and clearly labeled probe from ACTUAL sensor depth (not instance segmentation).
        mask=(depth>0)&(depth<=int(2.4*5000))
        T=poses.get(i)
        before=time.perf_counter()
        frame=SharedRGBDFrame(depth,K,T)
        timings["shared_backproject_ms"].append((time.perf_counter()-before)*1000)
        counter["frames_total"]+=1
        if T is None:
            counter["frames_without_pose"]+=1
            previous=None
            continue
        counter["frames_with_measured_pose"]+=1
        before=time.perf_counter()
        pc=frame.sample_mask(mask,max_points=4000)
        timings["shared_mask_ms"].append((time.perf_counter()-before)*1000)
        before=time.perf_counter()
        old=masked_pointcloud(depth,mask,K,T,max_points=4000)
        timings["legacy_mask_repeat_ms"].append((time.perf_counter()-before)*1000)
        world=pc@T[:3,:3].T+T[:3,3]
        if old.shape!=world.shape or not np.allclose(old,world,atol=1e-5):
            counter["parity_failures"]+=1
            raise RuntimeError(f"Shared geometric output differs from legacy mask projection on frame {i}")
        obs=frame.object_observation(mask,detection_id="depth_probe",
              label="NONSEMANTIC_DEPTH_PROBE",confidence=1.,timestamp=ts,
              frame_index=i,min_support=25,max_points=4000)
        if obs is None:
            counter["no_valid_depth_probe"]+=1
        else:
            counter["observations_probe"]+=1
            if previous is not None:
                begin=time.perf_counter()
                evidence=reproject_object(previous,frame,target_mask=mask)
                timings["reproject_ms"].append((time.perf_counter()-begin)*1000)
                counter["pairs_reprojected"]+=1
                if evidence.depth_consistency_ratio is not None:ratios.append(evidence.depth_consistency_ratio)
                if evidence.projected_mask_iou is not None:ious.append(evidence.projected_mask_iou)
            previous=obs
        if i%100==0:print("B0_FRAME",i,"/",len(pairs),flush=True)
        if i<4:probes.append({"frame":i,"num_cam_points":int(frame.valid.sum()),
                            "probe_points":len(pc)})
    assert counter["frames_total"]==len(pairs)==manifest["synchronized_rgbd_pairs"]
    assert counter["frames_with_measured_pose"]==len(poses)
    assert counter["parity_failures"]==0
    assert counter["frames_without_pose"]==1, counter
    report={"experiment":"B0_SHARED_ORGANIZED_RGBD_REAL_TUM",
            "dataset":"TUM_freiburg1_desk_all_original_paired_depth",
            "sensor_frames":manifest,"pose_source":str(args.poses),
            "ground_truth_used_as_input":False,
            "YOLO_masks_used":False,
            "mask_source":"depth_defined_nonsemantic_probe_only",
            "physical_ID_evaluation":"NOT_TESTED",
            "counts":counter,
            "timings":{k:summary(v) for k,v in timings.items()},
            "temporal_depth_consistency_median":float(np.median(ratios)) if ratios else None,
            "temporal_projected_mask_iou_median":float(np.median(ious)) if ious else None,
            "evidence_samples":probes,
            "caution":"Legacy comparison reprojects each mask separately; shared frame cost amortizes across masks. CPU-only; no RTAB-Map native point cloud reuse claimed."}
    out=args.output/"b0_full_tum_report.json"
    out.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print("B0_RESULT",json.dumps(report,ensure_ascii=False),flush=True)

if __name__=="__main__":main()
