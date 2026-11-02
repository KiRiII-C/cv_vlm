#!/usr/bin/env python3
"""B1/B2 real full TUM: YOLOE masks -> shared RGBD 3D objects -> short tracklets.

Every original synchronized RGB-D frame constructs ONE shared camera cloud.
YOLOE is deliberately run only on KEYFRAMES; its inference cadence is
explicitly counted and non-keyframes get depth-only reprojection of prior
objects (NOT fake YOLO detections). No real physical-instance GT in TUM:
short-track IDs are hypotheses and cannot be called verified physical UUIDs.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json,time
from pathlib import Path
import cv2,numpy as np
from semantic_map.geometry import Intrinsics
from semantic_map.shared_rgbd_geometry import SharedRGBDFrame,reproject_object
from semantic_map.yoloe26_detector import YOLOE26Segmenter,class_list
from semantic_map.short_term_semantics_b import ClassTolerantTemporalAssociator
from semantic_map.temporal_tracker import Detection,TemporalTracker
from semantic_map.reid_features import hsv_spatial
from tum_full_replay_a4 import full_rgbd_pairs

K=Intrinsics(517.3,516.5,318.6,255.3,640,480,1/5000)

def stats(a):
    if not a:return {"n":0}
    a=np.asarray(a,dtype=float)
    return {"n":int(len(a)),"mean_ms":float(np.mean(a)),
            "p50_ms":float(np.percentile(a,50)),
            "p95_ms":float(np.percentile(a,95))}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",type=Path,default=Path("data/tum"))
    ap.add_argument("--poses",type=Path,default=Path("results/a6/optimized_full_poses.json"))
    ap.add_argument("--out",type=Path,default=Path("results/b1"))
    ap.add_argument("--weights",default="yoloe-26s-seg.pt")
    ap.add_argument("--stride",type=int,default=5)
    ap.add_argument("--device",default="cpu")
    args=ap.parse_args()
    if args.stride<1 or args.stride>10:raise ValueError("Detector stride must be [1,10]")
    args.out.mkdir(parents=True,exist_ok=True)
    pairs,manifest=full_rgbd_pairs(args.data)
    src=json.loads(args.poses.read_text())
    if src.get("GT_used_as_input") is not False:
        raise RuntimeError("GT-derived camera poses forbidden in B1")
    pose={int(x["index"]):np.asarray(x["T_map_cam"],dtype=float) for x in src["poses"]}
    configured=class_list("config/yoloe26_classes.json")
    detector=YOLOE26Segmenter(args.weights,configured)
    tracker=ClassTolerantTemporalAssociator(
       TemporalTracker(high_thresh=.27,low_thresh=.10,match_score=.57,
                       max_lost=40,min_hits=2))
    last_observation={}  # short-term only; no permanent physical identity
    measures={"geometry_ms":[],"detection_ms":[],"association_ms":[],"reproject_ms":[]}
    counts=Counter()
    objects=Counter()
    track_status=Counter()
    reprojection_iou=[];reprojection_depth=[]
    rows=[]
    for i,(stamp,rgb_path,depth_path,depth_stamp) in enumerate(pairs):
        t0=time.perf_counter()
        depth=cv2.imread(str(depth_path),cv2.IMREAD_UNCHANGED)
        if depth is None:raise RuntimeError(f"Failed REAL depth at frame {i}")
        f=SharedRGBDFrame(depth,K,pose.get(i))
        measures["geometry_ms"].append((time.perf_counter()-t0)*1000)
        counts["rgbd_frames_processed"]+=1
        record={"idx":i,"timestamp":stamp,"has_measured_pose":i in pose,
                "detection_run":False,"yoled_masks":0,"object3d":0,
                "associated":0,"propagated_observed":0}
        # EVERY frame: predict prior object footprints by reprojecting REAL 3D
        # points against this frame's REAL depth. This is not a new mask!
        if i in pose and last_observation:
            for last in last_observation.values():
                st=time.perf_counter()
                ev=reproject_object(last,f)
                measures["reproject_ms"].append((time.perf_counter()-st)*1000)
                if ev.geometrically_visible_pixels>=8:
                    record["propagated_observed"]+=1
        if i in pose and (i==1 or i%args.stride==1):
            rgb=cv2.imread(str(rgb_path),cv2.IMREAD_COLOR)
            if rgb is None or rgb.shape[:2]!=depth.shape:
                raise RuntimeError(f"RGB/Depth mismatch on frame {i}")
            begin=time.perf_counter()
            res=detector.predict(rgb,conf=.10,device=args.device,imgsz=640)
            measures["detection_ms"].append((time.perf_counter()-begin)*1000)
            counts["detector_keyframes"]+=1
            record["detection_run"]=True
            dets=[];obs_by_det={};masks_by_det={}
            if res.masks is not None and res.boxes is not None:
                masks=res.masks.data.cpu().numpy()
                classes=res.boxes.cls.int().cpu().tolist()
                scores=res.boxes.conf.cpu().tolist()
                if len(masks)!=len(classes) or len(classes)!=len(scores):
                    raise RuntimeError("Detector mask/class/bbox misalignment")
                record["yoled_masks"]=len(masks)
                counts["YOLOE_masks_raw"]+=len(masks)
                for j,(raw,category,conf) in enumerate(zip(masks,classes,scores)):
                    label=detector.class_name(category)
                    mask=raw>.5
                    if mask.shape!=depth.shape:
                        mask=cv2.resize(mask.astype(np.uint8),
                          (K.width,K.height),interpolation=cv2.INTER_NEAREST).astype(bool)
                    if not mask.any():continue
                    # remove depth background contamination inside mask; no GT.
                    valid_depth=depth[mask].astype(np.float64)*K.depth_unit_m
                    valid_depth=valid_depth[(valid_depth>.20)&(valid_depth<6.)]
                    if len(valid_depth)<40:continue
                    median=float(np.median(valid_depth))
                    depth_gate=np.abs(f.depth_m-median)<max(.18,.16*median)
                    mask=mask & depth_gate & f.valid
                    key=f"det_{j}"
                    o=f.object_observation(mask,detection_id=key,label=label,
                         confidence=float(conf),timestamp=stamp,
                         frame_index=i,min_support=50,max_points=2500)
                    if o is None:continue
                    y,x=np.where(mask)
                    feature=hsv_spatial(rgb,mask)
                    if feature is None:continue
                    d=Detection(key,label,np.asarray([x.min(),y.min(),x.max()+1,y.max()+1]),
                            o.center_map,float(conf),np.asarray(feature),mask)
                    dets.append(d)
                    obs_by_det[key]=o
                    masks_by_det[key]=mask
                    counts["3d_observations"]+=1
                    objects[label]+=1
                record["object3d"]=len(dets)
            st=time.perf_counter()
            associations=tracker.update(i,dets)
            measures["association_ms"].append((time.perf_counter()-st)*1000)
            seen=set()
            for assignment in associations:
                track_status[assignment.status]+=1
                if assignment.short_id is None:continue
                if assignment.short_id in seen:raise RuntimeError("Two detections assigned to one short track")
                seen.add(assignment.short_id)
                record["associated"]+=1
                new=obs_by_det[assignment.detection_id]
                old=last_observation.get(assignment.short_id)
                if old is not None:
                    st=time.perf_counter()
                    e=reproject_object(old,f,target_mask=masks_by_det[assignment.detection_id])
                    measures["reproject_ms"].append((time.perf_counter()-st)*1000)
                    if e.projected_mask_iou is not None:
                        reprojection_iou.append(e.projected_mask_iou)
                    if e.depth_consistency_ratio is not None:
                        reprojection_depth.append(e.depth_consistency_ratio)
                last_observation[assignment.short_id]=new
            for tid in list(last_observation):
                if tid not in tracker.tracker.tracks:last_observation.pop(tid)
        elif i not in pose:
            counts["frames_without_pose"]+=1
        rows.append(record)
        if i%50==0:
            print("B1_FRAME",i,"/",len(pairs),"detector_frames",counts["detector_keyframes"],flush=True)
    if len(rows)!=len(pairs) or counts["rgbd_frames_processed"]!=len(pairs):
        raise RuntimeError("Not all original RGB-D sensor frames processed")
    report={"experiment":"B1_YOLOE_SHARED_RGBD_OBJECT3D_SHORT_TERM",
       "source_dataset":"TUM_freiburg1_desk",
       "source_pose":"A6 full optimized RTAB-Map /odom + graph interpolation",
       "ground_truth_used_for_pose":False,
       "YOLOE_weights":args.weights,"device":args.device,
       "class_prompts":configured,
       "inference_every_nth_frame":args.stride,
       "keyframes_only_detector":True,
       "all_depth_frames_geometry_processed":True,
       "all_depth_frames_detected":False,
       "GT_physical_instance_ID_available":False,
       "physical_ID_accuracy":"NOT_EVALUATED",
       "persistent_UUID_accuracy":"NOT_EVALUATED",
       "counts":dict(counts),
       "detected_classes":dict(objects),
       "temporal_statuses":dict(track_status),
       "unique_short_tracklets_created":tracker.tracker._next_id,
       "timings":{k:stats(v) for k,v in measures.items()},
       "reprojection_mask_iou_median":
         float(np.median(reprojection_iou)) if reprojection_iou else None,
       "reprojection_depth_consistency_median":
         float(np.median(reprojection_depth)) if reprojection_depth else None,
       "limitations":"This is a detector keyframe policy, not full 595-frame YOLOE inference. "
          "A short tracklet is not a persistent physical ID, and TUM lacks physical instance GT."}
    (args.out/"b1_full_tum_report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    with (args.out/"b1_frame_coverage.jsonl").open("w") as out:
        for row in rows:out.write(json.dumps(row)+"\n")
    print("B1_RESULT",json.dumps(report,ensure_ascii=False),flush=True)

if __name__=="__main__":main()
