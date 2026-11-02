#!/usr/bin/env python3
"""C3: REAL independent query YOLOE-26x observations in saved A6/C1 map.

All 639 original desk2 frames are accounted, but semantic inference is only
performed at actual A8 accepted map-pose timestamps (not fabricated poses).
Cross-session geometric candidate association is NOT proof of physical ID.
No C1 persistent object state is changed and no absence/movement is asserted.
"""
from __future__ import annotations
import argparse,json,sqlite3,time
from pathlib import Path
from collections import Counter
import cv2,numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation
from semantic_map.geometry import Intrinsics
from semantic_map.shared_rgbd_geometry import SharedRGBDFrame
from semantic_map.yoloe26_detector import YOLOE26Segmenter,class_list
from c2_cross_sequence_depth_visibility import read_tum,read_ply

K=Intrinsics(517.3,516.5,318.6,255.3,640,480,1/5000)

def safe_pose(row):
    T=np.eye(4)
    T[:3,:3]=Rotation.from_quat(row["quaternion"]).as_matrix()
    T[:3,3]=row["xyz"]
    return T

def candidate_pair_scores(detections,landmarks):
    """No GT physical-ID input. Per-frame one-to-one with ambiguity abstention."""
    scores=np.full((len(detections),len(landmarks)),-1e8,dtype=float)
    for i,obs in enumerate(detections):
        for j,land in enumerate(landmarks):
            if obs["class"]!=land["class"]:continue
            d=float(np.linalg.norm(np.array(obs["center_map_m"])-land["center"]))
            if d>.45:continue
            support=0.
            pts=obs["xyz_sample"]
            if len(pts)>=15:
                near=land["tree"].query(pts,k=1,distance_upper_bound=.10,workers=1)[0]
                support=float(np.mean(np.isfinite(near)))
            # Pure geometric evidence. No invented learned appearance.
            scores[i,j]=.75*np.exp(-.5*(d/.23)**2)+.25*support
    selected=[{"status":"NO_SPATIAL_MATCH","old_uuid_candidate":None,
               "geometry_score":None} for _ in detections]
    ambiguous=set()
    for i in range(len(detections)):
        found=np.sort(scores[i,scores[i]>-1e7])
        if len(found)>1 and found[-1]-found[-2]<.08:
            ambiguous.add(i)
            selected[i]["status"]="AMBIGUOUS_NO_FORCED_UUID"
    if len(detections) and landmarks:
        rows=[i for i in range(len(detections)) if i not in ambiguous]
        if rows:
            cost=-scores[rows]
            rr,cc=linear_sum_assignment(cost)
            for ri,ci in zip(rr,cc):
                i=rows[ri]
                if scores[i,ci]<.48 or scores[i,ci]<-1e7:continue
                selected[i]={"status":"GEOMETRIC_MATCH_CANDIDATE_NOT_PHYSICAL_ID",
                    "old_uuid_candidate":landmarks[ci]["uuid"],
                    "geometry_score":float(scores[i,ci])}
    return selected

def get_candidate_landmarks(sqlite_path,ply_path,instance_meta):
    metadata=json.loads(Path(instance_meta).read_text())
    xyz,indices=read_ply(ply_path)
    with sqlite3.connect(f"file:{sqlite_path}?mode=ro",uri=True) as conn:
        saved={r[0]:{"class":r[1],"center":np.asarray(json.loads(r[2]),dtype=float)}
             for r in conn.execute("SELECT object_id,label,center FROM objects WHERE state='confirmed'")}
    if len(saved)!=15 or len(metadata["instances"])!=15:
        raise RuntimeError("Unexpected C1 candidate record count")
    all_old=[]
    for idx,source in metadata["instances"].items():
        uid=source["uuid"]
        if uid not in saved:raise RuntimeError("C1 PLY object not in SQLite")
        points=xyz[indices==int(idx)]
        if len(points)<20:raise RuntimeError("C1 missing UUID geometry")
        all_old.append({**saved[uid],"uuid":uid,"tree":cKDTree(points)})
    return all_old

def closest_record(t,records,tolerance=.0001):
    times=np.asarray([v[0] for v in records])
    idx=int(np.argmin(abs(times-t)))
    if abs(times[idx]-t)>tolerance:
        raise RuntimeError("No original TUM timestamp for accepted localization")
    return records[idx][1]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,default=Path("data/tum_a7/rgbd_dataset_freiburg1_desk2"))
    ap.add_argument("--a8",type=Path,default=Path("results/a8/saved_last"))
    ap.add_argument("--c1",type=Path,default=Path("results/c1"))
    ap.add_argument("--out",type=Path,default=Path("results/c3"))
    ap.add_argument("--weights",default="yoloe-26x-seg.pt")
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    rgbfiles=read_tum(args.root/"rgb.txt")
    depthfiles=read_tum(args.root/"depth.txt")
    if len(rgbfiles)!=640 or len(depthfiles)!=639:
        raise RuntimeError("Independent desk2 must be complete")
    pub=json.loads((args.a8/"full_desk2.publish.json").read_text())
    accepted=[json.loads(row) for row in (args.a8/"localization_pose.jsonl").read_text().splitlines() if row.strip()]
    if len(pub)!=639 or len(accepted)<20:
        raise RuntimeError("Missing complete A8 source replay or accepted map poses")
    source={int(p["stamp_ns"]):p for p in pub}
    old=get_candidate_landmarks(args.c1/"object_memory_with_surfaces.sqlite",
        args.c1/"hypothesis_surfaces.ply",args.c1/"hypothesis_surfaces.instances.json")
    det=YOLOE26Segmenter(args.weights,class_list("config/yoloe26_classes.json"))
    metrics=Counter();samples=[];new_records=[];modeled_old=set()
    for row in accepted:
        stamp=int(row["stamp_ns"])
        if stamp not in source:raise RuntimeError("Accepted map pose has no source frame")
        p=source[stamp]
        color=cv2.imread(str(args.root/closest_record(float(p["source_tum_timestamp"]),rgbfiles)),
                         cv2.IMREAD_COLOR)
        depth=cv2.imread(str(args.root/closest_record(float(p["source_depth_timestamp"]),depthfiles)),
                         cv2.IMREAD_UNCHANGED)
        if color is None or depth is None or color.shape[:2]!=(480,640) or depth.shape!=(480,640):
            raise RuntimeError("Missing RGB/Depth on accepted real frame")
        rgbd=SharedRGBDFrame(depth,K,safe_pose(row))
        result=det.predict(color,conf=.10,device="cpu",imgsz=640)
        detections=[]
        if result.masks is not None and result.boxes is not None:
            mask_data=result.masks.data.cpu().numpy()
            cl=result.boxes.cls.int().cpu().tolist()
            conf=result.boxes.conf.cpu().tolist()
            if not len(mask_data)==len(cl)==len(conf):
                raise RuntimeError("YOLO mask/box mismatch")
            for j,(raw,cls,confidence) in enumerate(zip(mask_data,cl,conf)):
                metrics["raw_masks"]+=1
                if confidence<.40:continue
                mask=raw>.5
                if mask.shape!=(480,640):
                    mask=cv2.resize(mask.astype(np.uint8),(640,480),
                                    interpolation=cv2.INTER_NEAREST).astype(bool)
                if mask.sum()<60:continue
                dvals=rgbd.depth_m[mask&rgbd.valid]
                if len(dvals)<60:continue
                med=float(np.median(dvals))
                mask=mask&rgbd.valid&(abs(rgbd.depth_m-med)<max(.18,.16*med))
                name=det.class_name(cls)
                ob=rgbd.object_observation(mask,detection_id=f"q_{j}",label=name,
                    confidence=float(confidence),timestamp=float(p["source_tum_timestamp"]),
                    frame_index=int(p["idx"]),min_support=60,max_points=3000)
                if ob is None:continue
                world=ob.points_camera@ob.T_map_camera[:3,:3].T+ob.T_map_camera[:3,3]
                detections.append({
                    "frame_index":int(p["idx"]),"class":name,"confidence":float(confidence),
                    "center_map_m":ob.center_map.tolist(),
                    "xyz_sample":world,
                    "point_count":len(world),
                    "source_detection_id":f"q_{j}"})
        decisions=candidate_pair_scores(detections,old)
        for d,result in zip(detections,decisions):
            if result["old_uuid_candidate"]:modeled_old.add(result["old_uuid_candidate"])
            item={key:value for key,value in d.items() if key!="xyz_sample"}
            item.update(result)
            new_records.append(item)
            samples.append((d["xyz_sample"],d["class"]))
            metrics[result["status"]]+=1
        metrics["accepted_pose_frames_with_YOLOE"]+=1
        metrics["valid_new_3D_observations"]+=len(detections)
        print("C3_REAL_FRAME",metrics["accepted_pose_frames_with_YOLOE"],
              "/",len(accepted),"new_observations",len(detections),flush=True)
    status={
        "experiment":"C3_CROSS_SESSION_REAL_YOLOE_REOBSERVATION",
        "map_session":"TUM_fr1_desk",
        "query_session":"TUM_fr1_desk2",
        "all_independent_RGBD_source_frames_accounted":len(pub),
        "query_frame_size":len(rgbfiles),
        "actual_YOLOE26x_inference_frames":len(accepted),
        "accepted_map_pose_only":True,
        "no_pose_interpolation_or_GT_for_association":True,
        "old_C1_UUID_hypotheses":len(old),
        "old_C1_hypotheses_with_any_candidate_match":len(modeled_old),
        "counts":dict(metrics),
        "no_new_physical_UUID_asserted":True,
        "no_existing_C1_SQLite_changes":True,
        "physical_ID_accuracy":"NOT_EVALUATED_NO_SHARED_INSTANCE_GT",
        "moved_objects":"NOT_INFERRED",
        "absence_when_unobserved":"UNKNOWN_NOT_ABSENT",
        "shortcoming":"Only 25 accepted A8 map poses have YOLOE; cannot infer missed objects on remaining 614 frames."}
    (args.out/"c3_report.json").write_text(json.dumps(status,indent=2,ensure_ascii=False)+"\n")
    with (args.out/"c3_independent_object_candidates.jsonl").open("w") as f:
        for row in new_records:f.write(json.dumps(row,ensure_ascii=False)+"\n")
    if samples:
        xyz=np.concatenate([x[0] for x in samples]).astype(np.float32)
        labels=np.concatenate([np.full(len(x[0]),x[1],dtype="U40") for x in samples])
    else:xyz=np.zeros((0,3),np.float32);labels=np.zeros(0,dtype="U40")
    np.savez_compressed(args.out/"c3_independent_3d_masks.npz",xyz=xyz,labels=labels)
    print("C3_REAL_CROSS_SEQUENCE",json.dumps(status,ensure_ascii=False),flush=True)

if __name__=="__main__":main()
