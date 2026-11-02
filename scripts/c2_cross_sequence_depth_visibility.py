#!/usr/bin/env python3
"""C2: independent TUM fr1 desk2 DEPTH evidence against earlier C1 object surfaces.

All 639 independent measured depth files are checked and accounted. Surface
verification uses only the map poses accepted by RTAB-Map A8 in localization
mode (25 sparse poses, NOT interpolated or fabricated for the other frames).
New YOLO detections on desk2 are NOT available; therefore this script MUST NOT
assert that an object moved, disappeared, or is the same physical instance.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
import cv2
from scipy.spatial import cKDTree
from semantic_map.c2_depth_visibility import evidence,summarize_per_hypothesis

def read_tum(path):
    out=[]
    for line in Path(path).read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            t,p=line.strip().split()[:2]
            out.append((float(t),p))
    return out

def read_ply(filename):
    with Path(filename).open() as f:
        props=[];size=None;header_length=0
        while True:
            s=f.readline()
            if not s:raise ValueError("Truncated PLY")
            header_length+=1
            if s.startswith("element vertex "):size=int(s.split()[-1])
            if s.startswith("property "):props.append(s.split()[-1])
            if s.strip()=="end_header":break
    if not {"x","y","z","instance_index"}.issubset(props):
        raise ValueError("C1 PLY lacks instance provenance")
    data=np.loadtxt(filename,skiprows=header_length,dtype=float)
    if data.shape[0]!=size:raise ValueError("C1 PLY vertex count mismatch")
    points=data[:,[props.index(k) for k in ("x","y","z")]]
    ids=data[:,props.index("instance_index")].astype(int)
    return points,ids

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,default=Path("data/tum_a7/rgbd_dataset_freiburg1_desk2"))
    ap.add_argument("--loc",type=Path,default=Path("results/a8/saved_last"))
    ap.add_argument("--surface",type=Path,default=Path("results/c1/hypothesis_surfaces.ply"))
    ap.add_argument("--instances",type=Path,default=Path("results/c1/hypothesis_surfaces.instances.json"))
    ap.add_argument("--out",type=Path,default=Path("results/c2"))
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    depth_list=read_tum(args.root/"depth.txt")
    rgb_list=read_tum(args.root/"rgb.txt")
    if len(depth_list)!=639 or len(rgb_list)!=640:
        raise RuntimeError(f"Expected whole independent TUM desk2, got {len(rgb_list)}/{len(depth_list)}")
    publish=json.loads((args.loc/"full_desk2.publish.json").read_text())
    if len(publish)!=639:raise RuntimeError("Missing query source frame publications")
    stamps={int(x["stamp_ns"]):x for x in publish}
    depth_ts=np.asarray([x[0] for x in depth_list])
    used=set();mapping={}
    for item in publish:
        stamp=int(item["stamp_ns"])
        query=float(item["source_depth_timestamp"])
        idx=int(np.argmin(abs(depth_ts-query)))
        if abs(depth_ts[idx]-query)>.0001 or idx in used:
            raise RuntimeError("TUM depth source time not uniquely matched")
        used.add(idx);mapping[stamp]=args.root/depth_list[idx][1]
    if len(used)!=639 or not all(x.is_file() for x in mapping.values()):
        raise RuntimeError("Incomplete or missing independent depth")
    accepted=[json.loads(s) for s in (args.loc/"localization_pose.jsonl").read_text().splitlines() if s.strip()]
    if not accepted:raise RuntimeError("No accepted RTAB-Map poses")
    c1=json.loads(args.instances.read_text())
    pts,ids=read_ply(args.surface)
    if len(pts)!=c1["vertices"] or len(np.unique(ids))!=c1["unique_ids"]:
        raise RuntimeError("C1 object hypothesis vertices mismatch")
    cloud={int(i):pts[ids==int(i)] for i in np.unique(ids)}
    if len(cloud)!=15:raise RuntimeError("Expected all C1 hypotheses")
    names={int(i):v["uuid"] for i,v in c1["instances"].items()}
    observation_count=Counter();per_uuid=defaultdict(list);audit=[]
    for row in accepted:
        ns=int(row["stamp_ns"])
        if ns not in mapping:raise RuntimeError("Accepted localization has no real query depth")
        depth=cv2.imread(str(mapping[ns]),cv2.IMREAD_UNCHANGED)
        if depth is None or depth.shape!=(480,640) or depth.dtype!=np.uint16:
            raise RuntimeError("TUM independent 16bit depth missing or invalid")
        obs=depth.astype(np.float32)/5000.0
        frame=int(stamps[ns]["idx"])
        observation_count["accepted_relocalization_frames"]+=1
        for instance,points in cloud.items():
            data=evidence(points,row["xyz"],row["quaternion"],obs,
                          pose_xyz_cov=row.get("cov_xyz"))
            data.update({"frame":frame,"original_tum_timestamp":stamps[ns]["source_tum_timestamp"],
                         "source_uuid_hypothesis":names[instance],
                         "instance_index":instance,
                         "accepted_map_pose_only":True})
            per_uuid[instance].append(data)
            audit.append(data)
        if observation_count["accepted_relocalization_frames"]%5==0:
            print("C2_FRAME",observation_count["accepted_relocalization_frames"],"from",len(accepted),flush=True)
    output=[]
    for instance,states in per_uuid.items():
        result=summarize_per_hypothesis(states)
        result.update({"uuid_hypothesis":names[instance],
                       "instance_index":instance,"accepted_pose_frames_checked":len(states),
                       "is_physical_UUID_verified":False,
                       "state_update_committed_to_C1_SQLite":False})
        output.append(result)
    status=Counter(x["decision"] for x in output)
    per_view=Counter(x["state"] for x in audit)
    report={"experiment":"C2_CROSS_RECORDING_TUM_DEPTH_CHECK_OF_C1_SURFACE",
        "source_map":"C1 hypotheses from B7 fr1_desk REAL RGBD",
        "independent_query":"TUM_freiburg1_desk2",
        "real_independent_RGB_frames":len(rgb_list),
        "real_independent_depth_frames_processed_for_source_validation":len(used),
        "accepted_relocalization_poses_evaluated":len(accepted),
        "frames_without_accepted_pose_not_used_for_3D_evidence":len(used)-len(accepted),
        "measured_C1_voxel_points":int(len(pts)),
        "hypothesis_count":len(cloud),
        "total_hypothesis_view_checks":len(audit),
        "per_view_evidence_states":dict(per_view),
        "per_hypothesis_decisions":dict(status),
        "visual_appearance_confirmed":"NOT_AVAILABLE_NO_QUERY_YOLOE",
        "physical_gt_verified":False,"moved_objects_established":False,
        "deleted_objects_established":False,
        "old_memory_mutated":False,
        "pose_uncertainty_inflation":"3sigma_max_linear_covariance",
        "limitations":"25 sparse accepted A8 poses only; unknown camera extrinsics/GT cross-session; "
           "no query instance segmentation; geometric depth evidence is not physical ID proof",
        "hypotheses":output}
    (args.out/"c2_cross_session_visibility.json").write_text(
      json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    with (args.out/"c2_frame_uuid_evidence.jsonl").open("w") as f:
        for x in audit:f.write(json.dumps(x)+"\n")
    print("C2_REAL_INDEPENDENT_DEPTH_RESULT",json.dumps({
       k:report[k] for k in ("real_independent_depth_frames_processed_for_source_validation",
         "accepted_relocalization_poses_evaluated","hypothesis_count",
         "per_view_evidence_states","per_hypothesis_decisions")},ensure_ascii=False),flush=True)

if __name__=="__main__":main()
