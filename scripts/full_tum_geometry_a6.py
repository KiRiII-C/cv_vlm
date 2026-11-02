#!/usr/bin/env python3
"""A6 full-sequence RTAB-Map graph correction and geometry evidence.

ALL available uniquely matched RGB-D pairs are accounted for. RTAB-Map's
measured /odom (not TUM GT) supplies poses for ordinary frames. Optimized
RTAB-Map keyframes anchor a continuous map<-odom correction; interpolate
corrections between anchors rather than silently replacing all in-between
poses with their nearest keyframe. Exclude (but count) odometry-invalid frames.
Reference ground-truth is read AFTER both SLAM-derived pose sets are frozen.
A TSDF made with GT poses and the SAME RGB-D depth is a pose reference map,
not an independently scanned 3D ground truth.
"""
from __future__ import annotations
import argparse, json, math, sqlite3
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from scipy.spatial import cKDTree

from semantic_map.offline_mapping import validate_pose
from semantic_map.tum_yolo_pipeline import parse, download_scene
from tum_full_replay_a4 import full_rgbd_pairs


def mat(xyz, quat):
    xyz=np.asarray(xyz,dtype=float);q=np.asarray(quat,dtype=float)
    if xyz.shape!=(3,) or q.shape!=(4,) or not np.isfinite(xyz).all() or not np.isfinite(q).all():
        raise ValueError("Malformed pose")
    n=np.linalg.norm(q)
    if n<.95 or n>1.05:raise ValueError("Invalid quaternion norm")
    T=np.eye(4)
    T[:3,:3]=Rotation.from_quat(q/n).as_matrix()
    T[:3,3]=xyz
    validate_pose(T)
    return T


def aligned(X,Y):
    X=np.asarray(X,dtype=float);Y=np.asarray(Y,dtype=float)
    if X.shape!=Y.shape or X.ndim!=2 or X.shape[1]!=3 or len(X)<5:
        raise ValueError("ATE requires >=5 common measured poses")
    a=X.mean(axis=0);b=Y.mean(axis=0)
    U,_,Vt=np.linalg.svd((X-a).T@(Y-b))
    R=Vt.T@np.diag([1.,1.,np.sign(np.linalg.det(U@Vt))])@U.T
    t=b-R@a
    e=np.linalg.norm(X@R.T+t-Y,axis=1)
    return {"rmse_m":float(np.sqrt(np.mean(e**2))),
            "median_m":float(np.median(e)),
            "p95_m":float(np.percentile(e,95)),
            "max_m":float(np.max(e)),
            "n":int(len(e))},R,t


def prepare(a4, graph_file, database, data):
    report=json.loads((a4/"full_sequence.json").read_text())
    published=json.loads((a4/"full_sequence.publish.json").read_text())
    odometry=json.loads((a4/"full_sequence.odom.json").read_text())
    graph=json.loads(graph_file.read_text())
    pairs,manifest=full_rgbd_pairs(data)
    if (not report.get("GT_used_as_input") is False or
        report["synchronized_rgbd_pairs"]!=len(pairs) or
        len(published)!=len(pairs) or
        report["odominfo_ack_frames"]!=len(pairs) or
        manifest["frame_stride"]!=1 or manifest["max_frames"] is not None):
        raise RuntimeError("A6 refuses incomplete/subsampled source A4")
    for i,(rec,pair) in enumerate(zip(published,pairs)):
        if rec["idx"]!=i or abs(rec["source_tum_timestamp"]-pair[0])>1e-5:
            raise RuntimeError(f"Replay is not on all corresponding real RGB-D frames: {i}")
    by_odom={int(o["stamp_ns"]):o for o in odometry if o["numerically_valid"]}
    rows=[]
    for item,pair in zip(published,pairs):
        o=by_odom.get(int(item["stamp_ns"]))
        rows.append({
            "index":int(item["idx"]), "source_tum_timestamp":float(pair[0]),
            "ros_stamp_sec":float(item["stamp_ns"])/1e9,
            "rgb_path":pair[1],"depth_path":pair[2],
            "O":mat(o["xyz"],o["quaternion_xyzw"]) if o else None,
        })
    if sum(row["O"] is not None for row in rows)<.98*len(rows):
        raise RuntimeError("A6 requires >=98% independently measured input odometry")
    if graph.get("GT_used_as_input") is not False or graph.get("source")!="live_rtabmap_ros_GetMap":
        raise ValueError("Graph origin not verifiable")
    g=graph["optimized"]
    if not g["poses"] or len(g["poses"])<12:
        raise RuntimeError("Too few optimized keyframe poses")
    with sqlite3.connect(database) as conn:
        nodes={int(x[0]):(float(x[1]),int(x[2])) for x in
               conn.execute("SELECT id,stamp,weight FROM Node WHERE id>0")}
    stamp=np.asarray([r["ros_stamp_sec"] for r in rows],dtype=float)
    if np.any(np.diff(stamp)<=0):
        raise ValueError("Replay ROS stamps not strictly increasing")
    used=set()
    anchors=[]
    for node_id,p in g["poses"].items():
        node_id=int(node_id)
        if node_id not in nodes or nodes[node_id][1]<0:continue
        at,weight=nodes[node_id]
        nearest=int(np.argmin(np.abs(stamp-at)))
        if abs(float(stamp[nearest])-at)>.05:continue
        if nearest in used:
            raise RuntimeError("Duplicate optimized graph keyframe assignment")
        if rows[nearest]["O"] is None:continue
        used.add(nearest)
        G=mat(p["xyz"],p["xyzw"])
        A=G@np.linalg.inv(rows[nearest]["O"])
        validate_pose(A)
        anchors.append((nearest,node_id,A))
    anchors.sort(key=lambda x:x[0])
    if len(anchors)<12:
        raise RuntimeError(f"Too few matched and measured graph anchors: {len(anchors)}")
    # Detect suspicious discontinuous corrections; keep evidence, do not hide.
    translations=[float(np.linalg.norm(B[:3,3]-A[:3,3])) for (_,_,A),(_,_,B) in zip(anchors,anchors[1:])]
    rotations=[float(np.degrees(np.linalg.norm(Rotation.from_matrix(
                 A[:3,:3].T@B[:3,:3]).as_rotvec()))) for (_,_,A),(_,_,B) in zip(anchors,anchors[1:])]
    first=anchors[0][2]
    raw=[];opt=[];extrapolated=0
    keyindices=np.array([x[0] for x in anchors],dtype=int)
    for row in rows:
        if row["O"] is None:continue
        idx=row["index"]
        # Raw baseline has ONE fixed global gauge; graph version uses local corrections.
        base=first@row["O"]
        if idx<=keyindices[0]:A=anchors[0][2];extrapolated+=int(idx<keyindices[0])
        elif idx>=keyindices[-1]:A=anchors[-1][2];extrapolated+=int(idx>keyindices[-1])
        else:
            k=int(np.searchsorted(keyindices,idx))
            ia,_,Ta=anchors[k-1];ib,_,Tb=anchors[k]
            alpha=(stamp[idx]-stamp[ia])/(stamp[ib]-stamp[ia])
            if alpha<0 or alpha>1:raise RuntimeError("Bad correction interpolation")
            interp=Slerp([0.,1.],Rotation.from_matrix(
                  np.stack([Ta[:3,:3],Tb[:3,:3]])))([alpha])[0]
            A=np.eye(4)
            A[:3,:3]=interp.as_matrix()
            A[:3,3]=(1-alpha)*Ta[:3,3]+alpha*Tb[:3,3]
        corrected=A@row["O"]
        validate_pose(base);validate_pose(corrected)
        raw.append({"index":idx,"timestamp":row["source_tum_timestamp"],
                    "T_map_cam":base.tolist()})
        opt.append({"index":idx,"timestamp":row["source_tum_timestamp"],
                    "T_map_cam":corrected.tolist()})
    if [x["index"] for x in raw]!=[x["index"] for x in opt]:
        raise AssertionError("Raw/optimized frame sets differ")
    return rows,raw,opt,{
       "total_dataset_rgbd_pairs":len(rows),
       "observed_odom_frames":len(raw),
       "odometry_invalid_or_missing":len(rows)-len(raw),
       "observed_fraction":len(raw)/len(rows),
       "optimized_graph_nodes":len(g["poses"]),
       "matched_optimized_graph_anchors":len(anchors),
       "first_anchor_frame":int(anchors[0][0]),
       "last_anchor_frame":int(anchors[-1][0]),
       "extrapolated_pose_count":extrapolated,
       "max_adjacent_anchor_translation_correction_m":max(translations,default=0.),
       "max_adjacent_anchor_rotation_correction_deg":max(rotations,default=0.),
       "pose_graph_keyframe_indices":[int(x[0]) for x in anchors],
       "pose_correction_method":"map_odom_SE3_keyframe_interpolated",
       "pose_graph_association_tolerance_s":.05,
       "no_groundtruth_fallback":True,
       "frame_stride":1,"GT_used_as_input":False}


def reference_pose_lookup(rows,source):
    # This routine is only invoked after saving raw and corrected pose files.
    root=download_scene(source)
    gt=parse(root/"groundtruth.txt")
    ts=np.asarray([x[0] for x in gt])
    matches={}
    missing=[]
    for row in rows:
        idx=int(np.argmin(np.abs(ts-row["source_tum_timestamp"])))
        if abs(ts[idx]-row["source_tum_timestamp"])>.03:
            missing.append(row["index"])
            continue
        g=gt[idx][1]
        matches[row["index"]]=mat(g[:3],g[3:7])
    return matches,missing


def mesh_vs_reference(surface,ref,R,t):
    import open3d as o3d
    x=np.asarray(o3d.io.read_point_cloud(str(surface)).points,dtype=float)
    y=np.asarray(o3d.io.read_point_cloud(str(ref)).points,dtype=float)
    if len(x)<1000 or len(y)<1000:raise RuntimeError("Map too small to compare")
    x=x@R.T+t
    dx=cKDTree(y).query(x)[0]
    dy=cKDTree(x).query(y)[0]
    return {"reference":"GT_POSES_WITH_IDENTICAL_DEPTH_NOT_INDEPENDENT_SCAN",
            "surface_points":len(x),"reference_points":len(y),
            "raw_to_reference_mean_m":float(np.mean(dx)),
            "raw_to_reference_p95_m":float(np.percentile(dx,95)),
            "reference_to_raw_mean_m":float(np.mean(dy)),
            "symmetric_mean_nn_m":float((dx.mean()+dy.mean())/2),
            "symmetric_rmse_nn_m":float(np.sqrt((np.mean(dx*dx)+np.mean(dy*dy))/2)),
            "alignment":"SE3_FROM_ALL_COMMON_TRAJECTORY_POSES_NOT_SURFACES"}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--a4",type=Path,default=Path("results/a4"))
    parser.add_argument("--graph",type=Path,default=Path("results/a4/pose_graph_service.json"))
    parser.add_argument("--database",type=Path,default=Path("results/a4/rtabmap.db"))
    parser.add_argument("--data",type=Path,default=Path("data/tum"))
    parser.add_argument("--output",type=Path,default=Path("results/a6"))
    parser.add_argument("--reconstruct",action="store_true")
    parser.add_argument("--voxel",type=float,default=.06)
    parser.add_argument("--tsdf-side",type=int,default=160)
    parser.add_argument("--ray-stride",type=int,default=40)
    args=parser.parse_args()
    rows,raw,opt,stats=prepare(args.a4,args.graph,args.database,args.data)
    args.output.mkdir(parents=True,exist_ok=True)
    for name,subset in [("raw",raw),("optimized",opt)]:
        path=args.output/(name+"_full_poses.json")
        path.write_text(json.dumps({
            "source":"RTABMap_ODOM_PLUS_GETMAP_"+name.upper(),
            "GT_used_as_input":False,
            "poses":subset},indent=2)+"\n")
    # Reference is imported strictly AFTER both SLAM-derived pose files exist.
    GT,missing=reference_pose_lookup(rows,args.data)
    raw_matches=[p for p in raw if p["index"] in GT]
    opt_matches=[p for p in opt if p["index"] in GT]
    if [p["index"] for p in raw_matches]!=[p["index"] for p in opt_matches]:
        raise RuntimeError("Different evaluation subsets")
    truth=np.asarray([GT[p["index"]][:3,3] for p in raw_matches])
    err_raw,RR,tR=aligned([np.asarray(p["T_map_cam"])[:3,3] for p in raw_matches],truth)
    err_opt,RO,tO=aligned([np.asarray(p["T_map_cam"])[:3,3] for p in opt_matches],truth)
    report={
       "experiment":"A6_FULL_TUM_GRAPH_CORRECTED_GEOMETRY",
       "status":"POSE_EVALUATED_GEOMETRY_NOT_YET_VALIDATED",
       "dataset":"TUM_freiburg1_desk_full_RGBD",
       "GT_used_as_input":False,
       "GT_used_only_for_evaluation":True,
       "GT_evaluation_missing":missing,
       "pose_graph_coverage":stats,
       "raw_ATE_SE3":err_raw,
       "optimized_ATE_SE3":err_opt,
       "ATE_RMSE_improvement_m":err_raw["rmse_m"]-err_opt["rmse_m"],
       "map_geometric_accuracy":"NOT_INDEPENDENTLY_VALIDATED",
       "relocalization_after_restart":"NOT_TESTED",
    }
    summary=args.output/"a6_report.json"
    summary.write_text(json.dumps(report,indent=2)+"\n")
    print("A6_FULL_POSE_RESULT",json.dumps({
        "published_frames":len(rows),"pose_frames":len(raw),
        "anchors":stats["matched_optimized_graph_anchors"],
        "raw_ate":err_raw["rmse_m"],"opt_ate":err_opt["rmse_m"]}),flush=True)
    if args.reconstruct:
        from semantic_map.offline_mapping import reconstruct
        # Only frames whose REAL odometry was measured are integrated, with
        # all others explicitly counted. No temporal subsampling whatsoever.
        poses=[x["index"] for x in raw]
        pairs=[(rows[i]["source_tum_timestamp"],rows[i]["rgb_path"],
                rows[i]["depth_path"],np.eye(4)) for i in poses]
        for tag in ("raw","optimized"):
            result=reconstruct(pairs,args.output/(tag+"_geometry"),
                  voxel_m=args.voxel,occupancy_m=.12,
                  pose_file=args.output/(tag+"_full_poses.json"),
                  tsdf_side=args.tsdf_side,rays_stride=args.ray_stride,
                  frame_limit=0,integration_stride=1)
            report[tag+"_geometry"]={
                "status":result["geometric_mapping_status"],
                "frames_requested":result["frames_requested"],
                "frames_integrated":result["frames_integrated"],
                "mesh_vertices":int(result["mesh_vertices"]),
                "surface_points":int(result["surface_points"]),
                "navigation_ready":False,
            }
        gt_records=[{"index":i,"timestamp":rows[i]["source_tum_timestamp"],
               "T_map_cam":GT[i].tolist()} for i in poses if i in GT]
        if len(gt_records)==len(pairs):
            gt_file=args.output/"gt_pose_REFERENCE_only.json"
            gt_file.write_text(json.dumps({"source":"GT_EVALUATION_ONLY",
                         "never_used_for_SLAM":True,"poses":gt_records},indent=2)+"\n")
            gtmap=reconstruct(pairs,args.output/"gt_pose_REFERENCE_geometry",
                  voxel_m=args.voxel,occupancy_m=.12,pose_file=gt_file,
                  tsdf_side=args.tsdf_side,rays_stride=args.ray_stride,
                  frame_limit=0,integration_stride=1)
            report["GT_pose_reference_geometry"]={
                  "status":"REFERENCE_FROM_SAME_DEPTH_NOT_INDEPENDENT_GEOMETRY",
                  "frames":gtmap["frames_integrated"],
                  "mesh_vertices":gtmap["mesh_vertices"]}
            reference=args.output/"gt_pose_REFERENCE_geometry"/"tsdf_cloud.ply"
            for tag,R,t in [("raw",RR,tR),("optimized",RO,tO)]:
                report[tag+"_surface_error_against_GT_pose_reference"]=mesh_vs_reference(
                   args.output/(tag+"_geometry")/"tsdf_cloud.ply",
                   reference,R,t)
    summary.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print("A6_FULL_REPORT",json.dumps(report,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
