#!/usr/bin/env python3
"""A3: evaluate REAL RTAB-Map optimized pose graph on matched TUM RGB-D keyframes.

Mapping uses images/depth plus the returned graph poses. TUM ground-truth
poses are loaded ONLY in evaluate_ate() after constructing the pose exports.
Never fill missing graph nodes with ground truth or raw odometry.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sqlite3

import numpy as np
from scipy.spatial.transform import Rotation

from semantic_map.offline_mapping import validate_pose
from semantic_map.tum_yolo_pipeline import download_scene, synchronized
from rtabmap_geometry_a2 import dataset_rgbd_for_poses


def transform(pose):
    xyz=np.asarray(pose["xyz"], dtype=np.float64)
    q=np.asarray(pose["xyzw"], dtype=np.float64)
    if xyz.shape!=(3,) or q.shape!=(4,) or not np.isfinite(xyz).all() or not np.isfinite(q).all():
        raise ValueError("Nonfinite graph pose")
    if not 0.90 < np.linalg.norm(q) < 1.10:
        raise ValueError("Invalid graph quaternion")
    T=np.eye(4)
    T[:3,:3]=Rotation.from_quat(q/np.linalg.norm(q)).as_matrix()
    T[:3,3]=xyz
    validate_pose(T)
    return T


def match_keyframes(graph, database, published, min_nodes=12, max_stamp_lag_s=.05):
    if graph.get("GT_used_as_input") is not False:
        raise ValueError("Cannot prove GT exclusion from graph export")
    unopt=graph["unoptimized"]
    opt=graph["optimized"]
    if unopt["frame_id"]!=opt["frame_id"]:
        raise ValueError("Unoptimized/optimized frames differ")
    keys=sorted(set(unopt["poses"]) & set(opt["poses"]),key=int)
    if len(keys)<min_nodes:
        raise ValueError("Too few common real graph nodes")
    stamps=np.asarray([x["ros_time"] for x in published],dtype=float)
    if len(stamps)<30 or not np.all(np.diff(stamps)>0):
        raise ValueError("Invalid replay timestamp table")
    conn=sqlite3.connect(str(database))
    try:
        rows={int(r[0]):(float(r[1]),int(r[2])) for r in
              conn.execute("SELECT id,stamp,weight FROM Node WHERE id>0")}
    finally:
        conn.close()
    matches=[]
    seen=set()
    for key in keys:
        node_id=int(key)
        if node_id not in rows or rows[node_id][1]<0:
            continue
        stamp,weight=rows[node_id]
        i=int(np.argmin(abs(stamps-stamp)))
        lag=abs(float(stamps[i])-stamp)
        if lag>max_stamp_lag_s:
            continue
        if i in seen:
            raise RuntimeError("Two distinct RTAB-Map keyframes matched one RGB-D frame")
        seen.add(i)
        matches.append({
            "node_id":node_id,
            "published_index":i,
            "tum_timestamp":float(published[i]["source_tum_timestamp"]),
            "ros_timestamp":float(stamp),
            "stamp_error_s":lag,
            "raw_T":transform(unopt["poses"][key]),
            "opt_T":transform(opt["poses"][key]),
        })
    matches.sort(key=lambda m:m["published_index"])
    if len(matches)<min_nodes:
        raise RuntimeError(f"Only {len(matches)} RTAB-Map keyframes timestamp-matched")
    return matches


def aligned_errors(source_xyz,target_xyz):
    """SE(3), without scale. Comparison never imposes GT on SLAM estimation."""
    X=np.asarray(source_xyz,dtype=float)
    Y=np.asarray(target_xyz,dtype=float)
    if X.shape!=Y.shape or X.ndim!=2 or X.shape[1]!=3 or len(X)<3:
        raise ValueError("Need >=3 corresponding 3D points")
    mx=X.mean(0);my=Y.mean(0)
    U,_,Vt=np.linalg.svd((X-mx).T@(Y-my))
    d=np.linalg.det(U@Vt)
    R=Vt.T@np.diag([1.,1.,np.sign(d)])@U.T
    e=np.linalg.norm((X-mx)@R.T+my-Y,axis=1)
    return {"rmse_m":float(np.sqrt(np.mean(e*e))),
            "median_m":float(np.median(e)),
            "p95_m":float(np.percentile(e,95)),
            "max_m":float(np.max(e)),
            "sample_count":int(len(e)),
            "alignment":"SE3_without_scale"}


def evaluate_ate(matches, data):
    """The only function in A3 allowed to access groundtruth.txt."""
    pairs=synchronized(download_scene(data))
    lookup=np.asarray([p[0] for p in pairs])
    gt=[]
    for m in matches:
        index=int(np.argmin(abs(lookup-m["tum_timestamp"])))
        if abs(float(lookup[index])-m["tum_timestamp"])>0.0001:
            raise RuntimeError(f"GT pair missing for evaluation: {m['tum_timestamp']}")
        gt.append(pairs[index][3][:3,3])
    gt=np.asarray(gt)
    raw=np.asarray([m["raw_T"][:3,3] for m in matches])
    opt=np.asarray([m["opt_T"][:3,3] for m in matches])
    return {
        "raw_ate":aligned_errors(raw,gt),
        "optimized_ate":aligned_errors(opt,gt),
        "ATE_RMSE_improvement_m":float(aligned_errors(raw,gt)["rmse_m"]-
                                       aligned_errors(opt,gt)["rmse_m"]),
        "ground_truth_usage":"POST_HOC_METRICS_ONLY",
        "same_exact_node_ids_for_both_trajectories":True,
    }


def graph_topology(opt_graph, matches):
    ids={int(m["node_id"]) for m in matches}
    parents={i:i for i in ids}
    def root(i):
        while parents[i]!=i:
            parents[i]=parents[parents[i]]
            i=parents[i]
        return i
    types=Counter()
    active_links=0
    for item in opt_graph["links"]:
        a=int(item["from_id"]);b=int(item["to_id"])
        if a not in ids or b not in ids:
            continue
        types[str(int(item["type"]))]+=1
        if a!=b:
            parents[root(a)]=root(b)
        active_links+=1
    n_comp=len({root(x) for x in ids})
    return {
        "graph_nodes_matched":len(ids),
        "graph_links_among_matched_nodes":active_links,
        "connected_components":n_comp,
        "link_types":{"neighbor":types.get("0",0),
                      "global_closure":types.get("1",0),
                      "local_space_closure":types.get("2",0),
                      "local_time_closure":types.get("3",0)},
        "graph_connected_for_selected_nodes":n_comp==1,
        "loop_closure_correctness":"NOT_VALIDATED",
    }




def surface_disagreement(raw_ply, optimized_ply, iterations=12):
    """Rigid SE3 surface disagreement. This is NOT independent map accuracy."""
    import open3d as o3d
    from scipy.spatial import cKDTree
    raw=np.asarray(o3d.io.read_point_cloud(str(raw_ply)).points,dtype=float)
    opt=np.asarray(o3d.io.read_point_cloud(str(optimized_ply)).points,dtype=float)
    if len(raw)<100 or len(opt)<100:
        raise RuntimeError("Cannot compare small or empty reconstructed surfaces")
    R=np.eye(3);t=np.zeros(3)
    reference=cKDTree(opt)
    for _ in range(iterations):
        transformed=raw@R.T+t
        dist,idx=reference.query(transformed)
        keep=dist<=np.quantile(dist,.85)
        a=transformed[keep];b=opt[idx[keep]]
        ma=a.mean(axis=0);mb=b.mean(axis=0)
        U,_,Vt=np.linalg.svd((a-ma).T@(b-mb))
        step=Vt.T@np.diag([1.,1.,np.sign(np.linalg.det(Vt.T@U.T))])@U.T
        dt=mb-ma@step.T
        R=step@R;t=step@t+dt
    aligned=raw@R.T+t
    d_raw=reference.query(aligned)[0]
    d_opt=cKDTree(aligned).query(opt)[0]
    return {
        "interpretation":"SURFACE_DISAGREEMENT_NOT_GEOMETRIC_ACCURACY",
        "SE3_aligned":True,
        "raw_points":int(len(raw)),
        "optimized_points":int(len(opt)),
        "symmetric_mean_nn_m":float((d_raw.mean()+d_opt.mean())/2),
        "symmetric_rms_nn_m":float(np.sqrt((np.mean(d_raw*d_raw)+np.mean(d_opt*d_opt))/2)),
        "raw_to_optimized_median_nn_m":float(np.median(d_raw)),
        "raw_to_optimized_p95_nn_m":float(np.percentile(d_raw,95)),
    }

def run(args):
    graph=json.loads(args.graph.read_text())
    published=json.loads(args.published.read_text())
    matches=match_keyframes(graph,args.database,published,args.min_nodes)
    ids=[m["node_id"] for m in matches]
    if len(set(ids))!=len(ids):
        raise RuntimeError("Duplicate graph node ID")
    args.output.mkdir(exist_ok=True,parents=True)
    poses={}
    for label,key in [("raw","raw_T"),("optimized","opt_T")]:
        rows=[{"timestamp":m["tum_timestamp"],
               "T_map_cam":m[key].tolist(),
               "rtabmap_node_id":m["node_id"],
               "published_index":m["published_index"]} for m in matches]
        path=args.output/(label+"_poses.json")
        path.write_text(json.dumps({
            "source":"RTAB-Map GetMap(optimized="+str(label=="optimized")+")",
            "GT_used_as_input":False,
            "reference_timestamps_used_for_estimation":False,
            "frame_id":graph["optimized"]["frame_id"],
            "poses":rows},indent=2)+"\n")
        poses[label]=path
    correction=aligned_errors(
        [m["raw_T"][:3,3] for m in matches],
        [m["opt_T"][:3,3] for m in matches])
    topology=graph_topology(graph["optimized"],matches)
    report={
        "experiment":"A3_RTABMap_optimized_vs_raw_real_TUM_keyframes",
        "status":"POSE_GRAPHS_EXPORTED_NOT_GEOMETRY_VALIDATED",
        "input":"real_TUM_freiburg1_desk_RGBD",
        "GT_used_as_input":False,
        "graph_service_provenance":graph.get("source"),
        "graph_frame":graph["optimized"]["frame_id"],
        "common_keyframes":len(matches),
        "raw_pose_count":graph["unoptimized"]["pose_count"],
        "optimized_pose_count":graph["optimized"]["pose_count"],
        "published_frames":len(published),
        "matched_keyframe_fraction":len(matches)/len(published),
        "keyframe_published_indices":[m["published_index"] for m in matches],
        "graph":topology,
        "raw_vs_opt_se3_aligned_correction":correction,
        "geometry_accuracy":"NOT_VALIDATED",
        "relocalization":"NOT_TESTED",
    }
    # Preserve the GT isolation contract: GT enters only this evaluator,
    # after both pose files are written.
    if args.evaluate_gt:
        report["ATE"]=evaluate_ate(matches,args.data)
    if args.reconstruct:
        from semantic_map.offline_mapping import reconstruct
        rgbd=dataset_rgbd_for_poses(args.data,[
            {"timestamp":m["tum_timestamp"]} for m in matches])
        for label in ("raw","optimized"):
            result=reconstruct(rgbd,args.output/(label+"_geometry"),
                pose_file=poses[label],voxel_m=args.voxel,
                occupancy_m=.12,tsdf_side=320)
            report[label+"_geometry"]={
                "status":result["geometric_mapping_status"],
                "frames_integrated":result["frames_integrated"],
                "mesh_vertices":result["mesh_vertices"],
                "mesh_triangles":result["mesh_triangles"],
                "surface_points":result["surface_points"],
                "occupied_3d_voxels":result["occupied_3d_voxels"],
                "navigation_ready":result["occupancy_2d"]["nav_ready"],
            }
        report["raw_vs_opt_surface_disagreement"]=surface_disagreement(
            args.output/"raw_geometry"/"tsdf_cloud.ply",
            args.output/"optimized_geometry"/"tsdf_cloud.ply")
    output=args.output/"a3_report.json"
    output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print("A3_RESULT",json.dumps(report,ensure_ascii=False),flush=True)
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--graph",type=Path,default=Path("results/a3/pose_graph_service.json"))
    p.add_argument("--database",type=Path,default=Path("results/a1/rtabmap.db"))
    p.add_argument("--published",type=Path,
                   default=Path("results/a1/rtabmap_real_tum.publish.json"))
    p.add_argument("--data",type=Path,default=Path("data/tum"))
    p.add_argument("--output",type=Path,default=Path("results/a3"))
    p.add_argument("--min-nodes",type=int,default=12)
    p.add_argument("--voxel",type=float,default=.06)
    p.add_argument("--evaluate-gt",action="store_true")
    p.add_argument("--reconstruct",action="store_true")
    args=p.parse_args()
    run(args)


if __name__=="__main__":
    main()
