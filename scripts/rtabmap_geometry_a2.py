#!/usr/bin/env python3
"""A2 bridge: real RTAB-Map odometry -> TUM-stamped poses -> geometric map.

Never substitute GT pose for a missing odometry pose. A1 captures raw RTAB-Map
outputs; this converter selects only valid, timestamp-matched observations.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from semantic_map.tum_yolo_pipeline import download_scene, parse
from semantic_map.offline_mapping import validate_pose


def load_a1(prefix: Path):
    metrics=json.loads(prefix.with_suffix(".json").read_text())
    published=json.loads(prefix.with_suffix(".publish.json").read_text())
    odom=json.loads(prefix.with_suffix(".odom.json").read_text())
    if metrics.get("GT_used_as_input") is not False:
        raise ValueError("A1 evidence does not prove GT was excluded from estimation")
    if metrics.get("RGBD_transport")!="rtabmap_msgs/RGBDImage_one_atomic_message":
        raise ValueError("Unsupported A1 sensor transport version")
    return metrics,published,odom


def associate(metrics,published,odom,min_matches=20,max_lag=0.05):
    valid=[x for x in odom if x.get("valid")]
    if len(valid)<min_matches:
        raise RuntimeError(f"Too few valid measured odometry poses: {len(valid)}")
    stamps=np.asarray([p["ros_time"] for p in published],dtype=float)
    if not len(stamps) or not np.all(np.diff(stamps)>0):
        raise ValueError("Replay stamps must be strictly increasing")
    assignments={}
    for row in valid:
        t=float(row["timestamp"])
        idx=int(np.argmin(np.abs(stamps-t)))
        lag=abs(t-stamps[idx])
        if lag>max_lag:
            continue
        if idx in assignments and assignments[idx]["lag"]<=lag:
            continue
        xyz=np.asarray(row["translation_xyz"],dtype=float)
        quat=np.asarray(row["quaternion_xyzw"],dtype=float)
        if not np.isfinite(xyz).all() or not np.isfinite(quat).all():
            continue
        if not np.isclose(np.linalg.norm(quat),1.,atol=.04):
            continue
        T=np.eye(4)
        T[:3,:3]=Rotation.from_quat(quat/np.linalg.norm(quat)).as_matrix()
        T[:3,3]=xyz
        validate_pose(T)
        assignments[idx]={"lag":lag,"T_map_cam":T.tolist()}
    # Disjoint tracks after visual odometry loss are NOT safely co-registered.
    # Reconstruct only the longest continuous, physically plausible segment.
    indices=sorted(assignments)
    segments=[]
    for idx in indices:
        if not segments:
            segments.append([idx])
            continue
        prev=segments[-1][-1]
        dt=float(published[idx]["source_tum_timestamp"])-float(published[prev]["source_tum_timestamp"])
        pa=np.asarray(assignments[prev]["T_map_cam"],dtype=float)
        pb=np.asarray(assignments[idx]["T_map_cam"],dtype=float)
        translation=np.linalg.norm(pb[:3,3]-pa[:3,3])
        angle=np.linalg.norm(Rotation.from_matrix(pa[:3,:3].T@pb[:3,:3]).as_rotvec())
        if idx==prev+1 and 0<dt<0.2 and translation<0.5 and angle<1.5:
            segments[-1].append(idx)
        else:
            segments.append([idx])
    chosen=max(segments,key=len) if segments else []
    poses=[]
    for idx in chosen:
        entry=assignments[idx]
        poses.append({"timestamp":float(published[idx]["source_tum_timestamp"]),
                      "T_map_cam":entry["T_map_cam"],
                      "association_lag_s":entry["lag"],
                      "published_index":idx})
    if len(poses)<min_matches:
        raise RuntimeError(
            f"No long enough CONTINUOUS odometry segment: longest={len(poses)}, "
            f"total_valid_matched={len(assignments)}")
    return poses


def dataset_rgbd_for_poses(data,poses,max_match=0.0001):
    root=download_scene(data)
    rgb=parse(root/"rgb.txt")
    depth=parse(root/"depth.txt")
    rgb_ts=np.asarray([x[0] for x in rgb],dtype=float)
    depth_ts=np.asarray([x[0] for x in depth],dtype=float)
    pairs=[]
    for pose in poses:
        t=float(pose["timestamp"])
        i=int(np.argmin(np.abs(rgb_ts-t)))
        j=int(np.argmin(np.abs(depth_ts-t)))
        if abs(rgb_ts[i]-t)>max_match or abs(depth_ts[j]-t)>.03:
            raise RuntimeError(f"Missing measured RGB-D observation at {t}")
        # Dummy pose is replaced strictly by external odometry in reconstruct().
        # Not ground truth. No positional reference data are loaded here.
        pairs.append((t,root/rgb[i][1][0],root/depth[j][1][0],np.eye(4)))
    return pairs


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--a1-prefix",type=Path,default=Path("results/a1/rtabmap_real_tum"))
    p.add_argument("--output",type=Path,default=Path("results/a2"))
    p.add_argument("--data",type=Path,default=Path("data/tum"))
    p.add_argument("--min-poses",type=int,default=20)
    p.add_argument("--reconstruct",action="store_true")
    p.add_argument("--voxel",type=float,default=.06)
    args=p.parse_args()
    metrics,published,odom=load_a1(args.a1_prefix)
    poses=associate(metrics,published,odom,min_matches=args.min_poses)
    pairs=dataset_rgbd_for_poses(args.data,poses)
    args.output.mkdir(parents=True,exist_ok=True)
    posepath=args.output/"rtabmap_odometry_poses.json"
    posepath.write_text(json.dumps({
        "source":"RTABMap measured RGB-D odometry, not loop-optimized poses",
        "GT_used_as_input":False,
        "poses":poses},indent=2)+"\n")
    report={"status":"POSE_CONVERSION_READY",
            "camera_pose_source":"RTABMap visual odometry only",
            "map_pose_graph_optimized":False,
            "GT_used_as_input":False,
            "published_frames":len(published),
            "real_odometry_messages":len(odom),
            "selected_real_pose_frames":len(pairs),
            "excluded_noncontiguous_valid_frames":sum(bool(x.get("valid")) for x in odom)-len(pairs),
            "selected_published_index_interval":[poses[0]["published_index"],poses[-1]["published_index"]],
            "input_ATE_RMSE_m":metrics.get("ATE_RMSE_m"),
            "geometry_accuracy":"NOT_EVALUATED",
            "reconstruction_run":bool(args.reconstruct)}
    if args.reconstruct:
        from semantic_map.offline_mapping import reconstruct
        result=reconstruct(pairs,args.output/"geometry",pose_file=posepath,
                           voxel_m=args.voxel,occupancy_m=.12,tsdf_side=320)
        report["geometry_status"]=result["geometric_mapping_status"]
        report["mesh_vertices"]=result["mesh_vertices"]
        report["surface_points"]=result["surface_points"]
    (args.output/"a2_report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("A2_RESULT",json.dumps(report),flush=True)


if __name__=="__main__":
    main()
