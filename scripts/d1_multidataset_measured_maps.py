#!/usr/bin/env python3
"""D1 — measured RGB-D colored geometry and YOLOE-26x semantic maps on REAL datasets.

Purpose: cross-dataset DATA + perception robustness, NOT SLAM benchmarking.
All global camera transforms are generated from dataset ground-truth trajectories;
Bonn additionally needs the published marker-to-camera transform. Ground-truth
trajectories are an ORACLE, never claimed to be RTAB-Map estimates.

Sensor depth is not synthesized; unobserved surfaces are not filled; semantic
labels are predictions from real YOLOE masks, not semantic GT or verified UUID.
"""
from __future__ import annotations

import argparse
import bisect
from collections import Counter
import json
import os
from pathlib import Path
import struct
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

DATASETS = {
    "tum_xyz": {"K": (517.3, 516.5, 318.6, 255.3), "depth_scale": 5000.,
                "calibration": "TUM Freiburg1 RGB-D, camera GT trajectory"},
    "tum_desk2": {"K": (517.3, 516.5, 318.6, 255.3), "depth_scale": 5000.,
                  "calibration": "TUM Freiburg1 RGB-D, camera GT trajectory"},
    "bonn_box": {"K": (542.822841, 542.576870, 315.593520, 237.756098),
                 "depth_scale": 5000.,
                 "calibration": "Bonn RGB-D, mocap marker pose + published camera extrinsic"},
}

# The marker-to-camera convention was previously selected with a held-out,
# real measured Bonn depth-reprojection audit. NOT suitable for other sensors.
BONN_MARKER_TO_CAMERA = np.array([
    [1.0157, .1828, -.2389, .0113],
    [.0009, -.8431, -.6413, -.0098],
    [-.3009, .6147, -.8085, .0111],
    [0., 0., 0., 1.],
], dtype=float)
_u, _s, _v = np.linalg.svd(BONN_MARKER_TO_CAMERA[:3, :3])
BONN_MARKER_TO_CAMERA[:3, :3] = _u @ _v
BONN_ROS_AXIS = np.array([[-1, 0, 0, 0],
                           [0, 0, 1, 0],
                           [0, 1, 0, 0],
                           [0, 0, 0, 1]], dtype=float)
BONN_EXTRINSIC = BONN_ROS_AXIS @ BONN_MARKER_TO_CAMERA


def index_file(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        tokens = line.split()
        if len(tokens) < 2:
            raise ValueError(f"Malformed {path}: {line}")
        rows.append((float(tokens[0]), tokens[1:]))
    if not rows:
        raise ValueError("Empty TUM-format time index " + str(path))
    rows.sort(key=lambda z: z[0])
    return rows


def closest(rows, stamps, timestamp, tolerance):
    i = bisect.bisect_left(stamps, timestamp)
    candidates = [j for j in (i - 1, i) if 0 <= j < len(rows)]
    if not candidates:
        return None
    j = min(candidates, key=lambda k: abs(stamps[k] - timestamp))
    if abs(stamps[j] - timestamp) > tolerance:
        return None
    return rows[j]


def matched_sensor_rows(root: Path, time_tolerance=.045):
    rgbs, depths, gt = (index_file(root / x) for x in
                        ("rgb.txt", "depth.txt", "groundtruth.txt"))
    sd = [v[0] for v in depths]
    sp = [v[0] for v in gt]
    out = []
    for rgb_t, rgb_data in rgbs:
        d = closest(depths, sd, rgb_t, time_tolerance)
        pose = closest(gt, sp, rgb_t, time_tolerance)
        if d is None or pose is None or len(pose[1]) < 7:
            continue
        out.append({
            "timestamp": rgb_t,
            "rgb_path": root / rgb_data[0],
            "depth_path": root / d[1][0],
            "pose": np.array([float(x) for x in pose[1][:7]], dtype=float),
        })
    return out, {"rgb_index_entries": len(rgbs), "depth_index_entries": len(depths),
                 "gt_pose_index_entries": len(gt), "synchronized_matched": len(out)}


def camera_transform(pose7, dataset):
    pose = np.asarray(pose7, dtype=float)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError("Invalid ground truth pose")
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()
    T[:3, 3] = pose[:3]
    if dataset == "bonn_box":
        T = T @ BONN_EXTRINSIC
    if not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-5):
        raise ValueError("Invalid camera rotation")
    return T


def measured_xyz(depth_m, K, step=4):
    """Only valid original depth pixels; no interpolation or synthetic voxels."""
    h, w = depth_m.shape
    yy, xx = np.mgrid[0:h:step, 0:w:step]
    z = depth_m[::step, ::step]
    valid = np.isfinite(z) & (z >= .25) & (z <= 5.0)
    xyz = np.column_stack((
        (xx[valid] - K[2]) * z[valid] / K[0],
        (yy[valid] - K[3]) * z[valid] / K[1],
        z[valid],
    )).astype(np.float32)
    return xyz, yy[valid], xx[valid]


def apply_pose(points, T):
    return (points @ T[:3, :3].T + T[:3, 3]).astype(np.float32)


def aggregate_voxels(points, colors, voxel_m):
    if len(points) == 0:
        raise ValueError("No real measured points to voxelize")
    points = np.asarray(points, np.float32)
    colors = np.asarray(colors, np.uint8)
    if points.ndim != 2 or points.shape[1] != 3 or colors.shape != points.shape:
        raise ValueError("Invalid XYZ/RGB sampling dimensions")
    q = np.floor(points / float(voxel_m)).astype(np.int32)
    uq, inverse, count = np.unique(q, axis=0, return_inverse=True, return_counts=True)
    weights = count.astype(np.float64)
    xyz = np.column_stack([
        np.bincount(inverse, weights=points[:, i], minlength=len(uq)) / weights
        for i in range(3)
    ]).astype(np.float32)
    rgb = np.column_stack([
        np.bincount(inverse, weights=colors[:, i], minlength=len(uq)) / weights
        for i in range(3)
    ]).clip(0, 255).astype(np.uint8)
    return xyz, rgb, count


def write_binary_ply(path, xyz, rgb):
    path = Path(path)
    if len(xyz) != len(rgb):
        raise ValueError("Mismatched PLY color and positions")
    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                      ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    rows = np.empty(len(xyz), dtype=dtype)
    for i, c in enumerate("xyz"):
        rows[c] = xyz[:, i]
    for i, c in enumerate(("red", "green", "blue")):
        rows[c] = rgb[:, i]
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(xyz)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              "end_header\n").encode()
    with path.open("wb") as f:
        f.write(header)
        rows.tofile(f)


def class_palette(n):
    import matplotlib.pyplot as plt
    colormap = plt.get_cmap("tab20")
    return np.round(255 * np.asarray([colormap(i % 20)[:3] for i in range(n)])).astype(np.uint8)


def render_views(folder, name, geometry_xyz, geometry_rgb, semantic_xyz,
                 semantic_class, classes, counts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    folder = Path(folder)
    rng = np.random.default_rng(20261103)
    geo_ids = rng.choice(len(geometry_xyz), min(35000, len(geometry_xyz)), replace=False)
    vis_geo = geometry_xyz[geo_ids]
    vis_rgb = geometry_rgb[geo_ids] / 255.
    if len(semantic_xyz):
        sem_ids = rng.choice(len(semantic_xyz), min(25000, len(semantic_xyz)), replace=False)
        vis_sem = semantic_xyz[sem_ids]
        vis_cls = semantic_class[sem_ids]
    else:
        vis_sem = np.empty((0, 3), dtype=float)
        vis_cls = np.empty((0,), dtype=np.int32)
    palette = class_palette(len(classes))
    fig = plt.figure(figsize=(17, 8), facecolor="#101a27")
    ax = fig.add_subplot(121, projection="3d")
    ax.scatter(vis_geo[:, 0], vis_geo[:, 1], vis_geo[:, 2],
               c=vis_rgb, s=.36, alpha=.45, linewidths=0, rasterized=True)
    if len(vis_sem):
        ax.scatter(vis_sem[:, 0], vis_sem[:, 1], vis_sem[:, 2],
                   c=palette[vis_cls]/255, s=1.4, alpha=.96, linewidths=0, rasterized=True)
    ax.set_title("3D RGB-D + YOLOE 26x mask surfaces", color="white", pad=16)
    bounds = np.quantile(geometry_xyz, [.005, .995], axis=0)
    mid = bounds.mean(axis=0)
    span = max(.1, float(np.max(bounds[1] - bounds[0]))) * .58
    for fun,center in ((ax.set_xlim,mid[0]), (ax.set_ylim,mid[1]), (ax.set_zlim,mid[2])):
        fun(center-span, center+span)
    ax.set_xlabel("X [m]");ax.set_ylabel("Y [m]");ax.set_zlabel("Z [m]")
    ax.view_init(elev=25, azim=40)
    ax.set_box_aspect((1,1,1))
    ax2 = fig.add_subplot(122)
    ax2.set_facecolor("#101a27")
    ax2.scatter(vis_geo[:, 0], vis_geo[:, 1], c=vis_rgb, s=.25, alpha=.25, rasterized=True)
    if len(vis_sem):
        ax2.scatter(vis_sem[:,0],vis_sem[:,1],c=palette[vis_cls]/255,
                    s=1.2,alpha=.8,rasterized=True)
    ax2.set_aspect("equal", adjustable="datalim")
    ax2.set_xlabel("World X [m]");ax2.set_ylabel("World Y [m]")
    ax2.set_title("XY projection (world axes, not necessarily floor)", color="white")
    for axis in (ax,ax2):
        axis.tick_params(colors="#c5d1df")
        if axis==ax2:axis.grid(alpha=.12)
    used = sorted(set(semantic_class.tolist())) if len(semantic_class) else []
    handles=[Line2D([],[],linestyle="",marker="o",markersize=7,color=palette[i]/255,
                    label=classes[i]) for i in used]
    if handles:
        fig.legend(handles=handles,loc="lower center",ncol=min(6,len(handles)),
                   frameon=False,labelcolor="white")
    fig.text(.03,.96,f"{name} — REAL DATASET",color="white",fontsize=19,weight="bold")
    fig.text(.03,.922,
       f"GT camera poses (ORACLE, NOT SLAM) | geometry frames: {counts['geometry_frames']} | YOLO frames: {counts['yolo_frames']}",
       color="#b5c9dc",fontsize=10)
    fig.text(.03,.035,
       "Class colors = model predictions, not GT. No physical instance-ID or scene-change GT. Sampled RGB-D depth only.",
       color="#ffc5a7",fontsize=9)
    fig.subplots_adjust(left=.04,right=.96,top=.88,bottom=.15,wspace=.08)
    fig.savefig(folder/"map_overview.png",dpi=150,facecolor=fig.get_facecolor())
    plt.close(fig)

    # HTML is a selectable, zoomable point-cloud visualization.
    import plotly.graph_objects as go
    fig=go.Figure()
    fig.add_trace(go.Scatter3d(
        x=vis_geo[:,0],y=vis_geo[:,1],z=vis_geo[:,2],mode="markers",
        marker=dict(size=1.3,color=["rgb(%d,%d,%d)" % tuple(c) for c in geometry_rgb[geo_ids]],
                    opacity=.32),name="Measured RGB-D geometry"))
    for i in used:
        selected = vis_cls == i
        xyz = vis_sem[selected]
        if not len(xyz):continue
        r,g,b=map(int,palette[i])
        fig.add_trace(go.Scatter3d(
            x=xyz[:,0],y=xyz[:,1],z=xyz[:,2],mode="markers",
            marker=dict(size=2.2,color=f"rgb({r},{g},{b})",opacity=.90),
            name=f"YOLOE: {classes[i]}"))
    fig.update_layout(
        title=f"{name} | GT-pose reconstruction, semantic predictions",
        template="plotly_dark",height=850,
        scene=dict(aspectmode="data",xaxis_title="X m",yaxis_title="Y m",
                   zaxis_title="Z m"))
    fig.write_html(str(folder/"interactive_map.html"),include_plotlyjs="cdn",
                   full_html=True,auto_open=False)


def process(args):
    dataset = args.dataset
    config = DATASETS[dataset]
    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows,counts = matched_sensor_rows(root)
    if len(rows) < 80:
        raise RuntimeError("Too few real synchronized RGB-D+GT frames: " + str(len(rows)))
    # Verify actual sensor provenance: source RGB and depth always exist.
    for x in rows:
        if not x["rgb_path"].is_file() or not x["depth_path"].is_file():
            raise FileNotFoundError("Original RGB-D source missing on disk")
    from semantic_map.yoloe26_detector import YOLOE26Segmenter, class_list
    names = class_list("config/yoloe26_classes.json")
    detector = YOLOE26Segmenter(args.weights, names)
    geo_xyz, geo_rgb = [], []
    sem_xyz, sem_class, sem_scores = [], [], []
    detections = Counter()
    duration = []
    valid_depth_fracs = []
    raw_masks = 0
    geometry_frames = 0
    yolo_frames = 0
    poses_valid = 0
    source_h = []
    K = config["K"]

    for i, row in enumerate(rows):
        selected_geom = (i % args.geometry_stride == 0)
        selected_yolo = (i % args.yolo_stride == 0)
        if not selected_geom and not selected_yolo:
            continue
        raw = cv2.imread(str(row["depth_path"]), cv2.IMREAD_UNCHANGED)
        if raw is None or raw.ndim != 2 or raw.dtype != np.uint16:
            raise RuntimeError(f"Invalid source raw 16UC1 depth frame {i}")
        bgr = cv2.imread(str(row["rgb_path"]),cv2.IMREAD_COLOR)
        if bgr is None or bgr.shape[:2] != raw.shape:
            raise RuntimeError(f"Invalid synchronized source color frame {i}")
        T = camera_transform(row["pose"], dataset)
        poses_valid += 1
        depth_m = raw.astype(np.float32)/float(config["depth_scale"])
        good=(depth_m>=.25)&(depth_m<=5.0)
        valid_depth_fracs.append(float(good.mean()))
        if selected_geom:
            p,yy,xx=measured_xyz(depth_m,K,step=args.pixel_stride)
            if len(p):
                geo_xyz.append(apply_pose(p,T))
                geo_rgb.append(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)[yy,xx])
            geometry_frames += 1
        if selected_yolo:
            start = time.perf_counter()
            result = detector.predict(bgr,conf=.10,device=args.device,imgsz=640)
            duration.append(time.perf_counter()-start)
            yolo_frames += 1
            if result.masks is not None and result.boxes is not None:
                masks=result.masks.data.cpu().numpy()
                cls=result.boxes.cls.int().cpu().numpy()
                scores=result.boxes.conf.cpu().numpy()
                if len(masks)!=len(cls) or len(cls)!=len(scores):
                    raise RuntimeError("YOLOE segmentation mask/label mismatch")
                raw_masks += len(masks)
                for maskarr,category,score in zip(masks,cls,scores):
                    if float(score) < args.confidence:
                        continue
                    label = detector.class_name(category)
                    if label not in [__import__("semantic_map.yoloe26_detector",fromlist=["canonical"]).canonical(v) for v in names]:
                        raise RuntimeError("Unexpected YOLOE class")
                    mask = maskarr > .5
                    if mask.shape!=raw.shape:
                        mask=cv2.resize(mask.astype(np.uint8),(raw.shape[1],raw.shape[0]),
                                        interpolation=cv2.INTER_NEAREST).astype(bool)
                    dvalues=depth_m[mask & good]
                    if len(dvalues)<60:
                        continue
                    median=float(np.median(dvalues))
                    mask = mask & good & (np.abs(depth_m-median)<max(.18,.16*median))
                    ys,xs=np.where(mask)
                    if len(xs)<60:continue
                    if len(xs)>args.object_point_cap:
                        keep=np.linspace(0,len(xs)-1,args.object_point_cap,dtype=np.int64)
                        ys,xs=ys[keep],xs[keep]
                    z=depth_m[ys,xs]
                    xyz = np.column_stack([(xs-K[2])*z/K[0],
                                            (ys-K[3])*z/K[1],z]).astype(np.float32)
                    xyz=apply_pose(xyz,T)
                    sem_xyz.append(xyz)
                    sem_class.append(np.full(len(xyz),label, dtype="U40"))
                    sem_scores.append(np.full(len(xyz),float(score),dtype=np.float32))
                    detections[label] += 1
        if i % 120 == 0:
            print("D1_MAP_PROGRESS",dataset,i,len(rows),
                  "geometry",geometry_frames,"yolo",yolo_frames,flush=True)

    if not geo_xyz or not sem_xyz:
        raise RuntimeError("Missing valid measured geometry or predicted semantic object points")
    all_geo=np.concatenate(geo_xyz,axis=0)
    all_rgb=np.concatenate(geo_rgb,axis=0)
    all_sem=np.concatenate(sem_xyz,axis=0)
    all_labels=np.concatenate(sem_class,axis=0)
    all_conf=np.concatenate(sem_scores,axis=0)
    if not (np.isfinite(all_geo).all() and np.isfinite(all_sem).all()):
        raise RuntimeError("Nonfinite REAL reconstructed camera point")
    xyz,rgb,gcnt=aggregate_voxels(all_geo,all_rgb,args.geometry_voxel)
    write_binary_ply(out/"geometry_rgb.ply",xyz,rgb)
    from semantic_map.yoloe26_detector import canonical
    classes=list(dict.fromkeys(canonical(n) for n in names))
    label_ids=np.asarray([classes.index(l) for l in all_labels],dtype=np.int32)
    palette=class_palette(len(classes))
    # Keep per-category spatial supports separate. This avoids erasing real
    # class conflicts and explicitly does NOT infer persistent instance UUIDs.
    labels=np.empty((0,),dtype=np.int32)
    per_class_points=[]
    per_class_ids=[]
    for cls_idx in range(len(classes)):
        ix=np.flatnonzero(label_ids==cls_idx)
        if not len(ix):continue
        # High-resolution semantic accumulation, separately for each class.
        sx,sc,scnt=aggregate_voxels(all_sem[ix],
            np.broadcast_to(palette[cls_idx],(len(ix),3)).copy(),
            args.semantic_voxel)
        per_class_points.append(sx)
        per_class_ids.append(np.full(len(sx),cls_idx,dtype=np.int32))
    if not per_class_points:
        raise RuntimeError("No valid semantic class voxels")
    sem_pts=np.concatenate(per_class_points)
    sem_ids=np.concatenate(per_class_ids)
    write_binary_ply(out/"semantic_predicted.ply",sem_pts,palette[sem_ids])
    ct={"geometry_frames":geometry_frames,"yolo_frames":yolo_frames}
    render_views(out,dataset,xyz,rgb,sem_pts,sem_ids,classes,ct)
    b0=np.quantile(xyz,.01,axis=0)
    b1=np.quantile(xyz,.99,axis=0)
    report = {
      "experiment":"D1_REAL_MULTIDATASET_MEASURED_SEMANTIC_3D_MAPS",
      "dataset":dataset,
      "input": counts,
      "pose_source":config["calibration"],
      "GT_used_for_mapping":True,
      "SLAM_evaluated":False,
      "not_comparable_to_RTABMap_ATE":True,
      "independent_instance_ground_truth":False,
      "semantic_class_accuracy":"NOT_EVALUATED_NO_MATCHED_GT_LABELS",
      "physical_IDF1":"NOT_EVALUATED",
      "scene_change_F1":"NOT_EVALUATED",
      "density_policy":{
        "geometry_keyframe_stride":args.geometry_stride,
        "geometry_pixel_stride":args.pixel_stride,
        "geometry_voxel_m":args.geometry_voxel,
        "YOLOE_keyframe_stride":args.yolo_stride,
        "semantic_voxel_m":args.semantic_voxel,
        "per_mask_measured_depth_points_cap":args.object_point_cap,
      },
      "model":"yoloe-26x-seg.pt",
      "model_prompt_classes":names,
      "confidence_threshold":args.confidence,
      "measured_geometry_frames":geometry_frames,
      "actual_model_inference_frames":yolo_frames,
      "source_valid_depth_fraction_mean":float(np.mean(valid_depth_fracs)),
      "source_valid_depth_fraction_min":float(np.min(valid_depth_fracs)),
      "raw_YOLOE_masks":raw_masks,
      "valid_predicted_3D_masks":sum(detections.values()),
      "valid_measured_depth_xyz_count_before_voxels":len(all_geo),
      "geometry_voxels":len(xyz),
      "predicted_semantic_points_before_voxels":len(all_sem),
      "predicted_semantic_voxels_per_class":len(sem_pts),
      "per_class_mask_observations":dict(detections),
      "per_class_measured_voxels":{classes[i]:int(np.sum(sem_ids==i)) for i in range(len(classes))},
      "geometry_bbox_1_to_99_percentile_world_m":[b0.tolist(),b1.tolist()],
      "model_inference_seconds_median":float(np.median(duration)),
      "model_inference_seconds_p95":float(np.percentile(duration,95)),
      "outputs":["geometry_rgb.ply","semantic_predicted.ply","map_overview.png","interactive_map.html"],
      "limitations":("Visual 3D mapping uses original real RGB-D and GT/mocap poses as "
         "offline registration oracle, not RTAB-Map. Dynamic agents/objects may create "
         "ghosts in a globally fused geometry cloud. XY is world XY, not a validated "
         "floor occupancy map. Semantic masks are YOLOE predictions, no annotated GT. "
         "No autonomous relocalization, persistent physical-ID, or Nav2 safety test.")
    }
    (out/"metrics.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print("D1_REAL_MAP_RESULT",json.dumps({
      k:report[k] for k in ("dataset","measured_geometry_frames",
        "actual_model_inference_frames","geometry_voxels",
        "valid_predicted_3D_masks","predicted_semantic_voxels_per_class",
        "per_class_mask_observations")},ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",choices=DATASETS,required=True)
    p.add_argument("--root",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--weights",default="yoloe-26x-seg.pt")
    p.add_argument("--device",default="cpu")
    p.add_argument("--confidence",type=float,default=.40)
    p.add_argument("--geometry-stride",type=int,default=4)
    p.add_argument("--pixel-stride",type=int,default=4)
    p.add_argument("--yolo-stride",type=int,default=20)
    p.add_argument("--geometry-voxel",type=float,default=.03)
    p.add_argument("--semantic-voxel",type=float,default=.02)
    p.add_argument("--object-point-cap",type=int,default=3000)
    args=p.parse_args()
    if min(args.geometry_stride,args.pixel_stride,args.yolo_stride,args.object_point_cap)<1:
        p.error("Pixel/frame stride and point cap must be positive")
    if args.geometry_voxel<=0 or args.semantic_voxel<=0:
        p.error("Voxel dimensions must be positive")
    process(args)


if __name__ == "__main__":
    main()
