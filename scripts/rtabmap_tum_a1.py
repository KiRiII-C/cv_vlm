#!/usr/bin/env python3
"""Replay REAL TUM Freiburg1 desk RGB-D into ROS 2 RTAB-Map, without injecting GT poses.

Reference poses are used ONLY for post-run evaluation. Not a simulator, not a
proof of persistent map correctness, dynamic identity or loop-closure accuracy.
"""
from __future__ import annotations
import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
from semantic_map.tum_yolo_pipeline import download_scene, synchronized

def dataset_pairs(dataset: Path, count: int, every: int):
    root = download_scene(dataset)
    pairs = synchronized(root)
    pairs = pairs[::every]
    if count > 0:
        pairs = pairs[:count]
    if len(pairs) < 30:
        raise RuntimeError(f"Not enough synchronized real RGB-D+GT frames: {len(pairs)}")
    return pairs

def evaluate_poses(odometry: list, pairs: list):
    """Rigid SE(3) alignment; GT never used for odometry or map optimization."""
    from scipy.spatial.transform import Rotation
    gt_times = np.asarray([p[0] for p in pairs])
    est, ref = [], []
    used = set()
    for item in odometry:
        t = item["timestamp"]
        i = int(np.argmin(abs(gt_times - t)))
        if abs(gt_times[i] - t) > 0.018 or i in used:
            continue
        used.add(i)
        est.append(item["translation_xyz"])
        ref.append(pairs[i][3][:3, 3])
    if len(est) < 3:
        return {"matched_gt_frames":len(est),"ATE_RMSE_m":None,"ATE_median_m":None}
    a=np.asarray(est,dtype=np.float64); b=np.asarray(ref,dtype=np.float64)
    ma=a.mean(axis=0);mb=b.mean(axis=0)
    U,S,Vt=np.linalg.svd((a-ma).T @ (b-mb))
    diag=np.diag([1.,1.,np.linalg.det(U @ Vt)])
    rot=Vt.T @ diag @ U.T
    aligned=(a-ma)@rot.T + mb
    errs=np.linalg.norm(aligned-b,axis=1)
    return {"matched_gt_frames":int(len(errs)),
            "ATE_RMSE_m":float(np.sqrt(np.mean(errs**2))),
            "ATE_median_m":float(np.median(errs)),
            "GT_used_for_estimation":False,
            "alignment":"SE3_no_scale"}

def replay(args, pairs):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Image, CameraInfo
    from nav_msgs.msg import Odometry

    rclpy.init()
    qos=QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
    node=Node("tum_rgbd_a1_replay")
    pub_rgb=node.create_publisher(Image,"/camera/rgb/image",qos)
    pub_depth=node.create_publisher(Image,"/camera/depth/image",qos)
    pub_info=node.create_publisher(CameraInfo,"/camera/rgb/camera_info",qos)
    odom_log=[]

    def on_odometry(msg):
        stamp=msg.header.stamp.sec + msg.header.stamp.nanosec*1e-9
        p=msg.pose.pose.position
        q=msg.pose.pose.orientation
        odom_log.append({"timestamp":stamp,"translation_xyz":[p.x,p.y,p.z],
                         "quaternion_xyzw":[q.x,q.y,q.z,q.w]})

    sub=node.create_subscription(Odometry,"/odom",on_odometry,50)
    # BOTH odometry and mapping subscribe independently to each camera topic.
    # The first A1 attempt started with only one subscriber per topic, before
    # the full pipeline had completed ROS discovery.
    for _ in range(300):
        rclpy.spin_once(node,timeout_sec=0.1)
        if (pub_rgb.get_subscription_count() >= 2
                and pub_depth.get_subscription_count() >= 2
                and pub_info.get_subscription_count() >= 2):
            break
    counts={name:pub.get_subscription_count() for name,pub in
            [("rgb",pub_rgb),("depth",pub_depth),("camera_info",pub_info)]}
    if min(counts.values()) < 2:
        node.destroy_node()
        rclpy.shutdown()
        raise RuntimeError(f"RTAB-Map subscribers did not become ready: {counts}")

    fx,fy,cx,cy=517.3,516.5,318.6,255.3
    frame="camera_color_optical_frame"
    started=time.monotonic()
    for idx,(stamp,rgb_path,depth_path,_GT_EVALUATION_ONLY) in enumerate(pairs):
        rgb=cv2.imread(str(rgb_path),cv2.IMREAD_COLOR)
        depth=cv2.imread(str(depth_path),cv2.IMREAD_UNCHANGED)
        if rgb is None or depth is None or rgb.shape[:2]!=depth.shape or depth.dtype!=np.uint16:
            raise RuntimeError(f"Bad TUM RGB-D frame {idx}: {rgb_path} {depth_path}")
        h,w=depth.shape
        sec=math.floor(stamp)
        nanosec=int(round((stamp-sec)*1_000_000_000))
        if nanosec >= 1_000_000_000:
            sec+=1;nanosec-=1_000_000_000

        rgb_msg=Image()
        rgb_msg.header.stamp.sec=sec;rgb_msg.header.stamp.nanosec=nanosec
        rgb_msg.header.frame_id=frame
        rgb_msg.width=w;rgb_msg.height=h;rgb_msg.encoding="bgr8"
        rgb_msg.is_bigendian=0;rgb_msg.step=3*w
        rgb_msg.data=rgb.tobytes()
        depth_msg=Image()
        depth_msg.header=rgb_msg.header
        depth_msg.width=w;depth_msg.height=h;depth_msg.encoding="16UC1"
        depth_msg.is_bigendian=0;depth_msg.step=2*w
        depth_msg.data=depth.tobytes()
        ci=CameraInfo()
        ci.header=rgb_msg.header
        ci.width=w;ci.height=h
        ci.distortion_model="plumb_bob"
        ci.d=[0.,0.,0.,0.,0.]
        ci.k=[fx,0.,cx,0.,fy,cy,0.,0.,1.]
        ci.r=[1.,0.,0.,0.,1.,0.,0.,0.,1.]
        ci.p=[fx,0.,cx,0.,0.,fy,cy,0.,0.,0.,1.,0.]
        pub_rgb.publish(rgb_msg)
        pub_depth.publish(depth_msg)
        pub_info.publish(ci)
        until=time.monotonic() + 1.0/args.rate
        while time.monotonic() < until:
            rclpy.spin_once(node,timeout_sec=0.02)
        if (idx+1)%25==0:
            print(f"REAL_TUM_PUBLISHED={idx+1} ODOM_RECEIVED={len(odom_log)}",flush=True)
    # Wait for the RTAB-Map callback queue to drain before final evaluation.
    for _ in range(200):
        rclpy.spin_once(node,timeout_sec=0.10)
    elapsed=time.monotonic()-started
    metrics=evaluate_poses(odom_log,pairs)
    report={
      "experiment":"A1_real_TUM_RTABMap_RGBD_odometry_and_SLAM",
      "input":"real_TUM_freiburg1_desk_RGBD",
      "pose_estimator":"rtabmap_odom/rgbd_odometry",
      "mapper":"rtabmap_slam/rtabmap",
      "GT_used_as_input":False,
      "GT_only_for_metrics":True,
      "camera_intrinsics":"TUM_Freiburg1_factory_registered_RGBD",
      "published_frames":len(pairs),
      "odometry_messages":len(odom_log),
      "publish_rate_hz":args.rate,
      "wall_elapsed_s":elapsed,
      "subscriber_counts_initial":counts,
      "loop_closure_verified":False,
      "semantic_identity_verified":False,
      "localization_after_restart_verified":False,
      **metrics
    }
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+"\n")
    Path(args.output).with_suffix(".odom.json").write_text(json.dumps(odom_log,indent=2)+"\n")
    print("A1_REPORT",json.dumps(report,ensure_ascii=False),flush=True)
    node.destroy_node();rclpy.shutdown()
    if len(odom_log)<args.min_odom or metrics.get("matched_gt_frames",0)<args.min_odom:
        raise RuntimeError(f"Insufficient REAL odometry messages: {len(odom_log)}; matched {metrics.get('matched_gt_frames',0)}")
    if not math.isfinite(metrics["ATE_RMSE_m"]):
        raise RuntimeError("ATE is not finite")
    translations=np.asarray([o["translation_xyz"] for o in odom_log])
    spread=np.linalg.norm(translations.max(axis=0)-translations.min(axis=0))
    if spread < 0.02:
        raise RuntimeError(f"Odometry did not show real camera movement: spread={spread:.5f}m")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",type=Path,default=Path("data/tum"))
    p.add_argument("--frames",type=int,default=180)
    p.add_argument("--every",type=int,default=3)
    p.add_argument("--rate",type=float,default=5.)
    p.add_argument("--min-odom",type=int,default=20)
    p.add_argument("--output",default="results/a1/rtabmap_real_tum.json")
    p.add_argument("--download-only",action="store_true")
    args=p.parse_args()
    if args.frames < 30 or args.rate <= 0 or args.every < 1:p.error("Invalid frames/rate/every")
    pairs=dataset_pairs(args.data,args.frames,args.every)
    print(f"REAL_TUM_SYNCED_PAIRS={len(pairs)} FIRST={pairs[0][0]} LAST={pairs[-1][0]}",flush=True)
    if not args.download_only:
        replay(args,pairs)
if __name__=="__main__":main()
