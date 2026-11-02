#!/usr/bin/env python3
"""A4: FULL TUM RGB-D sequence with per-frame RTAB-Map OdomInfoLite feedback.

No temporal subsampling, no arbitrary frame limit and NO GT dependency for
feeding SLAM. TUM reference poses are read only after acquisition for metrics.
A packet is considered processed only when an OdomInfoLite message with its
exact capture header is observed. Never infer processing from DDS subscribers.
"""
from __future__ import annotations
import argparse
import json
import math
import time
from bisect import bisect_left
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
from semantic_map.tum_yolo_pipeline import download_scene, parse


def full_rgbd_pairs(folder: Path):
    root=download_scene(folder)
    rgb=parse(root/"rgb.txt")
    depth=parse(root/"depth.txt")
    dts=[d[0] for d in depth]
    if not rgb or not depth:
        raise RuntimeError("Empty RGB/depth dataset source")
    pairs=[]
    missing=0
    reused=0
    used=set()
    for t,paths in rgb:
        j=bisect_left(dts,t)
        nearby=[k for k in (j-1,j) if 0<=k<len(depth) and abs(depth[k][0]-t)<=.03]
        if not nearby:
            missing+=1
            continue
        k=min(nearby,key=lambda v:abs(dts[v]-t))
        if k in used:
            reused+=1
            continue
        used.add(k)
        pairs.append((float(t),root/paths[0],root/depth[k][1][0],float(dts[k])))
    if len(pairs)<30:
        raise RuntimeError("Too few RGB-D pairs")
    if any(pairs[i][0]>=pairs[i+1][0] for i in range(len(pairs)-1)):
        raise ValueError("Dataset RGB stamps are nonmonotonic")
    return pairs, {
        "source_rgb_frames":len(rgb),
        "source_depth_frames":len(depth),
        "synchronized_rgbd_pairs":len(pairs),
        "rgb_without_nearby_depth":missing,
        "rgb_depth_reuse_blocked":reused,
        "unpaired_depth":len(depth)-len(used),
        "pairing_tolerance_s":.03,
        "all_paired_frames_selected":True,
        "frame_stride":1,
        "max_frames":None,
        "groundtruth_access_during_pairing":False,
    }


def stamp_key(stamp):
    return int(stamp.sec)*1_000_000_000+int(stamp.nanosec)


def match_groundtruth(pairs, published, odom):
    """Post-hoc ATE on measured /odom only; no pose reference in SLAM input."""
    from scipy.spatial.transform import Rotation
    root=Path(pairs[0][1]).parent.parent
    reference=parse(root/"groundtruth.txt")
    ref_t=np.asarray([x[0] for x in reference],dtype=float)
    odom_by_key={o["stamp_ns"]:o for o in odom}
    gt=[]; est=[]
    no_gt=0; not_measured=0; rejected=0
    for p in published:
        t=p["source_tum_timestamp"]
        j=int(np.argmin(abs(ref_t-t)))
        if abs(ref_t[j]-t)>.03:
            no_gt+=1
            continue
        o=odom_by_key.get(p["stamp_ns"])
        if o is None:
            not_measured+=1
            continue
        if not o["numerically_valid"]:
            rejected+=1
            continue
        est.append(o["xyz"])
        gt.append([float(x) for x in reference[j][1][:3]])
    out={"evaluation_reference":"TUM_groundtruth.txt_POST_HOC_ONLY",
         "evaluation_no_gt":no_gt,
         "evaluation_no_odometry":not_measured,
         "evaluation_numerical_reject":rejected,
         "evaluation_matched":len(est),
         "ATE_RMSE_m":None, "ATE_median_m":None}
    if len(est)>=3:
        X=np.asarray(est,dtype=float);Y=np.asarray(gt,dtype=float)
        mx=X.mean(0);my=Y.mean(0)
        U,_,Vt=np.linalg.svd((X-mx).T@(Y-my))
        R=Vt.T@np.diag([1.,1.,np.sign(np.linalg.det(U@Vt))])@U.T
        errors=np.linalg.norm((X-mx)@R.T+my-Y,axis=1)
        out["ATE_RMSE_m"]=float(np.sqrt(np.mean(errors**2)))
        out["ATE_median_m"]=float(np.median(errors))
    return out


def replay(args, pairs, dataset_stats):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import CameraInfo
    from nav_msgs.msg import Odometry
    from rtabmap_msgs.msg import RGBDImage, OdomInfo
    rclpy.init()
    node=Node("full_tum_diagnostic_replay_a4")
    qos=QoSProfile(depth=10,reliability=ReliabilityPolicy.RELIABLE)
    rgbd=node.create_publisher(RGBDImage,"/camera/rgbd_image",qos)
    info_by_ns={}
    odom_by_ns={}
    info_rows=[]
    odom_rows=[]
    def on_info(msg):
        k=stamp_key(msg.header.stamp)
        q={
            "stamp_ns":k,
            "lost":bool(msg.lost),
            "inliers":int(msg.inliers),
            "matches":int(msg.matches),
            "features":int(msg.features),
            "local_map_size":int(msg.local_map_size),
            "time_estimation_s":float(msg.time_estimation),
            "cov_xyz_diagonal":[float(msg.covariance[i]) for i in (0,7,14)],
            "key_frame_added":bool(msg.key_frame_added),
            "odometry_type":int(msg.type),
            "icp_inliers_ratio":float(msg.icp_inliers_ratio),
        }
        if k not in info_by_ns:
            info_by_ns[k]=q
            info_rows.append(q)
    def on_odom(msg):
        k=stamp_key(msg.header.stamp)
        if k in odom_by_ns:
            return
        p=msg.pose.pose.position;q=msg.pose.pose.orientation
        xyz=[float(p.x),float(p.y),float(p.z)]
        quat=[float(q.x),float(q.y),float(q.z),float(q.w)]
        covariance=[float(msg.pose.covariance[i]) for i in (0,7,14)]
        ok=(all(math.isfinite(v) for v in xyz+quat+covariance)
            and .5<sum(v*v for v in quat)<1.5
            and max(covariance)<1000)
        o={"stamp_ns":k,"xyz":xyz,"quaternion_xyzw":quat,
           "cov_xyz_diagonal":covariance,"numerically_valid":bool(ok)}
        odom_by_ns[k]=o
        odom_rows.append(o)
    subs=[
        node.create_subscription(OdomInfo,"/odom_info_lite",on_info,qos),
        node.create_subscription(Odometry,"/odom",on_odom,qos),
    ]
    try:
        for _ in range(300):
            rclpy.spin_once(node,timeout_sec=.1)
            if rgbd.get_subscription_count()>=2 and node.count_publishers("/odom_info_lite")>=1:
                break
        consumers=rgbd.get_subscription_count()
        diagnostics=node.count_publishers("/odom_info_lite")
        if consumers<2 or diagnostics<1:
            raise RuntimeError(f"Missing subscribers/diagnostic publisher: RGBD={consumers}, info={diagnostics}")
        published=[]
        no_info=[]
        started=time.monotonic()
        last_pub=0.
        for i,(tum_t,rgb_file,depth_file,depth_t) in enumerate(pairs):
            rgb=cv2.imread(str(rgb_file),cv2.IMREAD_COLOR)
            depth=cv2.imread(str(depth_file),cv2.IMREAD_UNCHANGED)
            if rgb is None or depth is None or rgb.shape[:2]!=depth.shape or depth.dtype!=np.uint16:
                raise RuntimeError(f"Corrupt RGB-D frame {i}: {rgb_file}, {depth_file}")
            height,width=depth.shape
            depth_mm=np.rint(depth.astype(np.float32)/5.).clip(0,65535).astype(np.uint16)
            ok_rgb,jpeg=cv2.imencode(".jpg",rgb,[cv2.IMWRITE_JPEG_QUALITY,95])
            ok_depth,png=cv2.imencode(".png",depth_mm,[cv2.IMWRITE_PNG_COMPRESSION,1])
            if not ok_rgb or not ok_depth:
                raise RuntimeError(f"Cannot compress frame {i}")
            # Publish every frame exactly once; pace only if prior processing was fast.
            while time.monotonic()-last_pub < 1./args.max_rate_hz:
                rclpy.spin_once(node,timeout_sec=.01)
            now=node.get_clock().now().to_msg()
            key=stamp_key(now)
            cam=CameraInfo()
            cam.header.stamp=now
            cam.header.frame_id="camera_color_optical_frame"
            cam.width=width;cam.height=height
            cam.distortion_model="plumb_bob"
            cam.d=[0.,0.,0.,0.,0.]
            cam.k=[517.3,0.,318.6,0.,516.5,255.3,0.,0.,1.]
            cam.r=[1.,0.,0.,0.,1.,0.,0.,0.,0.,1.]
            cam.p=[517.3,0.,318.6,0.,0.,516.5,255.3,0.,0.,0.,1.,0.]
            msg=RGBDImage()
            msg.header=cam.header
            msg.rgb_camera_info=cam
            msg.depth_camera_info=cam
            msg.rgb_compressed.header=cam.header
            msg.rgb_compressed.format="bgr8; jpeg compressed bgr8"
            msg.rgb_compressed.data=jpeg.tobytes()
            msg.depth_compressed.header=cam.header
            msg.depth_compressed.format="png"
            msg.depth_compressed.data=png.tobytes()
            rgbd.publish(msg)
            last_pub=time.monotonic()
            published.append({"idx":i,"stamp_ns":key,
                              "source_tum_timestamp":tum_t,
                              "source_depth_timestamp":depth_t})
            deadline=time.monotonic()+args.ack_timeout_s
            while key not in info_by_ns and time.monotonic()<deadline:
                rclpy.spin_once(node,timeout_sec=.02)
            if key not in info_by_ns:
                no_info.append(i)
                print(f"NO_ODOMINFO_ACK frame={i}",flush=True)
            if (i+1)%50==0 or i+1==len(pairs):
                seen=sum(p["stamp_ns"] in info_by_ns for p in published)
                print(f"FULL_TUM_PROGRESS {i+1}/{len(pairs)} ODOMINFO_ACK={seen} "
                      f"LOST={sum(x['lost'] for x in info_rows)} "
                      f"ODOM={len(odom_rows)}",flush=True)
        for _ in range(150):
            rclpy.spin_once(node,timeout_sec=.02)
        acked=sum(p["stamp_ns"] in info_by_ns for p in published)
        infos=list(info_rows)
        numerical=sum(o["numerically_valid"] for o in odom_rows)
        not_lost=sum(not row["lost"] for row in infos)
        matches_gt=match_groundtruth(pairs,published,odom_rows)
        report={
            "experiment":"A4_FULL_TUM_RGBD_RTABMAP_ODOMINFO",
            "source_dataset":"TUM_RGBD_freiburg1_desk",
            **dataset_stats,
            "GT_used_as_input":False,
            "GT_used_only_posthoc":True,
            "published_frames":len(published),
            "odominfo_ack_frames":acked,
            "odominfo_ack_rate":acked/len(pairs),
            "missing_info_ack_count":len(no_info),
            "missing_info_ack_indices":no_info,
            "odominfo_lost_true_count":sum(row["lost"] for row in infos),
            "odominfo_lost_false_count":not_lost,
            "odominfo_inliers_median":float(np.median([q["inliers"] for q in infos])) if infos else None,
            "odominfo_matches_median":float(np.median([q["matches"] for q in infos])) if infos else None,
            "odominfo_inliers_min":min((q["inliers"] for q in infos),default=None),
            "odom_messages":len(odom_rows),
            "numerically_valid_odom":numerical,
            "ack_timeout_s":args.ack_timeout_s,
            "max_replay_rate_hz":args.max_rate_hz,
            "duration_s":time.monotonic()-started,
            "technical_validity_filter":"finite_xyz_quat_cov;0.5<quat_squared_norm<1.5;max_cov_xyz<1000",
            "odominfo_source":"rtabmap_odom/odom_info_lite",
            "success_thresholds_are_not_groundtruth_accuracy":True,
            **matches_gt,
        }
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+"\n")
        args.output.with_suffix(".publish.json").write_text(json.dumps(published,indent=2)+"\n")
        args.output.with_suffix(".odom.json").write_text(json.dumps(odom_rows,indent=2)+"\n")
        args.output.with_suffix(".odominfo.json").write_text(json.dumps(infos,indent=2)+"\n")
        print("A4_REPORT",json.dumps({k:v for k,v in report.items()
                                  if k not in ("missing_info_ack_indices",)},ensure_ascii=False),flush=True)
        if len(published)!=dataset_stats["synchronized_rgbd_pairs"]:
            raise RuntimeError("INCOMPLETE_REPLAY: not all RGB-D pairs were published")
        if acked!=len(pairs):
            raise RuntimeError(f"INCOMPLETE_PROCESSING: {acked}/{len(pairs)} frames acknowledged")
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",type=Path,default=Path("data/tum"))
    p.add_argument("--output",type=Path,default=Path("results/a4/full_sequence.json"))
    p.add_argument("--max-rate-hz",type=float,default=15.0)
    p.add_argument("--ack-timeout-s",type=float,default=5.0)
    p.add_argument("--manifest-only",action="store_true")
    args=p.parse_args()
    if args.max_rate_hz<=0 or args.ack_timeout_s<=0:
        p.error("Invalid pacing or acknowledgement deadline")
    pairs,meta=full_rgbd_pairs(args.data)
    print("FULL_TUM_DATASET",json.dumps(meta),flush=True)
    if args.manifest_only:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(meta,indent=2)+"\n")
        return
    replay(args,pairs,meta)


if __name__=="__main__":
    main()
