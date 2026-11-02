#!/usr/bin/env python3
"""A1: replay real TUM RGB-D through RTAB-Map; GT is strictly evaluation-only."""
from __future__ import annotations
import argparse, json, math, time
from pathlib import Path
import cv2
import numpy as np
from semantic_map.tum_yolo_pipeline import download_scene, synchronized


def dataset_pairs(folder, frames, every):
    pairs=synchronized(download_scene(folder))[::every][:frames]
    if len(pairs)<30:
        raise RuntimeError("Too few paired real TUM frames")
    return pairs


def evaluate(odometry, published, pairs):
    """Rigid SE(3) ATE, measured only after RTAB-Map pose estimation."""
    if not published:
        return {"matched_gt_frames":0, "ATE_RMSE_m":None, "ATE_median_m":None}
    ts=np.asarray([x["ros_time"] for x in published])
    a=[]; b=[]; used=set()
    for o in odometry:
        if not o["valid"]:
            continue
        i=int(np.argmin(np.abs(ts-o["timestamp"])))
        if abs(ts[i]-o["timestamp"])>0.05 or i in used:
            continue
        used.add(i)
        a.append(o["translation_xyz"])
        b.append(pairs[published[i]["idx"]][3][:3,3])
    if len(a)<3:
        return {"matched_gt_frames":len(a), "ATE_RMSE_m":None,"ATE_median_m":None}
    a=np.asarray(a,dtype=float); b=np.asarray(b,dtype=float)
    ma=a.mean(axis=0); mb=b.mean(axis=0)
    U,_,Vt=np.linalg.svd((a-ma).T@(b-mb))
    R=Vt.T@np.diag([1.,1.,np.linalg.det(U@Vt)])@U.T
    e=np.linalg.norm((a-ma)@R.T+mb-b,axis=1)
    return {"matched_gt_frames":int(len(e)),
            "ATE_RMSE_m":float(np.sqrt(np.mean(e**2))),
            "ATE_median_m":float(np.median(e)),
            "GT_used_for_estimation":False, "alignment":"SE3_no_scale"}


def replay(args, pairs):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Image, CameraInfo
    from nav_msgs.msg import Odometry
    from rtabmap_msgs.msg import RGBDImage

    rclpy.init()
    node=Node("a1_tum_rgbd_atomic_replay")
    qos=QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
    pub=node.create_publisher(RGBDImage,"/camera/rgbd_image",qos)
    odom=[]; published=[]

    def odom_cb(m):
        p=m.pose.pose.position; q=m.pose.pose.orientation
        xyz=[float(p.x),float(p.y),float(p.z)]
        quat=[float(q.x),float(q.y),float(q.z),float(q.w)]
        cov=[float(m.pose.covariance[i]) for i in (0,7,14)]
        good=(all(math.isfinite(v) for v in xyz+quat+cov)
              and .5<sum(v*v for v in quat)<1.5 and max(cov)<1000)
        odom.append({"timestamp":m.header.stamp.sec+m.header.stamp.nanosec*1e-9,
                     "translation_xyz":xyz,"quaternion_xyzw":quat,
                     "cov_xyz_diagonal":cov, "valid":bool(good)})

    odom_sub=node.create_subscription(Odometry,"/odom",odom_cb,50)
    for _ in range(300):
        rclpy.spin_once(node,timeout_sec=0.1)
        if pub.get_subscription_count()>=2:
            break
    count=pub.get_subscription_count()
    if count<2:
        node.destroy_node();rclpy.shutdown()
        raise RuntimeError(f"Expected two RGBDImage subscribers, got {count}")
    start=time.monotonic()
    for i,(tum_stamp,rgb_path,depth_path,_) in enumerate(pairs):
        rgb=cv2.imread(str(rgb_path),cv2.IMREAD_COLOR)
        depth=cv2.imread(str(depth_path),cv2.IMREAD_UNCHANGED)
        if rgb is None or depth is None or rgb.shape[:2]!=depth.shape or depth.dtype!=np.uint16:
            raise RuntimeError(f"Invalid TUM RGB-D frame {i}")
        h,w=depth.shape
        # TUM raw depths are metres*5000, whereas ROS 16UC1 is millimetres.
        depth_mm=np.clip(np.rint(depth.astype(np.float32)/5.),0,65535).astype(np.uint16)
        # Live ROS time avoids giant 2011->2026 delays in RTAB-Map/TF;
        # retain source times only for post-hoc ground-truth correspondence.
        stamp=node.get_clock().now().to_msg()
        cam=CameraInfo()
        cam.header.stamp=stamp;cam.header.frame_id="camera_color_optical_frame"
        cam.width=w;cam.height=h;cam.distortion_model="plumb_bob"
        cam.d=[0.,0.,0.,0.,0.]
        cam.k=[517.3,0.,318.6,0.,516.5,255.3,0.,0.,1.]
        cam.r=[1.,0.,0.,0.,1.,0.,0.,0.,1.]
        cam.p=[517.3,0.,318.6,0.,0.,516.5,255.3,0.,0.,0.,1.,0.]
        # Compress before DDS to prevent 1.5MB raw sensor bursts.
        # JPEG quality 95 retains visual features; depth PNG is LOSSLESS.
        ok_rgb, jpeg=cv2.imencode(".jpg",rgb,[cv2.IMWRITE_JPEG_QUALITY,95])
        ok_depth, png=cv2.imencode(".png",depth_mm,[cv2.IMWRITE_PNG_COMPRESSION,1])
        if not ok_rgb or not ok_depth:
            raise RuntimeError(f"RGB-D compression failed at frame {i}")
        packet=RGBDImage()
        packet.header=cam.header
        packet.rgb_camera_info=cam
        packet.depth_camera_info=cam
        packet.rgb_compressed.header=cam.header
        packet.rgb_compressed.format="bgr8; jpeg compressed bgr8"
        packet.rgb_compressed.data=jpeg.tobytes()
        packet.depth_compressed.header=cam.header
        # RTAB-Map legacy PNG depth payload is a standard 16-bit grayscale PNG.
        packet.depth_compressed.format="png"
        packet.depth_compressed.data=png.tobytes()
        pub.publish(packet)
        published.append({"idx":i,"ros_time":stamp.sec+stamp.nanosec*1e-9,
                          "source_tum_timestamp":float(tum_stamp)})
        until=time.monotonic()+1./args.rate
        while time.monotonic()<until:
            rclpy.spin_once(node,timeout_sec=0.02)
        if (i+1)%25==0:
            print(f"REAL_TUM_PUBLISHED={i+1} ODOM_RECEIVED={len(odom)} "
                  f"VALID_ODOM={sum(x['valid'] for x in odom)}",flush=True)
    for _ in range(150):
        rclpy.spin_once(node,timeout_sec=0.1)
    metrics=evaluate(odom,published,pairs)
    valid=[o for o in odom if o["valid"]]
    spread=float(np.linalg.norm(np.ptp(
        np.asarray([o["translation_xyz"] for o in valid]),axis=0))) if valid else 0.
    report={"experiment":"A1_real_TUM_RTABMap_RGBD_odometry_and_SLAM",
      "input":"real_TUM_freiburg1_desk_RGBD",
      "pose_estimator":"rtabmap_odom/rgbd_odometry",
      "mapper":"rtabmap_slam/rtabmap",
      "RGBD_transport":"rtabmap_msgs/RGBDImage_one_atomic_message",
      "RGBD_compression":"JPEG95_RGB_and_lossless_PNG_depth",
      "depth_encoding":"16UC1_mm_converted_from_TUM_5000",
      "ROS_header_time":"live_replay",
      "GT_used_as_input":False,"GT_only_for_metrics":True,
      "published_frames":len(pairs),"odometry_messages":len(odom),
      "valid_odometry_messages":len(valid),
      "valid_odom_translation_spread_m":spread,
      "publish_rate_hz":args.rate,
      "wall_elapsed_s":time.monotonic()-start,
      "rgbd_subscriber_count_initial":count,
      "loop_closure_verified":False,
      "semantic_identity_verified":False,
      "localization_after_restart_verified":False,**metrics}
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+"\n")
    output.with_suffix(".odom.json").write_text(json.dumps(odom,indent=2)+"\n")
    output.with_suffix(".publish.json").write_text(json.dumps(published,indent=2)+"\n")
    print("A1_REPORT",json.dumps(report,ensure_ascii=False),flush=True)
    node.destroy_node();rclpy.shutdown()
    if len(valid)<args.min_odom or metrics["matched_gt_frames"]<args.min_odom:
        raise RuntimeError(f"Insufficient valid REAL odometry: {len(valid)}, GT matches={metrics['matched_gt_frames']}")
    if not math.isfinite(metrics["ATE_RMSE_m"]) or spread<0.02:
        raise RuntimeError(f"Degenerate real odometry: spread={spread}, ATE={metrics['ATE_RMSE_m']}")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",type=Path,default=Path("data/tum"))
    p.add_argument("--frames",type=int,default=150)
    p.add_argument("--every",type=int,default=3)
    p.add_argument("--rate",type=float,default=2.)
    p.add_argument("--min-odom",type=int,default=20)
    p.add_argument("--output",default="results/a1/rtabmap_real_tum.json")
    p.add_argument("--download-only",action="store_true")
    a=p.parse_args()
    if a.frames<30 or a.every<1 or a.rate<=0:p.error("Invalid arguments")
    pairs=dataset_pairs(a.data,a.frames,a.every)
    print(f"REAL_TUM_SYNCED_PAIRS={len(pairs)} FIRST={pairs[0][0]} LAST={pairs[-1][0]}",flush=True)
    if not a.download_only:
        replay(a,pairs)


if __name__=="__main__":
    main()
