#!/usr/bin/env python3
"""A6-L monitoring of actual loop closure, map correction and localization pose.

A pose estimate from /odom alone is NOT relocalization. No initial pose is
supplied in this monitor; it only subscribes to RTAB-Map output.
"""
from __future__ import annotations
import argparse,json,signal,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,default=Path("results/a6loc"))
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    import rclpy
    from rclpy.node import Node
    from rtabmap_msgs.msg import Info
    from geometry_msgs.msg import PoseWithCovarianceStamped
    from tf2_msgs.msg import TFMessage
    rclpy.init()
    node=Node("blind_relocalization_monitor_a6")
    files={name:(args.out/(name+".jsonl")).open("w") for name in ("info","localization_pose","map_odom_tf")}
    counts={name:0 for name in files}
    stopped=False
    def on_stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,on_stop)
    def save(name,obj):
        counts[name]+=1
        files[name].write(json.dumps(obj)+"\n")
        files[name].flush()
    def info(m):
        save("info",{
            "stamp_ns":m.header.stamp.sec*10**9+m.header.stamp.nanosec,
            "reference_node_id":int(m.ref_id),
            "loop_closure_id":int(m.loop_closure_id),
            "proximity_detection_id":int(m.proximity_detection_id),
            "posterior_max":max(m.posterior_values,default=0.),
            "loop_probability":float(dict(zip(m.stats_keys,m.stats_values)).get(
                "Loop/Highest_hypothesis_value",float("nan"))),
            "stats":{k:float(v) for k,v in zip(m.stats_keys,m.stats_values)
                     if ("Localization" in k or "Loop" in k)},
        })
    def pose(m):
        q=m.pose.pose.orientation;v=m.pose.pose.position
        save("localization_pose",{
            "stamp_ns":m.header.stamp.sec*10**9+m.header.stamp.nanosec,
            "frame":m.header.frame_id,
            "xyz":[float(v.x),float(v.y),float(v.z)],
            "quaternion":[float(q.x),float(q.y),float(q.z),float(q.w)],
            "cov_xyz":[float(m.pose.covariance[i]) for i in (0,7,14)],
        })
    def tf(msg):
        for t in msg.transforms:
            if t.header.frame_id.lstrip("/")=="map" and t.child_frame_id.lstrip("/")=="odom":
                q=t.transform.rotation;p=t.transform.translation
                save("map_odom_tf",{
                    "stamp_ns":t.header.stamp.sec*10**9+t.header.stamp.nanosec,
                    "xyz":[float(p.x),float(p.y),float(p.z)],
                    "quaternion":[float(q.x),float(q.y),float(q.z),float(q.w)]})
    subs=[node.create_subscription(Info,"/info",info,20),
          node.create_subscription(PoseWithCovarianceStamped,
                                   "/localization_pose",pose,20),
          node.create_subscription(TFMessage,"/tf",tf,100)]
    started=time.monotonic()
    try:
        while not stopped:
            rclpy.spin_once(node,timeout_sec=.2)
    finally:
        for f in files.values():f.close()
        summary={"events":counts,"monitor_elapsed_s":time.monotonic()-started,
                 "no_initial_pose_published":True,"source":"RTABMap_msgs_and_TF_live"}
        (args.out/"monitor_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
        print("A6_LOC_MONITOR",json.dumps(summary),flush=True)
        node.destroy_node();rclpy.shutdown()

if __name__=="__main__":main()
