#!/usr/bin/env python3
"""A3: query live RTAB-Map pose graph in BOTH unoptimized and optimized modes.

The two GetMap calls are independent of TUM ground truth and the RGB-D
replay's saved odometry. Node stamps and pose IDs are resolved against the
persisted RTAB-Map SQLite database during a separate offline step.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def pose_dict(pose):
    p = pose.position
    q = pose.orientation
    values = [p.x, p.y, p.z, q.x, q.y, q.z, q.w]
    if not all(math.isfinite(float(v)) for v in values):
        raise ValueError("RTAB-Map graph contains non-finite pose")
    norm = sum(float(v) * float(v) for v in (q.x, q.y, q.z, q.w))
    if not 0.5 < norm < 1.5:
        raise ValueError("RTAB-Map graph contains invalid quaternion")
    return {
        "xyz": [float(p.x), float(p.y), float(p.z)],
        "xyzw": [float(q.x), float(q.y), float(q.z), float(q.w)],
    }


def graph_dict(msg):
    graph = msg.graph
    if len(graph.poses_id) != len(graph.poses):
        raise ValueError("Graph ID/pose vector length mismatch")
    poses = {}
    for id_, pose in zip(graph.poses_id, graph.poses):
        id_ = int(id_)
        if id_ <= 0 or str(id_) in poses:
            raise ValueError(f"Invalid or duplicate graph node {id_}")
        poses[str(id_)] = pose_dict(pose)
    links = [
        {"from_id": int(link.from_id),
         "to_id": int(link.to_id),
         "type": int(link.type)}
        for link in graph.links
    ]
    return {"frame_id": str(msg.header.frame_id), "poses": poses,
            "links": links, "pose_count": len(poses),
            "link_count": len(links)}


def request_graph(node, service, optimized, timeout_s):
    from rtabmap_msgs.srv import GetMap
    req = GetMap.Request()
    req.global_map = True
    req.optimized = bool(optimized)
    req.graph_only = True
    future = service.call_async(req)
    import rclpy
    rclpy.spin_until_future_complete(node, future, timeout_sec=timeout_s)
    if not future.done():
        raise TimeoutError(f"GetMap timed out, optimized={optimized}")
    response = future.result()
    if response is None:
        raise RuntimeError(f"GetMap returned no response, optimized={optimized}")
    return graph_dict(response.data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--service", default="/rtabmap/get_map_data")
    parser.add_argument("--output", type=Path,
                        default=Path("results/a3/pose_graph_service.json"))
    parser.add_argument("--timeout", type=float, default=40.0)
    parser.add_argument("--min-nodes", type=int, default=8)
    args = parser.parse_args()

    import rclpy
    from rclpy.node import Node
    from rtabmap_msgs.srv import GetMap
    rclpy.init()
    node = Node("a3_rtabmap_graph_exporter")
    try:
        client = node.create_client(GetMap, args.service)
        if not client.wait_for_service(timeout_sec=args.timeout):
            raise TimeoutError(f"Service unavailable: {args.service}")
        # Capture both graph versions after replay but before shutting down SLAM.
        raw = request_graph(node, client, False, args.timeout)
        optimized = request_graph(node, client, True, args.timeout)
        if raw["pose_count"] < args.min_nodes or optimized["pose_count"] < args.min_nodes:
            raise RuntimeError("RTAB-Map graph contains too few nodes for A3")
        if raw["frame_id"] != optimized["frame_id"]:
            raise RuntimeError("Unoptimized and optimized graph frames differ")
        record = {
            "source": "live_rtabmap_ros_GetMap",
            "service": args.service,
            "global_map": True, "graph_only": True,
            "GT_used_as_input": False,
            "poses_are_keyframes_only": True,
            "unoptimized": raw, "optimized": optimized,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2) + "\n")
        print("A3_GRAPH_EXPORT", json.dumps({
            "unoptimized_nodes": raw["pose_count"],
            "optimized_nodes": optimized["pose_count"],
            "unoptimized_links": raw["link_count"],
            "optimized_links": optimized["link_count"],
            "frame_id": raw["frame_id"],
        }), flush=True)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
