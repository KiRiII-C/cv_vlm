#!/usr/bin/env python3
"""A0: reproducible CPU microbenchmark for RGB-D back-projection reuse.

This benchmark uses deterministic SYNTHETIC depth and masks. It does NOT run
RTAB-Map odometry, loop closure, TSDF or object re-identification.
"""
import argparse
import json
import os
import platform
import statistics
import time
from pathlib import Path

import numpy as np


def backproject(depth, u, v, fx, fy, cx, cy):
    """Organized XYZ cloud; invalid depths are NaN, preserving pixel indexing."""
    xyz = np.stack(((u - cx) * depth / fx,
                    (v - cy) * depth / fy,
                    depth), axis=-1)
    xyz[~np.isfinite(depth) | (depth <= 0)] = np.nan
    return xyz


def select_points(xyz, mask):
    return xyz[mask & np.isfinite(xyz[..., 2])]


def run(iterations, output):
    height, width = 480, 640
    fx, fy, cx, cy = 525.0, 525.0, 319.5, 239.5
    u, v = np.meshgrid(np.arange(width, dtype=np.float32),
                       np.arange(height, dtype=np.float32))
    depth = (1.5 + 0.0003 * u + 0.001 * v).astype(np.float32)
    depth[((u + v) % 53) == 0] = 0
    mask_1 = (((u - 200) / 95) ** 2 + ((v - 210) / 100) ** 2) <= 1
    mask_2 = (u >= 310) & (u <= 540) & (v >= 125) & (v <= 350)

    def duplicate():
        # Simulates two consumers each independently lifting the same RGB-D frame.
        cloud_mapping = backproject(depth, u, v, fx, fy, cx, cy)
        cloud_instance = backproject(depth, u, v, fx, fy, cx, cy)
        return select_points(cloud_mapping, mask_1), select_points(cloud_instance, mask_2)

    def shared():
        # Exactly one organized back-projection is shared by two mask consumers.
        cloud = backproject(depth, u, v, fx, fy, cx, cy)
        return select_points(cloud, mask_1), select_points(cloud, mask_2)

    ref = duplicate()
    reused = shared()
    max_abs_error = max(float(np.max(np.abs(a-b))) for a,b in zip(ref, reused))
    assert ref[0].shape == reused[0].shape
    assert ref[1].shape == reused[1].shape
    assert max_abs_error < 1e-6

    # Verify pinhole reprojection after a known camera-frame translation.
    p = np.array([0., 0., 2.], dtype=np.float64)
    p_new = p + np.array([0.1, 0., 0.], dtype=np.float64)
    projected_u = fx * p_new[0] / p_new[2] + cx
    assert abs(projected_u - (cx + 26.25)) < 1e-10

    timings = {"duplicate_ms": [], "shared_ms": []}
    for i in range(iterations + 3):
        order = ("duplicate", "shared") if i % 2 == 0 else ("shared", "duplicate")
        for name in order:
            start = time.perf_counter_ns()
            result = duplicate() if name == "duplicate" else shared()
            elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000.0
            assert result[0].shape == ref[0].shape
            if i >= 3:
                timings[f"{name}_ms"].append(elapsed_ms)

    dup = statistics.median(timings["duplicate_ms"])
    shr = statistics.median(timings["shared_ms"])
    report = {
        "experiment": "A0_shared_organized_rgbd_cloud_cpu",
        "status": "SYNTHETIC_MICROBENCHMARK_ONLY",
        "rtabmap_slam_evaluated": False,
        "real_dataset_used": False,
        "pose_graph_evaluated": False,
        "github_sha": os.environ.get("GITHUB_SHA", "unknown"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "image_width": width,
        "image_height": height,
        "iterations_per_arm": iterations,
        "valid_mask_points": [int(len(x)) for x in reused],
        "duplicate_backprojections_per_frame": 2,
        "shared_backprojections_per_frame": 1,
        "max_absolute_xyz_error_m": max_abs_error,
        "reprojection_known_translation_error_px": abs(projected_u - (cx + 26.25)),
        "duplicate_median_ms": dup,
        "shared_median_ms": shr,
        "relative_speedup": dup / shr if shr > 0 else None,
        "duplicate_raw_ms": timings["duplicate_ms"],
        "shared_raw_ms": timings["shared_ms"],
        "interpretation": "CPU cost of two identical geometry consumers, not RTAB-Map internal work",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if not k.endswith("_raw_ms")},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=20)
    ap.add_argument("--output", type=Path, default=Path("results/rtabmap_geometry_a0.json"))
    args = ap.parse_args()
    if args.iterations < 3:
        ap.error("--iterations must be >= 3")
    run(args.iterations, args.output)
