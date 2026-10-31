# Stage A0: Docker/ROS 2 Jazzy geometry baseline

## Objective

Establish a reproducible, CPU-only execution environment for the first RTAB-Map
integration experiments and quantify the isolated computational cost of
duplicate RGB-D depth-to-3D projection versus sharing an organized cloud.

## Reproducibility

GitHub Actions workflow: `.github/workflows/rtabmap-geometry-a0.yml`.
Base image: existing repository `Dockerfile` (`ros:jazzy-ros-base` plus
`ros-jazzy-rtabmap-ros` and Python dependencies).
The workflow builds this image with Docker on `ubuntu-24.04`, checks the ROS
package installation and runs `scripts/benchmark_geometry_reuse.py` inside
the container. Reports and logs are uploaded as Actions artifacts.

The script uses deterministic **synthetic** 640x480 depth and two masks, with
same pixel grid and exact pinhole camera intrinsics in both arms:

- **duplicate**: create an organized XYZ cloud twice for one depth image.
- **shared**: create an organized XYZ cloud once and let both consumers index it.

Both outputs are checked for exact geometry equivalence (within 1e-6 m).
Timing arms alternate execution order after warmup; comparisons use medians.
Reprojection of a known 0.1 m camera-frame translation is checked independently.

## Claims the workflow cannot establish

This A0 check **does not** execute RGB-D SLAM, produce an RTAB-Map database,
assess odometry drift, measure RTAB-Map's internal cloud allocation, map-update
correctness, loop closure, object association, or GPU inference. Its timings
are synthetic CPU microbenchmark numbers for **our candidate shared geometry
pipeline**, not RTAB-Map performance estimates.

The next experiments must:
1. Run RTAB-Map RGB-D on a recorded TUM sequence with clocked RGB/Depth and
   camera calibration; save its database and keyframe/odometry diagnostics.
2. Compare raw RGB-D organized clouds, `rtabmap_util` clouds and map clouds.
   Verify pixel-to-point indexing and any filtering/downsampling.
3. Profile actual repeated projection and ROS transport/copies in standard
   nodes versus composed components; distinguish CPU and memory overhead.
4. Evaluate camera pose accuracy with ATE/RPE against ground truth, and publish
   a report explicitly indicating when loop closure could not be tested.
5. Only then compare modifying RTAB-Map internals against using existing APIs.

## Run on a local Docker host

```bash
docker build -t cv-vlm:jazzy-a0 .
docker run --rm -v "$PWD:/workspace" -w /workspace cv-vlm:jazzy-a0 \
  bash -lc 'source /opt/ros/jazzy/setup.bash; ros2 pkg prefix rtabmap_slam; python3 scripts/benchmark_geometry_reuse.py --iterations 20 --output results/rtabmap_geometry_a0.json'
```

No NVIDIA GPU is required for A0. GPU experiments on the RTX 4060 require a
separate NVIDIA-enabled self-hosted runner; the standard GitHub-hosted runner
must not be reported as a GPU benchmark.
