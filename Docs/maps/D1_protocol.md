# D1 — Cross-dataset test: real measured 3D maps with YOLOE-26x

**Branch:** `experiments/rtabmap-geometry-a0-20261031`.

The goal is to **build real image-derived maps**, not synthetic pictures or unexecuted diagrams, for:

1. TUM RGB-D Freiburg1 xyz (real physical camera motion, largely static);
2. TUM RGB-D Freiburg1 desk2 (a different real traversal of laboratory desks);
3. Bonn RGB-D moving_obstructing_box (real dynamic scene and a different RGB-D camera).

Each workflow job downloads the **original RGB+depth and original GT/mocap pose timestamp indexes**. The input is actual sensor data. TUM camera extrinsics come from the public Freiburg1 camera-pose ground truth; Bonn uses the previously audited mocap marker-to-camera extrinsic. Neither the model nor the 3D map algorithm receives synthetic observations.

**Important distinction from original A6/B7:** these three NEW maps are reconstructed in the GT-pose frame, to isolate the sensor-to-geometry and YOLOE semantic projections from SLAM pose drift. They are **offline oracle-registered maps, not successful RTAB-Map trials**. Original A6/B7 TUM desk map remains an RTAB-Map-pose baseline with no injected GT.

## Controlled procedure

- Enumerate all source timestamps and require real synchronized RGB, depth and camera GT (within 45 ms); no invented camera pose.
- Geometric keyframe every fourth matched frame; valid 16-bit depth (0.25–5 m) sampled at 4-pixel intervals, 3 cm spatial voxels, original RGB averaged within voxel.
- YOLOE-26x segmentation every twentieth matched frame (CPU); 12 fixed text prompts from the repository; confidence threshold 0.40.
- Semantic 3D points are **only** raw measured depth pixels passing the mask and local median-depth filtering. Maximum 3000 original depth points per valid mask, 2 cm voxels per class.
- No full-frame geometry synthesis, unknown-area fill, artificial upsampling or 3D GT semantic labels. Real RGB geometry is saved separately from YOLOE predicted colored surfaces.
- Each dataset exports `geometry_rgb.ply`, `semantic_predicted.ply`, `interactive_map.html`, `map_overview.png`, `metrics.json`. Absence of any target class is a legitimate negative result — do not hallucinate semantics or fail the geometric-map construction.
- The unit tests check calibration rigidity, valid-depth-only back-projection, actual timestamp matching, voxel averaging and PLY format. These toy-array software tests are not measured scientific results.
- Matrix CI only consolidates when all three real-data jobs are completed; missing data/maps/frames or a provenance mismatch fails closed. Real metrics and map PNGs will be stored in [D1_cross_dataset_results.md](D1_cross_dataset_results.md).

## Required scientific interpretation

TUM xyz/desk2 are two sequences in one dataset; Bonn is another sensor dataset. Dataset sample count, masked-3D observations, semantic voxel count and coverage are **descriptive engineering metrics only**. They are not semantic IoU, precision, object IDF1, change F1 or SLAM ATE. Do not assume all mapped surfaces are static in Bonn. XY is not a valid Nav2 costmap without floor alignment and a verified obstacle policy.

The next experiment for apples-to-apples localization would run actual RTAB-Map on each recording **without** GT camera poses, then compare estimates to GT posthoc. Only that could test ATE, map pose consistency and relocalization across datasets.
