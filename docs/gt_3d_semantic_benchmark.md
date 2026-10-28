# 3D semantic GT evaluation and dense color-coded instance mapping

Date: 2026-10-28. All numbers are measured from executed GitHub Actions and
should be interpreted within their clearly stated benchmark scopes.

## A. RGB-D Scenes v2 official 3D semantic GT (14 separate scans)

- Source: https://rgbd-dataset.cs.washington.edu/dataset/rgbd-scenes-v2/
- Authoritative PLY XYZRGB, per-point .label, PLY camera scene pose.
- Each class label is a **semantic category, not a GT physical instance ID**.
  Consequently no physical IDF1, false-split/false-merge or true instance mAP
  can legitimately be derived from this official annotation alone.
- Scene-disjoint split train scenes 1–10, test 11–14.
- 20,000 sampled training points per scene, up to 40,000 per test scene.
- Four controlled ExtraTrees ablations using the same train/test split:

| Features | Foreground semantic 3D mIoU | Point accuracy |
|---|---:|---:|
| RGB | 0.163293 | 0.375481 |
| Geometry | 0.135623 | 0.518100 |
| RGB + geometry | **0.229555** | 0.595169 |
| RGB + geometry + multiscale | 0.213938 | **0.601063** |

- CI result: https://github.com/KiRiII-C/cv_vlm/actions/runs/40752415584
- All per-scene confusion matrices, per-class IoU, 3D centroid/class-union AABB
  metrics, and predicted-vs-GT PLY/HTML are reproducible.
- Code: src/semantic_map/rgbd_scenes_gt.py ; workflow: rgbd-scenes-semantic-gt.yml

## B. Actual RGB-D RGB+Depth+pose+YOLO segmentation → physical 3D map → GT labels

- Frames publicly hosted by 3DMatch: https://3dvision.princeton.edu/projects/2016/3DMatch/downloads/rgbd-datasets/
- Separate GT PLY and labels from the RGB-D Scenes v2 authors.
- Official camera-to-world and calibration from the frame archive.
  **These experiments do not evaluate VO/SLAM pose estimation on scenes 11/14**.
- Actual pretrained YOLO11n-seg on 120 RGB-D frames per scene, all masks projected
  through measured depth and accumulated as 25mm global semantic voxels.
- Each real RGB-D object detection is also associated to existing SQLite physical
  track candidates and 3D colored points. These track UUIDs are **not GT verified**.

| Metric | Scene 11 | Scene 14 |
|---|---:|---:|
| Frames | 120 | 120 |
| Depth samples | 1,673,457 | 1,792,909 |
| Mapped 25mm voxels | 110,177 | 43,896 |
| YOLO masks detected | 441 | 393 |
| Predicted points within 8cm of GT geometry | 99.74% | 99.94% |
| Median surface nearest-distance to GT | 0.01140 m | 0.00999 m |
| All-class point accuracy | 0.81376 | 0.86539 |
| Foreground semantic mIoU | **0.18514** | **0.40787** |
| Confirmed track candidates (NOT GT instances) | 21 | 17 |

Source run: https://github.com/KiRiII-C/cv_vlm/actions/runs/40753853627
Source code: src/semantic_map/annotated_rgbd_frame_eval.py
Workflow: annotated-rgbd-frames.yml

Evaluation precautions:
- 99.74/99.94% measures fraction of **reconstructed map points** within 8cm
  of a reference surface, **not** completeness of GT geometry and not object
  identification accuracy.
- Overall semantic accuracy is strongly affected by background frequency.
- Foreground mIoU is the more informative metric; class 3 cereal_box and class 2
  cap are missed because they are absent from the detector's pretrained COCO
  vocabulary. The source's class 9 is ambiguously sofa/table; not silently
  reinterpreted.
- No physical GT instance IDs: cannot claim duplicate-free ID tracking, stable
  re-identification, object instance 3D mAP or IDF1. No robotic navigation was
  executed on these scenes.

## C. Denser TUM full offline map, colored physical-ID *candidates*

Modified full real TUM 596-frame e2e:
- TSDF voxel size: 6cm → **3cm**; input integration resolution: 320 → **640**.
- Full geometry mesh: 20,532 → **90,058 triangles** (4.39x).
- TSDF surface point cloud: 11,066 → **48,490 points** (4.38x).
- True segmented object mask depth samples fused at **1.5cm voxels**, stride 3:
  **108,562 colored foreground surface points**, 54 confirmed UUID candidates.
- Distinct stable RGB color keyed to physical track UUID; background is gray.
- 3D interactive HTML with selection/tooltip by UUID, points follow mask
  surfaces rather than just 3D centroid markers or rectangles.
- End-to-end workflow run:
  https://github.com/KiRiII-C/cv_vlm/actions/runs/40752675350
- Source: src/semantic_map/instance_cloud.py, dense_scene_viz.py, tum_yolo_pipeline.py

## D. True physical-instance ID evaluation on Hypersim (synthetic, metrically registered)

- Official Apple source: https://github.com/apple/ml-hypersim
- This **second dataset** provides per-pixel real physical instance-ID, NYU-style
  semantic class labels and exact world-space XYZ, unlike RGB-D Scenes v2.
- To avoid the 1.9 TB complete download, the CI reads ONLY selected compressed
  HDF5/RGB members from official per-scene 2.25GB ZIP over HTTP byte ranges.
- Scene ai_001_001, 19 sampled views (frame 36 unavailable), **54 unique physical
  GT IDs**, 272 object-visibility occurrences; GT scene scale 0.0254m/asset unit.
- Controlled inputs (same 3D GT world positions for *all* modes, so this is
  NOT VO/SLAM validation): (i) oracle semantic instance masks, (ii) actual
  pretrained YOLO11n-seg masks; (iii) larger YOLO11s-seg/conf0.15 masks.
- Actual predicted track UUIDs come from our *unchanged* global
  SemanticMapper/Hungarian/appearance+3D association; GT physical IDs are
  used ONLY for evaluation, not as predicted UUIDs. Oracle mode also uses GT
  semantic class for association, explicitly an upper-bound experiment.

| Per physical GT ID benchmark | Oracle GT masks | YOLO11n-seg | YOLO11s-seg conf0.15 |
|---|---:|---:|---:|
| GT instances visible over sample | 54 | 54 | 54 |
| GT instance observations | 272 | 272 | 272 |
| Predicted/matched observations | 272 | 53 | 75 |
| **IDF1** | **0.93050** | **0.22713** | **0.19315** |
| GT objects erroneously split into multiple UUIDs | 3 | 4 | 8 |
| Predicted UUIDs observed on two distinct GT instances | 0 | 0 | 0 |
| GT observation false negatives (IDFN) | 31 | 236 | 241 |

- Reproduced run: https://github.com/KiRiII-C/cv_vlm/actions/runs/40756179656
- Implementation: src/semantic_map/hypersim_range.py,
  src/semantic_map/hypersim_instance_eval.py,
  tests/test_hypersim_physical_identity.py.
- **Acceptance: FAIL for direct RGB+Depth YOLO-based physical object tracking.**
  The pretrained detector's low detection coverage and model vocabulary
  dominate errors, despite much better association under oracle masks.
  Larger model + lower threshold **did not improve** actual IDF1.
  Neither zero observed false merges nor a high oracle benchmark is proof
  of duplicate-free behavior on independent real-world scenes.

## Acceptance status

**PASS for these restricted measurements**: actual 3D semantic per-point
evaluation against official annotations, denser actual RGB-D TSDF geometry
production, UUID-colored per-object point-cloud export, and an end-to-end
two-scene 3D semantic GT evaluation.

**NOT PASSED**: robust unique physical object identity under return/occlusion
on a dataset with physical GT IDs, fully validated 3D instance AP, and real
robot Nav2 navigation. No "best final scientific architecture" claim.

Hypersim was successfully evaluated on a selective 19-frame sampled subset; model IDF1 is insufficient (see section D). A multi-scene, real-robot and GT instance-level end-to-end benchmark is still required. ShanghaiTech SSS data probe could not download its Google-Drive assets through CI; no fabricated SSS metrics.
