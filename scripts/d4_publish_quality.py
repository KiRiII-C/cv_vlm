#!/usr/bin/env python3
"""Publish D4 scientific facts only after verifying all original-source outputs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--results",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--run",required=True,type=int)
    a=p.parse_args()
    x=json.loads((a.results/"d4_metrics.json").read_text())
    d=x["held_out_raycast_reprojection"]
    g=x["independent_gt_mesh_surface"]
    m=x["mesh"]
    if x["full_source_frames_available"]!=2000 or x["actual_original_frames_integrated"]!=500:
        raise ValueError("D4 full original source not confirmed")
    if x["actual_YOLOE_inferences"]!=200 or d["heldout_original_frames"]!=25:
        raise ValueError("D4 incorrect independent test frame count")
    if x["RTABMap_run"] is not False or x["physical_instance_accuracy_evaluated"] is not False:
        raise ValueError("D4 source falsely asserts SLAM or physical-ID evidence")
    if m["full_mesh_faces"]<100000:
        raise ValueError("D4 scene reconstruction incomplete")
    if g["official_independent_GT_mesh_vertices"]<5000:
        raise ValueError("D4 GT mesh not actually used")
    if not 0<=d["raycast_depth_support_fraction"]<=1:
        raise ValueError("D4 impossible raycast coverage")
    for filename,size in (
      ("rgb_mesh.ply",20000),("semantic_mesh.ply",20000),
      ("rgb_mesh.glb",20000),("semantic_mesh.glb",20000),
      ("mesh_patch_and_semantic_labels.npz",20000),
      ("interactive_office0.html",20000),
      ("replica_office0_three_modes.png",30000),
      ("d4_camera_aligned_comparison.jpg",20000),
      ("d4_comparison_small.jpg",12000)):
        f=a.results/filename
        if not f.is_file() or f.stat().st_size<size:
            raise RuntimeError(f"Missing or truncated original map artifact {filename}")
    a.out.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(a.results/"d4_comparison_small.jpg",
                    a.out/"D4_office0_camera_aligned.jpg")
    (a.out/"D4_office0_metrics.json").write_text(
      json.dumps(x,ensure_ascii=False,indent=2)+"\n")
    runurl="https://github.com/KiRiII-C/cv_vlm/actions/runs/"+str(a.run)
    lines=[
      "# D4 — full official Replica office0: real reconstruction quality audit",
      "",
      "Status: PASS. The dataset is computer-rendered Replica, not physical RGB-D camera data.",
      "Actual source 2000 frames; 500 RGB-D TSDF camera frames; 200 independent YOLOE-26x inference calls.",
      "Camera poses are provided GT oracle. No RTAB-Map SLAM ran during this reconstruction.",
      "",
      "![Same-camera: original / reconstructed / geometry patches / YOLOE classes](D4_office0_camera_aligned.jpg)",
      "",
      "| Quantity | Verified outcome |",
      "|---|---:|",
      f"| Total original available frames | {x['full_source_frames_available']} |",
      f"| Original frame integrations into mesh | {x['actual_original_frames_integrated']} |",
      f"| YOLOE26x inference frames | {x['actual_YOLOE_inferences']} |",
      f"| Valid measured-depth semantic masks | {x['actual_valid_3D_YOLOE_masks']} |",
      f"| Original mesh triangle faces | {m['full_mesh_faces']} |",
      f"| Geometry patch mesh-face coverage | {100*x['mesh_face_geometric_region_coverage']:.2f}% |",
      f"| Semantic support mesh vertices | {100*m['semantic_supported_fraction']:.2f}% |",
      f"| Held-out original depth frames excluded from TSDF | {d['heldout_original_frames']} |",
      f"| Raycast coverage of valid held-out depth | {100*d['raycast_depth_support_fraction']:.2f}% |",
      f"| Supported ray median abs depth residual | {d['median_abs_sensor_depth_error_m_on_supported']:.4f} m |",
      f"| Supported ray p95 abs depth residual | {d['p95_abs_sensor_depth_error_m_on_supported']:.4f} m |",
      f"| Independent GT mesh one-way distance median | {g['reconstructed_to_independent_GT_triangle_distance_median']:.4f} m |",
      f"| Independent GT mesh one-way distance p95 | {g['reconstructed_to_independent_GT_triangle_distance_p95']:.4f} m |",
      "",
      "Interpretation: raycast support detects missing or perforated geometry in the reconstructed map.",
      "Depth residuals are evaluated only on raycast-supported pixels; both support and residuals must be reported.",
      "One-way GT triangle distance evaluates reconstructed points to the independent GT surface; it does not measure completeness of unseen areas.",
      "Geometric patches are not physical-instance labels; YOLOE colors are predictions, not GT.",
      "No physical UUID IDF1, instance IoU, open-set VLM mAP, RTAB-Map ATE or navigation validation.",
      "",
      "Full original measured-depth scene mesh PLY/GLB, aligned RGB/semantic renders,",
      f"interactive HTML and per-view metrics: [Actions run {a.run}]({runurl}),",
      "artifact D4-FULL-2000-REPLICA-500-RGBD-TSDF-QUALITY.",
      ""
    ]
    (a.out/"D4_full_office0_quality_results.md").write_text("\n".join(lines))
    print("D4_FULL_OFFICE0_PUBLISH_CHECK_PASS",{
      "mesh_faces":m["full_mesh_faces"],
      "raycast_coverage":d["raycast_depth_support_fraction"],
      "GT_surface_p95":g["reconstructed_to_independent_GT_triangle_distance_p95"]},flush=True)


if __name__=="__main__":
    main()
