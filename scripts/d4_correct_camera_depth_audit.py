#!/usr/bin/env python3
"""D4-M1: repair camera-z audit from an ALREADY RECONSTRUCTED REAL Replica map.

Original D4 computed z = Pw dot R[:,2] + tz. Correct world-to-camera
projection is z = Pw dot R[2,:] + tz. The original RGB mesh and model
outputs are unchanged, so do NOT train, generate or re-render YOLOE.
Downloads only 25 withheld original depth+RGB pairs, trajectory and official
GT scene mesh. The original bugged metric is preserved for traceability.
"""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
import numpy as np
from remotezip import RemoteZip
from d4_extract_full_replica import URL, indices
from d4_full_office0_map import (
    read_inputs,raycast_measured_quality,gt_mesh_oneway_distance,
)


def repair(args):
    import open3d as o3d
    oldpath=next(iter(sorted(args.original.rglob("d4_metrics.json"))),None)
    if oldpath is None:raise FileNotFoundError("D4 original verified metrics unavailable")
    source_folder=oldpath.parent
    original=json.loads(oldpath.read_text())
    if original.get("actual_original_frames_integrated")!=500 or (
         original.get("actual_YOLOE_inferences")!=200):
        raise RuntimeError("D4 source experiment provenance incorrect")
    if original.get("held_out_raycast_reprojection",{}).get("heldout_original_frames")!=25:
        raise RuntimeError("Expected exact original audit failure source")
    ply=source_folder/"rgb_mesh.ply"
    if not ply.is_file():raise RuntimeError("Original full reconstructed RGB triangle mesh unavailable")
    mesh=o3d.io.read_triangle_mesh(str(ply))
    if len(mesh.triangles)!=original["mesh"]["full_mesh_faces"]:
        raise RuntimeError("TSDF triangle count differs from original experiment")
    frames,geometry,yolo,heldout=indices()
    original_depth_source=args.source/"Replica"/"office0"
    files=[
      "Replica/cam_params.json","Replica/office0/traj.txt","Replica/office0_mesh.ply"
    ]
    for frame in sorted(heldout):
        files.extend([
         f"Replica/office0/results/frame{frame:06}.jpg",
         f"Replica/office0/results/depth{frame:06}.png"])
    with RemoteZip(URL,initial_buffer_size=1048576,timeout=180) as z:
        all_files={f.filename:f.file_size for f in z.infolist()}
        for i,name in enumerate(files):
            if name not in all_files:raise RuntimeError("Missing original withheld source "+name)
            dest=args.source/name
            dest.parent.mkdir(parents=True,exist_ok=True)
            payload=z.read(name)
            if len(payload)!=all_files[name]:
                raise RuntimeError("Incomplete original measurement "+name)
            dest.write_bytes(payload)
            if i%20==0:print("D4_RAYCAST_METRIC_REPAIR_SOURCE",i,len(files),flush=True)
    src,gt,K,poses,scale,meta=read_partial_source(args.source,original,heldout)
    new=raycast_measured_quality(mesh,src,sorted(heldout),
                                poses,K,scale,stride=8,max_depth=7.)
    old=original["held_out_raycast_reprojection"]
    if not (new["heldout_original_frames"]==25 and
            new["observed_valid_depth_samples"]==old["observed_valid_depth_samples"]):
        raise RuntimeError("Repair changed source measured depth population")
    # The raw raycast hit triangle set is constant, but VALID camera-depth
    # support also requires corrected optical z > 0.25m. Fixing R[2,:] rather
    # than R[:,2] is allowed to change this valid-support population.
    if not 0 < new["raycast_depth_support_fraction"] <= 1:
        raise RuntimeError("Corrected raycast source coverage is invalid")
    if not np.isfinite(new["median_abs_sensor_depth_error_m_on_supported"]) or (
            not np.isfinite(new["p95_abs_sensor_depth_error_m_on_supported"])):
        raise RuntimeError("Corrected optical camera-depth metrics invalid")
    corrected_support_delta=(new["raycast_depth_support_fraction"]-
                              old["raycast_depth_support_fraction"])
    print("D4_M1_CAMERA_Z_VALIDITY_CHANGED_SUPPORT",
          {"old":old["raycast_depth_support_fraction"],
           "new":new["raycast_depth_support_fraction"],
           "change_pp":100*corrected_support_delta},flush=True)
    new_gt=gt_mesh_oneway_distance(mesh,gt)
    if abs(new_gt["reconstructed_to_independent_GT_triangle_distance_median"]-
           original["independent_gt_mesh_surface"][
            "reconstructed_to_independent_GT_triangle_distance_median"])>1e-6:
        raise RuntimeError("Original independent mesh source changed")
    new["metric_revision"]="D4-M1_corrected_camera_R_row_2_not_column_2"
    original["held_out_raycast_reprojection"]=new
    original["metric_revision"]="D4-M1_fixed_camera_z_depth_projection"
    original["support_fraction_change_due_to_corrected_z_validity"]=float(corrected_support_delta)
    original["superseded_metric_error_m"]={
      "median":old["median_abs_sensor_depth_error_m_on_supported"],
      "p95":old["p95_abs_sensor_depth_error_m_on_supported"],
      "reason":"Original D4 used R[:,2] instead of R[2,:] to compute optical Z in world-to-camera transform"
    }
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/"D4_office0_metrics.json").write_text(json.dumps(original,indent=2,ensure_ascii=False)+"\n")
    shutil.copyfile(source_folder/"d4_comparison_small.jpg",
                    args.out/"D4_office0_camera_aligned.jpg")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    vals=np.asarray([v["median_z_abs_error_m"] for v in new["per_view"]])
    covers=np.asarray([v["fraction_depth_supported"] for v in new["per_view"]])
    fig,axes=plt.subplots(1,2,figsize=(12,4.2))
    axes[0].plot(np.arange(len(vals)),vals,marker="o",markersize=3)
    axes[0].set_title("Corrected median optical-depth residual by view")
    axes[0].set_ylabel("absolute z difference, m")
    axes[0].set_xlabel("held-out view index")
    axes[1].plot(np.arange(len(covers)),covers,marker="o",markersize=3)
    axes[1].set_title("Raycast coverage by held-out view")
    axes[1].set_xlabel("held-out view index")
    axes[1].set_ylabel("valid camera depth fraction")
    axes[1].set_ylim(0,1.05)
    for ax in axes:ax.grid(alpha=.15)
    fig.tight_layout()
    fig.savefig(args.out/"D4_corrected_raycast_diagnostics.png",dpi=140)
    plt.close(fig)
    run=args.run
    lines=[
      "# D4 — corrected 500-frame official Replica office0 TSDF reconstruction audit",
      "",
      f"**Original reconstruction:** [GitHub Actions run {run}](https://github.com/KiRiII-C/cv_vlm/actions/runs/{run}).",
      "**This correction reuses the exact original PLY/GLB and original YOLOE results.** "
      "Only the camera-z depth metric from a world-to-camera transform is recalculated.",
      "",
      "![Same camera: RGB reference / TSDF RGB / geometric regions / YOLOE categories](D4_office0_camera_aligned.jpg)",
      "",
      "![25 held-out original-depth views: corrected z residual and coverage](D4_corrected_raycast_diagnostics.png)",
      "",
      "| Experiment measure | Measured outcome |",
      "|---|---:|",
      "| Original complete available frames | 2000 |",
      "| Integrated GT-pose RGB-D frames | 500 |",
      "| YOLOE26x inference frames | 200 |",
      f"| Valid source-semantic masks | {original['actual_valid_3D_YOLOE_masks']} |",
      f"| TSDF triangular faces | {len(mesh.triangles)} |",
      f"| Surface vertex semantic support | {100*original['mesh']['semantic_supported_fraction']:.2f}% |",
      f"| Held-out frames | {new['heldout_original_frames']} |",
      f"| Raycast depth pixel support (including holes) | {100*new['raycast_depth_support_fraction']:.2f}% |",
      f"| CORRECTED median optical-depth abs residual on hits | {new['median_abs_sensor_depth_error_m_on_supported']:.4f} m |",
      f"| CORRECTED p95 optical-depth abs residual on hits | {new['p95_abs_sensor_depth_error_m_on_supported']:.4f} m |",
      f"| Supported-hit depth residual less than 5cm | {100*new['raycast_within_5cm_fraction_on_supported']:.2f}% |",
      f"| Official GT mesh one-way triangle distance p95 | {new_gt['reconstructed_to_independent_GT_triangle_distance_p95']:.4f} m |",
      "",
      "**Correction:** original invalid depth median "+
      f"{old['median_abs_sensor_depth_error_m_on_supported']:.4f}m and p95 "+
      f"{old['p95_abs_sensor_depth_error_m_on_supported']:.4f}m resulted from R[:,2] "
      "instead of R[2,:]. These old numbers are superseded, not silently deleted.",
      "",
      "**Remaining limitations:** rendering and oracle GT poses do not imply reliable physical object-instance labels. "
      "The original Replica GT mesh is a reference to the same synthetic scene; it "
      "is an independent mesh file, not a separate real sensor. "
      "GT p95 surface discrepancy may be affected by unobserved and inconsistent areas; further regional auditing is required.",
      "No RTAB-Map ATE, IDF1, physical object re-ID or Nav2 safety scores were evaluated.",
      ""
    ]
    (args.out/"D4_full_office0_quality_results.md").write_text("\n".join(lines))
    print("D4_REAL_SOURCE_RAYCAST_DEPTH_REPAIR_VERIFIED",json.dumps({
      "old_median":old["median_abs_sensor_depth_error_m_on_supported"],
      "new_median":new["median_abs_sensor_depth_error_m_on_supported"],
      "new_p95":new["p95_abs_sensor_depth_error_m_on_supported"],
      "coverage":new["raycast_depth_support_fraction"],
      "fraction_under_5cm":new["raycast_within_5cm_fraction_on_supported"],
    }),flush=True)


def read_partial_source(root,original,heldout):
    from d4_full_office0_map import read_inputs
    # D4-M1 only refetches withheld frames, not 500 integration and 200 detector
    # frames; read intrinsics and complete pose array explicitly without assuming
    # the original frames are locally present.
    source=root/"Replica"/"office0"
    camera=json.loads((root/"Replica"/"cam_params.json").read_text())["camera"]
    traj=np.loadtxt(source/"traj.txt").reshape(-1,4,4)
    if len(traj)!=2000:raise RuntimeError("Wrong original trajectory")
    K=np.array([[camera["fx"],0,camera["cx"]],
      [0,camera["fy"],camera["cy"]],[0,0,1]],float)
    return source,root/"Replica"/"office0_mesh.ply",K,traj,float(camera["scale"]),None


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--original",type=Path,required=True)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--run",type=int,required=True)
    repair(p.parse_args())


if __name__=="__main__":
    main()
