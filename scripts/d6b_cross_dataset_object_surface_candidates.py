#!/usr/bin/env python3
"""D6-B cross-dataset geometry-supported candidate surface mapping.

Same deterministic algorithm and ORIGINAL prior YOLOE evidence as D6 Replica.
Inputs are immutable D2 Open3D TSDF mesh and D2 measured YOLOE semantic vertex
labels for three REAL RGB-D sensor sequences (TUM xyz, TUM desk2, Bonn moving).
All D2 poses are original GT/mocap ORACLE, not estimated RTAB-Map. The spatial
group IDs are surface candidates, NOT physical instance GT or semantic recall.

Evaluation is intentionally descriptive because no independent 3D instance GT
exists for these matched original D2 sequences.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from d3_replica_office0_mesh_patches import segment_surface_patches
from d6_semantic_seeded_surface_candidates import (
    triangle_support,expand_near_surface,segment_class_surface,view_and_export
)

def source_geometry_baseline(mesh):
    verts=np.asarray(mesh.vertices)
    faces=np.asarray(mesh.triangles)
    simple=mesh.simplify_quadric_decimation(
       target_number_of_triangles=min(85000,len(faces)))
    sv=np.asarray(simple.vertices)
    sf=np.asarray(simple.triangles)
    if len(sf)<3000:
        raise RuntimeError("D2 real source mesh missing valid geometry")
    small_regions,_=segment_surface_patches(
        sv,sf,np.asarray(simple.vertex_colors),
        normal_cos=.82,color_delta=.40,min_faces=90,max_regions=75)
    _,index=cKDTree(sv[sf].mean(axis=1)).query(
         verts[faces].mean(axis=1),k=1)
    return small_regions[index]

def run(args):
    import open3d as o3d
    root=Path(args.d2)
    metrics=list(root.rglob("d2_metrics.json"))
    if len(metrics)!=1:
        raise RuntimeError("Exactly one D2 real sensor dataset artifact required")
    p=metrics[0].parent
    src=json.loads(metrics[0].read_text())
    if src.get("dataset")!=args.dataset or not src.get("known_GT_pose_used_for_integration"):
        raise RuntimeError("Incorrect original D2 dataset or pose provenance")
    if src.get("RTABMap_performed_in_D2") is not False:
        raise RuntimeError("Cannot mix RTAB-Map and GT pose results in D6-B")
    mesh_file=p/"rgb_mesh.ply"
    labels_file=p/"mesh_vertex_semantics.npz"
    if not mesh_file.is_file() or not labels_file.is_file():
        raise FileNotFoundError("Original immutable D2 full surface and semantic vertices")
    mesh=o3d.io.read_triangle_mesh(str(mesh_file))
    pred=np.load(labels_file,allow_pickle=False)
    source_verts=np.asarray(pred["vertices"])
    source_cat=np.asarray(pred["class_idx_plus_one"],dtype=np.int16)
    vertex=np.asarray(mesh.vertices,dtype=np.float32)
    faces=np.asarray(mesh.triangles,dtype=np.int32)
    if len(vertex)!=len(source_verts) or len(vertex)!=src["mesh_render"]["original_mesh_vertices"]:
        raise RuntimeError("Measured source mesh and YOLOE 3D semantic vertex count differs")
    if not np.allclose(vertex,source_verts,atol=1e-5):
        raise RuntimeError("3D vertex order changed; cannot transfer original semantic labels")
    if len(source_cat)!=len(vertex) or len(faces)!=src["mesh_render"]["original_mesh_triangles"]:
        raise RuntimeError("D2 geometry provenance or mesh faces changed")
    classes=np.asarray(pred["classes"]).tolist()
    geo=source_geometry_baseline(mesh)
    seeds,area,normal,centroid,_=triangle_support(vertex,faces,source_cat)
    rgb=np.asarray(mesh.vertex_colors,dtype=np.float32)[faces].mean(axis=1)
    initial,initial_meta=segment_class_surface(vertex,faces,seeds,area,normal)
    grown,inferred=expand_near_surface(seeds,centroid,normal,rgb)
    final,final_meta=segment_class_surface(vertex,faces,grown,area,normal)
    if not np.array_equal(seeds[seeds>0],grown[seeds>0]):
        raise AssertionError("D6-B changed original network evidence")
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    view_and_export(mesh,classes,geo,initial,final,out,
          scene_title="REAL "+args.dataset.upper()+" (GT pose oracle)")
    report={
      "experiment":"D6B_REAL_RGBD_MULTIDATASET_CLASS_SEEDED_SURFACE_CANDIDATES",
      "real_sensor_dataset":args.dataset,
      "original_source_D2_run":41637371732,
      "source_real_measured_sensor_frames":src["integrated_measured_RGBD_frames"],
      "source_D2_independent_GT_mocap_poses_used_oracle":True,
      "RTABMap_evaluated":False,
      "no_YOLOE_new_inference":True,
      "source_measured_YOLOE_3D_semantic_vertices":src["mesh_render"]["vertex_semantically_supported"],
      "original_mesh_triangles":len(faces),
      "original_mesh_vertices":len(vertex),
      "source_D2_mean_geometry_vertex_semantic_fraction":
        src["mesh_render"]["fraction_mesh_vertices_supported_by_predicted_classes"],
      "D6_original_geometric_regions":int(np.unique(geo[geo>0]).size),
      "original_strong_seed_faces":int((seeds>0).sum()),
      "original_strong_seed_area_fraction":float(area[seeds>0].sum()/area.sum()),
      "expanded_semantics_new_faces":int(inferred.sum()),
      "expanded_semantics_new_area_fraction":float(area[inferred].sum()/area.sum()),
      "expanded_semantics_area_fraction":float(area[grown>0].sum()/area.sum()),
      "unexpanded_region_stats":initial_meta,
      "grown_region_stats":final_meta,
      "physical_instance_IDF1":"NOT_EVALUATED",
      "independent_GT_instance_PQ":"NOT_EVALUATED",
      "no_verified_physical_instance_IDs":True,
      "dynamic_Bonn_ghost_surfaces_still_possible":args.dataset=="bonn_box",
      "limitation":("Original D2 GT poses used for geometry. Category-seeded graph "
        "components may split/merge objects. Metric labels on new faces are "
        "inferred only from neighboring source YOLOE, never independent GT. "
        "No true object-instance annotations or persistence in these sequences."),
      "outputs":["D6_instance_candidates_visual.png","D6_interactive_candidates.html",
                 "candidate_surface_instances.glb","candidate_face_evidence.npz",
                 "d6b_metrics.json"]
    }
    (out/"d6b_metrics.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print("D6B_REAL_RGBD_MULTIDATASET_CANDIDATE_RESULT",
          json.dumps({
            "dataset":args.dataset,
            "original_mesh_triangles":len(faces),
            "source_YOLOE_surface_vertices":src["mesh_render"]["vertex_semantically_supported"],
            "geometric_region_candidates":report["D6_original_geometric_regions"],
            "strong_semantic_faces":report["original_strong_seed_faces"],
            "grown_new_faces":report["expanded_semantics_new_faces"],
            "class_conditioned_candidate_regions_before":
              initial_meta["accepted_candidate_regions"],
            "class_conditioned_candidate_regions_after":
              final_meta["accepted_candidate_regions"],
          },ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--d2",required=True)
    p.add_argument("--dataset",required=True,choices=["tum_xyz","tum_desk2","bonn_box"])
    p.add_argument("--out",required=True)
    q=p.parse_args()
    run(q)

if __name__=="__main__":
    main()
