#!/usr/bin/env python3
"""D5: independent official OVI-MAP Replica-office0 instance GT audit.

Strictly EVALUATION ONLY: independent GT instance labels and GT mesh are
never inputs to instance proposal generation. Use immutable D4 predicted
RGB-D TSDF mesh / proposed class-agnostic face patches.

Three predeclared geometry-only variants are compared under exactly the same
source surface, GT matching rules and area-weighted evaluation:
  published_D4, finer_normal_color, moderate_normal_color.
No chosen 'best' method is a holdout-generalizable scientific result.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment

VARIANTS = {
    "published_D4": None,
    "finer_normal_color": {"normal_cos": .93, "color_delta": .22,
                            "min_faces": 75, "max_regions": 400},
    "moderate_normal_color": {"normal_cos": .88, "color_delta": .32,
                               "min_faces": 110, "max_regions": 400},
}
MAX_GT_DIST_M = .05
MIN_GT_INSTANCE_AREA_M2 = .018
MIN_PROPOSAL_AREA_M2 = .018


def area3(vertices, tri):
    vertices = np.asarray(vertices, dtype=np.float32)
    tri = np.asarray(tri, dtype=np.int32)
    r = vertices[tri]
    return .5 * np.linalg.norm(np.cross(r[:, 1] - r[:, 0], r[:, 2] - r[:, 0]), axis=1)


def gt_surface_labels(query, gtmesh, gtlabels, *, maxdist=.05, chunk=30000):
    """Transfer independent GT object identity by closest ORIGINAL GT TRIANGLE.

    Only GT mesh triangles with all THREE matching instance labels are
    accepted. Boundary triangles, missing surfaces and >5cm are unknown,
    not silently assigned. This is an EVALUATION association, not prediction.
    """
    import open3d as o3d
    gtvertices = np.asarray(gtmesh.vertices)
    faces = np.asarray(gtmesh.triangles, dtype=np.int32)
    label = np.asarray(gtlabels, np.int64)
    points = np.asarray(query, np.float32)
    if len(gtvertices) != len(label):
        raise RuntimeError("OVI-MAP instance annotation is not aligned to official mesh")
    if len(faces) < 1000 or len(points) == 0:
        raise ValueError("Insufficient original geometric data")
    if not np.isfinite(points).all():
        raise ValueError("Invalid query points")
    source_ids = label[faces]
    homogeneous = np.all(source_ids == source_ids[:, :1], axis=1)
    source_labels = np.where(homogeneous, source_ids[:, 0], -1)
    if (source_labels >= 0).sum() < 100:
        raise RuntimeError("GT mesh instance annotation too sparse/ambiguous")
    ts = o3d.t.geometry.RaycastingScene()
    ts.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(gtmesh))
    out = np.full(len(points), -1, dtype=np.int64)
    distance = np.full(len(points), np.nan, dtype=np.float32)
    for i in range(0, len(points), chunk):
        answer = ts.compute_closest_points(o3d.core.Tensor(points[i:i+chunk]))
        hit = answer["primitive_ids"].numpy().astype(np.int64)
        closest = answer["points"].numpy()
        dist = np.linalg.norm(closest - points[i:i+chunk], axis=1)
        good = (hit < len(faces)) & (dist <= maxdist)
        q = np.full(len(hit), -1, dtype=np.int64)
        q[good] = source_labels[hit[good]]
        out[i:i+len(hit)] = q
        distance[i:i+len(hit)] = dist
    return out, distance


def evaluate_instance_regions(gt, proposals, weights, *, gmin=.018, pmin=.018):
    """Area-weighted matched-source 3D instance-like panoptic audit.

    Faces with missing/ambiguous GT labels are excluded from the denominator
    but reported through the external association coverage metric. Zero is a
    valid source instance ID, so UNKNOWN is specifically -1.
    Predicted class-independent patch 0 = abstention.
    """
    gt = np.asarray(gt, np.int64)
    pred = np.asarray(proposals, np.int32)
    w = np.asarray(weights, np.float64)
    if not (gt.shape == pred.shape == w.shape) or not np.isfinite(w).all():
        raise ValueError("Misaligned GT, predictions or triangle area")
    if (w <= 0).all() or (pred < 0).any():
        raise ValueError("Invalid source geometry or proposal")
    matched = (gt >= 0) & (w > 0)
    if matched.sum() < 100:
        raise RuntimeError("Too few independently GT-labeled original surfaces")
    g,ginv = np.unique(gt[matched],return_inverse=True)
    p,pinv = np.unique(pred[matched],return_inverse=True)
    mat = np.zeros((len(g),len(p)),np.float64)
    np.add.at(mat,(ginv,pinv),w[matched])
    garea = mat.sum(axis=1)
    parea = mat.sum(axis=0)
    gkeep = garea >= gmin
    pkeep = (p > 0) & (parea >= pmin)
    target_g = g[gkeep]
    target_p = p[pkeep]
    if not len(target_g):
        raise RuntimeError("No sufficiently observed source GT object")
    intersection = mat[np.ix_(gkeep,pkeep)]
    union = garea[gkeep,None]+parea[pkeep][None,:]-intersection
    iou = np.divide(intersection,union,out=np.zeros_like(intersection),where=union>0)
    if iou.size:
        rr,cc = linear_sum_assignment(-iou)
        candidates = sorted(
            [(int(target_g[i]),int(target_p[j]),float(iou[i,j])) for i,j in zip(rr,cc)],
            key=lambda x:-x[2])
    else:
        candidates=[]
    match05 = [x for x in candidates if x[2]>=.50]
    tp = len(match05)
    n_gt = len(target_g)
    n_pred = len(target_p)
    # PQ = sum matched IoU / (TP + .5 FP + .5 FN)
    denom = tp + .5 * (n_pred - tp) + .5 * (n_gt - tp)
    pq = sum(a[2] for a in match05)/denom if denom else 0.
    # Area-weighted majority class consistency, regardless of GT ID order.
    no_label = np.flatnonzero(p==0)
    proposal_mat = mat[:,pkeep]
    area_total = float(garea[gkeep].sum())
    matched_agreement = float(proposal_mat.max(axis=0).sum()) if proposal_mat.size else 0.
    purity = matched_agreement/max(float(proposal_mat.sum()),1e-10)
    # Splitting: mean number of predicted regions with >10% of an
    # eligible GT object area; merging: mean number of GT objects contributing
    # >10% of each predicted patch's area.
    gt_split = []
    for a,idx in zip(garea,gkeep):
        if idx and n_pred:
            pass
    for gidx,gv in enumerate(g):
        if not gkeep[gidx]:continue
        gt_split.append(int((mat[gidx,pkeep] >= .10 * garea[gidx]).sum()))
    pred_merges = []
    for pi,pv in enumerate(p):
        if not pkeep[pi]:continue
        pred_merges.append(int((mat[gkeep,pi] >= .10*parea[pi]).sum()))
    return {
      "independent_GT_objects_with_min_observed_area":n_gt,
      "proposed_regions_with_min_observed_area":n_pred,
      "GT_original_instance_ids_considered":[int(x) for x in target_g],
      "physical_instance_PQ_at_IoU50":float(pq),
      "physical_instance_RQ_at_IoU50":float(tp/denom) if denom else 0.,
      "physical_instance_SQ_at_IoU50":float(np.mean([x[2] for x in match05])) if tp else 0.,
      "TP":tp,"FP":n_pred-tp,"FN":n_gt-tp,
      "pairwise_independent_gt_purity_of_proposed_regions":float(purity),
      "majority_GT_agreement_area_fraction_of_covered_GT_surface":
        float(matched_agreement/max(area_total,1e-10)),
      "abstained_area_fraction_of_GT_matched":
        float(mat[:,no_label].sum()/max(mat.sum(),1e-10)) if len(no_label) else 0.,
      "mean_proposal_regions_covering_GT_at_least_10_percent":
        float(np.mean(gt_split)) if gt_split else None,
      "mean_GT_instances_contributing_at_least_10_percent_to_a_proposal":
        float(np.mean(pred_merges)) if pred_merges else None,
      "iou_matching_pairs":[{"gt_id":a,"patch_id":b,"iou":c} for a,b,c in candidates[:40]],
      "evaluation_surface_area_m2":float(mat.sum()),
      "does_not_validate_persistent_UUID":True,
      "does_not_validate_instance_recall_outside_observed_mesh":True,
    }


def create_variants(mesh, original_labels):
    from d3_replica_office0_mesh_patches import segment_surface_patches
    v = np.asarray(mesh.vertices)
    f = np.asarray(mesh.triangles)
    original_labels = np.asarray(original_labels,dtype=np.int32)
    if len(original_labels)!=len(f):
        raise ValueError("Original D4 patch map is not aligned with original mesh faces")
    out={"published_D4":original_labels}
    meta={"published_D4":{"method":"original preregistered D4 predicted mesh patches",
                           "num_regions":int(np.unique(original_labels[original_labels>0]).size)}}
    # Multiple class-agnostic proposals from the SAME RGB-D mesh,
    # no GT labels used until the later evaluator. 125k simplification follows D4.
    simple=mesh.simplify_quadric_decimation(target_number_of_triangles=125000)
    sv=np.asarray(simple.vertices);sf=np.asarray(simple.triangles)
    scol=np.asarray(simple.vertex_colors)
    if len(sf)<20000:
        raise RuntimeError("Unexpectedly small D4 simplified evaluation surface")
    tri_centre=sv[sf].mean(axis=1)
    full_centre=v[f].mean(axis=1)
    _,nearest=cKDTree(tri_centre).query(full_centre,k=1)
    for name in ("finer_normal_color","moderate_normal_color"):
        group,summary=segment_surface_patches(sv,sf,scol,**VARIANTS[name])
        out[name]=np.asarray(group[nearest],np.int32)
        meta[name]=summary
    return out,meta


def render_gt_diagnostic(mesh, gt_surface, predictions, path):
    """One GT diagnostic panel and predicted region panels; no cherry-picking."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    vertices=np.asarray(mesh.vertices)
    faces=np.asarray(mesh.triangles)
    k=min(11000,len(faces))
    # Uniform face index sample is VIEW-only. Quantitative scores area-weighted.
    pick=np.linspace(0,len(faces)-1,k,dtype=np.int64)
    local=vertices[faces[pick]]
    normal=np.cross(local[:,1]-local[:,0],local[:,2]-local[:,0])
    normal/=np.maximum(np.linalg.norm(normal,axis=1)[:,None],1.e-8)
    L=np.array([.4,-.5,.82]);L/=np.linalg.norm(L)
    shade=np.clip(.72+.24*np.maximum(normal@L,0),.65,1.)
    cm=plt.get_cmap("tab20")
    panels=[("OVI-MAP independent GT instance (EVALUATION)",gt_surface)]
    panels.extend((k,predictions[k]) for k in predictions)
    fig=plt.figure(figsize=(18,9.5),facecolor="#101923",dpi=140)
    center=np.median(vertices,axis=0)
    axes=np.quantile(vertices,.99,axis=0)-np.quantile(vertices,.01,axis=0)
    span=np.maximum(.6,axes)*1.2
    for j,(heading,values) in enumerate(panels):
        ax=fig.add_subplot(2,2,j+1,projection="3d",proj_type="ortho")
        c=np.asarray(values)[pick].astype(int)
        # Official instance id can be 0; unknown GT -1.
        unknown=(c<0) if j==0 else (c==0)
        color=np.asarray([cm(abs(int(u))%20)[:3] for u in c])
        color[unknown]=[.6,.63,.69]
        color=np.column_stack((np.clip(color*shade[:,None],0,1),
                               np.ones(len(c))))
        ax.add_collection3d(Poly3DCollection(local,facecolors=color,
                                            linewidths=0,rasterized=True))
        for setlim,v,extent in zip((ax.set_xlim,ax.set_ylim,ax.set_zlim),center,span):
            setlim(v-extent/2,v+extent/2)
        ax.set_box_aspect(span)
        ax.view_init(elev=31,azim=-55)
        ax.set_axis_off()
        ax.set_title(heading,color="white",fontsize=12)
    fig.text(.04,.966,"D5 - Same observed Replica office0 mesh, independent physical-instance GT audit",
             fontsize=15,color="white",weight="bold")
    fig.text(.04,.028,
       "Top-left GT instance colors are diagnostic ONLY. All other panels are class-agnostic predictions. Gray = unsupported/abstained.",
       fontsize=10,color="#ffcfac")
    fig.tight_layout(rect=[0,.05,1,.94])
    fig.savefig(path,facecolor=fig.get_facecolor())
    plt.close(fig)


def run(args):
    import open3d as o3d
    p=Path(args.d4)
    found=list(p.rglob("d4_metrics.json"))
    if len(found)!=1:
        raise RuntimeError("D5 requires exactly ONE completed D4 experiment artifact")
    folder=found[0].parent
    old=json.loads(found[0].read_text())
    if old.get("actual_original_frames_integrated")!=500 or old.get("actual_YOLOE_inferences")!=200:
        raise RuntimeError("D4 source does not contain complete controlled mapping")
    if old.get("RTABMap_run") is not False:
        raise RuntimeError("Incorrect source SLAM provenance")
    npz=folder/"mesh_patch_and_semantic_labels.npz"
    rgbmesh=folder/"rgb_mesh.ply"
    if not npz.is_file() or not rgbmesh.is_file():
        raise RuntimeError("Original D4 mesh and predicted patch IDs must both exist")
    source=np.load(npz,allow_pickle=False)
    mesh=o3d.io.read_triangle_mesh(str(rgbmesh))
    pids=np.asarray(source["surface_patch_ids"],np.int32)
    verts=np.asarray(mesh.vertices)
    tris=np.asarray(mesh.triangles)
    if len(tris)!=len(pids) or len(tris)!=old["mesh"]["full_mesh_faces"]:
        raise RuntimeError("D4 saved prediction to mesh mismatched; cannot audit")
    gt=o3d.io.read_triangle_mesh(str(args.gt_mesh))
    official=np.loadtxt(args.gt_labels,dtype=np.int64)
    if len(official)!=len(gt.vertices):
        raise RuntimeError("Official OVI-MAP instance labels not aligned to original independent GT mesh")
    # GT faces themselves may straddle distinct object IDs. Do not invent
    # class assignment for those triangles.
    face_centers=verts[tris].mean(axis=1)
    oracle,dist=gt_surface_labels(face_centers,gt,official,
                                   maxdist=MAX_GT_DIST_M)
    real_area=area3(verts,tris)
    if not (np.isfinite(real_area).all() and real_area.sum()>2):
        raise RuntimeError("Corrupt original measured mesh")
    matched=oracle>=0
    match_coverage=float(real_area[matched].sum()/real_area.sum())
    gt_visible=len(np.unique(oracle[matched]))
    if match_coverage<.05 or gt_visible<3:
        raise RuntimeError("Independent GT mesh does not correspond to this reconstructed scene")
    variants,source_meta=create_variants(mesh,pids)
    results={name:evaluate_instance_regions(oracle,ids,real_area)
             for name,ids in variants.items()}
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/"d5_instance_audit_labels.npz",
       gt_instance_on_predicted_faces=oracle,
       source_distance_to_gt_mesh_m=dist,
       **{f"candidate_{name}":v.astype(np.int32) for name,v in variants.items()})
    render_gt_diagnostic(mesh,oracle,variants,out/"D5_instance_audit_visual_comparison.png")
    report={
      "experiment":"D5_OVI_MAP_OFFICIAL_INSTANCE_GT_AUDIT_SAME_D4_REPLICA",
      "source_d4_run_id":41645799954,
      "OVI_MAP_official_GT_labels_uri":
         "https://github.com/OVI-MAP/OVI-MAP/blob/main/scripts/datasets/replica_gt_instances/instance_labels_office0.txt",
      "GT_mesh_source":"official NICE-SLAM Replica office0_mesh.ply (same full dataset as D4)",
      "source_rendered_RGBD_not_physical_camera":True,
      "poses_GT_ORACLE_not_SLAM":True,
      "read_only_original_D4_source":True,
      "GT_used_for_training_or_proposal_generation":False,
      "no_GT_based_hyperparameter_selection":True,
      "independent_source_GT_mesh_vertex_count":len(gt.vertices),
      "independent_source_GT_mesh_faces":len(gt.triangles),
      "source_unique_physical_GT_IDs":len(np.unique(official)),
      "source_reconstructed_triangle_count":len(tris),
      "source_reconstructed_surface_area_m2":float(real_area.sum()),
      "GT_supported_reconstruction_area_fraction_5cm":match_coverage,
      "GT_visible_IDs_on_reconstructed_surface":gt_visible,
      "candidate_hypotheses":source_meta,
      "quantitative_independent_GT_instance_results":results,
      "metrics_are_not_cross_session_IDF1":True,
      "evaluation_GT_mesh_coverage_excludes_unknown_unseen_surfaces":True,
      "physical_instance_IDF1":"NOT_EVALUATED_NO_CROSS_SESSION_INSTANCE_ID_GT",
      "limitations":("Matched GT covers only <=5cm and unambiguous original GT triangles. "
          "Source RGB-D is computer-rendered with camera GT poses, not real RGB-D SLAM. "
          "Mesh vertex/triangle resolution affects GT match. RGB-only mesh patch variants "
          "are not complete learned class-agnostic instance segmentation, and the GT of "
          "this same scene must never be used to select universal hyperparameters."),
      "outputs":["D5_instance_audit_visual_comparison.png",
                 "d5_instance_audit_labels.npz","d5_instance_audit_results.json"]
    }
    (out/"d5_instance_audit_results.json").write_text(
       json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print("D5_OFFICIAL_OVIMAP_INSTANCE_GT_RESULTS",json.dumps({
       "source_reconstructed_triangle_count":len(tris),
       "matched_surface_fraction":match_coverage,
       "gt_instances_visible":gt_visible,
       "results":{name:{"n_pred":x["proposed_regions_with_min_observed_area"],
                        "GT_PQ_50":x["physical_instance_PQ_at_IoU50"],
                        "TP":x["TP"],"FP":x["FP"],"FN":x["FN"],
                        "purity":x["pairwise_independent_gt_purity_of_proposed_regions"]}
                  for name,x in results.items()}
       },ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--d4",required=True)
    p.add_argument("--gt-mesh",required=True)
    p.add_argument("--gt-labels",required=True)
    p.add_argument("--out",required=True)
    q=p.parse_args()
    run(q)

if __name__=="__main__":
    main()
