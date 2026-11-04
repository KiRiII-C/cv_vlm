#!/usr/bin/env python3
"""D6: class-seeded 3D instance CANDIDATE map from immutable real-source D4 mesh.

Not a semantic detector, not SAM and not full class-agnostic segmentation.
Actual YOLOE-26x 3D class labels from D4 are retained as immutable measured
semantic evidence. A conservative RGB/normal/metric 3D region grow provides
optional inferred face labels, stored separately; no missing label is a
physical object absence. Connected components are candidate surface groups,
NOT verified physical identities or persistent cross-session object UUIDs.

Compares:
  B0: YOLOE strong-seed topology components
  B1: topology after conservative class/normal/RGB boundary-aware growth
  G0: D4 geometry-only surface patches
on the SAME previously reconstructed 500-frame D4 full Replica office0 mesh.
"""
from __future__ import annotations

import argparse
import colorsys
import json
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy import sparse
from scipy.sparse.csgraph import connected_components

def triangle_support(vertices, faces, category):
    v=np.asarray(vertices,np.float32)
    tri=np.asarray(faces,np.int32)
    cat=np.asarray(category,np.int16)
    if len(cat)!=len(v) or tri.ndim!=2 or tri.shape[1]!=3 or tri.max()>=len(v):
        raise ValueError("Mismatched D4 triangle mesh and semantic vertex evidence")
    if not np.isfinite(v).all() or (cat<0).any():
        raise ValueError("Invalid source numerical mesh")
    triangles=v[tri]
    normals=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    area=.5*np.linalg.norm(normals,axis=1)
    normals=normals/np.maximum(np.linalg.norm(normals,axis=1)[:,None],1.e-10)
    categories=cat[tri]
    labels=np.zeros(len(tri),dtype=np.int16)
    for i in range(3):
        # Require agreement of at least TWO nonzero vertex semantic predictions
        # and NO conflicting third class (unknown zero is okay).
        chosen=categories[:,i]
        matches=(categories==chosen[:,None]).sum(axis=1)>=2
        conflict=np.any((categories>0)&(categories!=chosen[:,None]),axis=1)
        select=(chosen>0)&matches&(~conflict)&(labels==0)
        labels[select]=chosen[select]
    face_color=np.zeros((len(tri),3),np.float32)
    return labels,area,normals,triangles.mean(axis=1),face_color


def expand_near_surface(seeds,center,normals,face_rgb, *,
                        radius=.055,cosine=.92,color_delta=.27,k=5,
                        purity=.8):
    seed=np.asarray(seeds,np.int16)
    c=np.asarray(center,np.float32)
    n=np.asarray(normals,np.float32)
    col=np.asarray(face_rgb,np.float32)
    if not(len(seed)==len(c)==len(n)==len(col)):
        raise ValueError("D6 source face geometry shape mismatch")
    if len(seed)<100:
        raise ValueError("Insufficient original mapped triangles")
    out=seed.copy()
    hard=seed>0
    if hard.sum()==0:
        return out,np.zeros_like(hard,dtype=bool)
    roots=np.flatnonzero(hard)
    unknown=np.flatnonzero(~hard)
    tr=cKDTree(c[roots])
    grow=np.zeros(len(seed),dtype=bool)
    # Batch avoids excessive RAM on large real source surface.
    for st in range(0,len(unknown),35000):
        loc=unknown[st:st+35000]
        dist,ind=tr.query(c[loc],k=k,distance_upper_bound=radius,workers=1)
        if dist.ndim==1:
            dist=dist[:,None];ind=ind[:,None]
        valid=np.isfinite(dist)&(ind<len(roots))
        safe=np.minimum(ind,len(roots)-1)
        neigh=roots[safe]
        normal_agree=np.einsum("nkd,nd->nk",n[neigh],n[loc])>=cosine
        color_agree=np.linalg.norm(col[neigh]-col[loc,None,:],axis=2)<=color_delta
        valid &=normal_agree&color_agree
        candidates=seed[neigh]
        candidates=np.where(valid,candidates,0)
        valid_count=valid.sum(axis=1)
        if not np.any(valid_count):continue
        names=np.unique(seed[roots])
        votes=np.zeros((len(loc),len(names)),np.uint8)
        for j,name in enumerate(names):
            votes[:,j]=(candidates==name).sum(axis=1)
        winner_idx=votes.argmax(axis=1)
        winners=votes[np.arange(len(loc)),winner_idx]
        good=(winners>=3)&(winners/np.maximum(valid_count,1)>=purity)
        # Strong seeds on opposite sides of a furniture boundary must
        # not diffuse across disagreeing local classes.
        good &= (dist[:,0]<.04)
        out[loc[good]]=names[winner_idx[good]]
        grow[loc[good]]=True
    if np.any(out[hard]!=seed[hard]):
        raise AssertionError("D6 attempted to mutate original semantic evidence")
    return out,grow


def segment_class_surface(vertices,faces,category,area,normals,
                          *,min_faces=18,min_area_m2=.022,normal_cos=.80):
    """Connected components along actual ORIGINAL triangle SHARED EDGES only.
    No different-class face can be merged into a single object candidate.
    """
    f=np.asarray(faces,np.int32)
    label=np.asarray(category,np.int16)
    area=np.asarray(area,np.float32)
    n=np.asarray(normals,np.float32)
    if len(f)!=len(label) or len(area)!=len(label) or len(n)!=len(label):
        raise ValueError("Broken original face alignment")
    graph_edges=np.concatenate((f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]))
    graph_edges.sort(axis=1)
    owner=np.tile(np.arange(len(f),dtype=np.int32),3)
    order=np.lexsort((graph_edges[:,1],graph_edges[:,0]))
    graph_edges=graph_edges[order]
    owner=owner[order]
    pairs=np.flatnonzero(np.all(graph_edges[1:]==graph_edges[:-1],axis=1))
    u=owner[pairs];v=owner[pairs+1]
    keep=(label[u]>0)&(label[u]==label[v])&(
         np.einsum("ij,ij->i",n[u],n[v])>=normal_cos)&(u!=v)
    u=u[keep];v=v[keep]
    adjacency=sparse.coo_matrix((
      np.ones(2*len(u),dtype=np.uint8),
      (np.concatenate((u,v)),np.concatenate((v,u)))),
      shape=(len(f),len(f))).tocsr()
    count,component=connected_components(adjacency,directed=False)
    size=np.bincount(component,minlength=count)
    surface=np.bincount(component,weights=area,minlength=count)
    valid=(size>=min_faces)&(surface>=min_area_m2)
    valid[component[label==0]]=False
    region_candidates=np.flatnonzero(valid)
    # Stable deterministic ID ordering is based on increasing component index.
    lookup=np.zeros(count,np.int32)
    lookup[region_candidates]=np.arange(1,len(region_candidates)+1)
    return lookup[component], {
      "connected_components_raw":int(count),
      "accepted_candidate_regions":int(len(region_candidates)),
      "accepted_surface_area_m2":float(surface[region_candidates].sum()),
      "accepted_faces":int(size[region_candidates].sum()),
      "candidate_min_area_m2":min_area_m2,
      "candidate_min_faces":min_faces,
      "class_conditioned_seed_topology":True,
      "physical_instances_verified":False,
    }


def colors_for_ids(groups):
    x=np.asarray(groups,dtype=int)
    u=np.unique(x)
    table=np.zeros((int(u.max())+1,3),np.float32)
    table[0]=[.70,.72,.76]
    for v in u:
        if v>0:
            table[v]=colorsys.hsv_to_rgb(float((v*.61803398875)%1),.76,.91)
    return table[x]


def view_and_export(mesh,classes,baseline_patch,b0,b1,output,*,scene_title="REPLICA OFFICE0"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import trimesh
    import plotly.graph_objects as go
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    v=np.asarray(mesh.vertices)
    f=np.asarray(mesh.triangles)
    rgb=np.clip(np.asarray(mesh.vertex_colors),0,1)
    geo=colors_for_ids(baseline_patch)
    raw=colors_for_ids(b0)
    grown=colors_for_ids(b1)
    # FULL geometry is preserved; different FACE colors do not alter vertex
    # positions, normals, triangles or the original RGB/YOLOE evidence.
    tri=trimesh.Trimesh(vertices=v,faces=f,process=False)
    tri.visual.face_colors=np.column_stack((
        np.round(grown*255).astype(np.uint8),
        np.full(len(f),255,dtype=np.uint8)))
    tri.export(out/"candidate_surface_instances.glb")
    np.savez_compressed(out/"candidate_face_evidence.npz",
       vertices=v.astype(np.float32),
       triangle_faces=f.astype(np.int32),
       d4_geometric_region=baseline_patch,
       raw_YOLOE_connected_region=b0,
       grown_YOLOE_connected_region=b1)
    # Downsample only for visualization. All metrics and PLY positions were
    # computed on the full 555k-face original RGB-D TSDF surface.
    count=min(len(f),22000)
    idx=np.linspace(0,len(f)-1,count,dtype=np.int64)
    tri_xyz=v[f[idx]]
    tri_rgb=np.mean(np.clip(.05+rgb[f[idx]]**.73,0,1),axis=1)
    display_colors=[
      tri_rgb, geo[idx],raw[idx],grown[idx]]
    names=["RGB-D ORIGINAL TSDF", "D4 GEOMETRY PATCHES",
           "D6 STRONG YOLOE 3D SEEDS","D6 BOUNDARY-AWARE CANDIDATES"]
    center=np.median(v,axis=0)
    rg=np.maximum(np.quantile(v,.99,axis=0)-np.quantile(v,.01,axis=0),.5)*1.12
    norm=np.cross(tri_xyz[:,1]-tri_xyz[:,0],tri_xyz[:,2]-tri_xyz[:,0])
    norm/=np.maximum(np.linalg.norm(norm,axis=1)[:,None],1.e-8)
    l=np.array([.5,-.2,.85]);l/=np.linalg.norm(l)
    shade=np.clip(.76+.28*np.maximum(norm@l,0),.75,1.0)
    fig=plt.figure(figsize=(19,10),facecolor="#141d29",dpi=140)
    for i,(name,col) in enumerate(zip(names,display_colors)):
        ax=fig.add_subplot(2,2,i+1,projection="3d",proj_type="ortho")
        facecol=np.column_stack((np.clip(col*shade[:,None],0,1),
                                 np.ones(len(idx))))
        ax.add_collection3d(Poly3DCollection(
            tri_xyz,facecolors=facecol,linewidths=0,rasterized=True))
        for setlim,c,d in zip((ax.set_xlim,ax.set_ylim,ax.set_zlim),center,rg):
            setlim(c-d/2,c+d/2)
        ax.set_box_aspect(rg);ax.view_init(elev=27,azim=-61)
        ax.set_axis_off();ax.set_title(name,color="white",fontsize=12,pad=12)
    fig.text(.03,.96,"D6 | "+scene_title+" — SAME RECONSTRUCTED RGB-D MESH, 4 MODES",
             color="white",fontsize=16,weight="bold")
    fig.text(.03,.025,
       "D6 candidate colors are unverified object-like surface regions; generated from original YOLOE observations and geometry, not GT instance IDs.",
       fontsize=10,color="#ffc7af")
    fig.tight_layout(rect=[0,.05,1,.94])
    fig.savefig(out/"D6_instance_candidates_visual.png",facecolor=fig.get_facecolor())
    plt.close(fig)
    import open3d as o3d
    simple=mesh.simplify_quadric_decimation(target_number_of_triangles=50000)
    sv=np.asarray(simple.vertices);sf=np.asarray(simple.triangles)
    center_full=v[f].mean(axis=1)
    center_short=sv[sf].mean(axis=1)
    _,nearest=cKDTree(center_full).query(center_short,k=1)
    simrgb=np.asarray(simple.vertex_colors)
    colors=[np.clip(.05+np.mean(simrgb[sf]**.73,axis=1),0,1),
            geo[nearest],raw[nearest],grown[nearest]]
    fig=go.Figure()
    for i,(name,col) in enumerate(zip(names,colors)):
        fig.add_trace(go.Mesh3d(x=sv[:,0],y=sv[:,1],z=sv[:,2],
           i=sf[:,0],j=sf[:,1],k=sf[:,2],
           facecolor=["rgb(%d,%d,%d)"%tuple(c) for c in np.round(col*255).astype(np.uint8)],
           lighting=dict(ambient=.66,diffuse=.66,roughness=.85,specular=.14),
           name=name,visible=(i==0)))
    fig.update_layout(template="plotly_dark",height=850,
      title="D6: RGB / geometry-only / strong seeds / candidate surfaces (not verified physical ID)",
      updatemenus=[dict(type="buttons",direction="right",x=.03,y=1.09,
        buttons=[dict(label=names[i],method="update",
            args=[{"visible":[i==j for j in range(4)]}]) for i in range(4)])],
      scene=dict(aspectmode="data",xaxis=dict(visible=False),
                 yaxis=dict(visible=False),zaxis=dict(visible=False)),
      margin=dict(l=0,r=0,t=83,b=0))
    fig.write_html(out/"D6_interactive_candidates.html",include_plotlyjs="cdn")


def run(args):
    import open3d as o3d
    root=Path(args.d4)
    meta_files=list(root.rglob("d4_metrics.json"))
    if len(meta_files)!=1:
        raise RuntimeError("D6 requires exactly one independently verified D4 source")
    folder=meta_files[0].parent
    meta=json.loads(meta_files[0].read_text())
    if meta.get("actual_original_frames_integrated")!=500 or (
        meta.get("actual_YOLOE_inferences")!=200):
        raise RuntimeError("D6 source model or actual RGB-D frame provenance mismatch")
    mesh=o3d.io.read_triangle_mesh(str(folder/"rgb_mesh.ply"))
    data=np.load(folder/"mesh_patch_and_semantic_labels.npz",allow_pickle=False)
    f=np.asarray(mesh.triangles,dtype=np.int32)
    v=np.asarray(mesh.vertices,dtype=np.float32)
    src=np.asarray(data["category_ids"],np.int16)
    geom=np.asarray(data["surface_patch_ids"],np.int32)
    if len(f)!=meta["mesh"]["full_mesh_faces"] or len(geom)!=len(f):
        raise RuntimeError("D4 original mesh and predictions do not align")
    if len(src)!=len(v) or len(src)!=meta["mesh"]["full_mesh_vertices"]:
        raise RuntimeError("Original 3D YOLOE surface vertex labels do not align")
    classes=np.asarray(data["category_names"]).tolist()
    seeds,area,normals,center,_=triangle_support(v,f,src)
    face_rgb=np.asarray(mesh.vertex_colors,dtype=np.float32)[f].mean(axis=1)
    raw,raw_stats=segment_class_surface(v,f,seeds,area,normals)
    extended,inferred=expand_near_surface(seeds,center,normals,face_rgb)
    grouped,grown_stats=segment_class_surface(v,f,extended,area,normals)
    assert (extended[seeds>0]==seeds[seeds>0]).all()
    assert (extended[seeds==0]>=0).all()
    assert (area>0).any()
    result={
      "experiment":"D6_CATEGORY_SEEDED_3D_SURFACE_INSTANCE_CANDIDATES",
      "source_D4_original_success_run":41645799954,
      "same_full_replica_mesh_as_D4":True,
      "source_GT_pose_oracle_not_RTABMap":True,
      "original_source_500_RGBD_200_YOLOE_inferences_reused":True,
      "retrained_or_inferred_YOLOE_this_run":False,
      "reconstructed_original_mesh_faces":int(len(f)),
      "reconstructed_original_mesh_vertices":int(len(v)),
      "unmodified_original_YOLOE_supported_mesh_vertices":int((src>0).sum()),
      "source_D4_geometry_patch_regions":int(np.unique(geom[geom>0]).size),
      "raw_seed_faces":int((seeds>0).sum()),
      "raw_seed_area_fraction":float(area[seeds>0].sum()/area.sum()),
      "conservatively_expanded_faces":int(inferred.sum()),
      "expansion_only_area_fraction":float(area[inferred].sum()/area.sum()),
      "expanded_class_supported_area_fraction":float(area[extended>0].sum()/area.sum()),
      "raw_candidate_instance_map":raw_stats,
      "grown_candidate_instance_map":grown_stats,
      "surface_area_m2":float(area.sum()),
      "new_physical_UUIDs_claimed":False,
      "physical_IDF1":"NOT_EVALUATED",
      "independent_GT_instance_PQ":"NOT_EVALUATED_NO_VALID_ANNOTATION_IN_D6",
      "scientific_interpretation":"Descriptive source-conditioned candidate region stability; "
           "NOT measured segmentation accuracy or physical objects. "
           "Inferred labels are separately marked and may be incorrect.",
      "outputs":["D6_instance_candidates_visual.png","D6_interactive_candidates.html",
         "candidate_surface_instances.glb","candidate_face_evidence.npz",
         "d6_metrics.json"]
    }
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    view_and_export(mesh,classes,geom,raw,grouped,out)
    (out/"d6_metrics.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print("D6_SAME_ORIGINAL_REPLICA_MEASURED_MESH",
          json.dumps(result,ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--d4",required=True)
    p.add_argument("--out",required=True)
    a=p.parse_args()
    run(a)

if __name__=="__main__":
    main()
