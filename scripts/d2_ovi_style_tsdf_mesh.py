#!/usr/bin/env python3
"""D2: OVI-inspired *triangle mesh visualization* of real RGB-D maps.

This is a viewer/reconstruction experiment, NOT an OVI-MAP reproduction:
- the mesh integrates actual depth/color via Open3D ScalableTSDFVolume;
- all camera poses are explicitly GT/mocap for these D1 datasets (ORACLE);
- semantic surface colors come only from existing D1 YOLOE predictions;
- unknown geometry remains unlabeled, no imagined object/instance identifiers;
- offline HTML and full-resolution colored PLY/GLB are saved.

Caution: a fused TSDF creates interpolated zero-crossing SURFACE triangles
between real depth rays. It does not measure new points. Dynamic objects may
leave ghost surfaces. Do not use these geometry-only maps as Nav2 costmaps.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import cv2
import numpy as np
from scipy.spatial import cKDTree

from d1_multidataset_measured_maps import (
    DATASETS, camera_transform, class_palette, matched_sensor_rows,
)


def label_mesh_vertices(verts, source_points, source_colors, class_rgb,
                        *, max_dist=.055, min_votes=2, purity=.68):
    """Conservative nearest measured semantic-surface vote; 0 means unknown.

    Surface RGB and semantic mask support are independent. Reject vertices
    outside measured semantic coverage or with competing class assignments.
    Labels use 1-based indexes; no tracking or physical instance identity.
    """
    verts=np.asarray(verts,np.float32)
    points=np.asarray(source_points,np.float32)
    cols=np.asarray(source_colors,np.uint8)
    class_rgb=np.asarray(class_rgb,np.uint8)
    if verts.ndim!=2 or verts.shape[1]!=3 or points.ndim!=2 or points.shape[1]!=3:
        raise ValueError("Invalid vertices or D1 measured mask coordinates")
    if cols.shape!=points.shape or class_rgb.ndim!=2 or class_rgb.shape[1]!=3:
        raise ValueError("Invalid predicted RGB palette dimensions")
    if not np.isfinite(verts).all() or not np.isfinite(points).all():
        raise ValueError("NaN in measured 3D reconstruction")
    result=np.zeros(len(verts),dtype=np.int16)
    if len(points)==0:
        return result
    # One source RGB must correspond to exactly one configured class.
    color_sq=((cols.astype(np.int16)[:,None,:]-
               class_rgb.astype(np.int16)[None,:,:])**2).sum(axis=2)
    idx=np.argmin(color_sq,axis=1)
    if not np.all(np.min(color_sq,axis=1)<=3):
        raise ValueError("D1 semantic point colors do not match configured classes")
    tree=cKDTree(points)
    ds,neighbors=tree.query(verts,k=5,distance_upper_bound=max_dist,
                            workers=1)
    if ds.ndim==1:
        ds=ds[:,None];neighbors=neighbors[:,None]
    valid=np.isfinite(ds)&(neighbors<len(points))
    for j in range(len(verts)):
        selected=np.flatnonzero(valid[j])
        if len(selected)<min_votes or ds[j,selected[0]]>.035:
            continue
        # Force majority agreement: no label on a mixed nearby boundary.
        classes=idx[neighbors[j,selected]]
        # Avoid class bias from dense overlapping source mask points:
        # vote on available neighbors only, and abstain on conflict.
        counts=np.bincount(classes,minlength=len(class_rgb))
        winning=int(np.argmax(counts))
        if counts[winning]/len(selected)>=purity:
            result[j]=winning+1
    return result


def centered_limits(vertices, semantic_vertices=None, *, margin=.20):
    """Robust 3D camera framing; never include extreme wall outliers for focus."""
    source=np.asarray(semantic_vertices if semantic_vertices is not None and
                      len(semantic_vertices)>=100 else vertices,dtype=float)
    if source.ndim!=2 or source.shape[1]!=3 or not np.isfinite(source).all():
        raise ValueError("Bad mesh camera framing source")
    lo=np.quantile(source,.025,axis=0)
    hi=np.quantile(source,.975,axis=0)
    lo-=margin;hi+=margin
    mid=(lo+hi)/2
    ranges=np.maximum(hi-lo,.4)
    return mid,ranges


def mesh_display(mesh, labels, color_palette, out, *, max_faces=40000):
    import open3d as o3d
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    triangles=np.asarray(mesh.triangles,dtype=np.int32)
    original_vertices=np.asarray(mesh.vertices,dtype=np.float64)
    original_colors=np.asarray(mesh.vertex_colors,dtype=np.float64)
    if not len(triangles) or len(labels)!=len(original_vertices):
        raise ValueError("Invalid reconstructed mesh for rendering")

    semantic_rgb=np.tile(np.array([.52,.55,.60],dtype=np.float64),
                         (len(labels),1))
    supported=labels>0
    semantic_rgb[supported]=color_palette[labels[supported]-1]/255.
    # Show the full-resolution, reconstructed, color-textured mesh. No
    # interpolation from a coarse point-cloud rendering in this stage.
    semantic_mesh=o3d.geometry.TriangleMesh(mesh)
    semantic_mesh.vertex_colors=o3d.utility.Vector3dVector(semantic_rgb)
    o3d.io.write_triangle_mesh(str(out/"semantic_mesh.ply"),semantic_mesh,
                              write_vertex_colors=True)
    o3d.io.write_triangle_mesh(str(out/"rgb_mesh.ply"),mesh,
                              write_vertex_colors=True)
    import trimesh
    for filename, rgb in (("rgb_mesh.glb",original_colors),
                          ("semantic_mesh.glb",semantic_rgb)):
        mesh_trimesh=trimesh.Trimesh(
           vertices=original_vertices,faces=triangles,
           vertex_colors=np.column_stack([
             np.clip(np.round(rgb*255),0,255).astype(np.uint8),
             np.full(len(rgb),255,dtype=np.uint8)]),process=False)
        mesh_trimesh.export(out/filename)
    # The PLY/GLB above are NOT downsampled. Display-only mesh is reduced to
    # make browser manipulation practical.
    simple=mesh.simplify_quadric_decimation(target_number_of_triangles=max_faces)
    vertices=np.asarray(simple.vertices)
    faces=np.asarray(simple.triangles,dtype=np.int32)
    if len(faces)<100:
        raise RuntimeError("Insufficient triangles after display mesh decimation")
    # Re-associate to original colored mesh after quadric vertex movement.
    _,orig_id=cKDTree(original_vertices).query(vertices,k=1,workers=1)
    rgb=np.clip(original_colors[orig_id],0,1)
    col2=np.clip(semantic_rgb[orig_id],0,1)
    focus=original_vertices[supported] if int(supported.sum())>=100 else None
    center,limits=centered_limits(original_vertices,focus)
    camera_box=limits*1.45
    # A mesh is made of REAL integrated RGB-D rays, but triangle faces are
    # TSDF-interpolated zero-level surfaces, NOT independently measured points.
    n0=faces.shape[0]
    n=np.cross(vertices[faces[:,1]]-vertices[faces[:,0]],
               vertices[faces[:,2]]-vertices[faces[:,0]])
    n=n/np.maximum(np.linalg.norm(n,axis=1)[:,None],1e-8)
    light=np.asarray([.55,-.45,.75]);light/=np.linalg.norm(light)
    illumination=np.clip(.46+.64*np.maximum(n@light,0.),.34,1.)
    fig=plt.figure(figsize=(18,9),facecolor="#151c27")
    for column,(colors,heading) in enumerate(((rgb,"RGB-D TSDF SURFACE"),
                                              (col2,"YOLOE CATEGORY ON MESH")),1):
        ax=fig.add_subplot(1,2,column,projection="3d",proj_type="ortho")
        ax.set_facecolor("#151c27")
        rgb_face=np.mean(colors[faces],axis=1)
        face_colors=np.column_stack((np.clip(rgb_face*illumination[:,None],0,1),
                                     np.ones(n0)))
        collection=Poly3DCollection(vertices[faces],facecolors=face_colors,
                                    linewidths=0,zsort="average",rasterized=True)
        ax.add_collection3d(collection)
        for setlim,v,d in zip((ax.set_xlim,ax.set_ylim,ax.set_zlim),
                              center,camera_box):
            setlim(v-d/2,v+d/2)
        ax.set_box_aspect(camera_box)
        ax.view_init(elev=24,azim=-62,vertical_axis="z")
        ax.set_axis_off()
        ax.set_title(heading,pad=12,color="#f0f5fb",fontsize=14)
    fig.text(.03,.964,"MEASURED RGB-D → TSDF MESH",color="white",
             fontsize=19,weight="bold")
    fig.text(.03,.925,"Source GT/mocap pose ORACLE (not RTAB-Map) · "
              "gray surfaces = no reliable YOLOE class",color="#b3c6d9",fontsize=10)
    fig.text(.03,.033,"3D mesh triangles are interpolated TSDF surfaces; labels are model predictions, NOT physical-instance IDs.",
             color="#ffc5a7",fontsize=10)
    fig.savefig(out/"mesh_vs_semantics.png",dpi=150,facecolor=fig.get_facecolor())
    plt.close(fig)

    import plotly.graph_objects as go
    fig=go.Figure()
    for i,(col,label) in enumerate(((rgb,"Original RGB surface"),
                                     (col2,"Predicted semantic surface"))):
        fig.add_trace(go.Mesh3d(x=vertices[:,0],y=vertices[:,1],z=vertices[:,2],
           i=faces[:,0],j=faces[:,1],k=faces[:,2],
           vertexcolor=np.round(col*255).astype(np.uint8),
           lighting=dict(ambient=.65,diffuse=.65,roughness=.92,specular=.12,
                         fresnel=.05),
           lightposition=dict(x=10,y=-10,z=15),
           flatshading=False,
           name=label,showlegend=True,visible=(i==0)))
    fig.update_layout(
       title="RGB-D TSDF mesh | real sensor measurements + YOLOE category (not instance)",
       template="plotly_dark",height=850,
       updatemenus=[dict(type="buttons",direction="right",x=.08,y=1.10,
          buttons=[
           dict(label="Natural RGB surface",method="update",
                args=[{"visible":[True,False]}]),
           dict(label="YOLOE semantic surface",method="update",
                args=[{"visible":[False,True]}]),
          ])],
       scene=dict(aspectmode="data",
                  xaxis=dict(showbackground=False,showgrid=False,visible=False),
                  yaxis=dict(showbackground=False,showgrid=False,visible=False),
                  zaxis=dict(showbackground=False,showgrid=False,visible=False)),
       margin=dict(l=0,r=0,t=85,b=0))
    fig.write_html(str(out/"mesh_interactive.html"),
                   include_plotlyjs=True,full_html=True)
    return {"display_triangles":len(faces),
            "original_mesh_triangles":len(triangles),
            "original_mesh_vertices":len(original_vertices),
            "vertex_semantically_supported":int(supported.sum()),
            "fraction_mesh_vertices_supported_by_predicted_classes":
              float(supported.mean())}


def run(args):
    import open3d as o3d
    ds=DATASETS[args.dataset]
    rows,source=matched_sensor_rows(Path(args.root))
    if len(rows)<80:
        raise RuntimeError("D2 requires complete real RGB-D dataset")
    source_meta=json.loads((Path(args.semantic_source)/"metrics.json").read_text())
    if source_meta.get("dataset")!=args.dataset or source_meta.get("GT_used_for_mapping") is not True:
        raise ValueError("D2 semantic source does not match original dataset and GT pose convention")
    cloud=o3d.io.read_point_cloud(str(Path(args.semantic_source)/"semantic_predicted.ply"))
    observed=np.asarray(cloud.points,dtype=np.float32)
    observed_colors=np.round(np.asarray(cloud.colors)*255).astype(np.uint8)
    if len(observed)!=source_meta["predicted_semantic_voxels_per_class"]:
        raise RuntimeError("D1 semantically measured source cloud differs from immutable metrics")
    from semantic_map.yoloe26_detector import canonical
    classes=list(dict.fromkeys(canonical(x) for x in source_meta["model_prompt_classes"]))
    palette=class_palette(len(classes))
    depth_intr=ds["K"]
    engine=o3d.pipelines.integration.ScalableTSDFVolume(
       voxel_length=args.tsdf_voxel,
       sdf_trunc=args.sdf_trunc,
       color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
       volume_unit_resolution=16,
       depth_sampling_stride=4)
    count=0
    t0=time.perf_counter()
    for i,row in enumerate(rows):
        if i%args.frame_stride:continue
        depth=cv2.imread(str(row["depth_path"]),cv2.IMREAD_UNCHANGED)
        color=cv2.imread(str(row["rgb_path"]),cv2.IMREAD_COLOR)
        if depth is None or color is None or depth.dtype!=np.uint16 or color.shape[:2]!=depth.shape:
            raise RuntimeError("Missing measured 16bit depth/RGB at original timestamp")
        height,width=depth.shape
        K=o3d.camera.PinholeCameraIntrinsic(width,height,*depth_intr)
        color_rgb=cv2.cvtColor(color,cv2.COLOR_BGR2RGB)
        rgbd=o3d.geometry.RGBDImage.create_from_color_and_depth(
             o3d.geometry.Image(np.ascontiguousarray(color_rgb)),
             o3d.geometry.Image(np.ascontiguousarray(depth)),
             depth_scale=ds["depth_scale"],depth_trunc=args.max_depth,
             convert_rgb_to_intensity=False)
        T=camera_transform(row["pose"],args.dataset)
        engine.integrate(rgbd,K,np.linalg.inv(T))
        count+=1
        if count%30==0:
            print("D2_TSDF_REAL_FRAMES",args.dataset,count,flush=True)
    mesh=engine.extract_triangle_mesh()
    if len(mesh.triangles)<5000 or len(mesh.vertices)<2000:
        raise RuntimeError("TSDF failed to reconstruct a real scene mesh")
    if len(mesh.vertex_colors)!=len(mesh.vertices):
        raise RuntimeError("TSDF failed to reconstruct original sensor colors")
    mesh.compute_vertex_normals()
    vertices=np.asarray(mesh.vertices)
    labels=label_mesh_vertices(vertices,observed,observed_colors,palette,
                               max_dist=args.semantic_max_dist)
    if len(observed)>0 and int(np.count_nonzero(labels))<50:
        raise RuntimeError("Measured YOLOE semantic cloud cannot register to reconstructed mesh")
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    view=mesh_display(mesh,labels,palette,out,max_faces=args.display_triangles)
    np.savez_compressed(out/"mesh_vertex_semantics.npz",
       classes=np.asarray(classes,dtype="U64"),vertices=vertices.astype(np.float32),
       class_idx_plus_one=labels)
    report={
       "experiment":"D2_REAL_RGBD_TSDF_MESH_AND_CONSERVATIVE_SEMANTIC_OVERLAY",
       "dataset":args.dataset,
       "pose_source":"GT/mocap camera poses (ORACLE) from original D1 dataset",
       "RTABMap_performed_in_D2":False,
       "known_GT_pose_used_for_integration":True,
       "original_real_RGBD_synchronized_frames":len(rows),
       "integrated_measured_RGBD_frames":count,
       "frame_stride":args.frame_stride,
       "TSDF_voxel_m":args.tsdf_voxel,
       "TSDF_truncation_m":args.sdf_trunc,
       "input_D1_semantic_voxels":len(observed),
       "semantic_nearest_surface_max_distance_m":args.semantic_max_dist,
       "semantic_category_names":classes,
       "mesh_render":view,
       "runtime_seconds":float(time.perf_counter()-t0),
       "physical_instance_labels_constructed":False,
       "semantic_class_accuracy":"NOT_EVALUATED_NO_GT",
       "IDF1":"NOT_EVALUATED_NO_PHYSICAL_GT",
       "geometry_mode":"surface_from_TSDF_zero_crossing_interpolated_between_measured_depth_rays",
       "unknown_geometry_policy":"unlabeled neutral surface, no invented classes",
       "dynamic_ghosting_not_corrected":True,
       "outputs":["rgb_mesh.ply","rgb_mesh.glb","semantic_mesh.ply",
                  "semantic_mesh.glb","mesh_vertex_semantics.npz",
                  "mesh_vs_semantics.png","mesh_interactive.html","d2_metrics.json"],
    }
    (out/"d2_metrics.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print("D2_REAL_TSDF_MESH_RESULT",json.dumps(report,ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",required=True,choices=DATASETS)
    p.add_argument("--root",required=True)
    p.add_argument("--semantic-source",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--frame-stride",type=int,default=4)
    p.add_argument("--tsdf-voxel",type=float,default=.02)
    p.add_argument("--sdf-trunc",type=float,default=.08)
    p.add_argument("--max-depth",type=float,default=4.5)
    p.add_argument("--semantic-max-dist",type=float,default=.055)
    p.add_argument("--display-triangles",type=int,default=40000)
    a=p.parse_args()
    if a.frame_stride<1 or not .008<=a.tsdf_voxel<=.1 or a.sdf_trunc<a.tsdf_voxel*2:
        p.error("Invalid actual integration parameters")
    run(a)


if __name__=="__main__":
    main()
