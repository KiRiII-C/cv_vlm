#!/usr/bin/env python3
"""D4: full original 2000-frame NICE-SLAM Replica office0; robust map-quality audit.

Integrate 500 actual photorealistic RGB-D frames; 200 original RGB frames for
YOLOE-26x, 25 independent depth frames withheld from TSDF.
Unlike the prior D3 100-frame downsample, D4 uses the ORIGINAL OVI-MAP input
distribution. GT poses are supplied, so this is NOT a RTAB-Map SLAM benchmark.
Raycast the reconstructed mesh back into held-out RGB-D camera views. This
measures coverage and z-depth error, exposing holes hidden by NN point scores.
An independently supplied GT reference mesh is used for one-way surface check.
"""
from __future__ import annotations
import argparse, json, time
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial import cKDTree
from d3_replica_office0_mesh_patches import (
    mask_to_measured_world, transfer_semantic, segment_surface_patches,
    render, palette,
)


def read_inputs(root):
    root=Path(root)
    provenance=json.loads((root/"d4_provenance.json").read_text())
    data=root/"Replica"
    camera=json.loads((data/"cam_params.json").read_text())["camera"]
    sequence=data/"office0"
    poses=np.loadtxt(sequence/"traj.txt").reshape(-1,4,4)
    if len(poses)!=2000 or not np.isfinite(poses).all():
        raise RuntimeError("Official complete 2000-frame Replica trajectory not verified")
    if not np.allclose(poses[:,3,:],np.array([0.,0.,0.,1.]),atol=1e-4):
        raise RuntimeError("Incorrect OVI Replica original pose format")
    K=np.array([[camera["fx"],0,camera["cx"]],
                [0,camera["fy"],camera["cy"]],
                [0,0,1]],float)
    if camera["fx"]<300 or camera["scale"]<=1:
        raise ValueError("Bad intrinsics or source depth-unit scale")
    for kind in ("integration_original_indices","YOLOE_original_indices",
                 "independent_depth_validation_indices"):
        if not provenance[kind]:
            raise ValueError("Original source missing "+kind)
    train=set(provenance["integration_original_indices"])
    heldout=set(provenance["independent_depth_validation_indices"])
    if train & heldout:
        raise RuntimeError("Held-out depth appeared in mapping")
    for i in sorted(train|heldout|set(provenance["YOLOE_original_indices"])):
        if not (sequence/"results"/f"frame{i:06d}.jpg").is_file() or not (
                sequence/"results"/f"depth{i:06d}.png").is_file():
            raise FileNotFoundError(f"Incomplete original Replica source for {i}")
    return sequence, data/"office0_mesh.ply",K,poses,float(camera["scale"]),provenance


def get_frame(folder,idx,depth_scale):
    bgr=cv2.imread(str(folder/"results"/f"frame{idx:06d}.jpg"),
                   cv2.IMREAD_COLOR)
    raw=cv2.imread(str(folder/"results"/f"depth{idx:06d}.png"),
                   cv2.IMREAD_UNCHANGED)
    if bgr is None or raw is None or raw.dtype!=np.uint16 or raw.shape!=bgr.shape[:2]:
        raise RuntimeError(f"Missing or invalid measured Replica RGB-Depth {idx}")
    z=raw.astype(np.float32)/depth_scale
    return bgr,z


def cast_rays(mesh, intrinsics, world_to_camera, width, height):
    """Return camera-z range and reconstructed RGB/face indices per image pixel."""
    import open3d as o3d
    vertices=np.asarray(mesh.vertices,dtype=np.float32)
    faces=np.asarray(mesh.triangles,dtype=np.int32)
    original_colors=np.asarray(mesh.vertex_colors,dtype=np.float32)
    if len(vertices)<1000 or len(faces)<1000:
        raise ValueError("Mesh must have measured RGB and triangle surfaces")
    tmesh=o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    sc=o3d.t.geometry.RaycastingScene()
    sc.add_triangles(tmesh)
    return cast_with_scene(sc,vertices,faces,original_colors,
                           intrinsics,world_to_camera,width,height)


def cast_with_scene(scene,vertices,faces,original_colors,
                    intrinsics,world_to_camera,width,height,category=None,patches=None):
    import open3d as o3d
    T=np.asarray(world_to_camera,dtype=np.float32)
    K=np.asarray(intrinsics,dtype=np.float32)
    rays=scene.create_rays_pinhole(o3d.core.Tensor(K),
                                  o3d.core.Tensor(T),width_px=int(width),
                                  height_px=int(height))
    ans=scene.cast_rays(rays)
    t=ans["t_hit"].numpy().reshape(-1)
    face_id=ans["primitive_ids"].numpy().reshape(-1).astype(np.int64)
    uv=ans["primitive_uvs"].numpy().reshape(-1,2)
    N=len(t)
    ok=np.isfinite(t)&(t>0)&(face_id<len(faces))
    idx=np.flatnonzero(ok)
    out_depth=np.full(N,np.nan,dtype=np.float32)
    rgb=np.zeros((N,3),dtype=np.uint8)
    sem=np.zeros((N,3),dtype=np.uint8)
    patchrgb=np.zeros((N,3),dtype=np.uint8)
    if len(idx):
        batch=faces[face_id[idx]]
        uv0=uv[idx,0];uv1=uv[idx,1]
        bary=np.stack([1-uv0-uv1,uv0,uv1],axis=1).astype(np.float32)
        cols=original_colors[batch]
        rgb[idx]=np.clip(np.einsum("ij,ijk->ik",bary,cols)*255,0,255).astype(np.uint8)
        # Camera depth from ray hit in original w2c projection, not t_hit
        # (which depends on the ray direction normalization convention).
        origin=rays.numpy().reshape(-1,6)[idx,:3]
        direction=rays.numpy().reshape(-1,6)[idx,3:]
        hit=origin+direction*t[idx,None]
        zcam=hit @ T[2,:3]+T[2,3]
        out_depth[idx]=zcam
        if category is not None:
            names=category[batch]
            # Nearest actual vertex label (not barycentric category blending).
            sel=np.argmax(bary,axis=1)
            values=names[np.arange(len(idx)),sel]
            col=palette(max(int(category.max()),3))
            sem[idx]=col[values]
            unknown=values==0
            sem[idx[unknown]]=np.array([191,196,204],np.uint8)
        if patches is not None:
            face_patch=patches[face_id[idx]]
            col=palette(max(3,int(patches.max())+1))
            patchrgb[idx]=col[face_patch]
            patchrgb[idx[face_patch==0]]=[150,160,173]
    return {"z":out_depth.reshape(height,width),
            "hit":ok.reshape(height,width),
            "rgb":rgb.reshape(height,width,3),
            "semantic":sem.reshape(height,width,3),
            "patches":patchrgb.reshape(height,width,3)}


def raycast_measured_quality(mesh,folder,frame_ids,poses,K,depth_scale,
                             *,stride=8,max_depth=7.):
    import open3d as o3d
    verts=np.asarray(mesh.vertices,dtype=np.float32)
    faces=np.asarray(mesh.triangles,dtype=np.int32)
    C=np.asarray(mesh.vertex_colors,dtype=np.float32)
    sc=o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    stats=[];all_errors=[];total_valid=0;total_hit=0
    for idx in frame_ids:
        _,depth=get_frame(folder,idx,depth_scale)
        H,W=depth.shape
        h,w=H//stride,W//stride
        k=np.array(K,dtype=np.float32);k[:2,:]/=stride
        world_to_camera=np.linalg.inv(poses[idx]).astype(np.float32)
        result=cast_with_scene(sc,verts,faces,C,k,world_to_camera,w,h)
        target=cv2.resize(depth,(w,h),interpolation=cv2.INTER_NEAREST)
        observed=np.isfinite(target)&(target>.25)&(target<max_depth)
        cover=observed & result["hit"] & np.isfinite(result["z"]) & (result["z"]>.25)
        error=np.abs(result["z"][cover]-target[cover])
        if not len(error):
            raise RuntimeError(f"No raycast depth support for real withheld frame {idx}")
        total_valid+=int(observed.sum());total_hit+=int(cover.sum())
        all_errors.append(error)
        stats.append({"original_frame":int(idx),
                      "valid_sensor_depth_pixels":int(observed.sum()),
                      "raycast_supported_pixels":int(cover.sum()),
                      "fraction_depth_supported":float(cover.sum()/max(1,observed.sum())),
                      "median_z_abs_error_m":float(np.median(error)),
                      "p95_z_abs_error_m":float(np.percentile(error,95)),
                      "fraction_hits_depth_residual_under_5cm":float((error<.05).mean())})
    errors=np.concatenate(all_errors)
    return {"heldout_original_frames":len(stats),
            "depth_sampling_stride":stride,
            "observed_valid_depth_samples":total_valid,
            "supported_samples_by_reconstructed_mesh":total_hit,
            "raycast_depth_support_fraction":float(total_hit/total_valid),
            "median_abs_sensor_depth_error_m_on_supported":float(np.median(errors)),
            "p95_abs_sensor_depth_error_m_on_supported":float(np.percentile(errors,95)),
            "raycast_within_5cm_fraction_on_supported":float((errors<.05).mean()),
            "raycast_within_10cm_fraction_on_supported":float((errors<.10).mean()),
            "per_view":stats,
            "GT_reference_camera_poses_used":True,
            "not_SLAM_ATE":True}


def gt_mesh_oneway_distance(mesh, reference, *,max_samples=18000):
    import open3d as o3d
    if not reference.is_file():
        raise FileNotFoundError("Official independent office0 reference mesh")
    gt=o3d.io.read_triangle_mesh(str(reference))
    if len(gt.vertices)<5000 or len(gt.triangles)<5000:
        raise RuntimeError("Replica original reference mesh invalid")
    xyz=np.asarray(mesh.vertices)
    pick=np.linspace(0,len(xyz)-1,min(len(xyz),max_samples),dtype=int)
    # Point-to-TRIANGLE distance, not nearest-vertex. Only from reconstructed
    # surface to GT: reverse would unfairly penalize unobserved GT room areas.
    ts=o3d.t.geometry.RaycastingScene()
    ts.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(gt))
    d=ts.compute_distance(o3d.core.Tensor(xyz[pick].astype(np.float32))).numpy()
    return {"official_independent_GT_mesh_vertices":len(gt.vertices),
            "official_independent_GT_mesh_triangles":len(gt.triangles),
            "reconstruction_sampled_surface_vertices":len(pick),
            "reconstructed_to_independent_GT_triangle_distance_median":float(np.median(d)),
            "reconstructed_to_independent_GT_triangle_distance_p95":float(np.percentile(d,95)),
            "reconstructed_points_within_GT_5cm_fraction":float((d<.05).mean()),
            "evaluation":"one_way_only_without_unobserved_room_penalty",
            "not_complete_3D_chamfer_or_instance_accuracy":True}


def comparison_from_camera(mesh,folder,poses,K,scale,frames,semantic,patches,out):
    import open3d as o3d
    from PIL import Image, ImageDraw, ImageFont
    out=Path(out)
    verts=np.asarray(mesh.vertices,dtype=np.float32)
    faces=np.asarray(mesh.triangles,dtype=np.int32)
    cols=np.asarray(mesh.vertex_colors,dtype=np.float32)
    sc=o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    # Each picture is the SAME camera: reference/meshed render/geometry/YOLOE.
    pages=[]
    for i in frames:
        bgr,depth=get_frame(folder,i,scale)
        h,w=bgr.shape[:2]
        downw,downh=w//2,h//2
        k=np.asarray(K,dtype=np.float32).copy()
        k[:2,:]/=2
        r=cast_with_scene(sc,verts,faces,cols,k,np.linalg.inv(poses[i]),
                          downw,downh,semantic,patches)
        reference=cv2.cvtColor(cv2.resize(bgr,(downw,downh)),cv2.COLOR_BGR2RGB)
        panels=[reference,r["rgb"],r["patches"],r["semantic"]]
        # Visible unknown mesh stays gray to reveal incomplete segmentation.
        result=np.concatenate(panels,axis=1)
        pages.append(result)
    canvas=np.concatenate(pages,axis=0)
    im=Image.fromarray(canvas)
    draw=ImageDraw.Draw(im)
    text="REFERENCE RGB / TSDF RGB / GEOMETRY REGIONS / YOLOE SEMANTIC"
    draw.rectangle((0,0,im.width,29),fill=(11,23,39))
    draw.text((18,7),text,fill=(245,248,252))
    im.save(out/"d4_camera_aligned_comparison.jpg",quality=90,optimize=True)
    # A small faithful preview can be inspected directly on GitHub while the
    # full-resolution aligned view remains available in Actions artifacts.
    thumb=im.resize((1200,680),Image.Resampling.LANCZOS)
    thumb.save(out/"d4_comparison_small.jpg",quality=82,optimize=True)
    # Source is 100% original Replica render data; jpg is VIEW ONLY.
    return {"comparison_original_frame_ids":frames,
            "reference_reconstructed_same_pose":True,
            "image_original_depth_not_modified":True,
            "output":"d4_camera_aligned_comparison.jpg"}


def main():
    import open3d as o3d
    from semantic_map.yoloe26_detector import YOLOE26Segmenter, class_list, canonical
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--voxel",type=float,default=.018)
    p.add_argument("--max-depth",type=float,default=7.)
    p.add_argument("--confidence",type=float,default=.30)
    args=p.parse_args()
    source,gtpath,K,poses,scale,manifest=read_inputs(args.source)
    args.out.mkdir(parents=True,exist_ok=True)
    t0=time.perf_counter()
    indices=manifest["integration_original_indices"]
    seg_indices=set(manifest["YOLOE_original_indices"])
    hold=manifest["independent_depth_validation_indices"]
    model=YOLOE26Segmenter("yoloe-26x-seg.pt",
                           class_list("config/yoloe26_classes.json"))
    classes=list(dict.fromkeys(canonical(x) for x in
                 class_list("config/yoloe26_classes.json")))
    intr=o3d.camera.PinholeCameraIntrinsic(
      1200,680,float(K[0,0]),float(K[1,1]),float(K[0,2]),float(K[1,2]))
    tsdf=o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=args.voxel,sdf_trunc=args.voxel*4,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        volume_unit_resolution=16,depth_sampling_stride=4)
    sem_xyz=[];sem_cls=[];cls_counts=Counter()
    all_frames=sorted(set(indices)|seg_indices)
    count=0;n_infer=0
    for frame in all_frames:
        bgr,z=get_frame(source,frame,scale)
        if bgr.shape[:2]!=(680,1200):
            raise RuntimeError("Unexpected original Replica source resolution")
        T=np.asarray(poses[frame],np.float64)
        if frame in indices:
            z0=z.copy()
            z0[(z0<.2)|(z0>args.max_depth)|(~np.isfinite(z0))]=0
            rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            im=o3d.geometry.RGBDImage.create_from_color_and_depth(
              o3d.geometry.Image(np.ascontiguousarray(rgb)),
              o3d.geometry.Image(np.ascontiguousarray(z0)),
              depth_scale=1.,depth_trunc=args.max_depth,
              convert_rgb_to_intensity=False)
            tsdf.integrate(im,intr,np.linalg.inv(T))
            count+=1
        if frame in seg_indices:
            n_infer+=1
            res=model.predict(bgr,conf=args.confidence,device="cpu",imgsz=640)
            if res.masks is not None and res.boxes is not None:
                for mask,cat,confidence in zip(res.masks.data.cpu().numpy(),
                  res.boxes.cls.int().cpu().numpy(),
                  res.boxes.conf.cpu().numpy()):
                    lbl=canonical(model.class_name(cat))
                    if lbl not in classes:raise RuntimeError("Invalid YOLOE class output")
                    mask=(mask>.5).astype(np.uint8)
                    if mask.shape!=z.shape:
                        mask=cv2.resize(mask,(1200,680),interpolation=cv2.INTER_NEAREST)
                    pts=mask_to_measured_world(mask.astype(bool),z,
                        np.isfinite(z)&(z>.2)&(z<args.max_depth),K,T,max_pts=1700)
                    if len(pts)<55:continue
                    sem_xyz.append(pts)
                    sem_cls.append(np.full(len(pts),classes.index(lbl)+1,np.int16))
                    cls_counts[lbl]+=1
        if count and count%75==0 and frame in indices:
            print("D4_FULL_REPLICA_MESH_PROGRESS",count,len(indices),
                  n_infer,flush=True)
    mesh=tsdf.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    if len(mesh.triangles)<100000:
        raise RuntimeError("500 original measured frames did not produce a dense mesh")
    v=np.asarray(mesh.vertices)
    faces=np.asarray(mesh.triangles)
    sx=np.concatenate(sem_xyz) if sem_xyz else np.empty((0,3),np.float32)
    sc=np.concatenate(sem_cls) if sem_cls else np.empty((0,),np.int16)
    sem=transfer_semantic(v,sx,sc,max_distance=.065,
                          k=5,min_votes=3,purity=.6)
    simple=mesh.simplify_quadric_decimation(target_number_of_triangles=125000)
    sv=np.asarray(simple.vertices);sf=np.asarray(simple.triangles)
    face_regions,patch_meta=segment_surface_patches(sv,sf,
           np.asarray(simple.vertex_colors),normal_cos=.82,color_delta=.40,
           min_faces=150,max_regions=75)
    full_faces_center=v[faces].mean(axis=1)
    simple_faces_center=sv[sf].mean(axis=1)
    _,map_id=cKDTree(simple_faces_center).query(full_faces_center,k=1)
    reg=face_regions[map_id]
    render_meta=render(mesh,sem,reg,args.out,classes)
    quality=raycast_measured_quality(mesh,source,hold,poses,K,scale,max_depth=args.max_depth)
    GT=gt_mesh_oneway_distance(mesh,gtpath)
    preview=comparison_from_camera(mesh,source,poses,K,scale,
                           [0,400,800,1200],sem,reg,args.out)
    summary={
       "experiment":"D4_REAL_OFFICIAL_FULL_REPLICA_OFFICE0",
       "source":"NICE-SLAM official Replica.zip 12.4GB selective HTTP-range extraction",
       "scene":"Replica office0",
       "rendered_synthetic_sensor_not_real_camera":True,
       "full_source_frames_available":2000,
       "actual_original_frames_integrated":count,
       "actual_YOLOE_inferences":n_infer,
       "actual_valid_3D_YOLOE_masks":int(sum(cls_counts.values())),
       "class_mask_counts":dict(cls_counts),
       "tsdf_voxel_m":args.voxel,
       "source_depth_scale":scale,
       "camera_pose_source":"original GT/c2w, inverted for Open3D integration",
       "RTABMap_run":False,
       "physical_instance_accuracy_evaluated":False,
       "geometric_regions_are_physical_instances":False,
       "mesh":render_meta,
       "geometric_patch_statistics":patch_meta,
       "mesh_face_geometric_region_coverage":float((reg>0).mean()),
       "held_out_raycast_reprojection":quality,
       "independent_gt_mesh_surface":GT,
       "visible_camera_comparisons":preview,
       "seconds_elapsed":round(time.perf_counter()-t0,3),
       "no_hallucinated_unknown_geometry":True,
       "semantic_accuracy":"UNKNOWN_NO_MATCHED_GT",
       "object_IDF1":"NOT_EVALUATED",
       "results_are_not_direct_OVI_MAP_model_performance":True
    }
    (args.out/"d4_metrics.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n")
    print("D4_REAL_FULL_REPLICA_SUMMARY",json.dumps({
      k:summary[k] for k in ("actual_original_frames_integrated",
          "actual_YOLOE_inferences","actual_valid_3D_YOLOE_masks","mesh",
          "mesh_face_geometric_region_coverage")},ensure_ascii=False),flush=True)
    print("D4_RAYCAST_DEPTH_HELDOUT",json.dumps({
        k:quality[k] for k in ("heldout_original_frames","raycast_depth_support_fraction",
            "median_abs_sensor_depth_error_m_on_supported",
            "p95_abs_sensor_depth_error_m_on_supported")}),flush=True)
    print("D4_INDEPENDENT_GT_SURFACE",json.dumps(GT),flush=True)


if __name__=="__main__":
    main()
