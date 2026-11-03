#!/usr/bin/env python3
"""D3: reproducible Replica office0 mesh/geometry patches/YOLOE surface comparison.

Dataset: public research-only 100-frame derived subset of NICE-SLAM Replica;
photorealistic *rendered* RGB-D, NOT a raw physical camera sequence.
The provided w2c poses are used as ORACLE, not RTAB-Map.
A class-agnostic mesh patch is NOT guaranteed to be a physical object.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
from collections import Counter
import numpy as np
import cv2
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

def load_replica(scene):
    scene=Path(scene)
    depth=np.load(scene/"depth.npy",mmap_mode="r")
    valid=np.load(scene/"valid_mask.npy",mmap_mode="r")
    world_to_cam=np.load(scene/"extrinsics.npy",mmap_mode="r")
    cameras=np.load(scene/"intrinsics.npy",mmap_mode="r")
    meta=json.loads((scene/"meta.json").read_text())
    images=sorted((scene/"images").glob("*.jpg"))
    n=len(images)
    if n!=100 or depth.shape!=(100,680,1200) or valid.shape!=depth.shape:
        raise ValueError("Expected complete REAL Replica-rendered 100-frame office0 source")
    if cameras.shape!=(n,3,3) or world_to_cam.shape!=(n,4,4):
        raise ValueError("Pose or camera intrinsics mismatch")
    if len(meta.get("frame_ids",[]))!=n:
        raise ValueError("Missing original frame-id provenance")
    if len(set(meta["frame_ids"]))!=n:
        raise ValueError("Duplicated original Replica source frames")
    fx=cameras[:,0,0];fy=cameras[:,1,1]
    if np.max(np.abs(fx-600))>2 or np.max(np.abs(fy-600))>2:
        raise ValueError("Not expected original Replica camera calibration")
    if not np.all(np.isfinite(world_to_cam)) or not np.allclose(
          np.asarray(world_to_cam)[:,3,:],np.array([0,0,0,1]),atol=1e-5):
        raise ValueError("Invalid world-to-camera matrix")
    for T in world_to_cam:
        if not np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=.01):
            raise ValueError("Invalid Replica rotation")
    return images,depth,valid,world_to_cam,cameras,meta

def mask_to_measured_world(mask, z, valid, K, Tcw, max_pts=3000):
    """No pseudo points: return actual valid z-depth pixels under a 2D mask."""
    yy,xx=np.where(mask & valid & np.isfinite(z) & (z>.15) & (z<8.))
    if len(xx)<55:
        return np.empty((0,3),dtype=np.float32)
    dz=z[yy,xx]
    median=float(np.median(dz))
    good=np.abs(dz-median)<max(.18,.16*median)
    yy,xx,dz=yy[good],xx[good],dz[good]
    if len(xx)<55:
        return np.empty((0,3),dtype=np.float32)
    if len(xx)>max_pts:
        choose=np.linspace(0,len(xx)-1,max_pts,dtype=np.int32)
        xx,yy,dz=xx[choose],yy[choose],dz[choose]
    cam=np.column_stack(((xx-K[0,2])*dz/K[0,0],
                         (yy-K[1,2])*dz/K[1,1],dz))
    world=(cam @ Tcw[:3,:3].T+Tcw[:3,3]).astype(np.float32)
    return world

def segment_surface_patches(vertices,faces,colors, *, normal_cos=.90,
                            color_delta=.35,min_faces=100,max_regions=40):
    """Class-agnostic local mesh regions, NOT full instance segmentation.

    Polygon adjacency with normal+appearance continuity. No image model,
    semantic class, physical GT identifier, or object shape prior is used.
    """
    verts=np.asarray(vertices,np.float32)
    faces=np.asarray(faces,np.int32)
    rgb=np.asarray(colors,np.float32)
    n=len(faces)
    if n<100 or verts.shape[1]!=3 or rgb.shape!=verts.shape:
        raise ValueError("Bad mesh data for patch proposal segmentation")
    if faces.min()<0 or faces.max()>=len(verts) or not np.isfinite(verts).all():
        raise ValueError("Invalid mesh topology")
    e1=verts[faces[:,1]]-verts[faces[:,0]]
    e2=verts[faces[:,2]]-verts[faces[:,0]]
    normal=np.cross(e1,e2)
    normal/=np.maximum(np.linalg.norm(normal,axis=1)[:,None],1e-10)
    mean_rgb=rgb[faces].mean(axis=1)
    edges=np.concatenate((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]))
    edges=np.sort(edges,axis=1)
    owner=np.tile(np.arange(n,dtype=np.int32),3)
    order=np.lexsort((edges[:,1],edges[:,0]))
    edges,owner=edges[order],owner[order]
    matching=np.flatnonzero(np.all(edges[1:]==edges[:-1],axis=1))
    a=owner[matching];b=owner[matching+1]
    keep=(a!=b)
    keep &= (np.einsum("ij,ij->i",normal[a],normal[b])>=normal_cos)
    keep &= (np.linalg.norm(mean_rgb[a]-mean_rgb[b],axis=1)<=color_delta)
    a,b=a[keep],b[keep]
    graph=sp.coo_matrix((np.ones(len(a)*2,dtype=np.uint8),
                      (np.concatenate((a,b)),np.concatenate((b,a)))),
                      shape=(n,n)).tocsr()
    count,grp=connected_components(graph,directed=False)
    sizes=np.bincount(grp,minlength=count)
    keep_groups=np.flatnonzero(sizes>=min_faces)
    keep_groups=keep_groups[np.argsort(-sizes[keep_groups])][:max_regions]
    # Region 0: not enough evidence to present a distinct coherent patch.
    mapped=np.zeros(count,dtype=np.int32)
    mapped[keep_groups]=np.arange(1,len(keep_groups)+1)
    patches=mapped[grp]
    return patches,{"candidate_components_before_filters":int(count),
                     "displayed_geometric_regions":int(len(keep_groups)),
                     "coverage_fraction_faces":float((patches>0).mean()),
                     "max_region_triangle_count":int(sizes.max()),
                     "class_agnostic":True,
                     "physical_instance_ids_claimed":False}

def transfer_semantic(verts,semantic_xyz,semantic_labels,*,max_distance=.052,
                      k=5,min_votes=3,purity=.70):
    """Conservative nearest-measured-RGBD instance mask support on mesh.
    Labels are *category indices*, not physical ID/instance GT.
    """
    result=np.zeros(len(verts),dtype=np.int32)
    if not len(semantic_xyz):return result
    tree=cKDTree(semantic_xyz)
    distances,inds=tree.query(verts,k=k,distance_upper_bound=max_distance)
    if distances.ndim==1:
        distances=distances[:,None];inds=inds[:,None]
    good=(np.isfinite(distances))&(inds<len(semantic_xyz))
    # Vectorized majority vote for arbitrary number of configured classes.
    ncls=int(semantic_labels.max())
    votes=np.zeros((len(verts),ncls+1),dtype=np.uint8)
    for j in range(k):
        idx=np.flatnonzero(good[:,j])
        if len(idx):
            votes[idx,semantic_labels[inds[idx,j]]] += 1
    votes[:,0]=0
    winner=votes.argmax(axis=1)
    strength=votes[np.arange(len(verts)),winner]
    valid=(strength>=min_votes)&(strength/np.maximum(1,good.sum(axis=1))>=purity)
    valid &= (distances[:,0]<.04)
    result[valid]=winner[valid]
    return result

def voxel_surface_distance(mesh_vertices,images,depth,valid,world_to_cam,K,
                           test_indices,max_sample=7000):
    """Heldout measured depth→mesh-vertex NN distances, NOT GT mesh accuracy."""
    tree=cKDTree(np.asarray(mesh_vertices))
    measures=[]
    supports=[]
    for i in test_indices:
        z=np.asarray(depth[i]);mask=np.asarray(valid[i])
        yy,xx=np.where(mask[::8,::8]&np.isfinite(z[::8,::8])&
                        (z[::8,::8]>.15)&(z[::8,::8]<6.))
        if len(xx)<10:continue
        yy=yy*8;xx=xx*8
        if len(xx)>max_sample:
            selected=np.linspace(0,len(xx)-1,max_sample,dtype=int)
            yy,xx=yy[selected],xx[selected]
        zz=z[yy,xx]
        k=K[i];w2c=np.asarray(world_to_cam[i])
        pose=np.linalg.inv(w2c)
        pts=np.column_stack(((xx-k[0,2])*zz/k[0,0],
                             (yy-k[1,2])*zz/k[1,1],zz))
        pts=pts@pose[:3,:3].T+pose[:3,3]
        distances,_=tree.query(pts,k=1)
        measures.append(distances)
        supports.append(len(distances))
    if not measures:raise RuntimeError("No original held-out depth samples")
    d=np.concatenate(measures)
    return {"heldout_frame_count":len(measures),
            "heldout_projected_measured_points":len(d),
            "median_nn_vertex_m":float(np.median(d)),
            "p95_nn_vertex_m":float(np.percentile(d,95)),
            "coverage_5cm":float((d<.05).mean()),
            "coverage_10cm":float((d<.10).mean()),
            "metric_type":"heldout_measured_sensor_points_to_reconstructed_mesh_VERTEX_NN",
            "not_independent_geometry_GT":True}

def palette(n):
    from matplotlib import colormaps
    cm=colormaps["tab20"]
    return np.round(np.asarray([cm(i%20)[:3] for i in range(n+1)])*255).astype(np.uint8)

def render(mesh,labels,patches,out,classes):
    import open3d as o3d
    import trimesh
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import plotly.graph_objects as go
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    v=np.asarray(mesh.vertices);f=np.asarray(mesh.triangles)
    rgb=np.clip(np.asarray(mesh.vertex_colors),0,1)
    if len(labels)!=len(v) or len(patches)!=len(f):raise ValueError("Mesh label alignment error")
    C=palette(max(len(classes),int(labels.max())))
    sem=np.tile(np.array([.78,.79,.81]),(len(v),1))
    idx=labels>0
    sem[idx]=C[labels[idx]]/255.
    patches_p=palette(max(int(patches.max())+2,4))
    patchface=np.tile(np.array([.64,.67,.72]),(len(f),1))
    if len(patches):
        patchface[patches>0]=patches_p[patches[patches>0]]/255.
    # Mesh files are complete and carry faces rather than point-only vertices.
    for filename,cols in (("rgb_mesh",rgb),("semantic_mesh",sem)):
        m=o3d.geometry.TriangleMesh(mesh)
        m.vertex_colors=o3d.utility.Vector3dVector(cols)
        if not o3d.io.write_triangle_mesh(str(out/(filename+".ply")),m):
            raise RuntimeError("Could not write triangle PLY")
        colors=np.column_stack((np.round(cols*255).astype(np.uint8),
                                np.full(len(v),255,dtype=np.uint8)))
        tm=trimesh.Trimesh(vertices=v,faces=f,vertex_colors=colors,process=False)
        tm.export(out/(filename+".glb"))
    np.savez_compressed(out/"mesh_patch_and_semantic_labels.npz",
       vertices=v.astype(np.float32),faces=f.astype(np.int32),
       surface_patch_ids=patches.astype(np.int32),
       category_ids=labels.astype(np.int32),
       category_names=np.asarray(["unknown"]+classes,dtype="U48"))
    # Plotly: three separately selectable layers; preserve original 3D geometry.
    # Display simplification affects only interactivity; raw mesh unchanged.
    simple=mesh.simplify_quadric_decimation(target_number_of_triangles=45000)
    sv=np.asarray(simple.vertices);sf=np.asarray(simple.triangles)
    _,vx=cKDTree(v).query(sv,k=1)
    scol=np.clip(.05+np.power(rgb[vx],.80),0,1)
    ssem=sem[vx]
    # The displayed surface-patch colors are face labels, not vertex-smoothed.
    bary=sv[sf].mean(axis=1)
    original_bary=v[f].mean(axis=1)
    _,fi=cKDTree(original_bary).query(bary,k=1)
    spatch=patchface[fi]
    traces=[]
    for mode,colors,kind in (("RGB","rgb","vertex"),
                             ("Geometric regions",spatch,"face"),
                             ("YOLOE semantic",ssem,"vertex")):
        kw=dict(x=sv[:,0],y=sv[:,1],z=sv[:,2],
                i=sf[:,0],j=sf[:,1],k=sf[:,2],name=mode,
                lighting=dict(ambient=.68,diffuse=.7,specular=.12,roughness=.85),
                visible=(len(traces)==0))
        if kind=="vertex":
            C1=scol if colors=="rgb" else colors
            kw["vertexcolor"]=np.round(C1*255).astype(np.uint8)
        else:
            kw["facecolor"]=["rgb(%d,%d,%d)"%tuple(c) for c in
                             np.round(colors*255).astype(np.uint8)]
        traces.append(go.Mesh3d(**kw))
    fig=go.Figure(data=traces)
    fig.update_layout(template="plotly_dark",height=850,
      title="Replica office0 — TSDF surface / CLASS-INDEPENDENT patches / YOLOE categories",
      updatemenus=[dict(type="buttons",direction="right",x=.02,y=1.08,
         buttons=[dict(label=name,method="update",
                  args=[{"visible":[j==i for j in range(3)]}])
                for i,name in enumerate(("Natural RGB","Geometry-only patches","YOLOE classes"))])],
      scene=dict(aspectmode="data",xaxis=dict(visible=False),
                 yaxis=dict(visible=False),zaxis=dict(visible=False)),
      margin=dict(l=0,r=0,t=80,b=0))
    fig.write_html(out/"interactive_office0.html",include_plotlyjs="cdn")
    # Static overview: identical broad geometric framing across all panels.
    fig=plt.figure(figsize=(19,7.5),facecolor="#13202d",dpi=140)
    # Sparse enough for CPU matplotlib; no physical scene changes are concealed.
    sampled=np.linspace(0,len(f)-1,min(18000,len(f)),dtype=np.int32)
    v1=v[f[sampled]]
    ff=f[sampled]
    cent=np.median(v,axis=0)
    qlo=np.quantile(v,.01,axis=0);qhi=np.quantile(v,.99,axis=0)
    extent=np.maximum(qhi-qlo,.5)*1.18
    normals=np.cross(v1[:,1]-v1[:,0],v1[:,2]-v1[:,0])
    normals/=np.maximum(np.linalg.norm(normals,axis=1)[:,None],1e-8)
    light=np.array([.4,.3,.86]);light/=np.linalg.norm(light)
    intensity=np.clip(.76+.34*np.maximum(normals@light,0.),.62,1)
    for j,(name,colors) in enumerate((
       ("REAL RGB-D RECONSTRUCTION",np.mean(np.clip(.07+rgb[ff]**.75,0,1),axis=1)),
       ("CLASS-AGNOSTIC SURFACE PATCHES",patchface[sampled]),
       ("YOLOE CATEGORY OVERLAY",np.mean(sem[ff],axis=1)))):
        ax=fig.add_subplot(1,3,j+1,projection="3d",proj_type="ortho")
        rgbcolor=np.column_stack((np.clip(colors*intensity[:,None],0,1),
                                  np.ones(len(colors))))
        ax.add_collection3d(Poly3DCollection(v1,facecolors=rgbcolor,
                             linewidths=0,rasterized=True))
        for lim,c,size in zip((ax.set_xlim,ax.set_ylim,ax.set_zlim),cent,extent):
            lim(c-size/2,c+size/2)
        ax.set_box_aspect(extent)
        ax.view_init(elev=20,azim=-50)
        ax.set_axis_off()
        ax.set_title(name,color="white",fontsize=11,pad=14)
    fig.text(.027,.95,"Replica office0 | Reconstruction / Geometry / Semantics",
        color="white",fontsize=16,weight="bold")
    fig.text(.027,.914,"100 original rendered RGB-D frames | poses ORACLE | surface patches are NOT instance IDs",
        color="#acc4d8",fontsize=10)
    fig.text(.027,.025,"No GT instance accuracy or RTAB-Map ATE evaluated. Replica is photo-realistically RENDERED, not a real camera recording.",
        color="#ffc8aa",fontsize=9)
    fig.savefig(out/"replica_office0_three_modes.png",facecolor=fig.get_facecolor())
    plt.close(fig)
    return {"full_mesh_vertices":len(v),"full_mesh_faces":len(f),
            "display_mesh_faces":len(sf),
            "semantic_supported_vertices":int(np.count_nonzero(idx)),
            "semantic_supported_fraction":float(np.mean(idx)),
            "nonzero_geometric_patch_faces":int(np.count_nonzero(patches)),
            "class_independent_patch_fraction":float(np.mean(patches>0))}

def run(args):
    import open3d as o3d
    from semantic_map.yoloe26_detector import YOLOE26Segmenter,class_list
    scene=Path(args.scene)
    images,depth,valid,w2c,K,meta=load_replica(scene)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    classes=class_list("config/yoloe26_classes.json")
    from semantic_map.yoloe26_detector import canonical
    canonical_names=list(dict.fromkeys(canonical(x) for x in classes))
    model=YOLOE26Segmenter("yoloe-26x-seg.pt",classes)

    volume=o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=args.voxel,sdf_trunc=args.voxel*4,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        volume_unit_resolution=16,depth_sampling_stride=4)
    sem_xyz=[];sem_ids=[];masks=Counter()
    inference_frames=0;depth_integrated=0
    Kx=K; start=time.monotonic()
    # Even frames provide the mesh; odd frames are held out for geometric consistency.
    even_frames=range(0,len(images),2)
    for i in even_frames:
        z=np.asarray(depth[i],np.float32).copy()
        z[~np.asarray(valid[i],bool)]=0
        bgr=cv2.imread(str(images[i]),cv2.IMREAD_COLOR)
        if bgr is None or bgr.shape[:2]!=z.shape:raise RuntimeError("Replica image-depth mismatch")
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        h,w=z.shape
        k=np.asarray(K[i]);intr=o3d.camera.PinholeCameraIntrinsic(
            w,h,float(k[0,0]),float(k[1,1]),float(k[0,2]),float(k[1,2]))
        rgba=o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(rgb)),
            o3d.geometry.Image(np.ascontiguousarray(z)),
            depth_scale=1.0,depth_trunc=args.max_depth,
            convert_rgb_to_intensity=False)
        volume.integrate(rgba,intr,np.asarray(w2c[i],dtype=np.float64))
        depth_integrated+=1
        if i%args.inference_step==0:
            res=model.predict(bgr,conf=args.confidence,device="cpu",imgsz=640)
            inference_frames+=1
            if res.masks is None or res.boxes is None:continue
            for raw,lab,confidence in zip(res.masks.data.cpu().numpy(),
                        res.boxes.cls.int().cpu().numpy(),
                        res.boxes.conf.cpu().numpy()):
                label=canonical(model.class_name(lab))
                if label not in canonical_names:raise ValueError("unexpected YOLO class")
                mask=raw>.5
                if mask.shape!=z.shape:
                    mask=cv2.resize(mask.astype(np.uint8),(w,h),
                      interpolation=cv2.INTER_NEAREST).astype(bool)
                pts=mask_to_measured_world(mask,z,np.asarray(valid[i],bool),k,
                         np.linalg.inv(np.asarray(w2c[i])),max_pts=3000)
                if len(pts)==0:continue
                sem_xyz.append(pts)
                sem_ids.append(np.full(len(pts),canonical_names.index(label)+1,dtype=np.int16))
                masks[label]+=1
        if depth_integrated%10==0:
            print("D3_REPLICA_INTEGRATION_PROGRESS",depth_integrated,flush=True)
    oracle_even=volume.extract_triangle_mesh()
    if len(oracle_even.triangles)<1000:raise RuntimeError("Cannot recover Replica geometry")
    even_metric=voxel_surface_distance(np.asarray(oracle_even.vertices),images,
                depth,valid,w2c,K,range(1,100,2))
    # Now integrate the held-out real RGB-D depth too for dense final visualization.
    for i in range(1,len(images),2):
        z=np.asarray(depth[i],np.float32).copy()
        z[~np.asarray(valid[i],bool)]=0
        bgr=cv2.imread(str(images[i]),cv2.IMREAD_COLOR)
        if bgr is None or bgr.shape[:2]!=z.shape:raise RuntimeError("Replica RGB missing")
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        k=np.asarray(K[i]);h,w=z.shape
        intr=o3d.camera.PinholeCameraIntrinsic(w,h,float(k[0,0]),float(k[1,1]),
                                                float(k[0,2]),float(k[1,2]))
        rgba=o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(rgb)),
            o3d.geometry.Image(np.ascontiguousarray(z)),
            depth_scale=1.0,depth_trunc=args.max_depth,convert_rgb_to_intensity=False)
        volume.integrate(rgba,intr,np.asarray(w2c[i],dtype=np.float64))
        depth_integrated+=1
    mesh=volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    if len(mesh.triangles)<10000:raise RuntimeError("Insufficient Replica mesh")
    if len(sem_xyz):
        xyz=np.concatenate(sem_xyz)
        labels=np.concatenate(sem_ids)
    else:
        xyz=np.empty((0,3),np.float32);labels=np.empty((0,),np.int16)
    verts=np.asarray(mesh.vertices)
    pred=transfer_semantic(verts,xyz,labels)
    # Geometric patches independent of all YOLO outputs — segment simplified
    # display mesh separately, but full-fidelity mesh is saved intact.
    # Only if a patch graph successfully forms do we expose them as such.
    nfaces=min(110000,len(mesh.triangles))
    local_mesh=mesh.simplify_quadric_decimation(target_number_of_triangles=nfaces)
    lv=np.asarray(local_mesh.vertices);lf=np.asarray(local_mesh.triangles)
    lp,patch_stats=segment_surface_patches(lv,lf,
         np.asarray(local_mesh.vertex_colors),min_faces=100)
    # Map patch IDs from simplified surface triangle centroids onto full mesh,
    # explicitly a visualization heuristic (not persistent identities).
    cf=verts[np.asarray(mesh.triangles)].mean(axis=1)
    sf=lv[lf].mean(axis=1)
    _,ix=cKDTree(sf).query(cf,k=1)
    patches=lp[ix]
    mapped_patch_coverage=float((patches>0).mean())
    view=render(mesh,pred,patches,out,canonical_names)
    report={"experiment":"D3_REPLICA_OFFICE0_OVI_STYLE_THREE_LAYER_MESH",
       "source_dataset":"helioom/3dvlm-replica_subset/replica/office0.tar",
       "original_source_family":"NICE-SLAM Replica photorealistic rendered RGB-D",
       "research_only_license":True,
       "synthetic_rendered_dataset_not_physical_sensor":True,
       "complete_subset_frames":100,
       "camera_poses_from_dataset_ORACLE":True,
       "RTABMap_run":False,
       "pose_convention":"provided OpenCV world_to_camera applied directly to Open3D integration",
       "original_rgbd_frames_fused":depth_integrated,
       "even_training_frames":50,
       "odd_frames_held_out_for_geometry_nn":50,
       "TSDF_voxel_m":args.voxel,
       "source_YOLOE_model":"yoloe-26x-seg.pt",
       "YOLOE_inference_frames":inference_frames,
       "YOLOE_valid_object_masks":sum(masks.values()),
       "per_class_observation_masks":dict(masks),
       "measured_semantic_mask_pixels_transformed_to_world":len(xyz),
       "class_agnostic_patch_stats":patch_stats,
       "class_agnostic_patch_coverage_on_full_mesh":mapped_patch_coverage,
       "class_agnostic_patches_are_physical_instances":False,
       "geometry_heldout_depth_distance":even_metric,
       "mesh":view,
       "original_GT_instance_accuracy":"NOT_EVALUATED",
       "actual_OVI_MAP_runtime_or_accuracy_comparison":"NOT_EVALUATED_NO_MATCHED_OVI_OUTPUT",
       "limitations":"Class-agnostic mesh surface patches may split one object or join objects; "
                     "YOLOE is not class-agnostic panoptic segmentation. "
                     "No GT physical instance labels. Geometry metric uses "
                     "a *heldout depth vs mesh vertex* proxy, not GT mesh distance. "
                     "Synthetic Replica rendered room, oracle poses not RTAB-Map.",
       "outputs":["rgb_mesh.ply","rgb_mesh.glb","semantic_mesh.ply",
          "semantic_mesh.glb","mesh_patch_and_semantic_labels.npz",
          "interactive_office0.html","replica_office0_three_modes.png","d3_metrics.json"],
       "elapsed_seconds":round(time.monotonic()-start,3)}
    (out/"d3_metrics.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print("D3_REPLICA_OFFICE0_MESH_RESULT",json.dumps(report,ensure_ascii=False),flush=True)

def main():
    a=argparse.ArgumentParser()
    a.add_argument("--scene",required=True)
    a.add_argument("--out",required=True)
    a.add_argument("--voxel",type=float,default=.025)
    a.add_argument("--max-depth",type=float,default=7.)
    a.add_argument("--confidence",type=float,default=.35)
    a.add_argument("--inference-step",type=int,default=10)
    q=a.parse_args()
    if not .01<=q.voxel<=.10 or q.inference_step<2 or q.inference_step%2:
        a.error("Voxel and inference step constraints invalid")
    run(q)

if __name__=="__main__":main()
