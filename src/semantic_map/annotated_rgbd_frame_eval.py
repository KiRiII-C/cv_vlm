"""Real RGB-D frames -> YOLO masks -> associated 3D objects -> 3D GT semantic scores.

Evaluation uses independently annotated 3D point class labels from RGB-D Scenes
v2. Camera poses come from the 3DMatch sequence, not estimated SLAM. GT labels
are NEVER used for predictions/association. GT is not instance identity data.
"""
from __future__ import annotations
import argparse,io,json,urllib.request,zipfile
from pathlib import Path
from collections import defaultdict
import cv2,numpy as np
from scipy.spatial import cKDTree
from .rgbd_scenes_gt import download_archive,load_scene,metrics,save_colored_ply
from .geometry import Intrinsics,make_observation
from .engine import SemanticMapper
from .storage import MemoryStore
from .instance_cloud import InstanceCloud
from .tum_yolo_pipeline import hsv_feature

BASE='https://3dvision.princeton.edu/projects/2016/3DMatch/downloads/rgbd-datasets/'
# Shared COCO/official GT categories only; omitted categories are UNMAPPABLE,
# not silently reinterpreted as positive predictions.
COCO_GT={'bowl':1,'cup':4,'dining table':5,'chair':6,'bottle':7,'couch':8,'sofa':8}

def get_archive(scene,root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    path=root/f'rgbd-scenes-v2-scene_{scene:02d}.zip'
    if not path.exists() or path.stat().st_size<10_000_000:
        url=BASE+path.name
        tmp=path.with_suffix('.part');tmp.unlink(missing_ok=True)
        req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'})
        with urllib.request.urlopen(req,timeout=180) as src,open(tmp,'wb') as f:
            while True:
                b=src.read(1024*1024)
                if not b:break
                f.write(b)
        tmp.rename(path)
    return path

def read_video(archive,scene):
    root=f'rgbd-scenes-v2-scene_{scene:02d}/'
    names=archive.namelist()
    intrin_name=root+'camera-intrinsics.txt'
    mat=np.loadtxt(io.BytesIO(archive.read(intrin_name)))
    if mat.shape!=(3,3):raise ValueError('Unrecognized camera intrinsics')
    colors=sorted(x for x in names if x.endswith('.color.png'))
    records=[]
    for path in colors:
        depth=path.replace('.color.png','.depth.png')
        pose=path.replace('.color.png','.pose.txt')
        if depth not in names or pose not in names:continue
        records.append((path,depth,pose))
    if len(records)<30:raise ValueError('Not enough pose-aligned real RGB-D frames')
    return mat,records

def frame_sample(records,n):
    if n<=0 or len(records)<=n:return records
    inds=np.linspace(0,len(records)-1,n).round().astype(int)
    return [records[i] for i in np.unique(inds)]

def predict_scene(scene,out,datadir='data/annotated_rgbd',maxframes=180,
                  model_name='yolo11n-seg.pt',conf=.3,device='cpu',pixel_stride=4):
    from ultralytics import YOLO
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    pcdir=download_archive(Path(datadir)/'official_3d_gt')
    gt_xyz,gt_rgb,gt_y=load_scene(pcdir,scene)
    zippath=get_archive(scene,Path(datadir)/'video')
    archive=zipfile.ZipFile(zippath)
    camera,video=read_video(archive,scene)
    video=frame_sample(video,maxframes)
    k0=np.asarray(camera,float)
    detector=YOLO(model_name)
    memory=MemoryStore(out/'semantic_memory.sqlite')
    mapping=SemanticMapper(memory,min_hits=2)
    cloud=InstanceCloud(voxel_m=.018,stride=3)
    counts={} # (ix,iy,iz) -> per-voxel 3D class vote counts
    frames=0;detected=0;used_masks=0;source_points=0
    vis=[]
    for i,(r,d,p) in enumerate(video):
        rgb=cv2.imdecode(np.frombuffer(archive.read(r),dtype=np.uint8),cv2.IMREAD_COLOR)
        depth=cv2.imdecode(np.frombuffer(archive.read(d),dtype=np.uint8),cv2.IMREAD_UNCHANGED)
        T=np.loadtxt(io.BytesIO(archive.read(p)))
        if rgb is None or depth is None or T.shape!=(4,4) or not np.isfinite(T).all():raise ValueError('Corrupted synchronized frame')
        if rgb.shape[:2]!=depth.shape:raise ValueError('Invalid RGB-D registration')
        h,w=depth.shape
        K=Intrinsics(float(k0[0,0]),float(k0[1,1]),float(k0[0,2]),float(k0[1,2]),w,h,depth_unit_m=.001)
        result=detector.predict(source=rgb,conf=conf,retina_masks=True,verbose=False,device=device,imgsz=640)[0]
        labels=np.full(depth.shape,10,dtype='uint8')
        observations=[];accepted=[]
        if result.masks is not None:
            masks=result.masks.data.cpu().numpy()
            classes=result.boxes.cls.int().cpu().tolist()
            scores=result.boxes.conf.cpu().tolist()
            for j in np.argsort(scores)[::-1]:
                name=detector.names[classes[j]]
                if name in ('person','clock'):continue
                raw=masks[j]
                if raw.shape!=depth.shape:raw=cv2.resize(raw.astype('float32'),(w,h),interpolation=cv2.INTER_NEAREST)
                mask=raw>.5
                ds=depth[mask].astype('float32')*.001
                ds=ds[(ds>.25)&(ds<5)]
                if len(ds)<60:continue
                median=np.median(ds)
                mask &= np.abs(depth.astype('float32')*.001-median)<max(.20,.16*median)
                if mask.sum()<60:continue
                detected+=1
                official=COCO_GT.get(name)
                if official is not None:
                    labels[mask & (labels==10)]=official
                    used_masks+=1
                app=hsv_feature(rgb,mask)
                obs=make_observation(depth,mask,K,T,source=f'rgbd_scenes_v2_{scene:02d}',frame_id=str(i),
                    detection_id=f'{i}_{j}',label=str(name),timestamp=float(i),appearance=app,
                    description=str(name),confidence=float(scores[j]),min_support=50)
                if obs is not None:observations.append(obs);accepted.append(mask.copy())
        decisions=mapping.process_frame(observations)
        if len(decisions)!=len(accepted):raise ValueError('Observation association mismatch')
        for mask,decision in zip(accepted,decisions):
            if decision.object_id and decision.status not in ('ambiguous','unresolved','unassigned_conflict','duplicate_detection'):
                cloud.add(decision.object_id,mask,depth,K,T)
        ys,xs=np.mgrid[0:h:pixel_stride,0:w:pixel_stride]
        z=depth[::pixel_stride,::pixel_stride].astype('float32')*.001
        valid=(z>.25)&(z<4.5)
        x=xs[valid];y=ys[valid];z=z[valid]
        xyz=np.column_stack(((x-K.cx)*z/K.fx,(y-K.cy)*z/K.fy,z))
        world=xyz@T[:3,:3].T+T[:3,3]
        keys=np.floor(world/.025).astype('int32')
        raw_labels=labels[::pixel_stride,::pixel_stride][valid]
        # Compress repeated indices within each frame before Python dictionary update.
        comb=np.column_stack([keys,raw_labels])
        unique,votes=np.unique(comb,axis=0,return_counts=True)
        for (a,b,c,label),vote in zip(unique,votes):
            key=(int(a),int(b),int(c))
            if key not in counts:counts[key]=np.zeros(11,dtype=np.int32)
            counts[key][int(label)]+=int(vote)
        source_points+=len(z)
        frames+=1
        if frames%30==0:print('RGBD_GT_FRAMES',scene,frames,'/',len(video),'voxels',len(counts),'tracks',len(memory.tracks()),flush=True)
    if not counts:raise ValueError('Empty reconstructed pointcloud')
    xyz=np.asarray(list(counts.keys()),dtype='float32')*.025+.0125
    cls=np.array([np.argmax(counts[k]) for k in counts],dtype=np.int16)
    # GT lookup: do not train, calibrate or transform camera poses from GT geometry.
    gt_tree=cKDTree(gt_xyz)
    dd,nn=gt_tree.query(xyz,workers=2)
    covered=dd<.08
    if not np.any(covered):coverage=0.
    else:coverage=float(np.mean(covered))
    report={'dataset':'RGB-D Scenes v2','scene':scene,
            'gt':'official annotated 3D semantic class per PLY vertex, not physical instance',
            'rgbd_frames_mapped':frames,'dataset_video_frames':len(video),
            'gt_surface_points':len(gt_xyz),'depth_samples_integrated':source_points,
            'mapped_voxels_25mm':len(xyz),'geometry_match_to_GT_8cm':coverage,
            'median_GT_surface_distance_m':float(np.median(dd)),
            'masks_detected':detected,'masks_with_exact_official_gt_class_mapping':used_masks,
            'coco_to_official_class_mapping':COCO_GT,
            'gt_alignment_requirement':'Coordinate frames from public 3DMatch camera-to-world and RGBD-Scenes v2 PLY; no ICP to GT',
            'physical_id_gt_available':False,
            'unobserved_gt_points_not_scored':True}
    if coverage<.2:
        report.update(status='FAIL_POSE_OR_DATASET_FRAME_ALIGNMENT_UNVERIFIED',semantic_metrics=None)
    else:
        report.update(status='MEASURED_RGBD_TO_3D_GT_SEMANTICS',
                      semantic_metrics=metrics(gt_y[nn[covered]],cls[covered]))
    (out/'rgbd_to_3d_gt_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    save_colored_ply(out/f'scene_{scene:02d}_rgbd_voxel_classes.ply',xyz,cls)
    save_colored_ply(out/f'scene_{scene:02d}_gt_nearest_classes.ply',xyz,gt_y[nn])
    if report['status']=='MEASURED_RGBD_TO_3D_GT_SEMANTICS':
        from .rgbd_scenes_gt import plot_clouds
        plot_clouds(out,xyz,cls,gt_y[nn],scene,points=40000)
    tracks=[t for t in memory.tracks() if t.state=='confirmed']
    mapping_points=cloud.export(out/'dense_colored_object_instances.ply',track_filter={t.object_id for t in tracks})
    report['num_confirmed_tracks']=len(tracks)
    report['dense_point_instance_map']=mapping_points
    (out/'rgbd_to_3d_gt_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print('RGBD_TO_3D_GT_RESULT',json.dumps({k:v for k,v in report.items() if k not in ('semantic_metrics','dense_point_instance_map')}),flush=True)
    archive.close();memory.close()
    return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--scene',type=int,default=11)
    p.add_argument('--out',default='results/annotated_rgbd_scene11')
    p.add_argument('--data',default='data/annotated_rgbd')
    p.add_argument('--maxframes',type=int,default=180)
    p.add_argument('--device',default='cpu')
    a=p.parse_args()
    predict_scene(a.scene,a.out,a.data,maxframes=a.maxframes,device=a.device)
if __name__=='__main__':main()