"""Real TUM RGB-D + real pretrained YOLO instance masks -> 3D semantic memory.

External weights are downloaded by Ultralytics at first invocation. No training,
instance-level ground-truth labels, or Nav2 actuation are provided by TUM.
Data: TUM fr1_desk; optical camera GT poses nearest-neighbor with hard time gate.
"""
from __future__ import annotations
import argparse,json,tarfile,urllib.request,sys,os
from pathlib import Path
import cv2,numpy as np
from scipy.spatial.transform import Rotation
from .geometry import Intrinsics, make_observation
from .storage import MemoryStore
from .engine import SemanticMapper
from .pointmap import SurfaceVoxelMap

K=Intrinsics(517.3,516.5,318.6,255.3,640,480,depth_unit_m=1/5000)

def download_scene(out,name='rgbd_dataset_freiburg1_desk'):
    out=Path(out);out.mkdir(exist_ok=True,parents=True)
    root=out/name;archive=out/(name+'.tgz')
    if not (root/'rgb.txt').exists():
        if not archive.exists():
            errs=[]
            for base in ('https://vision.in.tum.de','https://webshare.cvg.cit.tum.de'):
                try:
                    url=f'{base}/rgbd/dataset/freiburg1/{name}.tgz'
                    print('Downloading',url,flush=True)
                    req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'})
                    with urllib.request.urlopen(req,timeout=180) as r,open(archive,'wb') as f:
                        while True:
                            part=r.read(2**20)
                            if not part:break
                            f.write(part)
                    break
                except Exception as e:
                    errs.append(str(e));archive.unlink(missing_ok=True)
            if not archive.exists():raise RuntimeError('TUM download failed: '+str(errs))
        print('Extracting',archive,flush=True)
        with tarfile.open(archive) as tar:
            for m in tar:
                p=Path(m.name)
                if p.is_absolute() or '..' in p.parts or p.parts[0]!=name:raise ValueError('Unsafe tar')
                if not m.isfile():continue
                target=out/p;target.parent.mkdir(exist_ok=True,parents=True)
                with tar.extractfile(m) as inp, open(target,'wb') as dst:
                    while True:
                        data=inp.read(2**20)
                        if not data:break
                        dst.write(data)
    return root

def parse(path):
    rows=[]
    for line in Path(path).read_text().splitlines():
        if line.lstrip().startswith('#') or not line.strip():continue
        p=line.split();rows.append((float(p[0]),p[1:]))
    return rows

def synchronized(root):
    rgb=parse(root/'rgb.txt');depth=parse(root/'depth.txt');pose=parse(root/'groundtruth.txt')
    dt=np.array([t for t,_ in depth]);pt=np.array([t for t,_ in pose])
    results=[]
    for t,src in rgb:
        j=int(np.searchsorted(dt,t));kk=[k for k in (j-1,j) if 0<=k<len(dt)]
        if not kk:continue
        j=min(kk,key=lambda k:abs(dt[k]-t))
        if abs(dt[j]-t)>.03:continue
        j2=int(np.searchsorted(pt,t));pp=[k for k in (j2-1,j2) if 0<=k<len(pt)]
        if not pp:continue
        j2=min(pp,key=lambda k:abs(pt[k]-t))
        if abs(pt[j2]-t)>.03:continue
        p=np.array([float(x) for x in pose[j2][1]],dtype=float)
        if len(p)!=7:continue
        T=np.eye(4);T[:3,:3]=Rotation.from_quat(p[3:7]).as_matrix();T[:3,3]=p[:3]
        results.append((t,root/src[0],root/depth[j][1][0],T))
    return results

def hsv_feature(rgb,mask):
    hsv=cv2.cvtColor(rgb,cv2.COLOR_BGR2HSV)
    # 8 H bins, 4 S bins, 4 V bins; fixed size normalized appearance descriptor
    values=hsv[mask]
    if len(values)<50:return None
    hist=cv2.calcHist([hsv],[0,1,2],mask.astype('uint8'),[8,4,4],[0,180,0,256,0,256])
    a=np.sqrt(hist.ravel().astype(float)+1e-8)
    return (a/np.linalg.norm(a)).tolist()

def infer_benchmark(dataset='data/tum',output='results/tum_real',model_name='yolo11n-seg.pt',
                    confidence=.30,max_frames=0,voxel=.07):
    from ultralytics import YOLO
    from matplotlib import pyplot as plt
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    root=download_scene(dataset)
    pairs=synchronized(root)
    if not pairs:raise RuntimeError('No synchronized posed camera frames')
    if max_frames>0:pairs=pairs[:max_frames]
    print('REAL_FRAMES',len(pairs),'model',model_name,flush=True)
    det=YOLO(model_name)
    mem=MemoryStore(output/'semantic_memory.sqlite')
    mapping=SemanticMapper(mem,min_hits=2)
    vox=SurfaceVoxelMap(output/'surface_voxels.sqlite',voxel_m=voxel)
    reports=[];evidence=[];stats={'frames':len(pairs),'detections':0,'invalid_masks':0,'labels':{},'statuses':{}}
    colors={'chair':'#f58518','dining table':'#4c78a8','tv':'#54a24b'}
    for i,(stamp,rgbname,depthname,T) in enumerate(pairs):
        rgb=cv2.imread(str(rgbname),cv2.IMREAD_COLOR)
        depth=cv2.imread(str(depthname),cv2.IMREAD_UNCHANGED)
        if rgb is None or depth is None or rgb.shape[:2]!=depth.shape:raise ValueError('Invalid TUM frame')
        vox.add_frame(depth,K,T,'tum_freiburg1_desk',str(stamp),stride=8)
        result=det.predict(source=rgb,conf=confidence,retina_masks=True,verbose=False,device='cpu',imgsz=640)[0]
        observations=[];display=rgb.copy()
        if result.masks is not None and result.boxes is not None:
            masks=result.masks.data.cpu().numpy();ids=result.boxes.cls.int().cpu().tolist()
            scores=result.boxes.conf.cpu().tolist()
            for j,(m,k,conf) in enumerate(zip(masks,ids,scores)):
                label=str(det.names[k])
                if label in ('person','clock'):
                    continue  # dynamic/hard-to-static in this baseline
                if m.shape!=depth.shape:m=cv2.resize(m.astype('float32'),(640,480),interpolation=cv2.INTER_NEAREST)
                mask=m>.5
                # Depth mode separation reduces background leakage from imprecise 2D masks.
                ds=depth[mask].astype('float32')/5000
                ds=ds[(ds>.25)&(ds<5)]
                if len(ds)<40:stats['invalid_masks']+=1;continue
                pivot=np.median(ds);mask=mask & (np.abs(depth.astype('float32')/5000-pivot)<max(.18,.16*pivot))
                feat=hsv_feature(rgb,mask)
                obs=make_observation(depth,mask,K,T,source='tum_freiburg1_desk',frame_id=str(stamp),
                                    detection_id=f'det{j}',label=label,timestamp=stamp,appearance=feat,
                                    description=label,confidence=float(conf),min_support=50)
                if obs is None:stats['invalid_masks']+=1;continue
                stats['detections']+=1;stats['labels'][label]=stats['labels'].get(label,0)+1
                observations.append(obs)
                if i%60==0:
                    contours,_=cv2.findContours(mask.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(display,contours,-1,(0,210,240),2)
                    cv2.putText(display,label,(int(np.median(np.where(mask)[1])),int(np.median(np.where(mask)[0]))),
                                cv2.FONT_HERSHEY_SIMPLEX,.6,(20,50,230),2)
        evidence.append([{'source':o.source,'frame_id':o.frame_id,
                          'detection_id':o.detection_id,'label':o.label,
                          'center':o.center.tolist(),'low':o.low.tolist(),'high':o.high.tolist(),
                          'timestamp':o.timestamp,'appearance':o.appearance.tolist() if o.appearance is not None else None,
                          'description':o.description,'confidence':o.confidence} for o in observations])
        decisions=mapping.process_frame(observations)
        for dec in decisions:stats['statuses'][dec.status]=stats['statuses'].get(dec.status,0)+1
        reports.append({'frame':i,'timestamp':stamp,'detections':len(observations),
                        'associations':[d.__dict__ for d in decisions]})
        if i%60==0:
            cv2.imwrite(str(output/f'overlay_{i:04}.jpg'),display)
        if (i+1)%50==0:print('REAL_PROGRESS',i+1,'/',len(pairs),'tracks',len(mem.tracks()),flush=True)
    tracks=mem.tracks();confirmed=[t for t in tracks if t.state=='confirmed']
    stats.update({'total_tracks':len(tracks),'confirmed_tracks':len(confirmed),
                  'tracks_reobserved_10plus':sum(t.hits>=10 for t in tracks),
                  'identity_gt_available':False,
                  'tracking_metrics':'Cannot derive GT IDF1 or false-merge rate without annotated physical identities',
                  'groundtruth_poses':len(pairs),'surface_voxels':vox.export_ply(output/'observed_surface.ply')})
    stats['objects']=[{'id':t.object_id,'label':t.label,'center_xyz_m':t.center.tolist(),
                       'low_xyz_m':t.low.tolist(),'high_xyz_m':t.high.tolist(),
                       'hits':t.hits,'state':t.state,'description':t.description} for t in tracks]
    (output/'real_mapping_report.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2))
    import gzip
    with gzip.open(output/'observation_evidence.json.gz','wt',encoding='utf-8') as f:
        json.dump(evidence,f,ensure_ascii=False,separators=(',',':'))
    (output/'real_frame_associations.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2))
    (output/'poses_and_dataset.json').write_text(json.dumps({'source':'TUM fr1_desk',
                'frames':[{'timestamp':float(t),'rgb':p.name,'depth':d.name,'T_map_cam':T.tolist()} for t,p,d,T in pairs]},indent=2))
    print('REAL_COMPLETE',json.dumps({k:v for k,v in stats.items() if k!='objects'}),flush=True)
    vox.close();mem.close()
    return stats

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',default='data/tum');p.add_argument('--output',default='results/tum_real');
    p.add_argument('--model',default='yolo11n-seg.pt');p.add_argument('--conf',type=float,default=.3);p.add_argument('--max-frames',type=int,default=0)
    a=p.parse_args();infer_benchmark(a.data,a.output,a.model,a.conf,a.max_frames)
if __name__=='__main__':main()
