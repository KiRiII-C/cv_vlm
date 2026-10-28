"""Real physical-instance ID benchmark on official Hypersim RGB-D/XYZ.

Hypersim provides stable physical semantic_instance IDs across camera frames.
Benchmark modes:
  - oracle_masks: GT physical instance pixels and GT semantic labels (isolates association)
  - predicted_yolo_masks: actual RGB YOLO segmentation, true physical-ID used ONLY to
    score detection grouping, with GT world positions as camera-pose control
Both modes use GT world XYZ and do NOT assess pose estimation or depth backprojection.
No detector is allowed to read physical IDs in its decisions.
"""
from __future__ import annotations
import argparse,csv,io,json
from collections import Counter,defaultdict
from pathlib import Path
import cv2,numpy as np
from scipy.optimize import linear_sum_assignment
from .hypersim_range import open_scene,hdf5_array
from .models import Observation
from .storage import MemoryStore
from .engine import SemanticMapper
from .tum_yolo_pipeline import hsv_feature

def instance_observation(world_m,rgb,mask,*,frame,detection,label,confidence,source):
    ys,xs=np.where(mask[::4,::4])
    if len(xs)<12:return None
    pts=world_m[::4,::4][ys,xs]
    pts=pts[np.isfinite(pts).all(axis=1)]
    if len(pts)<12:return None
    low=np.quantile(pts,.07,axis=0);high=np.quantile(pts,.93,axis=0)
    high=np.maximum(high,low+.018)
    center=np.median(pts,axis=0)
    descriptor=hsv_feature(rgb,mask)
    return Observation(source=source,frame_id=str(frame),detection_id=str(detection),
                       label=str(label),center=center,low=low,high=high,
                       timestamp=float(frame),appearance=descriptor,confidence=float(confidence),
                       map_id='hypersim_001001_gt_position')

def evaluate_identity(observations,total_gt):
    """Globally optimal GT ID ↔ predicted UUID matching on observation occurrences."""
    gt=sorted({int(a) for a,b in observations if a is not None})
    ids=sorted({b for a,b in observations if b is not None})
    aidx={x:i for i,x in enumerate(gt)};bidx={x:i for i,x in enumerate(ids)}
    matrix=np.zeros((len(gt),len(ids)),dtype=np.int64)
    for a,b in observations:
        if a is not None and b is not None:
            matrix[aidx[int(a)],bidx[b]]+=1
    if matrix.size:
        rows,cols=linear_sum_assignment(-matrix)
        tp=int(sum(matrix[r,c] for r,c in zip(rows,cols)))
    else:tp=0
    pred_count=int(sum(b is not None for a,b in observations))
    fpid=max(0,pred_count-tp)
    fnid=max(0,total_gt-tp)
    idf1=(2*tp/max(1,2*tp+fpid+fnid))
    ids_for_gt=defaultdict(set);gts_for_id=defaultdict(set)
    for a,b in observations:
        if a is not None and b is not None:
            ids_for_gt[int(a)].add(b);gts_for_id[b].add(int(a))
    return {'IDF1_observation_level':float(idf1),'IDTP':tp,'IDFP':fpid,'IDFN':fnid,
        'gt_physical_object_count':len(gt),'predicted_track_count':len(ids),
        'gt_observations':int(total_gt),'observations_recorded':len(observations),
        'false_split_gt_objects':sum(len(v)>1 for v in ids_for_gt.values()),
        'false_merge_predicted_ids':sum(len(v)>1 for v in gts_for_id.values()),
        'unassigned_observations':int(sum(b is None for a,b in observations)),
        'warning':'Oracle mode uses true GT instance masks and semantic classes, but ID mapping is predicted.'}

def run(out,frames=(0,1,2,3,4,5,6,8,10,12,15,18,20,24,28,32,36,40,44,48),
        mode='oracle_masks',max_objects=36):
    out=Path(out);out.mkdir(exist_ok=True,parents=True)
    from PIL import Image
    remote,z=open_scene()
    detector=None
    if mode=='predicted_yolo_masks':
        from ultralytics import YOLO
        detector=YOLO('yolo11n-seg.pt')
    elif mode!='oracle_masks':raise ValueError('Unknown mode '+mode)
    store=MemoryStore(out/'memory.sqlite')
    mapper=SemanticMapper(store,min_hits=2)
    evidence=[];gt_frame_ids=defaultdict(list);first_rgb=None;first_world=None;first_gt=None
    try:
        root='ai_001_001/images/'
        scene_meta=z.read('ai_001_001/_detail/metadata_scene.csv').decode('utf-8')
        metadata=dict(csv.reader(io.StringIO(scene_meta)))
        scale=float(metadata['meters_per_asset_unit'])
        framecount=0;total_gt=0;missing=[]
        for k in frames:
            prefix=root+f'scene_cam_00_geometry_hdf5/frame.{k:04d}.'
            rgbname=root+f'scene_cam_00_final_preview/frame.{k:04d}.tonemap.jpg'
            needed=[prefix+'semantic_instance.hdf5',prefix+'semantic.hdf5',prefix+'position.hdf5',rgbname]
            if not all(n in z.namelist() for n in needed):
                missing.append(k);continue
            gt=hdf5_array(z,prefix+'semantic_instance.hdf5').astype('int32')
            sem=hdf5_array(z,prefix+'semantic.hdf5').astype('int32')
            world=hdf5_array(z,prefix+'position.hdf5').astype('float32')*scale
            im=cv2.cvtColor(np.array(Image.open(io.BytesIO(z.read(rgbname))).convert('RGB')),cv2.COLOR_RGB2BGR)
            if gt.shape!=sem.shape or gt.shape!=im.shape[:2] or world.shape[:2]!=gt.shape:
                raise ValueError('Hypersim input misregistered')
            if first_rgb is None:first_rgb=im;first_world=world;first_gt=gt
            unique,counts=np.unique(gt,return_counts=True)
            visible=[int(uid) for uid,c in zip(unique,counts) if int(uid)>=0 and c>=250]
            visible=sorted(visible,key=lambda x:-int((gt==x).sum()))[:max_objects]
            gt_frame_ids[k]=visible
            total_gt+=len(visible)
            observations=[];physical_ids=[]
            if mode=='oracle_masks':
                for uid in visible:
                    mask=(gt==uid)
                    cat=sem[mask];cat=cat[cat>=0]
                    if not len(cat):continue
                    category=int(Counter(cat.tolist()).most_common(1)[0][0])
                    o=instance_observation(world,im,mask,frame=k,detection=f'oracle_{uid}',
                       label=f'gt_semantic_{category}',confidence=1.,source=mode)
                    if o is not None:observations.append(o);physical_ids.append(uid)
            else:
                result=detector.predict(im,conf=.25,imgsz=640,device='cpu',retina_masks=True,verbose=False)[0]
                if result.masks is not None:
                    masks=result.masks.data.cpu().numpy();classes=result.boxes.cls.int().cpu().tolist()
                    probs=result.boxes.conf.cpu().tolist()
                    for jj,(m,c,prob) in enumerate(zip(masks,classes,probs)):
                        if m.shape!=gt.shape:m=cv2.resize(m.astype('float32'),(gt.shape[1],gt.shape[0]),interpolation=cv2.INTER_NEAREST)
                        mask=m>.5
                        if int(mask.sum())<250:continue
                        associated=gt[mask];associated=associated[associated>=0]
                        uid=None
                        if len(associated)>0:
                            top,n=Counter(associated.tolist()).most_common(1)[0]
                            if n/max(1,int(mask.sum()))>.55 and int(top) in visible:
                                uid=int(top)
                        o=instance_observation(world,im,mask,frame=k,detection=f'yolo_{jj}',
                           label=str(detector.names[c]),confidence=float(prob),source=mode)
                        if o is not None:observations.append(o);physical_ids.append(uid)
            decisions=mapper.process_frame(observations)
            if len(decisions)!=len(physical_ids):raise AssertionError('Observation association mismatch')
            evidence.extend([(uid,d.object_id) for uid,d in zip(physical_ids,decisions)])
            framecount+=1
            if framecount%5==0:
                print('HYPERSIM_ID_PROGRESS',mode,framecount,'/',len(frames),'tracks',len(store.tracks()),flush=True)
        if framecount<10:raise RuntimeError('Too few valid GT instance frames')
        stats=evaluate_identity(evidence,total_gt)
        gt_appearances=defaultdict(list)
        for k,ids in gt_frame_ids.items():
            for uid in ids:gt_appearances[uid].append(k)
        return_after_gap=[uid for uid,times in gt_appearances.items()
                          if any(b-a>=4 for a,b in zip(times[:-1],times[1:]))]
        stats.update({'status':'PHYSICAL_INSTANCE_GT_EVALUATION',
            'mode':mode,'scene':'ai_001_001','metric_world_scale':scale,
            'gt_is_physically_stable_across_frames':True,
            'used_actual_rgb_in_oracle_appearance':True,
            'uses_GT_semantic_classes_for_oracle':mode=='oracle_masks',
            'uses_GT_world_3D_positions_for_both_modes':True,
            'estimated_camera_pose':'NOT_TESTED',
            'frames_evaluated':framecount,'missing_frames':missing,
            'gt_objects_reappearing_after_sampled_gap':len(return_after_gap),
            'reentering_gt_ids':return_after_gap,
            'physical_gt_id_evaluation':'YES',
            'independent_real_rgbd_semantic_GT':'SYNTHETIC_HYPERSIM_GT_NOT_REAL_SENSOR'})
        (out/'hypersim_physical_id_gt.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2))
        # Score only observed GT instance occurrences, not unobserved room objects.
        print('HYPERSIM_PHYSICAL_ID_GT',json.dumps(stats,ensure_ascii=False),flush=True)
        return stats
    finally:
        store.close();z.close();remote.close()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',default='results/hypersim_physical_ids')
    p.add_argument('--mode',choices=['oracle_masks','predicted_yolo_masks'],default='oracle_masks')
    a=p.parse_args()
    run(a.out,mode=a.mode)

if __name__=='__main__':main()
