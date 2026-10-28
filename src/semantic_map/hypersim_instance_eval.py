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
from .reid_features import hsv_spatial,AppearanceEncoder
from .yoloe26_detector import build_detector,class_list
from .temporal_tracker import TemporalTracker,Detection

def instance_observation(world_m,rgb,mask,*,frame,detection,label,confidence,source,feature_mode='hsv',appearance_override=None):
    ys,xs=np.where(mask[::4,::4])
    if len(xs)<12:return None
    pts=world_m[::4,::4][ys,xs]
    pts=pts[np.isfinite(pts).all(axis=1)]
    if len(pts)<12:return None
    low=np.quantile(pts,.07,axis=0);high=np.quantile(pts,.93,axis=0)
    high=np.maximum(high,low+.018)
    center=np.median(pts,axis=0)
    descriptor=(appearance_override if appearance_override is not None else
                hsv_spatial(rgb,mask) if feature_mode=='hsv_spatial'
                else hsv_feature(rgb,mask))
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
        'warning':'Oracle mode uses GT masks/classes; predicted mode uses pretrained YOLO masks. All modes use GT world positions. Physical ID metrics are computed using GT only in scoring, with missed detections penalized.'}

def run(out,frames=(0,1,2,3,4,5,6,8,10,12,15,18,20,24,28,32,36,40,44,48),
        mode='oracle_masks',max_objects=36,model_name='yolo11n-seg.pt',confidence=.25,
        temporal_tracking=False,association_mode='fused',classes_json=None,
        scene='ai_001_001',feature_mode='hsv_spatial'):
    out=Path(out);out.mkdir(exist_ok=True,parents=True)
    from PIL import Image
    remote,z=open_scene(scene)
    detector=None
    if mode=='predicted_yolo_masks':
        detector=build_detector(model_name,class_list(classes_json),
                 legacy=not Path(model_name).name.startswith('yoloe-26'))
    elif mode!='oracle_masks':raise ValueError('Unknown mode '+mode)
    store=MemoryStore(out/'memory.sqlite')
    mapper=SemanticMapper(store,min_hits=2,association_mode=association_mode)
    temporal=TemporalTracker(high_thresh=confidence,
        low_thresh=min(.10,confidence*.55)) if temporal_tracking else None
    memory_hint={}
    encoder=AppearanceEncoder(feature_mode,device='cpu') if temporal_tracking else None
    # These 8 NYU40 classes have explicit semantics within our 12 YOLOE
    # prompts; monitor/cup/keyboard/laptop etc may collapse into otherprop.
    # Report both FULL-SCENE IDF1 and target-vocabulary-only IDF1; never
    # penalize intentionally unrequested GT instances as missed detections.
    target_nyu40={3,4,5,6,7,10,14,25}
    evidence=[];target_evidence=[];target_unique=set();target_gt_total=0
    gt_predictions_by_frame=defaultdict(list)
    gt_frame_ids=defaultdict(list);first_rgb=None;first_world=None;first_gt=None
    try:
        root=f'{scene}/images/'
        scene_meta=z.read(f'{scene}/_detail/metadata_scene.csv').decode('utf-8')
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
            target_visible=set()
            for uid in visible:
                cls=sem[gt==uid]
                cls=cls[cls>=0]
                if len(cls) and int(Counter(cls.tolist()).most_common(1)[0][0]) in target_nyu40:
                    target_visible.add(uid)
            target_gt_total+=len(target_visible)
            target_unique.update(target_visible)
            observations=[];physical_ids=[];short_inputs=[]
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
                result=detector.predict(im,conf=min(.10,confidence*.55) if temporal else confidence,
                                        imgsz=640,device='cpu')
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
                           label=detector.class_name(c),confidence=float(prob),source=mode,
                           feature_mode=feature_mode if temporal else 'hsv',
                           appearance_override=encoder(im,mask) if encoder is not None else None)
                        if o is not None:
                            observations.append(o);physical_ids.append(uid)
                            short_inputs.append(Detection(o.detection_id,o.label,
                               np.array([np.where(mask)[1].min(),np.where(mask)[0].min(),
                                         np.where(mask)[1].max()+1,np.where(mask)[0].max()+1],float),
                               o.center,o.confidence,o.appearance,
                               cv2.resize(mask.astype('uint8'),(80,60),
                                          interpolation=cv2.INTER_NEAREST).astype(bool)))
            hint={}
            if temporal is not None:
                assigned=temporal.update(k,short_inputs)
                keep=[i for i,a in enumerate(assigned) if a.short_id is not None]
                observations=[observations[i] for i in keep]
                physical_ids=[physical_ids[i] for i in keep]
                assigned=[assigned[i] for i in keep]
                hint={o.detection_id:memory_hint[a.short_id]
                      for o,a in zip(observations,assigned) if a.short_id in memory_hint}
            decisions=mapper.process_frame(observations,prior_ids=hint)
            if temporal is not None:
                for a,d in zip(assigned,decisions):
                    if d.object_id and a.short_id not in memory_hint:
                        memory_hint[a.short_id]=d.object_id
            if len(decisions)!=len(physical_ids):raise AssertionError('Observation association mismatch')
            evidence.extend([(uid,d.object_id) for uid,d in zip(physical_ids,decisions)])
            for uid,decision in zip(physical_ids,decisions):
                if uid is not None:
                    gt_predictions_by_frame[int(uid)].append((int(k),decision.object_id))
            if mode=="oracle_masks":
                # The oracle emits all NYU40 categories, unlike the bounded
                # 12-class model. Non-target oracle masks are out of scope,
                # NOT false positives of a target-only detector.
                target_evidence.extend([(uid,d.object_id)
                    for uid,d in zip(physical_ids,decisions) if uid in target_visible])
            else:
                # YOLOE explicitly predicts ONLY target classes. Its masks
                # on other physical categories are true target false positives.
                target_evidence.extend([(uid if uid in target_visible else None,d.object_id)
                    for uid,d in zip(physical_ids,decisions)])
            framecount+=1
            if framecount%5==0:
                print('HYPERSIM_ID_PROGRESS',mode,framecount,'/',len(frames),'tracks',len(store.tracks()),flush=True)
        if framecount<10:raise RuntimeError('Too few valid GT instance frames')
        stats=evaluate_identity(evidence,total_gt)
        if not target_gt_total:
            raise RuntimeError("No NYU40 target-vocabulary GT instances in selected frames")
        stats["target_vocabulary_gt"]=evaluate_identity(target_evidence,target_gt_total)
        # Re-entry statistics count re-observations after a frame gap;
        # missing GT IDs are reflected in the IDF1 denominator separately.
        n_reentries=0;n_consistent=0;n_inconsistent=0;n_unresolved=0;switches=0
        for uid,observed in gt_predictions_by_frame.items():
            observed.sort(key=lambda item:item[0])
            for (fa,ua),(fb,ub) in zip(observed[:-1],observed[1:]):
                if ua is not None and ub is not None and ua!=ub:
                    switches+=1
                if fb-fa<4:continue
                n_reentries+=1
                if ua is None or ub is None:n_unresolved+=1
                elif ua==ub:n_consistent+=1
                else:n_inconsistent+=1
        stats["observed_reentry_pairs_after_gap_4_frames"]=n_reentries
        stats["observed_reentry_same_uuid"]=n_consistent
        stats["observed_reentry_changed_uuid"]=n_inconsistent
        stats["observed_reentry_unassigned"]=n_unresolved
        stats["observed_reentry_consistency"]=float(n_consistent/n_reentries) if n_reentries else None
        stats["observed_gt_to_uuid_switches"]=switches
        stats["reentry_warning"]="Scores only reobservations with a model detection; unobserved GT frame occurrences remain penalized by IDFN in IDF1."
        stats["target_gt_nyu40_ids"]=sorted(target_nyu40)
        stats["target_gt_unique_physical_ids"]=len(target_unique)
        stats["target_scope_note"]="Target subset = physical IDs whose modal NYU40 class matches our 12 configured prompts; other detections are counted as target false positives, other GT misses are excluded."
        gt_appearances=defaultdict(list)
        for k,ids in gt_frame_ids.items():
            for uid in ids:gt_appearances[uid].append(k)
        return_after_gap=[uid for uid,times in gt_appearances.items()
                          if any(b-a>=4 for a,b in zip(times[:-1],times[1:]))]
        stats.update({'unique_GT_visible_objects_all_frames':len(gt_appearances),
            'unique_GT_objects_detected_in_observations':len({a for a,b in evidence if a is not None}),
            'detector_model':model_name if mode=='predicted_yolo_masks' else 'ORACLE_GT_MASKS',
            'detector_confidence':confidence if mode=='predicted_yolo_masks' else None,
            'status':'PHYSICAL_INSTANCE_GT_EVALUATION',
            'mode':mode,'scene':scene,'metric_world_scale':scale,
            'gt_is_physically_stable_across_frames':True,
            'used_actual_rgb_in_oracle_appearance':True,
            'uses_GT_semantic_classes_for_oracle':mode=='oracle_masks',
            'uses_GT_world_3D_positions_for_both_modes':True,
            'estimated_camera_pose':'NOT_TESTED',
            'short_term_temporal_enabled':bool(temporal_tracking),
            'global_association_mode':association_mode,
            'appearance_mode':feature_mode if temporal_tracking else 'hsv',
            'closed_vocabulary_class_list':class_list(classes_json) if mode=='predicted_yolo_masks' else None,
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
    p.add_argument('--model',default='yolo11n-seg.pt')
    p.add_argument('--conf',type=float,default=.25)
    p.add_argument('--classes-json',default=None)
    p.add_argument('--scene',default='ai_001_001')
    p.add_argument('--max-objects',type=int,default=36)
    p.add_argument('--feature-mode',choices=['hsv','hsv_spatial','mobilenetv3'],default='hsv_spatial')
    p.add_argument('--temporal',action='store_true')
    p.add_argument('--association',choices=['fused','multiview','geometry','appearance','adaptive_reid'],default='fused')
    a=p.parse_args()
    run(a.out,mode=a.mode,model_name=a.model,confidence=a.conf,
        classes_json=a.classes_json,temporal_tracking=a.temporal,
        association_mode=a.association,scene=a.scene,max_objects=a.max_objects,
        feature_mode=a.feature_mode)

if __name__=='__main__':main()
