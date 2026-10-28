"""YOLOE-26x factorial GT diagnostic: detection, mask, class, instance splitting.
Hypersim physical GT is for scoring and explicitly marked oracle interventions
ONLY. No GT physical ID enters ObjectMemoryV2.process_frame.
"""
import argparse,csv,gzip,io,json,time
from collections import Counter,defaultdict
from pathlib import Path
import cv2,numpy as np
from scipy.optimize import linear_sum_assignment
from .hypersim_range import open_scene,hdf5_array
from .hypersim_instance_eval import instance_observation
from .memory_v2_replay import replay
from .yoloe26_detector import build_detector,class_list,canonical

NYU={3:"cabinet",4:"bed",5:"chair",6:"sofa",7:"table",10:"bookshelf",14:"table",25:"tv"}
VARIANTS=["raw","oracle_class","oracle_mask","oracle_mask_class",
          "oracle_dedup","oracle_recall","oracle_all"]
FRAMES=(0,1,2,3,4,5,6,8,10,12,15,18,20,24,28,32,36,40,44,48)

def target_instances(gt,sem,limit=130):
    uid,counts=np.unique(gt,return_counts=True)
    all_visible=sorted([(int(i),int(n)) for i,n in zip(uid,counts)
        if i>=0 and n>=250],key=lambda x:-x[1])[:limit]
    selected=[]
    for k,n in all_visible:
        cls=Counter(sem[gt==k].tolist()).most_common(1)[0][0]
        if int(cls) in NYU:selected.append((k,NYU[int(cls)],n))
    return selected

def match_masks(masks,gt,targets,threshold=.10):
    gt_ids=[v[0] for v in targets];n=len(masks);m=len(gt_ids)
    ious=np.zeros((n,m),float);inter=np.zeros((n,m),int)
    if not n or not m:return {},ious,inter
    lookup={v:i for i,v in enumerate(gt_ids)}
    for i,mask in enumerate(masks):
        a=int(mask.sum())
        ids,counts=np.unique(gt[mask],return_counts=True)
        for uid,count in zip(ids,counts):
            j=lookup.get(int(uid))
            if j is None:continue
            inter[i,j]=int(count)
            ious[i,j]=count/max(1,a+targets[j][2]-count)
    padded=np.concatenate([ious,np.full((n,n),threshold-1e-6)],axis=1)
    rows,cols=linear_sum_assignment(-padded)
    return {int(i):int(j) for i,j in zip(rows,cols) if j<m and
            ious[i,j]>=threshold},ious,inter

def encode(o):
    return dict(source=o.source,frame_id=o.frame_id,detection_id=o.detection_id,
        label=o.label,center=o.center.tolist(),low=o.low.tolist(),
        high=o.high.tolist(),timestamp=o.timestamp,confidence=o.confidence,
        appearance=o.appearance.tolist() if o.appearance is not None else None,
        map_id=o.map_id,map_version=o.map_version)

def observation(world,rgb,mask,frame,index,label,confidence,scene):
    o=instance_observation(world,rgb,mask,frame=frame,detection=str(index),
        label=label,confidence=float(confidence),source="yoloe26x_factorial")
    if o is None:return None
    o.map_id="hypersim_"+scene+"_gt_xyz"
    return encode(o)

def evaluate(scene,out,model="yoloe-26x-seg.pt",conf=.10,
             frames=FRAMES,imgsz=640,max_objects=130):
    if model!="yoloe-26x-seg.pt":raise ValueError("Only YOLOE-26x-seg is accepted")
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    prompts=class_list("config/yoloe26_classes.json")
    detector=build_detector(model,prompts)
    remote,archive=open_scene(scene)
    streams={k:[] for k in VARIANTS};frame_metrics=[]
    try:
        names=set(archive.namelist())
        meta=dict(csv.reader(io.StringIO(archive.read(
            scene+"/_detail/metadata_scene.csv").decode("utf-8"))))
        scale=float(meta["meters_per_asset_unit"])
        for frame in frames:
            prefix=f"{scene}/images/scene_cam_00_geometry_hdf5/frame.{frame:04d}."
            rgbname=f"{scene}/images/scene_cam_00_final_preview/frame.{frame:04d}.tonemap.jpg"
            if not all(x in names for x in
                [prefix+"semantic_instance.hdf5",prefix+"semantic.hdf5",
                 prefix+"position.hdf5",rgbname]):
                print("MISSING_FRAME",scene,frame,flush=True);continue
            gt=hdf5_array(archive,prefix+"semantic_instance.hdf5").astype("int32")
            sem=hdf5_array(archive,prefix+"semantic.hdf5").astype("int32")
            world=hdf5_array(archive,prefix+"position.hdf5").astype("float32")*scale
            rgb=cv2.imdecode(np.frombuffer(archive.read(rgbname),"uint8"),cv2.IMREAD_COLOR)
            if rgb is None or gt.shape!=rgb.shape[:2] or world.shape[:2]!=gt.shape:
                raise ValueError("Misregistered RGBD/GT")
            targets=target_instances(gt,sem,max_objects)
            t0=time.monotonic()
            result=detector.predict(rgb,conf=conf,device="cpu",imgsz=imgsz)
            latency=time.monotonic()-t0
            masks=[];labels=[];scores=[]
            if result.masks is not None:
                for r in result.masks.data.cpu().numpy():
                    if r.shape!=gt.shape:
                        r=cv2.resize(r.astype("float32"),(gt.shape[1],gt.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)
                    masks.append(r>.5)
                labels=[detector.class_name(x) for x in result.boxes.cls.cpu().numpy()]
                scores=[float(x) for x in result.boxes.conf.cpu().numpy()]
            match,iou,intersection=match_masks(masks,gt,targets)
            pred_area=[int(m.sum()) for m in masks]
            count_gt=len(targets)
            tp50=sum(iou[i,j]>=.5 for i,j in match.items())
            tp25=sum(iou[i,j]>=.25 for i,j in match.items())
            tp75=sum(iou[i,j]>=.75 for i,j in match.items())
            class_hits=sum(canonical(labels[i])==targets[j][1] for i,j in match.items())
            off=[1-intersection[i,j]/max(1,pred_area[i]) for i,j in match.items()]
            # Fragmentation: two distinct predicted masks cover >=15% of same
            # GT instance; merging: a predicted mask covers >=10% of two GT.
            splits=sum(sum(intersection[i,j]/max(1,t[2])>=.15 and intersection[i,j]>=30
                      for i in range(len(masks)))>=2 for j,t in enumerate(targets))
            merges=sum(sum(intersection[i,j]/max(1,t[2])>=.10
                  and intersection[i,j]/max(1,pred_area[i])>=.08
                  and intersection[i,j]>=30 for j,t in enumerate(targets))>=2
                  for i in range(len(masks)))
            frame_metrics.append(dict(frame=frame,gt=count_gt,predictions=len(masks),
                matched=len(match),tp25=tp25,tp50=tp50,tp75=tp75,
                correctly_classified=class_hits,
                mask_iou=[float(iou[i,j]) for i,j in match.items()],
                contamination=[float(v) for v in off],
                GT_instances_split_by_multiple_masks=int(splits),
                predicted_masks_spanning_multiple_GT=int(merges),
                detector_time_seconds=latency))
            for variant in VARIANTS:
                observations=[];ids=[]
                for i,mask in enumerate(masks):
                    if variant=="oracle_all":break
                    j=match.get(i)
                    if variant=="oracle_dedup" and j is None and len(targets):
                        close=int(np.argmax(iou[i]))
                        if iou[i,close]>=.10 and any(
                            jj==close and iou[h,jj]>iou[i,close] for h,jj in match.items()):
                            continue
                    label=labels[i]
                    if j is not None and variant in ("oracle_class","oracle_mask_class"):
                        label=targets[j][1]
                    if j is not None and variant in ("oracle_mask","oracle_mask_class"):
                        mask=(gt==targets[j][0])
                    o=observation(world,rgb,mask,frame,"pred_"+str(i),label,
                                  scores[i],scene)
                    if o is not None:
                        observations.append(o)
                        ids.append(int(targets[j][0]) if j is not None else None)
                if variant in ("oracle_recall","oracle_all"):
                    seen=set(match.values()) if variant=="oracle_recall" else set()
                    for j,(uid,label,_) in enumerate(targets):
                        if j in seen:continue
                        o=observation(world,rgb,gt==uid,frame,"oracle_"+str(j),label,1.,scene)
                        if o is not None:
                            observations.append(o);ids.append(int(uid))
                streams[variant].append(dict(frame=frame,observations=observations,
                    physical_gt=ids,target_visible=[t[0] for t in targets],
                    all_gt_visible=[t[0] for t in targets]))
            print("YOLOE26X_FRAME",scene,frame,"gt",count_gt,
                "pred",len(masks),"tp50",tp50,"sec",round(latency,2),flush=True)
    finally:
        archive.close();remote.close()
    if len(frame_metrics)<10:raise RuntimeError("Fewer than 10 real GT frames")
    evaluations={}
    for variant in VARIANTS:
        cache=dict(scene=scene,mode="oracle_masks" if variant=="oracle_all" else
                   "predicted_yolo_masks",frames=streams[variant],
                   detector=model,gt_labels_are_scoring_only=True,
                   intervention=variant,
                   gt_xyz_used_in_every_variant=True)
        with gzip.open(out/(variant+".json.gz"),"wt",encoding="utf8") as f:
            json.dump(cache,f)
        result=replay(cache,"v2")
        evaluations[variant]={k:v for k,v in result.items() if k!="timeline"}
        print("YOLOE26X_FACTORIAL_IDF1",scene,variant,
              result["target_gt"]["IDF1"],flush=True)
    N=sum(d["gt"] for d in frame_metrics)
    total_pred=sum(d["predictions"] for d in frame_metrics)
    all_iou=[v for d in frame_metrics for v in d["mask_iou"]]
    all_off=[v for d in frame_metrics for v in d["contamination"]]
    aligned=sum(d["matched"] for d in frame_metrics)
    report=dict(status="EVALUATED_ORACLE_INTERVENTIONS_NOT_DEPLOYABLE",
       model=model,scene=scene,frames=len(frame_metrics),prompt_classes=prompts,
       GT_world_xyz_for_all_variants=True,estimated_slam="NOT_TESTED",
       image_size=imgsz,detector_confidence=conf,
       diagnostics=dict(gt_target_occurrences=N,total_predicted_masks=total_pred,
           matched_with_IoU_0_10=aligned,
           recall_IoU25=sum(d["tp25"] for d in frame_metrics)/max(1,N),
           recall_IoU50=sum(d["tp50"] for d in frame_metrics)/max(1,N),
           recall_IoU75=sum(d["tp75"] for d in frame_metrics)/max(1,N),
           class_accuracy_aligned=sum(d["correctly_classified"] for d in frame_metrics)/max(1,aligned),
           mask_IoU_mean=float(np.mean(all_iou)) if all_iou else None,
           pixel_contamination_mean=float(np.mean(all_off)) if all_off else None,
           gt_instance_split_occurrences=sum(d["GT_instances_split_by_multiple_masks"] for d in frame_metrics),
           prediction_merge_occurrences=sum(d["predicted_masks_spanning_multiple_GT"] for d in frame_metrics),
           detector_mean_time_sec=float(np.mean([d["detector_time_seconds"] for d in frame_metrics]))),
       variants=evaluations,per_frame=frame_metrics,
       variant_definitions=dict(raw="Original YOLOE-26x masks, classes, detections",
         oracle_class="GT labels for IoU>=0.10 matched masks; masks/number fixed",
         oracle_mask="GT masks for IoU>=0.10 matched predictions; labels/count fixed",
         oracle_mask_class="GT masks+labels for matched predictions; count fixed",
         oracle_dedup="GT-guided removal of extra overlapping proposals, not deployable",
         oracle_recall="Add missed GT target instances with perfect mask+label; upper bound",
         oracle_all="Only all GT instances with GT mask+label; memory upper control"),
       caveats=["GT physical identities never enter ObjectMemoryV2 matching",
         "Oracle variants use GT to alter inference and are not deployable",
         "Matched IoU>=0.10 pairs may be poor and only define a diagnostic intervention",
         "Hypersim synthetic data and GT world XYZ; not real RGB-D SLAM",
         "Delta IDF1 is not a causal additive decomposition of errors"])
    (out/"factorial_results.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print("YOLOE26X_RESULTS",json.dumps(dict(scene=scene,model=model,
        recall50=report["diagnostics"]["recall_IoU50"],
        raw_idf1=evaluations["raw"]["target_gt"]["IDF1"],
        oracle_all_idf1=evaluations["oracle_all"]["target_gt"]["IDF1"])),flush=True)
    return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--scene",default="ai_002_001")
    p.add_argument("--out",default="results/yoloe26x_factorial")
    p.add_argument("--model",default="yoloe-26x-seg.pt")
    p.add_argument("--conf",type=float,default=.10)
    p.add_argument("--imgsz",type=int,default=640)
    p.add_argument("--max-objects",type=int,default=130)
    args=p.parse_args()
    evaluate(args.scene,args.out,model=args.model,conf=args.conf,
             imgsz=args.imgsz,max_objects=args.max_objects)
if __name__=="__main__":main()
