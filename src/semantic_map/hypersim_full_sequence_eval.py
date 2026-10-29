"""Full (not subsampled) Hypersim physical-ID sequence validation.

Each selected camera trajectory is enumerated from the official scene ZIP:
RGB, GT XYZ, semantic and instance labels must be available for EVERY frame.
YOLOE-26x runs on all frames, then frozen model predictions and oracle
instance masks are independently associated by legacy and ObjectMemoryV2.
Only evaluation receives GT identity. Report includes gaps/re-entries, ID
switches, fragmentation, false merges, cumulative and window identity scores.

Important: 3D XYZ is Hypersim GT, not estimated camera trajectory or SLAM.
"""
from __future__ import annotations
import argparse,csv,gzip,io,json,time
from collections import Counter,defaultdict
from pathlib import Path
import cv2,numpy as np
from .hypersim_range import open_scene,hdf5_array
from .hypersim_full_sequences import discover
from .yoloe26x_factorial import target_instances,match_masks,observation
from .yoloe26_detector import build_detector,class_list
from .memory_v2_replay import replay,score_identity

def _serialize(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,
        default=lambda a:a.item() if isinstance(a,np.generic) else
                       a.tolist() if isinstance(a,np.ndarray) else str(a)))

def _cache(path,data):
    with gzip.open(path,"wt",encoding="utf8",compresslevel=5) as f:json.dump(data,f)

def gap_stats(cache,timeline):
    """GT used exclusively for evaluation, never the tracker.
    Identity switch is between *observed* detections of the same physical
    object, not the official HOTA/TrackEval CLEAR MOT event definition.
    """
    if len(cache["frames"])!=len(timeline):
        raise ValueError("Decision timeline is not aligned to all input frames")
    gt_seen=defaultdict(list)
    pred_seen=defaultdict(list)
    per_frame=[]
    thresholds=(5,10,20,30,50)
    for frame,dec in zip(cache["frames"],timeline):
        t=int(frame["frame"]); vis=set(map(int,frame["target_visible"]))
        for uid in vis:gt_seen[int(uid)].append(t)
        if len(dec["uuids"])!=len(frame["physical_gt"]):
            raise ValueError("Wrong decision count")
        for gt,uid in zip(frame["physical_gt"],dec["uuids"]):
            if gt is not None and gt in vis:
                pred_seen[int(gt)].append((t,uid))
        per_frame.append(dict(frame=t,gt_visible=len(vis),
             assigned=sum(x is not None for x in dec["uuids"]),
             predicted=len(dec["uuids"]),
             matched_target=sum(g is not None and u is not None for g,u
                 in zip(frame["physical_gt"],dec["uuids"]))))
    records={}
    for label in ("detector_gaps","gt_out_of_view_reentry"):
        for bound in thresholds:
            allpairs=0;retained=0;switched=0;unknown=0
            for physical,records_for_id in pred_seen.items():
                # Consecutive detector observations measure detector-miss
                # or out-of-view gaps; consecutive GT-visible frames
                # measure true out-of-view re-entry specifically.
                byframe=defaultdict(list)
                for f,uid in records_for_id:byframe[f].append(uid)
                usable={f:next((u for u in uids if u is not None),None)
                         for f,uids in byframe.items()}
                if label=="detector_gaps":
                    candidates=sorted(usable)
                else:
                    candidates=gt_seen.get(physical,[])
                for a,b in zip(candidates[:-1],candidates[1:]):
                    if b-a-1<bound:continue
                    allpairs+=1
                    u=usable.get(a);v=usable.get(b)
                    if u is None or v is None:unknown+=1
                    elif u==v:retained+=1
                    else:switched+=1
            records[f"{label}_at_least_{bound}_missing_frames"]={
               "eligible_physical_reappearance_pairs":allpairs,
               "uuid_retained":retained,"uuid_switched":switched,
               "unresolved_endpoint":unknown,
               "retention_given_two_assigned_endpoints":
                   retained/(retained+switched) if retained+switched else None}
    consecutive_switches=0;observed_pair_count=0
    objects_with_split=0
    for physical,seq in pred_seen.items():
        nonempty=[(f,u) for f,u in seq if u is not None]
        distinct={u for _,u in nonempty}
        if len(distinct)>1:objects_with_split+=1
        for (_,a),(_,b) in zip(nonempty[:-1],nonempty[1:]):
            observed_pair_count+=1
            consecutive_switches+=a!=b
    records["per_physical_object"]={
      "gt_unique_visible":len(gt_seen),
      "gt_unique_ever_matched":len(pred_seen),
      "gt_objects_with_two_or_more_assigned_uuids":objects_with_split,
      "observed_valid_uuid_transition_pairs":observed_pair_count,
      "observed_uuid_switch_transitions":int(consecutive_switches),
      "warning":"Observation-level transitions, not official MOT ID-switch/HOTA."}
    return {"gaps":records,"per_frame":per_frame}

def _evolving_scores(cache,decision_timeline):
    """Separate local-window and cumulative IDF1; global mapping not shared
    across windows for window metric, but shared inside cumulative prefix."""
    n=len(cache["frames"])
    boundaries=sorted(set([0,n//4,n//2,3*n//4,n]))
    windows=[]
    def sliced_score(start,end):
        gt_total=sum(len(f["target_visible"]) for f in cache["frames"][start:end])
        items=[]
        for f,t in zip(cache["frames"][start:end],decision_timeline[start:end]):
            vis=set(f["target_visible"])
            items.extend((g if g in vis else None,uid) for g,uid in
                         zip(f["physical_gt"],t["uuids"]))
        return score_identity(items,gt_total)
    for a,b in zip(boundaries[:-1],boundaries[1:]):
        if b<=a:continue
        windows.append(dict(start_frame=cache["frames"][a]["frame"],
            end_frame=cache["frames"][b-1]["frame"],
            frames=b-a,local_window=sliced_score(a,b),
            cumulative=sliced_score(0,b)))
    return windows

def run(scene,output,*,camera="cam_00",model="yoloe-26x-seg.pt",
        conf=.10,imgsz=640,include_oracle=True):
    if model!="yoloe-26x-seg.pt":raise ValueError("Full-sequence benchmark requires requested YOLOE-26x")
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    clips=class_list("config/yoloe26_classes.json")
    predictor=build_detector(model,clips)
    remote,z=open_scene(scene)
    raw_frames=[];oracle_frames=[];frame_rows=[]
    try:
        cameras=discover(z,scene)
        if camera not in cameras:raise ValueError(f"Missing camera {camera}; available {sorted(cameras)}")
        metadata=cameras[camera]
        frames=metadata["frames_complete"]
        if not frames or metadata["missing_modalities"] or metadata["missing_frame_indices"]:
            raise RuntimeError("Not an entire contiguous fully annotated camera sequence")
        if frames!=list(range(frames[0],frames[-1]+1)):
            raise RuntimeError("Non-contiguous camera trajectory")
        scale=float(dict(csv.reader(io.StringIO(z.read(
            f"{scene}/_detail/metadata_scene.csv").decode("utf8"))))["meters_per_asset_unit"])
        for n,frame in enumerate(frames):
            pre=f"{scene}/images/scene_{camera}_geometry_hdf5/frame.{frame:04d}."
            jpg=f"{scene}/images/scene_{camera}_final_preview/frame.{frame:04d}.tonemap.jpg"
            gt=hdf5_array(z,pre+"semantic_instance.hdf5").astype("int32")
            sem=hdf5_array(z,pre+"semantic.hdf5").astype("int32")
            xyz=hdf5_array(z,pre+"position.hdf5").astype("float32")*scale
            rgb=cv2.imdecode(np.frombuffer(z.read(jpg),"uint8"),cv2.IMREAD_COLOR)
            if rgb is None or not (gt.shape==sem.shape==rgb.shape[:2]==xyz.shape[:2]):
                raise ValueError(f"RGB/XYZ/instance frame alignment error {frame}")
            targets=target_instances(gt,sem,limit=1_000_000)
            started=time.monotonic()
            result=predictor.predict(rgb,conf=conf,device="cpu",imgsz=imgsz)
            dt=time.monotonic()-started
            masks=[];labels=[];scores=[]
            if result.masks is not None:
                masks=result.masks.data.cpu().numpy()
                masks=[cv2.resize(m.astype("float32"),(gt.shape[1],gt.shape[0]),
                          interpolation=cv2.INTER_NEAREST)>.5
                       if m.shape!=gt.shape else m>.5 for m in masks]
                labels=[predictor.class_name(x) for x in result.boxes.cls.cpu().numpy()]
                scores=[float(x) for x in result.boxes.conf.cpu().numpy()]
            pairing,ious,_=match_masks(masks,gt,targets,threshold=.10)
            raw=[];physical=[]
            for i,mask in enumerate(masks):
                if mask.sum()<250:continue
                o=observation(xyz,rgb,mask,frame,f"det_{i}",labels[i],scores[i],scene)
                if o is None:continue
                raw.append(o)
                j=pairing.get(i)
                physical.append(int(targets[j][0]) if j is not None else None)
            oracle=[];oracle_ids=[]
            if include_oracle:
                for i,(uid,label,pixels) in enumerate(targets):
                    o=observation(xyz,rgb,gt==uid,frame,f"oracle_{i}",label,1.,scene)
                    if o is None:
                        raise RuntimeError(f"Valid GT target {uid} discarded by observation extraction at {frame}")
                    oracle.append(o);oracle_ids.append(int(uid))
            common={"frame":int(frame),"target_visible":[int(t[0]) for t in targets],
                    "all_gt_visible":[int(t[0]) for t in targets]}
            raw_frames.append({**common,"observations":raw,"physical_gt":physical})
            if include_oracle:oracle_frames.append({**common,"observations":oracle,"physical_gt":oracle_ids})
            frame_rows.append(dict(frame=frame,gt_target_instances=len(targets),
               actual_predicted_masks=len(masks),usable_raw_observations=len(raw),
               gt_matches_iou50=sum(ious[i,j]>=.5 for i,j in pairing.items()),
               gt_matches_iou25=sum(ious[i,j]>=.25 for i,j in pairing.items()),
               cpu_model_seconds=dt))
            if n%10==0 or n==len(frames)-1:
                print("FULL_SEQUENCE_PROGRESS",scene,camera,n+1,"/",len(frames),
                      "gt",len(targets),"detections",len(raw),"seconds",round(dt,2),flush=True)
    finally:
        z.close();remote.close()
    if len(raw_frames)!=len(frames):raise RuntimeError("Any lost frame invalidates full-sequence claim")
    all_sources={"raw":dict(scene=scene,mode="predicted_yolo_masks",
        detector=model,frames=raw_frames,gt_labels_are_scoring_only=True),
        **({"oracle":dict(scene=scene,mode="oracle_masks",detector="GT_ORACLE_MASKS",
          frames=oracle_frames,gt_labels_are_scoring_only=True)} if include_oracle else {})}
    report={"status":"COMPLETE_ANNOTATED_CAMERA_TRAJECTORY_EVALUATED",
       "model":model,"scene":scene,"camera":camera,"frames_expected":len(frames),
       "frames_processed":len(frame_rows),"first_frame":frames[0],"last_frame":frames[-1],
       "frame_indices_complete":frames==list(range(frames[0],frames[-1]+1)),
       "missing_modalities":metadata["missing_modalities"],
       "max_object_limit":None,"GT_world_XYZ_used":True,"SLAM":"NOT_TESTED",
       "12_model_prompts":clips,"confidence":conf,
       "observations_raw":sum(len(f["observations"]) for f in raw_frames),
       "physical_target_GT_occurrences":sum(len(f["target_visible"]) for f in raw_frames),
       "physical_GT_unique_target_IDs":len({v for f in raw_frames for v in f["target_visible"]}),
       "mask_recall_50":sum(r["gt_matches_iou50"] for r in frame_rows)/
          max(1,sum(r["gt_target_instances"] for r in frame_rows)),
       "mean_cpu_inference_seconds":float(np.mean([r["cpu_model_seconds"] for r in frame_rows])),
       "per_frame":frame_rows,"results":{},
       "scoring_note":"GT physical IDs never enter tracker; available only to evaluation. GT XYZ is input to all variants. IDF1 observation-level, not official HOTA."}
    for typ,cache in all_sources.items():
        _cache(root/f"{typ}_observations_and_GT_scoring.json.gz",cache)
        report["results"][typ]={}
        for engine in ("legacy","v2"):
            result=replay(cache,engine)
            diagnostics=gap_stats(cache,result["timeline"])
            windows=_evolving_scores(cache,result["timeline"])
            report["results"][typ][engine]={
                "scores":{k:v for k,v in result.items() if k!="timeline"},
                "time_windows":windows,
                "reidentification":diagnostics["gaps"],
                "per_frame_assignments":diagnostics["per_frame"]}
            print("FULL_SEQUENCE_IDENTITY",scene,typ,engine,
                  "IDF1",result["target_gt"]["IDF1"],
                  "false_merges",result["target_gt"]["false_merge_predicted_uuid"],
                  "false_splits",result["target_gt"]["false_split_physical_gt"],
                  "uuid_switches",diagnostics["gaps"]["per_physical_object"]["observed_uuid_switch_transitions"],flush=True)
    _serialize(root/"full_sequence_report.json",report)
    print("FULL_SEQUENCE_DONE",scene,camera,len(frames),
        "raw_IDF1",report["results"]["raw"]["v2"]["scores"]["target_gt"]["IDF1"],flush=True)
    return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--scene",required=True)
    p.add_argument("--camera",default="cam_00")
    p.add_argument("--output",default="results/hypersim_full_sequence")
    p.add_argument("--conf",type=float,default=.10)
    p.add_argument("--imgsz",type=int,default=640)
    p.add_argument("--model",default="yoloe-26x-seg.pt")
    p.add_argument("--skip-oracle",action="store_true")
    a=p.parse_args()
    run(a.scene,a.output,camera=a.camera,conf=a.conf,imgsz=a.imgsz,
        model=a.model,include_oracle=not a.skip_oracle)
if __name__=="__main__":main()
