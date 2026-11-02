#!/usr/bin/env python3
"""B5 real YOLOE-26x physical GT short-tracking across 1/3/5/10 detection cadence.

Input: frozen real YOLOE-26x predictions on all 100 frames of each complete
Hypersim scene, with observed 3D mask centroids from REAL renderer GT XYZ.
GT physical ID is used ONLY for scoring. No true instance boxes or masks
supplied to tracker. Cached predictions lack image-plane boxes/masks, so
B2 tests 3D+appearance-only ablation, NOT projected-mask fusion or SLAM.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import gzip,json
from pathlib import Path
import numpy as np
from semantic_map.temporal_tracker import Detection,TemporalTracker
from semantic_map.short_term_semantics_b import ClassTolerantTemporalAssociator,class_parent
from semantic_map.memory_v2_replay import score_identity

SCENES=("ai_002_001","ai_005_001","ai_017_001")

class GeometryOnlyTracker(TemporalTracker):
    """B2 comparison without fabricated camera-plane bboxes or masks."""
    def __init__(self,*,use_appearance,**kw):
        super().__init__(**kw)
        self.use_appearance=use_appearance

    def _score(self,track,d,frame):
        if class_parent(track.label)!=class_parent(d.label):
            return -1.
        dt=max(1,frame-track.last_frame)
        dist=float(np.linalg.norm(track.predict(frame)-d.xyz))
        limit=min(.8,.25+.09*np.sqrt(dt))
        if dist>limit:return -1.
        prox=float(np.exp(-.5*(dist/max(.10,limit*.55))**2))
        if not self.use_appearance:return prox
        if track.feature is None or d.feature is None:
            return prox
        a=float(np.clip(track.feature@d.feature,0.,1.))
        if a<.32 and dist>.10:return -1.
        return float(.78*prox+.22*a)


def evaluate_scene(path,mode,stride,max_lost=40):
    with gzip.open(path,"rt",encoding="utf8") as f:
        cache=json.load(f)
    assert cache["detector"]=="yoloe-26x-seg.pt"
    frames=cache["frames"]
    assert stride in (1,3,5,10)
    assert len(frames)==100
    assert [int(f["frame"]) for f in frames]==list(range(100))
    useapp=mode!="3d_only"
    tracker=ClassTolerantTemporalAssociator(
        GeometryOnlyTracker(use_appearance=useapp,
             high_thresh=.27,low_thresh=.10,match_score=.57,max_lost=max_lost,
             min_hits=2,ambiguity_margin=.065))
    all_items=[];count=Counter();all_gt=0
    for frame in frames:
        observations=frame["observations"]
        physical=frame["physical_gt"]
        if len(observations)!=len(physical):
            raise RuntimeError("GT/observations alignment failure")
        dets=[]
        detection_is_run=(int(frame["frame"])%stride==0)
        if detection_is_run: count["detector_keyframes"]+=1
        for i,row in enumerate(observations if detection_is_run else []):
            xyz=np.asarray(row["center"],dtype=float)
            # Neutral technical bbox is required by Detection constructor but
            # is NEVER consumed by GeometryOnlyTracker._score() and is NOT
            # interpreted as a real image-space bbox.
            bbox=np.array([0.,0.,1.,1.])
            emb=None
            if useapp and row.get("appearance") is not None:
                emb=np.asarray(row["appearance"],dtype=float)
            dets.append(Detection(str(row["detection_id"]),row["label"],
                                  bbox,xyz,float(row["confidence"]),feature=emb,mask=None))
        decisions=tracker.update(int(frame["frame"]),dets)
        if len(decisions)!=(len(physical) if detection_is_run else 0):
            raise RuntimeError("Missing decision")
        target=set(int(x) for x in frame["target_visible"])
        all_gt+=len(target)
        for physical_id,assignment in zip(physical if detection_is_run else [],decisions):
            # Groundtruth labels never accessed during matching.
            pid=int(physical_id) if physical_id is not None and int(physical_id) in target else None
            all_items.append((pid,assignment.short_id))
            count[assignment.status]+=1
        count["frames"]+=1
        count["detections"]+=len(dets)
    scores=score_identity(all_items,all_gt)
    return {
       "source":cache["scene"],"method":mode,"frames":count["frames"],
       "predicted_mask_observations":count["detections"],
       "detector_keyframes":count["detector_keyframes"],"inference_stride":stride,
       "all_100_source_frames_accounted":True,
       "GT_total_visible_instance_frames":all_gt,
       "tracklet_count":tracker.tracker._next_id,
       "association_statuses":{k:v for k,v in count.items() if k not in ("frames","detections")},
       "physical_GT_scoring_only":True,
       "camera_bboxes_or_masks_consumed":False,
       "geometry_origin":"official_Hypersim_GT_XYZ_for_observed_YOLO_MASK_POINTS_not_SLAM",
       "scores":scores,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,default=Path("results/b2_inputs"))
    p.add_argument("--out",type=Path,default=Path("results/b5_tracking"))
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    results={}
    strides=(1,3,5,10)
    models=("3d_only","3d_and_appearance")
    for scene in SCENES:
        cache=args.root/f"full_sequence_{scene}"/"raw_observations_and_GT_scoring.json.gz"
        if not cache.exists():
            cache=args.root/"results"/f"full_sequence_{scene}"/"raw_observations_and_GT_scoring.json.gz"
        if not cache.exists():raise RuntimeError("Missing real full YOLOE cache: "+str(cache))
        results[scene]={str(k):{m:evaluate_scene(cache,m,k) for m in models} for k in strides}
    totals={}
    for stride in strides:
        totals[str(stride)]={}
        for method in models:
            s=Counter()
            for scene in SCENES:
                row=results[scene][str(stride)][method]
                s.update({k:row["scores"][k] for k in ("IDTP","IDFP","IDFN")})
            totals[str(stride)][method]={
                "pooled_IDF1":2*s["IDTP"]/max(1,2*s["IDTP"]+s["IDFP"]+s["IDFN"]),
                "IDTP":s["IDTP"],"IDFP":s["IDFP"],"IDFN":s["IDFN"],
                "detector_frames_per_scene":results[SCENES[0]][str(stride)][method]["detector_keyframes"]}
    report={
      "experiment":"B5_FULL_SEQUENCE_PHYSICAL_GT_DETECTION_CADENCE",
      "source":"three_complete_100_frame_Hypersim_scenes_frozen_real_YOLOE_26x",
      "full_frames_accounted":300,
      "sensor_frames_per_scene":100,
      "detector_inference_policy":"every_1_3_5_10_frames_based_on_100_REAL_FROZEN_PREDICTIONS",
      "detector_reexecuted":False,"GT_used_for_tracking":False,
      "pose_source":"Hypersim_OFFICIAL_GT_XYZ_not_RTAB_map",
      "camera_bboxes_or_masks_consumed":False,
      "surface_reprojection_evaluated":False,
      "true_physical_IDF1_evaluated":True,
      "GT_misses_counted_on_ALL_100_frames":True,
      "physical_ground_truth_quality":"official_Hypersim_semantic_instance_label",
      "per_scene":results,"pooled":totals,
      "limitations":"Frozen real YOLOE predictions; missing masks, RTAB-Map pose error and long-term ID handling. "
                    "Changing detector cadence is an ablation; ALL original frame ground truths are scored."}
    (args.out/"b5_full_gt_tracking_cadence.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print("B5_TRACKING_CADENCE",json.dumps(totals,ensure_ascii=False),flush=True)

if __name__=="__main__":main()
