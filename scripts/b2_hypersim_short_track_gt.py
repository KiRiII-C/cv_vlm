#!/usr/bin/env python3
"""B2 real YOLOE-26x physical-instance GT evaluation of SHORT-TERM tracker.

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


def evaluate_scene(path,mode,max_lost=40):
    with gzip.open(path,"rt",encoding="utf8") as f:
        cache=json.load(f)
    assert cache["detector"]=="yoloe-26x-seg.pt"
    frames=cache["frames"]
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
        for i,row in enumerate(observations):
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
        if len(decisions)!=len(physical):
            raise RuntimeError("Missing decision")
        target=set(int(x) for x in frame["target_visible"])
        all_gt+=len(target)
        for physical_id,assignment in zip(physical,decisions):
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
    p.add_argument("--out",type=Path,default=Path("results/b2"))
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    results={}
    for scene in SCENES:
        cache=args.root/f"full_sequence_{scene}"/"raw_observations_and_GT_scoring.json.gz"
        if not cache.exists():raise RuntimeError("Missing real full YOLOE cache: "+str(cache))
        results[scene]={method:evaluate_scene(cache,method) for method in ("3d_only","3d_and_appearance")}
    totals={}
    for method in ("3d_only","3d_and_appearance"):
        s=Counter()
        for scene in SCENES:
            row=results[scene][method]
            s.update({k:row["scores"][k] for k in ("IDTP","IDFP","IDFN")})
        totals[method]={"pooled_IDF1":2*s["IDTP"]/max(1,2*s["IDTP"]+s["IDFP"]+s["IDFN"]),
                        "IDTP":s["IDTP"],"IDFP":s["IDFP"],"IDFN":s["IDFN"]}
    report={
      "experiment":"B2_PHYSICAL_GT_SHORT_TERM_TRACKER_ABLATION",
      "source":"three_complete_100_frame_Hypersim_scenes_frozen_real_YOLOE_26x",
      "full_frames_processed":sum(results[s]["3d_only"]["frames"] for s in SCENES),
      "detector_reexecuted":False,"GT_used_for_tracking":False,
      "pose_source":"Hypersim_OFFICIAL_GT_XYZ_not_RTAB_map",
      "mask_reprojection_evaluated":False,
      "true_short_tracking_IDF1_evaluated":True,
      "comparators":["3d_only","3d_and_appearance"],
      "per_scene":results,"pooled":totals,
      "caution":"GT XYZ is used for 3D measurement, and 2D boxes/masks are not cached. "
                "These figures do not represent RTAB-Map+YOLOE metric performance or full projected-mask tracker."}
    (args.out/"b2_physical_gt_short_tracker.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print("B2_RESULTS",json.dumps({"frames":report["full_frames_processed"],
          "pooled":totals,
          "scenes":{s:{k:{"IDF1":v["scores"]["IDF1"],
                          "false_merge":v["scores"]["false_merge_predicted_uuid"],
                          "false_split":v["scores"]["false_split_physical_gt"]}
                   for k,v in result.items()} for s,result in results.items()}},
         ensure_ascii=False),flush=True)

if __name__=="__main__":main()
