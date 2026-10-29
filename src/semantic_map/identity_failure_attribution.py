"""Disentangle physical-ID changes while visible, missed by detector, or truly out-of-view.

Uses frozen actual YOLOE-26x predictions and official Hypersim GT from the
completed 100/100-frame runs. Replays exact legacy and ObjectMemoryV2 engines.
The short-term TemporalTracker is NOT invoked in this diagnostic or original
full-Hypersim evaluation; cannot attribute errors to its internal tracking.

Same-object GT association is based on the original evaluation's 2D IoU >= .10
Hungarian assignment; this is an observation-level diagnostic, not HOTA.
"""
from __future__ import annotations
import argparse,gzip,json,math,tempfile
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from .models import Observation
from .storage import MemoryStore
from .object_memory_v2 import ObjectMemoryV2
from .engine import SemanticMapper
from .association import overlap_3d
from .memory_v2_replay import score_identity

def _cache(path):
    with gzip.open(path,"rt",encoding="utf8") as f:return json.load(f)

def _gate_diagnostic(old_track, obs, engine):
    """State BEFORE frame update: why prior same-GT global UUID is incompatible.

    Reports hard evidence, not guaranteed counterfactual root cause (Hungarian
    competition, other tracks and history can still change the decision).
    """
    if old_track is None:return {"gate":"old_global_id_absent"}
    if old_track.map_id!=obs.map_id or old_track.map_version!=obs.map_version:
        return {"gate":"map_version"}
    if engine._category(old_track.label)!=engine._category(obs.label):
        return {"gate":"category_changed"}
    d=float(np.linalg.norm(old_track.center-obs.center))
    radius=engine.DEFAULT_STATIC_REID_RADIUS.get(engine._category(old_track.label))
    if radius is not None and d>radius:return {"gate":"class_radius","distance_m":d,"cap_m":radius}
    gap=np.maximum(0.,np.maximum(old_track.low,obs.low)-np.minimum(old_track.high,obs.high))
    surface_gap=float(np.linalg.norm(gap))
    size=max(.18,min(1.2,float(np.linalg.norm(old_track.high-old_track.low)*.48)))
    if surface_gap>.21 or d>max(.52,1.55*size):
        return {"gate":"surface_or_centroid","distance_m":d,"surface_gap_m":surface_gap}
    overlap=overlap_3d(old_track.low,old_track.high,obs.low,obs.high)
    appearance=engine._gallery_similarity(old_track,obs)
    if appearance is not None and appearance<.2 and overlap<.64:
        return {"gate":"appearance_reject","appearance_similarity":appearance}
    score=engine._score(old_track,obs)
    if score<engine.candidate_floor:return {"gate":"score_below_candidate_floor","score":score}
    if score<engine.min_score:return {"gate":"score_below_match_threshold","score":score}
    return {"gate":"compatible_but_not_selected","score":score}

def classify_switch(prev, current, visible_frames, all_matched_gt_frames):
    """One and only one mutually exclusive visibility/event class."""
    fa,fb=prev["frame"],current["frame"]
    if fb==fa:return "multiple_masks_same_frame"
    if fb==fa+1:return "direct_consecutive_frames"
    between=range(fa+1,fb)
    if any(frame not in visible_frames for frame in between):
        return "true_out_of_view"
    missed=[f for f in between if f not in all_matched_gt_frames]
    if missed:return "still_visible_detector_gap"
    return "still_visible_but_unassigned_uuid"

def audit(cache, oracle=None, *,engine="v2"):
    if cache["detector"]!="yoloe-26x-seg.pt":
        raise ValueError("Only frozen actual YOLOE-26x predictions")
    frames=cache["frames"]
    if [f["frame"] for f in frames]!=list(range(100)):
        raise ValueError("Full 100 consecutive frames required")
    if oracle is not None and [f["frame"] for f in oracle["frames"]]!=list(range(100)):
        raise ValueError("Oracle frame alignment failed")
    gt_visible=defaultdict(set)
    obs_present=defaultdict(set)
    all_gt_classes={}
    for f in oracle["frames"] if oracle else []:
        for uid,obs in zip(f["physical_gt"],f["observations"]):
            if uid is not None:all_gt_classes[(f["frame"],uid)]=obs["label"]
    for f in frames:
        for uid in f["target_visible"]:gt_visible[int(uid)].add(f["frame"])
        for uid in f["physical_gt"]:
            if uid is not None and uid in f["target_visible"]:
                obs_present[int(uid)].add(f["frame"])
    diag=Counter();events=[];matched_by_gt=defaultdict(list)
    status_counts=Counter();gt_miss_count=0;class_counts=Counter()
    for f in frames:
        gt_miss_count+=len(set(f["target_visible"])-set(u for u in f["physical_gt"] if u is not None))
        for uid,obs in zip(f["physical_gt"],f["observations"]):
            if uid not in f["target_visible"]:continue
            expected=all_gt_classes.get((f["frame"],uid))
            if expected is not None:
                class_counts["class_matched" if expected==obs["label"] else "class_mismatched"]+=1
    with tempfile.TemporaryDirectory(prefix="hypersim_identity_attribution_") as d:
        store=MemoryStore(Path(d)/"map.sqlite")
        tracker=ObjectMemoryV2(store,min_hits=2) if engine=="v2" else SemanticMapper(store,min_hits=2,association_mode="fused")
        # Snapshot of latest assigned mask observation for each GT, for
        # comparing object geometry before vs after a potential split.
        first_seen_global={}
        old_id_by_gt={}
        for f in frames:
            obs=[Observation(**row) for row in f["observations"]]
            previous_track_by_uid={t.object_id:t for t in store.tracks()} if engine=="v2" else {}
            decisions=tracker.process_frame(obs)
            if len(decisions)!=len(obs):raise AssertionError("Decision length")
            for i,(gt,d,o) in enumerate(zip(f["physical_gt"],decisions,obs)):
                status="not_recorded"
                if engine=="v2":
                    q=store.db.execute("""SELECT status FROM object_evidence_v2
                       WHERE map_id=? AND source=? AND frame_id=? AND detection_id=?""",
                       o.key).fetchone()
                    if q is None:raise AssertionError("Missing V2 immutable evidence")
                    status=q["status"]
                else:
                    q=store.db.execute("""SELECT association_status FROM observations
                       WHERE map_id=? AND source=? AND frame_id=? AND detection_id=?""",
                       o.key).fetchone()
                    if q is not None:status=q["association_status"]
                status_counts[status]+=1
                if gt is None or gt not in f["target_visible"]:
                    continue
                curr={"frame":f["frame"],"gt":int(gt),"uuid":d.object_id,
                      "status":status,"model_class":o.label,
                      "center_m":o.center.tolist(),"confidence":o.confidence,
                      "global_id_previously_seen":d.object_id in first_seen_global}
                if d.object_id is not None:
                    first_seen_global.setdefault(d.object_id,(f["frame"],int(gt)))
                    old=old_id_by_gt.get(int(gt))
                    if old is not None and old["uuid"]!=curr["uuid"]:
                        kind=classify_switch(old,curr,gt_visible[int(gt)],obs_present[int(gt)])
                        is_created=(status=="created_tentative")
                        comp={"gate":"not_in_v2"}
                        if engine=="v2":
                            comp=_gate_diagnostic(previous_track_by_uid.get(old["uuid"]),o,tracker)
                        event={"scene":cache["scene"],"source":"raw_yoloe26x",
                               "engine":engine,"physical_GT_ID":int(gt),
                               "prev_frame":old["frame"],"frame":f["frame"],
                               "frames_between":f["frame"]-old["frame"]-1,
                               "previous_uuid":old["uuid"],"current_uuid":curr["uuid"],
                               "kind":kind,"category_switch":old["model_class"]!=curr["model_class"],
                               "prev_label":old["model_class"],"curr_label":curr["model_class"],
                               "prev_confidence":old["confidence"],"curr_confidence":curr["confidence"],
                               "3d_center_jump_m":float(np.linalg.norm(o.center-np.asarray(old["center_m"]))),
                               "created_new_global_uuid":is_created,
                               "assigned_preexisting_uuid":curr["global_id_previously_seen"],
                               "decision_status":status,
                               "previous_global_uuid_gate":comp}
                        events.append(event)
                        diag[kind]+=1
                        diag["uuid_switch_events"]+=1
                        diag["created_new_global_uuid" if is_created else "switched_to_existing_global_uuid"]+=1
                        diag["category_changed" if event["category_switch"] else "same_category"]+=1
                        diag["prior_gate_"+comp["gate"]]+=1
                    old_id_by_gt[int(gt)]=curr
                    matched_by_gt[int(gt)].append(curr)
        store.close()
    # Calculations deliberately exclude any inferred physical GT ID from
    # processing. The full sequence GT is only used to partition the errors.
    categorized=sum(diag[x] for x in (
        "direct_consecutive_frames","true_out_of_view",
        "still_visible_detector_gap","still_visible_but_unassigned_uuid",
        "multiple_masks_same_frame"))
    if categorized!=len(events):raise AssertionError("Switch partition not exhaustive")
    denominator_gt=sum(len(f["target_visible"]) for f in frames)
    matched_pairs=sum(1 for f in frames for g in f["physical_gt"]
                      if g is not None and g in f["target_visible"])
    return {
      "scene":cache["scene"],"engine":engine,"model":cache["detector"],
      "frames":len(frames),"short_term_tracker_executed":False,
      "physical_gt_target_occurrences":denominator_gt,
      "detector_gt_matched_observations":matched_pairs,
      "no_gt_matched_mask_occurrences":gt_miss_count,
      "model_class_diagnostic":dict(class_counts),
      "association_status_counts":dict(status_counts),
      "uuid_switch_counts":dict(diag),"events":events,
      "caveats":[
        "Each predicted mask matched to at most one GT physical instance by IoU>=0.10 Hungarian; duplicate masks may appear as false positives instead of same-GT double-matches",
        "Ground truth identities are used only for event attribution and scores, never in ObjectMemoryV2",
        "Status created_tentative comes from immutable per-observation SQLite evidence; GT is never a decision input",
        "GT frame visibility distinguishes absent from detector missed in view; out-of-view events may have mixed detector gaps",
        "Switches between consecutive assigned observations measure ID disagreement, not official TrackEval IDS",
        "No TemporalTracker in this full-sequence benchmark",
        "GT world XYZ supplied; SLAM errors not tested"]
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--scene",required=True)
    p.add_argument("--raw",required=True)
    p.add_argument("--oracle",required=True)
    p.add_argument("--out",required=True)
    a=p.parse_args()
    cache=_cache(a.raw);oracle=_cache(a.oracle)
    if cache["scene"]!=a.scene or oracle["scene"]!=a.scene:
        raise ValueError("Input scene/provenance mismatch")
    result={"scene":a.scene,
            "experimental_stages":["independent_detection_2d","3D_world_observation",
                                  "global_persistent_object_memory_only"],
            "attribution":{
                engine:audit(cache,oracle,engine=engine) for engine in ("legacy","v2")}}
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    for engine,diag in result["attribution"].items():
        print("FULL_SCENE_ID_CAUSAL_CATEGORIES",a.scene,engine,
              json.dumps(diag["uuid_switch_counts"]),flush=True)
        print("FULL_SCENE_OBSERVATION_STATUSES",a.scene,engine,
              json.dumps(diag["association_status_counts"]),flush=True)
    return result
if __name__=="__main__":main()
