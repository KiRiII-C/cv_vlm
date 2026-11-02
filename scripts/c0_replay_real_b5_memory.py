#!/usr/bin/env python3
"""C0: replay every real B5 YOLOE observation into existing ObjectMemoryV2.

No Ground Truth, no fake appearance vector, no invented point measurements.
Quality-rejected observations remain auditable, with no physical UUID.
Every frame is processed in chronological order. An unobserved object is
UNKNOWN, never marked removed. The resulting UUIDs remain HYPOTHESES until
independent physical-instance GT validation.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import numpy as np
from semantic_map.models import Observation
from semantic_map.storage import MemoryStore
from semantic_map.object_memory_v2 import ObjectMemoryV2

SOURCE="TUM_fr1_desk_B5_YOLOE26x"
MAP_ID="A6_RTABMAP_FULL_TUM_DESK"

def read_inputs(observations_path,poses_path):
    p=json.loads(Path(poses_path).read_text())
    if p.get("GT_used_as_input") is not False:
        raise RuntimeError("C0 refuses GT-derived world poses")
    transforms={}
    for row in p["poses"]:
        idx=int(row["index"])
        T=np.asarray(row["T_map_cam"],dtype=float)
        if T.shape!=(4,4) or not np.isfinite(T).all():
            raise RuntimeError("Bad A6 measured camera pose")
        transforms[idx]=T
    rows=[json.loads(line) for line in Path(observations_path).read_text().splitlines() if line.strip()]
    if len(rows)==0:raise RuntimeError("B5 actual real observations missing")
    rows.sort(key=lambda a:(int(a["frame_index"]),str(a["detection_id"])))
    if len({(int(e["frame_index"]),str(e["detection_id"])) for e in rows})!=len(rows):
        raise RuntimeError("Duplicate B5 frame/detection evidence")
    if any(int(row["frame_index"]) not in transforms for row in rows):
        raise RuntimeError("B5 observation without valid measured A6 pose")
    return rows,transforms

def convert(row,T):
    lo=np.asarray(row["aabb_min_map_m"],float)
    hi=np.asarray(row["aabb_max_map_m"],float)
    center=np.asarray(row["center_map_m"],float)
    return Observation(
        source=SOURCE,frame_id=str(int(row["frame_index"])),
        detection_id=str(row["detection_id"]),label=str(row["class"]),
        center=center,low=lo,high=hi,
        timestamp=float(row["timestamp_tum"]),
        confidence=float(row["confidence"]),
        T_map_cam=T,map_id=MAP_ID,map_version=0,
        # B5 never exported a learned visual embedding: do not fabricate one.
        appearance=None)

def replay(rows,transforms,db_path,*,min_conf=.40,min_depth_ratio=.45,min_pixels=60):
    if not (0<=min_conf<=1 and 0<=min_depth_ratio<=1 and min_pixels>=1):
        raise ValueError("Invalid immutable quality gate")
    db_path=Path(db_path)
    db_path.parent.mkdir(parents=True,exist_ok=True)
    db=MemoryStore(db_path)
    mem=ObjectMemoryV2(db,min_hits=3,conservative=True)
    frames=defaultdict(list)
    for row in rows: frames[int(row["frame_index"])].append(row)
    counters=Counter()
    events=[]
    for frame in sorted(frames):
        obs=[];keys=[]
        for item in frames[frame]:
            key={"frame_index":frame,"detection_id":str(item["detection_id"]),
                 "class":str(item["class"]),"confidence":float(item["confidence"]),
                 "short_track_id_unverified":item.get("short_track_id")}
            counters["input_observations"]+=1
            if (float(item["confidence"])<min_conf or
                float(item["valid_depth_ratio"])<min_depth_ratio or
                int(item["depth_support_pixels"])<min_pixels):
                events.append({**key,"object_uuid":None,"status":"DEFERRED_LOW_QUALITY"})
                counters["DEFERRED_LOW_QUALITY"]+=1
                continue
            o=convert(item,transforms[frame])
            obs.append(o);keys.append(key)
        if not obs: continue
        decisions=mem.process_frame(obs)
        if len(decisions)!=len(obs):raise RuntimeError("Identity engine dropped observations")
        for key,d in zip(keys,decisions):
            events.append({**key,"object_uuid":d.object_id,"status":d.status,
                           "association_score":float(d.confidence)})
            counters[d.status]+=1
    tracks=db.tracks()
    accepted=[x for x in tracks if x.state=="confirmed"]
    candidates=[x for x in tracks if x.state!="confirmed"]
    reviews=mem.review_queue()
    check=db.db.execute("SELECT COUNT(*) FROM observations WHERE map_id=?",(MAP_ID,)).fetchone()[0]
    if len(events)!=len(rows) or check != len(rows)-counters["DEFERRED_LOW_QUALITY"]:
        raise RuntimeError("C0 evidence conservation failure")
    # This is NOT an independent GT count of physical objects.
    summary={
      "experiment":"C0_REAL_TUM_B5_TO_EXISTING_OBJECT_MEMORY_V2",
      "source":SOURCE,"map_id":MAP_ID,
      "original_observations":len(rows),
      "source_rgbd_frames":595,"source_detector_keyframes":119,
      "source_YOLOE":"26x-seg",
      "measured_pose_source":"A6 graph-optimized RTAB-Map no GT",
      "observed_frame_count":len(frames),
      "quality_gates":{"min_conf":min_conf,
          "min_depth_ratio":min_depth_ratio,"min_pixels":min_pixels},
      "counts":dict(counters),
      "durable_tracks_total":len(tracks),
      "confirmed_hypotheses":len(accepted),
      "tentative_hypotheses":len(candidates),
      "identity_review_count":len(reviews),
      "observations_persisted":check,
      "no_image_appearance_embeddings_supplied":True,
      "new_uuid_is_not_verified_physical_ID":True,
      "absence_of_detection_does_not_mark_deleted":True,
      "moved_object_uuid_accuracy":"NOT_TESTED",
      "true_physical_instance_IDF1":"NOT_EVALUATED_TUM_HAS_NO_INSTANCE_GT",
      "independent_scene_relocalization_bridge":"NOT_YET_IMPLEMENTED",
      "nav_ready":False}
    (db_path.parent/"c0_event_audit.jsonl").write_text(
        "".join(json.dumps(e,ensure_ascii=False)+"\n" for e in events))
    (db_path.parent/"c0_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    with (db_path.parent/"c0_object_hypotheses.jsonl").open("w") as handle:
        for t in sorted(tracks,key=lambda x:x.object_id):
            payload={"object_id":t.object_id,"category":t.label,
                "center_map_m":t.center.tolist(),
                "bbox_min_map_m":t.low.tolist(),"bbox_max_map_m":t.high.tolist(),
                "hits":t.hits,"state":t.state,"map_version":t.map_version,
                "last_seen":t.last_seen,"is_verified_physical_ID":False}
            handle.write(json.dumps(payload,ensure_ascii=False)+"\n")
    db.close()
    print("C0_MEMORY_FROM_REAL_DATA",json.dumps(summary,ensure_ascii=False),flush=True)
    return summary

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--observations",type=Path,required=True)
    ap.add_argument("--poses",type=Path,required=True)
    ap.add_argument("--out",type=Path,default=Path("results/c0"))
    ap.add_argument("--min-confidence",type=float,default=.40)
    ap.add_argument("--min-valid-depth",type=float,default=.45)
    ap.add_argument("--min-support-pixels",type=int,default=60)
    args=ap.parse_args()
    rows,poses=read_inputs(args.observations,args.poses)
    replay(rows,poses,args.out/"object_memory.sqlite",
           min_conf=args.min_confidence,min_depth_ratio=args.min_valid_depth,
           min_pixels=args.min_support_pixels)

if __name__=="__main__":main()
