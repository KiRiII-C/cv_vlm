#!/usr/bin/env python3
"""A7 post-hoc analysis of cross-sequence accepted old-map node matches.

Must not alter localization. The independent query, its entire frame manifest
and the old map SQLite database are the only inputs. No TUM GT pose is used.
"""
from __future__ import annotations
import argparse, json, sqlite3
from pathlib import Path
import numpy as np

def load_jsonl(p):
    if not p.exists():return []
    return [json.loads(row) for row in p.read_text().splitlines() if row.strip()]

def evaluate(mapping,query):
    old=json.loads((mapping/"full_sequence.json").read_text())
    q=json.loads((query/"full_desk2.json").read_text())
    mf=json.loads((query/"independent_query_manifest.json").read_text())
    sent=json.loads((query/"full_desk2.publish.json").read_text())
    info=load_jsonl(query/"info.jsonl")
    poses=load_jsonl(query/"localization_pose.jsonl")
    tf=load_jsonl(query/"map_odom_tf.jsonl")
    odominfo=json.loads((query/"full_desk2.odominfo.json").read_text())
    if mf["source_dataset"]!="TUM_freiburg1_desk2_INDEPENDENT_QUERY":
        raise RuntimeError("Query is not the independent desk2 recording")
    if (old["published_frames"]!=old["synchronized_rgbd_pairs"] or
        q["published_frames"]!=len(sent) or
        q["published_frames"]!=mf["synchronized_rgbd_pairs"] or
        q["odominfo_ack_frames"]!=q["published_frames"] or
        mf["frame_stride"]!=1 or mf["max_frames"] is not None):
        raise RuntimeError("A7 input incomplete; do not claim all-frame result")
    with sqlite3.connect(mapping/"rtabmap.db") as db:
        old_ids={int(v[0]) for v in db.execute("SELECT id FROM Node WHERE id>0")}
    stamps={int(x["stamp_ns"]):int(x["idx"]) for x in sent}
    hits=[]
    missing_event_stamps=[]
    for e in info:
        g=int(e["loop_closure_id"])
        prox=int(e["proximity_detection_id"])
        if g<=0 and prox<=0:continue
        if g>0 and g not in old_ids:raise RuntimeError("Global hit to node not in old map")
        if prox>0 and prox not in old_ids:raise RuntimeError("Proximity hit not in old map")
        t=int(e["stamp_ns"])
        idx=stamps.get(t)
        if idx is None:missing_event_stamps.append(t)
        hits.append({"query_frame":idx,"global":g,"proximity":prox,
                     "timestamp_ns":t})
    localization_stamps={int(x["stamp_ns"]) for x in poses}
    matched_pose_count=sum(int(h["timestamp_ns"] in localization_stamps) for h in hits)
    first=min((h["query_frame"] for h in hits if h["query_frame"] is not None),
              default=None)
    lost_by_stamp={int(x["stamp_ns"]):bool(x["lost"]) for x in odominfo}
    lost_indices=[int(x["idx"]) for x in sent if lost_by_stamp.get(int(x["stamp_ns"]))]
    result={
      "experiment":"A7_CROSS_SESSION_RTABMAP_LOCALE",
      "map_dataset":"TUM_freiburg1_desk",
      "query_dataset":"TUM_freiburg1_desk2",
      "map_processed_frames":old["published_frames"],
      "query_processed_frames":q["published_frames"],
      "query_acked_frames":q["odominfo_ack_frames"],
      "query_unpaired_rgb":mf["source_rgb_frames"]-len(sent),
      "old_map_node_count":len(old_ids),
      "rtabmap_info_message_count":len(info),
      "accepted_event_count":len(hits),
      "global_accepted_count":sum(h["global"]>0 for h in hits),
      "proximity_accepted_count":sum(h["proximity"]>0 for h in hits),
      "first_hit_frame":first,
      "accepted_events_with_identical_stamp_localization_pose":matched_pose_count,
      "loc_pose_messages":len(poses),
      "map_odom_tf_messages":len(tf),
      "tracking_lost_frames":len(lost_indices),
      "first_five_lost_frame_indices":lost_indices[:5],
      "nonmatching_acceptance_event_timestamps":len(missing_event_stamps),
      "all_frame_stride":1,"no_external_initial_pose":True,
      "no_original_images_reused":True,
      "prior_free_localization":"NOT_PROVEN_START_AT_ORIGIN_IS_AN_ASSUMPTION",
      "cross_sequence_pose_GT_error":"NOT_EVALUATED_FRAME_ALIGNMENT_NOT_VERIFIED",
      "status":"PASS_CROSS_SEQUENCE_FUNCTIONAL_ONLY" if
        hits and first is not None and poses and tf and not missing_event_stamps else
        "FAIL_NO_VERIFIED_RELOCALIZATION"
    }
    query.mkdir(exist_ok=True,parents=True)
    (query/"a7_detailed_report.json").write_text(json.dumps(result,indent=2)+"\n")
    (query/"a7_accepted_events.json").write_text(json.dumps(hits,indent=2)+"\n")
    print("A7_DETAILED",json.dumps(result),flush=True)
    return result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--map",dest="mapping",type=Path,default=Path("results/a7_map"))
    ap.add_argument("--query",type=Path,default=Path("results/a7"))
    args=ap.parse_args()
    result=evaluate(args.mapping,args.query)
    if not result["status"].startswith("PASS"):
        raise SystemExit(1)

if __name__=="__main__":main()
