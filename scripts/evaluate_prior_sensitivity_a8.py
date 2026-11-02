#!/usr/bin/env python3
"""A8 analyze complete independent RGB-D relocalization under controlled priors.

No TUM GT or synthetic RGBD observations in the localization pipeline.
Do NOT mistake /odom or pre-initialized map->odom for a relocalization event.
A failed matching arm is a VALID negative result if every input frame was
processed and the ROS nodes were functioning.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3

MODES=("first_node","saved_last","wrong_8m")

def read_jsonl(path):
    if not path.exists():
        raise RuntimeError("Missing monitor file: "+str(path))
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def nearest_transform(rows, ns):
    if not rows:return None
    i=min(range(len(rows)),key=lambda k:abs(int(rows[k]["stamp_ns"])-int(ns)))
    return rows[i]

def evaluate(root:Path, mapdir:Path):
    mapping=json.loads((mapdir/"full_sequence.json").read_text())
    manifest=json.loads((root/"query_manifest.json").read_text())
    if not (mapping["published_frames"]==mapping["odominfo_ack_frames"]==595 and
            manifest["source_dataset"]=="TUM_freiburg1_desk2_INDEPENDENT_QUERY" and
            manifest["frame_stride"]==1 and manifest["max_frames"] is None and
            manifest["synchronized_rgbd_pairs"]>=600):
        raise RuntimeError("A8 must use whole verified distinct TUM sessions")
    with sqlite3.connect(mapdir/"rtabmap.db") as conn:
        mapids={int(r[0]) for r in conn.execute("SELECT id FROM Node WHERE id>0")}
    if len(mapids)<8:raise RuntimeError("A8 map has too few nodes")
    original_hash=hashlib.sha256((mapdir/"rtabmap.db").read_bytes()).hexdigest()
    cases={}
    for mode in MODES:
        folder=root/mode
        report=json.loads((folder/"full_desk2.json").read_text())
        sent=json.loads((folder/"full_desk2.publish.json").read_text())
        info=read_jsonl(folder/"info.jsonl")
        poses=read_jsonl(folder/"localization_pose.jsonl")
        tf=read_jsonl(folder/"map_odom_tf.jsonl")
        observations=json.loads((folder/"full_desk2.odominfo.json").read_text())
        evidence=(folder/"initialization_evidence.txt").read_text()
        hashes=(folder/"input_db_sha256.txt").read_text().splitlines()
        # Compare preserved original mapping copy BEFORE each arm runs.
        if len(hashes)<2 or original_hash not in hashes[0] or original_hash not in hashes[1]:
            raise RuntimeError("A8 map copies are not the same exact source")
        if (report["published_frames"]!=manifest["synchronized_rgbd_pairs"] or
            report["odominfo_ack_frames"]!=len(sent) or
            report["published_frames"]!=len(sent) or
            len(observations)!=len(sent)):
            raise RuntimeError("A8 arm did not process entire query: "+mode)
        if mode=="wrong_8m" and "Setting initial pose" not in evidence:
            raise RuntimeError("Explicit wrong prior injection not confirmed")
        if mode=="first_node" and "last localization pose is ignored" not in evidence:
            raise RuntimeError("First-node prior setting not confirmed")
        if mode=="saved_last" and "last localization pose is ignored" in evidence:
            raise RuntimeError("Saved-last arm unexpectedly ignored saved pose")
        stamps={int(row["stamp_ns"]):int(row["idx"]) for row in sent}
        hits=[]
        for inf in info:
            g=int(inf["loop_closure_id"])
            pr=int(inf["proximity_detection_id"])
            if g<=0 and pr<=0:continue
            if g>0 and g not in mapids:raise RuntimeError("Global event not in OLD map")
            if pr>0 and pr not in mapids:raise RuntimeError("Proximity event not in OLD map")
            t=int(inf["stamp_ns"])
            if t not in stamps:raise RuntimeError("Hit has no source RGB-D frame")
            hits.append({"frame":stamps[t],"stamp_ns":t,"global":g,"proximity":pr})
        idx=min((h["frame"] for h in hits),default=None)
        match_pose={int(x["stamp_ns"]) for x in poses}
        with_pose=sum(h["stamp_ns"] in match_pose for h in hits)
        lost=sum(bool(x["lost"]) for x in observations)
        first_tf=nearest_transform(tf,sent[0]["stamp_ns"])
        first_hit_tf=nearest_transform(tf,hits[0]["stamp_ns"]) if hits else None
        cases[mode]={
           "query_frames":len(sent),
           "processed_confirmations":report["odominfo_ack_frames"],
           "odominfo_lost_frames":lost,
           "first_accepted_frame":idx,
           "first_accepted_source_time_s":float(sent[idx]["source_tum_timestamp"]) if idx is not None else None,
           "accepted_event_count_unique":len(hits),
           "accepted_global":sum(h["global"]>0 for h in hits),
           "accepted_proximity":sum(h["proximity"]>0 for h in hits),
           "accepted_events_with_pose_same_stamp":with_pose,
           "localization_pose_messages":len(poses),
           "map_odom_tf_samples":len(tf),
           "first_map_odom_translation_m":first_tf["xyz"] if first_tf else None,
           "at_first_hit_map_odom_translation_m":first_hit_tf["xyz"] if first_hit_tf else None,
           "prior":("forced_first_map_node" if mode=="first_node" else
                    "saved_last_map_pose" if mode=="saved_last" else
                    "explicit_wrong_8m_180deg"),
           "status":"CROSS_RECORDING_MATCHED" if hits and poses and tf else "NO_RELOCALIZATION",
           "mean_odometry_ATE_m_separate_trajectory_not_global_map_error":report.get("ATE_RMSE_m"),
        }
    result={
       "experiment":"A8_CROSS_SEQUENCE_PRIOR_SENSITIVITY",
       "map_source":"TUM_freiburg1_desk",
       "query_source":"TUM_freiburg1_desk2",
       "map_frames":mapping["published_frames"],
       "query_frames_each":manifest["synchronized_rgbd_pairs"],
       "prior_free_relocalization":"NOT_TESTED_ALL_THREE_CONDITIONS_HAVE_EXPLICIT_OR_SAVED_PRIOR",
       "global_metric_relocalization_error":"NOT_EVALUATED",
       "source_image_overlap_between_map_and_query":False,
       "all_rgbd_frames_used":True,
       "saved_map_sha256_before_each_arm":original_hash,
       "cases":cases,
       "result":"COMPLETE_CONTROLLED_PRIOR_EXPERIMENT"
    }
    out=root/"a8_prior_sensitivity_report.json"
    out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print("A8_PRIOR_RESULTS",json.dumps(result,ensure_ascii=False),flush=True)
    return result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,default=Path("results/a8"))
    ap.add_argument("--map-dir",type=Path,default=Path("results/a8_map"))
    args=ap.parse_args()
    evaluate(args.root,args.map_dir)

if __name__=="__main__":main()
