#!/usr/bin/env python3
"""A8 post-hoc inter-arm consistency, no GT and no shared image reuse.

For each arm, associate native RTAB-Map localization_pose with all published
RGB-D frames by exact ROS stamp; compare map-frame poses at identical source
frame indices across independently restarted full-sequence query runs.
Do not mistake cross-run agreement for ground-truth accuracy.
"""
from __future__ import annotations
import argparse,json,sqlite3
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

MODES=("first_node","saved_last","wrong_8m")

def read_accepted_poses(root):
    sent=json.loads((root/"full_desk2.publish.json").read_text())
    stamp_to_idx={int(x["stamp_ns"]):int(x["idx"]) for x in sent}
    poses={}
    for row in (root/"localization_pose.jsonl").read_text().splitlines():
        if not row.strip():continue
        pose=json.loads(row)
        k=int(pose["stamp_ns"])
        if k not in stamp_to_idx:
            raise RuntimeError("Localization pose has no published source frame")
        idx=stamp_to_idx[k]
        if idx in poses:
            raise RuntimeError("Multiple accepted poses for the same source frame")
        poses[idx]=pose
    return poses,len(sent)

def quantify(poses_a,poses_b):
    common=sorted(set(poses_a)&set(poses_b))
    trans=[];angles=[]
    for idx in common:
        a=poses_a[idx];b=poses_b[idx]
        trans.append(float(np.linalg.norm(np.asarray(a["xyz"])-np.asarray(b["xyz"]))))
        qa=Rotation.from_quat(a["quaternion"])
        qb=Rotation.from_quat(b["quaternion"])
        angles.append(float(np.degrees((qa.inv()*qb).magnitude())))
    if not common:return {"n":0}
    return {
      "n":len(common),
      "common_source_frame_indices":common,
      "translation_mean_m":float(np.mean(trans)),
      "translation_rmse_m":float(np.sqrt(np.mean(np.square(trans)))),
      "translation_p95_m":float(np.percentile(trans,95)),
      "rotation_median_deg":float(np.median(angles)),
      "rotation_p95_deg":float(np.percentile(angles,95)),
      "same_saved_map_coordinate_system":True,
      "not_an_independent_gt_accuracy_measurement":True}

def evaluate(root,map_db):
    rows={}
    for mode in MODES:
        rows[mode],count=read_accepted_poses(root/mode)
        if count<600:
            raise RuntimeError("A8 is not full-dataset")
    with sqlite3.connect(f"file:{map_db}?mode=ro",uri=True) as db:
        records=db.execute(
            "SELECT length(opt_last_localization),length(opt_poses),length(opt_ids) FROM Admin"
        ).fetchall()
        n=db.execute("SELECT COUNT(*) FROM Node").fetchone()[0]
    if len(records)!=1 or n<10:raise RuntimeError("Unexpected map database format")
    output={
      "experiment":"A8_RELOCALIZATION_INTER_ARM_AGREEMENT",
      "database":str(map_db),
      "map_node_count":n,
      "Admin_opt_last_localization_blob_bytes":records[0][0],
      "Admin_opt_poses_blob_bytes":records[0][1],
      "Admin_opt_ids_blob_bytes":records[0][2],
      "no_saved_last_pose_in_this_database":records[0][0] is None,
      "accepted_pose_counts":{k:len(v) for k,v in rows.items()},
      "first_node_vs_no_saved":quantify(rows["first_node"],rows["saved_last"]),
      "first_node_vs_wrong_8m":quantify(rows["first_node"],rows["wrong_8m"]),
      "no_saved_vs_wrong_8m":quantify(rows["saved_last"],rows["wrong_8m"]),
      "interpretation":"Cross-arm same-map-frame agreement ONLY, not independent metric ground-truth",
      "all_query_images_are_independent_of_original_mapping_recording":True,
      "true_uniform_prior_over_all_map_nodes":"NOT_VALIDATED"}
    dest=root/"a8_pose_agreement.json"
    dest.write_text(json.dumps(output,ensure_ascii=False,indent=2)+"\n")
    print("A8_AGREEMENT",json.dumps({k:output[k] for k in
         ("first_node_vs_no_saved","first_node_vs_wrong_8m","no_saved_vs_wrong_8m")},
         ensure_ascii=False),flush=True)
    return output

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,default=Path("results/a8"))
    ap.add_argument("--map-db",type=Path,default=Path("results/a8_map/rtabmap.db"))
    args=ap.parse_args()
    evaluate(args.root,args.map_db)
