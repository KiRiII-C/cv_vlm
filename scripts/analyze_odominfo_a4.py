#!/usr/bin/env python3
"""A4 diagnostics: distinguish RTAB-Map tracking decisions from numeric odom filter.

Uses recorded, real, frame-aligned OdomInfoLite/odom observations. Does not
label a pose 'correct' based on covariance/inliers alone. GT not accessed.
"""
from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path
import numpy as np


def summarize(output):
    output=Path(output)
    base=output.with_suffix("")
    report=json.loads(output.read_text())
    info=json.loads(base.with_suffix(".odominfo.json").read_text())
    odom=json.loads(base.with_suffix(".odom.json").read_text())
    published=json.loads(base.with_suffix(".publish.json").read_text())
    by_info={int(x["stamp_ns"]):x for x in info}
    by_odom={int(x["stamp_ns"]):x for x in odom}
    states=Counter()
    joint=[]
    for item in published:
        key=int(item["stamp_ns"])
        inf=by_info.get(key)
        od=by_odom.get(key)
        if inf is None:
            state="NO_ODOMINFO"
        elif inf["lost"]:
            state="RTABMAP_REPORTED_LOST"
        elif od is None:
            state="RTABMAP_NOT_LOST_BUT_NO_ODOM_MESSAGE"
        elif not od["numerically_valid"]:
            state="RTABMAP_NOT_LOST_NUMERICAL_FILTER_REJECTED"
        else:
            state="RTABMAP_NOT_LOST_TECHNICALLY_VALID"
        states[state]+=1
        joint.append({"frame_index":item["idx"],
                      "source_tum_timestamp":item["source_tum_timestamp"],
                      "status":state,
                      "odominfo_inliers":inf["inliers"] if inf else None,
                      "odominfo_matches":inf["matches"] if inf else None,
                      "rtabmap_lost":inf["lost"] if inf else None,
                      "numerical_filter_passed":od["numerically_valid"] if od else None})

    if sum(states.values())!=report["published_frames"]:
        raise RuntimeError("Frame accounting inconsistent")

    sequences=[]
    previous_status=None;current=None
    for row in joint:
        if row["status"]!=previous_status:
            if current:
                sequences.append(current)
            current={"status":row["status"],"start":row["frame_index"],
                     "end":row["frame_index"],"count":1}
            previous_status=row["status"]
        else:
            current["end"]=row["frame_index"]
            current["count"]+=1
    if current:
        sequences.append(current)
    def quantiles(rows,name):
        a=np.asarray([x[name] for x in rows if x.get(name) is not None],dtype=float)
        if len(a)==0:return {"n":0}
        return {"n":int(len(a)),
                "min":float(np.min(a)), "median":float(np.median(a)),
                "p10":float(np.percentile(a,10)),
                "p90":float(np.percentile(a,90)), "max":float(np.max(a))}
    result={
        "source_report":str(output),
        "GT_used_as_input":False,
        "inferred_pose_accuracy_from_odominfo":False,
        "all_paired_frames_included":report["published_frames"]==report["synchronized_rgbd_pairs"],
        "frame_count":len(published),
        "frame_states":dict(states),
        "state_segments":sequences,
        "longest_not_lost_and_technical_segment":max((x["count"] for x in sequences
                       if x["status"]=="RTABMAP_NOT_LOST_TECHNICALLY_VALID"),default=0),
        "inliers_when_rtabmap_not_lost":quantiles([x for x in info if not x["lost"]],"inliers"),
        "inliers_when_rtabmap_lost":quantiles([x for x in info if x["lost"]],"inliers"),
        "matches_when_rtabmap_not_lost":quantiles([x for x in info if not x["lost"]],"matches"),
        "matches_when_rtabmap_lost":quantiles([x for x in info if x["lost"]],"matches"),
        "caution":"OdomInfo.lost is RTAB-Map's tracking status, not GT pose accuracy",
    }
    dest=output.parent/"frame_quality_summary.json"
    dest.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    (output.parent/"frame_quality_by_index.json").write_text(json.dumps(joint,indent=2)+"\n")
    print("A4_QUALITY_SUMMARY",json.dumps({k:v for k,v in result.items()
           if k not in ("state_segments",)},ensure_ascii=False),flush=True)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--report",type=Path,default=Path("results/a4/full_sequence.json"))
    args=p.parse_args()
    summarize(args.report)


if __name__=="__main__":
    main()
