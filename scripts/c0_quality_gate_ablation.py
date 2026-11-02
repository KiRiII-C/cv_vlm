#!/usr/bin/env python3
"""C0-A: pre-defined confidence gate ablation on the SAME 871 real B5 observations.

Important: not an accuracy sweep. Physical instance GT is not available.
Never choose a gate by the number of UUIDs or by number of confirmations.
"""
from __future__ import annotations
import argparse,importlib.util,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--observations",type=Path,required=True)
    p.add_argument("--poses",type=Path,required=True)
    p.add_argument("--out",type=Path,default=Path("results/c0_ablation"))
    args=p.parse_args()
    source=Path(__file__).with_name("c0_replay_real_b5_memory.py")
    spec=importlib.util.spec_from_file_location("c0_original",source)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rows,poses=mod.read_inputs(args.observations,args.poses)
    thresholds=(.25,.40,.60)
    outcomes={}
    ids={}
    for gate in thresholds:
        name=f"confidence_{gate:.2f}"
        path=args.out/name
        report=mod.replay(rows,poses,path/"object_memory.sqlite",
                          min_conf=gate,min_depth_ratio=.45,min_pixels=60)
        tracks=[json.loads(x) for x in (path/"c0_object_hypotheses.jsonl").read_text().splitlines()]
        ids[name]={x["object_id"] for x in tracks if x["state"]=="confirmed"}
        outcomes[name]={
            "confidence_threshold":gate,
            "input_observations":report["original_observations"],
            "deferred_quality":report["counts"].get("DEFERRED_LOW_QUALITY",0),
            "observations_recorded":report["observations_persisted"],
            "confirmed_hypotheses":report["confirmed_hypotheses"],
            "tentative_hypotheses":report["tentative_hypotheses"],
            "ambiguous_or_deferred_events":report["identity_review_count"],
            "not_verified_physical_instance_count":True,
            "not_physical_GT_accuracy_measure":True}
    mid=ids["confidence_0.40"]
    for key,subset in ids.items():
        outcomes[key]["baseline_confirmed_ID_key_jaccard"]=len(mid&subset)/max(1,len(mid|subset))
    recorded=[outcomes[f"confidence_{g:.2f}"]["observations_recorded"] for g in thresholds]
    if not (recorded[0]>=recorded[1]>=recorded[2]):
        raise RuntimeError("Confidence quality gate input acceptance must be monotone")
    summary={
       "experiment":"C0_REAL_B5_CONFIDENCE_GATE_PREDEFINED_ABLATION",
       "source":"FULL_871_YOLOE26X_B5_OBSERVATIONS",
       "real_frames":595,"detector_keyframes":119,
       "all_input_observations_accounted_in_every_arm":True,
       "GT_instance_ID_available":False,
       "thresholds_predefined":[.25,.40,.60],
       "outcomes":outcomes,
       "warning":"Counts and UUID-key Jaccard are NOT identity accuracy; no performance-based tuning permitted."}
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/"c0_quality_ablation.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    print("C0_QUALITY_SWEEP",json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=="__main__":main()
