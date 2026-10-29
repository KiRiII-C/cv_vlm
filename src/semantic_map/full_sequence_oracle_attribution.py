"""All-100-frame physical-ID error counterfactuals on the same YOLOE26x frames.

Physical GT never enters unmodified raw detector or ObjectMemoryV2. Explicit
oracle variants alter observation data for causal diagnostic intervention:
correct labels, full GT-mask-derived 3D observation, missing-GT insertions.
No such oracle variant is deployable. Pretrained model inference is not rerun.
"""
from __future__ import annotations
import argparse,copy,gzip,json
from pathlib import Path
from .memory_v2_replay import replay

VARIANTS=("raw","oracle_class","oracle_mask_geometry",
          "oracle_class_and_geometry","oracle_missing_detections",
          "oracle_all","no_visual_embedding")
FIELDS=("center","low","high","appearance")

def read(path):
    with gzip.open(path,"rt",encoding="utf8") as f:return json.load(f)

def prepare(raw,oracle,variant):
    if variant=="oracle_all":return copy.deepcopy(oracle),{"corrected":0,"added":0,"oracle_3d_missing":0}
    if variant not in VARIANTS:raise ValueError("Unknown intervention")
    cache=copy.deepcopy(raw)
    by_frame={f["frame"]:f for f in oracle["frames"]}
    ncorrected=nadded=nmiss=0
    for f in cache["frames"]:
        oracle_frame=by_frame[f["frame"]]
        ref={int(g):o for g,o in zip(oracle_frame["physical_gt"],oracle_frame["observations"])
             if g is not None}
        eligible=set(int(g) for g in f["target_visible"])
        present={int(g) for g in f["physical_gt"] if g is not None and g in eligible}
        for gt,obs in zip(f["physical_gt"],f["observations"]):
            if gt is None or int(gt) not in eligible:continue
            gt_ref=ref.get(int(gt))
            if gt_ref is None:
                nmiss+=1;continue
            if variant in ("oracle_class","oracle_class_and_geometry"):
                obs["label"]=gt_ref["label"]
                ncorrected+=1
            if variant in ("oracle_mask_geometry","oracle_class_and_geometry"):
                for key in FIELDS:obs[key]=copy.deepcopy(gt_ref[key])
                ncorrected+=1
        if variant=="oracle_missing_detections":
            for uid in sorted(eligible-present):
                gt_ref=ref.get(uid)
                if gt_ref is None:
                    nmiss+=1;continue
                o=copy.deepcopy(gt_ref)
                o["detection_id"]="corrective_oracle_missing_"+str(uid)
                f["observations"].append(o)
                f["physical_gt"].append(uid)
                nadded+=1
    cache["intervention"]=variant
    return cache,{"corrected":ncorrected,"added":nadded,"oracle_3d_missing":nmiss}

def run(raw_path,oracle_path,out_path):
    raw=read(raw_path);oracle=read(oracle_path)
    if raw["detector"]!="yoloe-26x-seg.pt" or len(raw["frames"])!=100:
        raise ValueError("Need full original YOLOE26x 100-frame predictions")
    assert [x["frame"] for x in raw["frames"]]==[x["frame"] for x in oracle["frames"]]==list(range(100))
    result={"scene":raw["scene"],"detector":"yoloe-26x-seg.pt",
            "scope":"100 consecutive frames; same frozen predictions; fixed GT world XYZ",
            "GT_oracle_3d_used_only_in_named_diagnostic_interventions":True,
            "short_term_temporal_tracker_included":False,
            "variants":{}}
    for variant in VARIANTS:
        modified,changed=prepare(raw,oracle,variant)
        out=replay(modified,"v2",remove_visual=(variant=="no_visual_embedding"))
        a=out["target_gt"]
        result["variants"][variant]={"target_gt":a,"full_gt":out["all_gt"],
            "input_detections":out["input_detections"],"counts":changed}
        print("FULL_SEQUENCE_OBSERVATION_INTERVENTION",raw["scene"],variant,
              "IDF1",a["IDF1"],"false_merge",a["false_merge_predicted_uuid"],
              "false_split",a["false_split_physical_gt"],
              "added",changed["added"],flush=True)
    baseline={"ai_002_001":.39074960127591707,
              "ai_005_001":.07178664527772208,
              "ai_017_001":.44888475836431224}.get(raw["scene"])
    if baseline is None or abs(result["variants"]["raw"]["target_gt"]["IDF1"]-baseline)>1e-9:
        raise AssertionError("Raw exact-baseline regression mismatch")
    out=Path(out_path);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print("FULL_SEQUENCE_ORACLE_CAUSAL_DIAGNOSTICS_DONE",raw["scene"],flush=True)
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--raw",required=True)
    p.add_argument("--oracle",required=True)
    p.add_argument("--out",required=True)
    a=p.parse_args()
    run(a.raw,a.oracle,a.out)
