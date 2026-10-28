"""Unmodified YOLOE-26x detections: 3D physical ReID spatial-gate sensitivity.

This is a *diagnostic parameter grid*, not an independent test or GT-augmented
inference. Changing the radius is a model hyperparameter; GT used only for
the scorer, never for ObjectMemoryV2.
"""
import argparse,gzip,json,tempfile
from pathlib import Path
from .storage import MemoryStore
from .models import Observation
from .object_memory_v2 import ObjectMemoryV2
from .memory_v2_replay import score_identity

def gate_replay(cache,radius,min_score=.65):
    with tempfile.TemporaryDirectory(prefix="reid_gate_") as d:
        store=MemoryStore(Path(d)/"map.sqlite")
        engine=ObjectMemoryV2(store,min_hits=2,min_score=min_score)
        engine.DEFAULT_STATIC_REID_RADIUS=dict(engine.DEFAULT_STATIC_REID_RADIUS)
        engine.DEFAULT_STATIC_REID_RADIUS["chair"]=float(radius)
        matches=[];gt_count=0
        for frame in cache["frames"]:
            observations=[Observation(**row) for row in frame["observations"]]
            dec=engine.process_frame(observations)
            if len(dec)!=len(frame["physical_gt"]):
                raise AssertionError("Observation and GT scoring index mismatch")
            target=set(frame["target_visible"])
            gt_count+=len(target)
            matches.extend((gt if gt in target else None,d.object_id)
                           for gt,d in zip(frame["physical_gt"],dec))
        out=score_identity(matches,gt_count)
        store.close()
    return out

def run(path_a,path_b,output):
    def read(path):
        with gzip.open(path,"rt",encoding="utf8") as f:return json.load(f)
    a=read(path_a);b=read(path_b)
    if not (a["detector"]==b["detector"]=="yoloe-26x-seg.pt"):
        raise ValueError("Not YOLOE-26x raw detection cache")
    if a["mode"]!="predicted_yolo_masks" or b["mode"]!="predicted_yolo_masks":
        raise ValueError("Oracle cache not allowed")
    results=[]
    for r in (.12,.18,.25,.35,.45,.60):
        for score in (.65,.72):
            qa=gate_replay(a,r,score);qb=gate_replay(b,r,score)
            results.append({"chair_radius_m":r,"min_score":score,
               "diverse":qa,"crowded":qb,
               "total_false_merges":qa["false_merge_predicted_uuid"]+
                                     qb["false_merge_predicted_uuid"],
               "mean_target_idf1":(qa["IDF1"]+qb["IDF1"])/2})
            print("YOLOE26X_REID_GATE",r,score,
                  "diverse",qa["IDF1"],"crowded",qb["IDF1"],
                  "false_merges",qa["false_merge_predicted_uuid"],
                  qb["false_merge_predicted_uuid"],flush=True)
    baseline=next(q for q in results if q["chair_radius_m"]==.45 and q["min_score"]==.65)
    if abs(baseline["diverse"]["IDF1"]-.398576512455516)>1e-7 or abs(
        baseline["crowded"]["IDF1"]-.19174434087882822)>1e-7:
        raise AssertionError("ReID replay is not identical to successful YOLOE26x raw factorial; abort comparison")
    result={"model":"yoloe-26x-seg.pt","dataset":"Hypersim ai_002_001 & ai_005_001; same exact detector frames; GT xyz",
        "scope":"Tuned on two previously examined scenes; NO independent unseen generalization",
        "GT_identity_as_model_input":False,
        "baseline":baseline,"variants":results}
    p=Path(output);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--diverse",required=True);p.add_argument("--crowded",required=True)
    p.add_argument("--out",default="results/reid_gate_sweep.json")
    a=p.parse_args();run(a.diverse,a.crowded,a.out)
