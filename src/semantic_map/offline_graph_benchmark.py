"""Reproduce full 100-frame physical ID GT outcomes of class-soft offline graph."""
import argparse,gzip,json
from pathlib import Path
from .offline_instance_graph import OfflineInstanceGraph
from .models import Observation
from .memory_v2_replay import score_identity

def load(path):
    with gzip.open(path,"rt",encoding="utf8") as f:return json.load(f)
def benchmark(cache,algorithm=None):
    algorithm=algorithm or OfflineInstanceGraph()
    frames=[(f["frame"],[Observation(**x) for x in f["observations"]])
            for f in cache["frames"]]
    assigned=algorithm.associate(frames)
    expected=sum(len(f["observations"]) for f in cache["frames"])
    if len(assigned)!=expected:raise AssertionError("Missing assignment")
    obs=[];ptr=0;gt_total=0
    for frame in cache["frames"]:
        visible=set(frame["target_visible"])
        gt_total+=len(visible)
        for g in frame["physical_gt"]:
            obs.append((g if g in visible else None,assigned[ptr].object_uuid))
            ptr+=1
    return {"frames":len(frames),"gt_total":gt_total,
            "raw_observations":expected,
            "target_gt":score_identity(obs,gt_total),
            "GT_as_algorithm_input":False}
def run(raw,oracle,out,*,export_sqlite=False):
    r=load(raw);o=load(oracle)
    assert r["detector"]=="yoloe-26x-seg.pt"
    assert len(r["frames"])==len(o["frames"])==100
    graph=OfflineInstanceGraph(radius_m=.65,edge_threshold=.40,
                                max_spread_m=.80)
    result={"scene":r["scene"],"type":"class_soft_offline_instance_graph",
         "params":dict(radius_m=.65,edge_threshold=.40,max_spread_m=.80),
         "model":"yoloe-26x-seg.pt",
         "GT_world_XYZ_used":True,"SLAM":"NOT_TESTED",
         "raw":benchmark(r,graph),"oracle":benchmark(o,graph)}
    if export_sqlite:
        frames=[(f["frame"],[Observation(**x) for x in f["observations"]])
            for f in r["frames"]]
        result["database"]=graph.export_sqlite(frames,Path(out).with_suffix(".sqlite"))
    output=Path(out);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print("OFFLINE_INSTANCE_GRAPH_FULL_GT",r["scene"],
        "raw_IDF1",result["raw"]["target_gt"]["IDF1"],
        "oracle_IDF1",result["oracle"]["target_gt"]["IDF1"],
        "raw_false_merge",result["raw"]["target_gt"]["false_merge_predicted_uuid"],
        "raw_false_split",result["raw"]["target_gt"]["false_split_physical_gt"],
        flush=True)
    return result
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--raw",required=True);p.add_argument("--oracle",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--export-sqlite",action="store_true")
    args=p.parse_args();run(args.raw,args.oracle,args.out,export_sqlite=args.export_sqlite)
