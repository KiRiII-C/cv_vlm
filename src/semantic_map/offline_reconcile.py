"""Conservative second-pass reconciliation of re-entry track fragments.

Never merge co-visible objects. Preserve old IDs as SQLite aliases. This is a
hypothesis-reconciliation stage, not measured physical-ID accuracy: real GT
annotations are required to establish false merges/splits.
"""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np
from .association import overlap_3d
from .storage import MemoryStore

def proposal(a,b,store,min_iou=.78,min_cos=.965,max_center_dist=.15):
    if a.object_id==b.object_id:return None
    if a.map_id!=b.map_id or a.map_version!=b.map_version or a.label!=b.label:return None
    if store.sightings_cooccur(a.object_id,b.object_id):return None
    if a.appearance is None or b.appearance is None:return None
    if a.appearance.shape!=b.appearance.shape:return None
    sim=float(a.appearance@b.appearance)
    iou=overlap_3d(a.low,a.high,b.low,b.high)
    delta=float(np.linalg.norm(a.center-b.center))
    if iou<min_iou or sim<min_cos or delta>max_center_dist:return None
    # High geometrical overlap, appearance and absence of simultaneous visibility.
    return dict(keep=a.object_id if a.hits>=b.hits else b.object_id,
                remove=b.object_id if a.hits>=b.hits else a.object_id,
                category=a.label,overlap_iou=iou,cosine=sim,distance_m=delta,
                reason="high_3d_overlap_and_visual_agreement_no_covision")

def reconcile(path,out_path=None,dry_run=False,max_merges=100):
    mem=MemoryStore(path)
    before=mem.tracks()
    merged=[]
    for _ in range(max_merges):
        tracks=sorted(mem.tracks(),key=lambda t:(-t.hits,t.object_id))
        proposals=[]
        for i,a in enumerate(tracks):
            for b in tracks[i+1:]:
                p=proposal(a,b,mem)
                if p:proposals.append(p)
        if not proposals:break
        proposals.sort(key=lambda x:(-x["overlap_iou"],-x["cosine"],x["distance_m"]))
        p=proposals[0]
        if dry_run:
            merged.append(dict(p,operation="PROPOSED_NOT_APPLIED"))
            break
        try:
            mem.merge(p["keep"],p["remove"],p["reason"],time.time())
            merged.append(dict(p,operation="MERGED_WITH_ALIAS"))
        except ValueError:
            # A new co-visibility contradiction invalidates this merger.
            merged.append(dict(p,operation="REJECTED_CONFLICT"))
            break
    after=mem.tracks()
    result={"algorithm":"strict_pairwise_multiview_alias_reconciliation",
            "pre_track_count":len(before),"post_track_count":len(after),
            "reconciliations":merged,"dry_run":dry_run,
            "hard_guards":["same_category","same_map_version","no_simultaneous_visibility",
                           "3d_iou>=0.78","appearance_cosine>=0.965","centroid_distance<=0.15m"],
            "real_gt_idf1":"NOT_AVAILABLE",
            "false_merge_free_guarantee":False}
    output=Path(out_path) if out_path else Path(path).parent/"reconciliation.json"
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print("RECONCILIATION",json.dumps({k:v for k,v in result.items() if k!="reconciliations"}),flush=True)
    mem.close()
    return result

if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("--db",default="results/tum_real/semantic_memory.sqlite")
    a.add_argument("--out",default=None)
    a.add_argument("--dry-run",action="store_true")
    args=a.parse_args()
    reconcile(args.db,args.out,args.dry_run)
