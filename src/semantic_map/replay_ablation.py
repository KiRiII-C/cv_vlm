"""Replay **real YOLO/TUM observation evidence** for different association policies.

No physical-ID annotations: output is structural diagnostics, never GT IDF1.
"""
import argparse,gzip,json,tempfile
from pathlib import Path
from collections import Counter
from .models import Observation
from .storage import MemoryStore
from .engine import SemanticMapper
from .association import overlap_3d
import numpy as np

def candidates(store,min_iou=.2,min_similarity=.75,max_dist=.5):
    t=store.tracks();o=[]
    for i,a in enumerate(t):
        for b in t[i+1:]:
            if a.map_id!=b.map_id or a.map_version!=b.map_version or a.label!=b.label:continue
            if store.sightings_cooccur(a.object_id,b.object_id):continue
            dist=float(np.linalg.norm(a.center-b.center));iou=overlap_3d(a.low,a.high,b.low,b.high)
            cos=float(a.appearance@b.appearance) if a.appearance is not None and b.appearance is not None else -1.
            if iou>=min_iou and cos>=min_similarity and dist<=max_dist:
                o.append({'label':a.label,'id1':a.object_id,'id2':b.object_id,
                          'distance_m':round(dist,3),'iou_3d_aabb':round(iou,3),
                          'appearance_similarity':round(cos,3),'manual_review_required':True})
    return o

def run(root='results/tum_real'):
    root=Path(root)
    with gzip.open(root/'observation_evidence.json.gz','rt',encoding='utf-8') as f:data=json.load(f)
    results={}
    for mode in ['geometry','appearance','fused','adaptive_reid']:
        with tempfile.TemporaryDirectory() as tmp:
            store=MemoryStore(Path(tmp)/'memory.sqlite')
            mapper=SemanticMapper(store,min_score=.68,ambiguity_margin=.1,association_mode=mode)
            status=Counter();det_count=0
            for f in data:
                if not f:continue
                obs=[Observation(**o) for o in f];det_count+=len(obs)
                for result in mapper.process_frame(obs):status[result.status]+=1
            tracks=store.tracks()
            proposal=candidates(store)
            results[mode]={'frames':len(data),'detected_observations':det_count,
                'canonical_tracks':len(tracks),'confirmed_tracks':sum(t.state=='confirmed' for t in tracks),
                'unresolved_observations':status['unresolved']+status['ambiguous']+status['unassigned_conflict'],
                'statuses':dict(status),'candidate_duplicate_pairs':len(proposal),
                'proposed_pairs':proposal[:30],
                'warning':'Counts are NOT GT false split/merge; lower track count could hide erroneous merges'}
            print('REAL_ASSOCIATION_ABLATION',mode,{k:v for k,v in results[mode].items() if k!='proposed_pairs'},flush=True)
            store.close()
    (root/'real_association_ablation.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
    return results
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='results/tum_real');a=p.parse_args();run(a.root)
