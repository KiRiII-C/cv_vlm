"""Controlled physical-ID benchmark replay on FIXED detector input observations.

The replay input stores physical GT only for scoring; it is never passed to a
memory engine. This prevents detector randomness and repeated network inference
from confounding comparisons of alternative memory mechanisms.
"""
from __future__ import annotations
import argparse,gzip,json,tempfile
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment
from .models import Observation
from .storage import MemoryStore
from .engine import SemanticMapper
from .object_memory_v2 import ObjectMemoryV2


def score_identity(items,gt_total):
    true=sorted({int(g) for g,p in items if g is not None})
    pred=sorted({p for g,p in items if p is not None})
    matrix=np.zeros((len(true),len(pred)),dtype=int)
    lookgt={a:i for i,a in enumerate(true)};lookid={b:j for j,b in enumerate(pred)}
    for gt,uid in items:
        if gt is not None and uid is not None:matrix[lookgt[int(gt)],lookid[uid]]+=1
    if matrix.size:
        i,j=linear_sum_assignment(-matrix)
        tp=int(sum(matrix[a,b] for a,b in zip(i,j)))
    else:tp=0
    fp=max(0,int(sum(uid is not None for _,uid in items))-tp)
    fn=max(0,int(gt_total)-tp)
    bygt=defaultdict(set);byid=defaultdict(set)
    for gt,uid in items:
        if gt is not None and uid is not None:
            bygt[gt].add(uid);byid[uid].add(gt)
    return {'IDF1':float(2*tp/max(1,2*tp+fp+fn)),'IDTP':tp,'IDFP':fp,'IDFN':fn,
            'false_split_physical_gt':sum(len(s)>1 for s in bygt.values()),
            'false_merge_predicted_uuid':sum(len(s)>1 for s in byid.values()),
            'gt_observations':int(gt_total),'predicted_tracks':len(pred)}


def replay(cached,variant,*,min_score=.65,visual_weight=.16,
           candidate_floor=.40,remove_visual=False):
    frames=cached['frames']; oracle=cached['mode']=='oracle_masks'
    with tempfile.TemporaryDirectory(prefix='object_memory_replay_') as td:
        db=MemoryStore(Path(td)/'memory.sqlite')
        if variant=='legacy':engine=SemanticMapper(db,min_hits=2,association_mode='fused')
        elif variant=='v2':engine=ObjectMemoryV2(db,min_hits=2,min_score=min_score,
                   visual_weight=visual_weight,candidate_floor=candidate_floor)
        else:raise ValueError('unknown variant')
        items=[];target=[];target_count=0;total_gt=0;timeline=[]
        for frame in frames:
            observations=[]
            for row in frame['observations']:
                v=dict(row)
                if remove_visual:v['appearance']=None
                observations.append(Observation(**v))
            gt=frame['physical_gt'];visible=set(frame['target_visible'])
            assert len(observations)==len(gt)
            decisions=engine.process_frame(observations)
            assert len(decisions)==len(gt)
            total_gt+=len(frame['all_gt_visible'])
            target_count+=len(visible)
            for physical,d in zip(gt,decisions):
                items.append((physical,d.object_id))
                if oracle:
                    if physical in visible:target.append((physical,d.object_id))
                else:target.append((physical if physical in visible else None,d.object_id))
            timeline.append({'frame':frame['frame'],
                             'uuids':[d.object_id for d in decisions],
                             'statuses':[d.status for d in decisions],
                             'physical_gt_scoring_only':gt})
        result={'variant':variant,'params':{'min_score':min_score,
                'visual_weight':visual_weight,'candidate_floor':candidate_floor,
                'remove_visual':remove_visual},
                'scene':cached['scene'],'source':cached['mode'],
                'all_gt':score_identity(items,total_gt),
                'target_gt':score_identity(target,target_count),
                'physical_target_count':len(set(g for f in frames for g in f['target_visible'])),
                'input_detections':len(items),'frames':len(frames),
                'gt_as_model_input':False,
                'status':'MEASURED_NOT_AUTOMATICALLY_ACCEPTED',
                'timeline':timeline}
        db.close()
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cache',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    with gzip.open(args.cache,'rt',encoding='utf-8') as f:cache=json.load(f)
    assert cache['gt_labels_are_scoring_only'] is True
    variants=[('legacy',{}),('v2_baseline',{}),
      ('v2_geometry_only',{'remove_visual':True}),
      ('v2_no_visual_weight',{'visual_weight':0.}),
      ('v2_strict_070',{'min_score':.70}),
      ('v2_strict_075',{'min_score':.75}),
      ('v2_visual_025',{'visual_weight':.25}),
      ('v2_defer_050',{'candidate_floor':.50})]
    results={}
    for name,kw in variants:
        result=replay(cache,'legacy' if name=='legacy' else 'v2',**kw)
        results[name]={k:v for k,v in result.items() if k!='timeline'}
        print('MEMORY_FIXED_INPUT_ABLATION',name,cache['scene'],cache['mode'],
              'target_idf1',result['target_gt']['IDF1'],
              'false_merges',result['target_gt']['false_merge_predicted_uuid'],
              'false_splits',result['target_gt']['false_split_physical_gt'],flush=True)
    report={'dataset':'Hypersim physical GT, 2D mask and GT XYZ extracted ONCE',
            'scene':cache['scene'],'mode':cache['mode'],
            'fixed_input':True,'gt_labels_input_to_model':False,
            'tuning_note':'Ablations are diagnostic; selecting a parameter on this scene is not independent generalization evidence.',
            'variants':results}
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    return report
if __name__=='__main__':main()
