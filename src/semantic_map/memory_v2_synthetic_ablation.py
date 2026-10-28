"""Reproducible synthetic stress tests (NOT real-data or physical GT benchmark).
Every memory variant consumes identical observations and their GT identities
are disclosed only to the evaluation function, not to the mapper.
"""
from __future__ import annotations
from pathlib import Path
from collections import defaultdict
import argparse,json,tempfile
import numpy as np
from scipy.optimize import linear_sum_assignment
from .storage import MemoryStore
from .models import Observation
from .engine import SemanticMapper
from .object_memory_v2 import ObjectMemoryV2


def sequence(seed, count=14, frames=75, no_color=False):
    rng=np.random.default_rng(seed)
    categories=['chair','chair','chair','table','sofa','cabinet','laptop']
    xy=rng.uniform(-.10,.10,(count,3))
    for i in range(count):
        xy[i]+=np.array([int(i%5)*.95,int(i//5)*.95,1.2])
    feature=rng.normal(0,1,(count,12))
    feature=feature/np.linalg.norm(feature,axis=1,keepdims=True)
    events=[];gt_count=0
    for t in range(frames):
        observations=[];gt=[]
        # Exercise partial reentries: same object disappears for 10-30 frames.
        active=((t//10)%3)
        for i in range(count):
            if (i%3)==active or rng.random()<.16:continue
            if rng.random()<.06:continue
            gt_count+=1
            ctr=xy[i]+rng.normal(0,.042,3)
            box=rng.uniform(.14,.22)
            z=feature[i]+rng.normal(0,.08,12)
            if rng.random()<.08:z=rng.normal(0,1,12)
            o=Observation(source='synthetic',frame_id=str(t),detection_id=f'det_{i}',
               label=categories[i%len(categories)],center=ctr,
               low=ctr-box,high=ctr+box,appearance=None if no_color else z,
               timestamp=float(t),confidence=.90 if rng.random()>.10 else .18,
               map_id='synthetic_GT_not_association')
            observations.append(o);gt.append(i)
        events.append((observations,gt))
    return events,gt_count


def score_ids(pairs,denominator):
    known=sorted({gt for gt,pred in pairs if gt is not None})
    predicted=sorted({pred for gt,pred in pairs if pred is not None})
    matrix=np.zeros((len(known),len(predicted)),int)
    ia={v:i for i,v in enumerate(known)};ib={v:i for i,v in enumerate(predicted)}
    for gt,pred in pairs:
        if gt is not None and pred is not None:matrix[ia[gt],ib[pred]]+=1
    if matrix.size:
        a,b=linear_sum_assignment(-matrix)
        matches=sum(matrix[i,j] for i,j in zip(a,b))
    else:matches=0
    pred_count=sum(p is not None for _,p in pairs)
    fpid=max(0,pred_count-matches);fnid=max(0,denominator-matches)
    ids_by_gt=defaultdict(set);gts_by_pred=defaultdict(set)
    for gt,pred in pairs:
        if gt is not None and pred is not None:
            ids_by_gt[gt].add(pred);gts_by_pred[pred].add(gt)
    return {'idf1':round(2*int(matches)/max(1,2*int(matches)+int(fpid)+int(fnid)),6),
            'idtp':int(matches),'idfp':int(fpid),'idfn':int(fnid),
            'false_splits':sum(len(x)>1 for x in ids_by_gt.values()),
            'false_merges':sum(len(x)>1 for x in gts_by_pred.values()),
            'track_count':len(predicted)}


def run(seed,mode='legacy',no_color=False):
    events,total=sequence(seed,no_color=no_color)
    with tempfile.TemporaryDirectory() as td:
        store=MemoryStore(Path(td)/'memory.sqlite')
        mapper=(SemanticMapper(store,min_hits=2,association_mode='fused') if mode=='legacy'
                else ObjectMemoryV2(store,min_hits=2,visual_weight=0. if mode=='geometry' else .16))
        hits=[]
        for obs,gt in events:
            decisions=mapper.process_frame(obs)
            hits.extend(zip(gt,[d.object_id for d in decisions]))
        return score_ids(hits,total)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default='results/memory_v2_synthetic_ablation.json')
    args=p.parse_args()
    results={}
    for profile,kwargs in [('appearance',{}),('geometry_only',{'no_color':True})]:
        cases=[]
        for seed in (301,302,303,304):
            base=run(seed,'legacy',**kwargs)
            full=run(seed,'v2',**kwargs)
            geom=run(seed,'geometry',**kwargs)
            cases.append({'seed':seed,'legacy':base,'v2_full':full,'v2_geometry_ablation':geom})
        mean=lambda key:float(np.mean([c[key]['idf1'] for c in cases]))
        results[profile]={'cases':cases,'mean_idf1_legacy':mean('legacy'),
                          'mean_idf1_v2_full':mean('v2_full'),
                          'mean_idf1_v2_geometry':mean('v2_geometry_ablation')}
        print('CONTROLLED_SYNTHETIC_MEMORY_ABLATION',profile,
              {k:results[profile][k] for k in ['mean_idf1_legacy','mean_idf1_v2_full','mean_idf1_v2_geometry']},flush=True)
    data={'provenance':'SYNTHETIC 3D physical identity GT, not real RGB-D',
          'fixed_seeds':[301,302,303,304],
          'identical_input_for_each_variant':True,
          'dataset_limits':'No real segmentation or SLAM, independent evidence-only memory test',
          'results':results}
    path=Path(args.out);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2))
    return data
if __name__=='__main__':main()
