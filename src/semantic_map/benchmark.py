"""Deterministic synthetic *identity* stress test; no claimed real-world benchmark score.

Stressors: returns after 30 frames, partial occlusion, camera viewpoint noise,
confusable same-class objects, 8-dimensional appearance, and missing features.
"""
from __future__ import annotations
import argparse,json,math,tempfile
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from .models import Observation
from .storage import MemoryStore
from .engine import SemanticMapper

LABELS=['chair','chair','table','lamp','cabinet','chair','plant','table','chair','sofa','chair','lamp']
BASE_XY=np.array([[2.0,2.0],[2.8,2.05],[4.8,3.1],[7.0,2.2],[9,2.0],[9.8,2.1],
                  [2.0,6.0],[4.7,6.4],[5.3,6.4],[8,6.2],[8.9,6.2],[10.5,6.2]])
SIZES={'chair':(.6,.6,.9),'table':(1.2,.7,.85),'lamp':(.2,.2,1.5),
       'cabinet':(1.,.55,1.8),'plant':(.55,.55,1.2),'sofa':(1.6,.8,.8)}
COLORS=['red','blue','brown','yellow','white','green','green','brown','blue','gray','red','yellow']

def synth_frames(seed=42,frames=120):
    rng=np.random.default_rng(seed)
    unique=rng.normal(0,1,(len(LABELS),12));unique=unique/np.linalg.norm(unique,axis=1,keepdims=True)
    records=[];paths=[]
    for t in range(frames):
        # repeated visits to two spatial clusters with full occlusion intervals
        sector=0 if (t//20)%2==0 else 1
        ids=range(0,6) if sector==0 else range(6,12)
        here=[]
        robot=(1.0+0.09*(t%20),1.0+sector*4.5)
        for i in ids:
            if rng.random()<.12:continue
            # deliberate co-visibility of similar objects, viewpoint & sensor jitter
            center=np.r_[BASE_XY[i],.70]
            center+=rng.normal(0,.035,3)
            if rng.random()<.035:center+=rng.normal(0,.07,3)
            sizes=np.asarray(SIZES[LABELS[i]])
            jitter=np.clip(rng.normal(0,.025,3),-.07,.07)
            low=center-(sizes+jitter)/2
            high=center+(sizes+jitter)/2
            embedding=unique[i]+rng.normal(0,.16,12)
            if rng.random()<.08:embedding=None
            # not consistent full RGB-D, this is observation-level sensor simulation
            o=Observation(source='synthetic',frame_id=str(t),detection_id=f'd{i}',
                          label=LABELS[i],center=center,low=low,high=high,
                          appearance=embedding,description=f'{COLORS[i]} {LABELS[i]}',
                          timestamp=float(t),confidence=.95)
            here.append((i,o))
        # Shuffle assignments to avoid giving track ordering an unfair advantage
        rng.shuffle(here);records.append(here);paths.append(robot)
    return records,paths

def evaluate(records,mode,seed=1):
    with tempfile.TemporaryDirectory() as tmp:
        store=MemoryStore(Path(tmp)/'eval.sqlite')
        if mode=='naive':mapper=None
        else:
            variant='fused' if mode in ('fused_no_hold','fused_hold') else mode
            mapper=SemanticMapper(store,association_mode=variant,
                                  min_score=.68,ambiguity_margin=.10 if mode=='fused_hold' else 0)
        contingency={};gt_pred={};pred_gt={};count=0;assigned=0;statuses={};trace=[]
        for frame in records:
            if mapper is None:
                from uuid import uuid4
                results=[dict(object_id='new_'+str(uuid4()),status='new') for _ in frame]
            else:results=mapper.process_frame([o for gt,o in frame])
            for (gt,o),result in zip(frame,results):
                count+=1
                ident=result['object_id'] if isinstance(result,dict) else result.object_id
                stat=result['status'] if isinstance(result,dict) else result.status
                statuses[stat]=statuses.get(stat,0)+1
                if ident:
                    assigned+=1
                    contingency[(gt,ident)]=contingency.get((gt,ident),0)+1
                    gt_pred.setdefault(gt,set()).add(ident)
                    pred_gt.setdefault(ident,set()).add(gt)
                trace.append({'frame':o.frame_id,'gt':int(gt),'label':o.label,
                              'predicted_id':ident,'association_status':stat})
        # IDF1 via global optimal one-to-one assignment, covering all detections.
        gtids=sorted({a for a,b in contingency});pids=sorted({b for a,b in contingency})
        mat=np.zeros((len(gtids),len(pids)),dtype=int)
        lutg={v:i for i,v in enumerate(gtids)};lutp={v:i for i,v in enumerate(pids)}
        for (g,p),n in contingency.items():mat[lutg[g],lutp[p]]=n
        if mat.size:
            rid,cid=linear_sum_assignment(-mat);idtp=int(mat[rid,cid].sum())
        else:idtp=0
        idfp=assigned-idtp;idfn=count-idtp
        out={'mode':mode,'detections':count,'assigned':assigned,
             'predicted_tracks':len(pids), 'gt_objects':len(LABELS),
             'false_split_objects':sum(len(v)>1 for v in gt_pred.values()),
             'false_merge_tracks':sum(len(v)>1 for v in pred_gt.values()),
             'idf1':round(2*idtp/max(2*idtp+idfp+idfn,1),5),
             'association_coverage':round(assigned/max(count,1),5),
             'statuses':statuses,
             'id_switch_count':sum(1 for g in range(len(LABELS)) for a,b in zip(
                 [x['predicted_id'] for x in trace if x['gt']==g and x['predicted_id']],
                 [x['predicted_id'] for x in trace if x['gt']==g and x['predicted_id']][1:]) if a!=b)}
        objects=[{'object_id':t.object_id,'label':t.label,'description':t.description,
                  'color':COLORS[min(range(len(LABELS)),key=lambda i:np.linalg.norm(t.center[:2]-BASE_XY[i]))],
                  'center':t.center.tolist(),'low':t.low.tolist(),'high':t.high.tolist(),
                  'state':t.state,'hits':t.hits} for t in store.tracks()]
        store.close()
        return out,objects,trace

def run(output='results/identity_benchmark',seed=42,frames=120):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    records,robot_path=synth_frames(seed,frames)
    modes=['naive','geometry','appearance','fused_no_hold','fused_hold']
    results={};stable=None;trace=None
    for mode in modes:
        result,objects,t=evaluate(records,mode)
        print(mode,result,flush=True)
        results[mode]=result
        if mode=='fused_hold':stable=objects;trace=t
    payload={'dataset':'deterministic synthetic object-observation stress test; NOT camera RGB-D evidence',
             'random_seed':seed,'frames':frames,'gt_objects':len(LABELS),
             'description':'repeat visits, non-visibility, confusing classes, appearance noise',
             'results':results}
    (out/'ablation.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2))
    (out/'objects.json').write_text(json.dumps(stable,ensure_ascii=False,indent=2))
    (out/'tracking.json').write_text(json.dumps(trace,ensure_ascii=False,indent=2))
    return payload,stable,robot_path

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='results/identity_benchmark');p.add_argument('--frames',type=int,default=120)
    a=p.parse_args();run(a.output,frames=a.frames)
if __name__=='__main__':main()
