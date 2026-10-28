"""Synthetic moved-object reidentification ablation with two chairs exchanging positions."""
import json,tempfile
from pathlib import Path
import numpy as np
from .models import Observation
from .engine import SemanticMapper
from .storage import MemoryStore

def test_mode(mode):
    with tempfile.TemporaryDirectory() as tmp:
        db=MemoryStore(Path(tmp)/'memory.sqlite');mapper=SemanticMapper(db,association_mode=mode,
                   ambiguity_margin=.10)
        bygt={0:[],1:[]};statuses={};last=[]
        for t in list(range(8))+list(range(25,40)):
            observations=[]
            for i in range(2):
                x=(3.0 if i==0 else 4.2) if t<25 else (4.2 if i==0 else 3.0)
                center=np.array([x,3.0,.6]);sz=np.array([.6,.6,1.])
                obs=Observation('dynamic','%03d'%t,'det%d'%i,'chair',center,center-sz/2,
                                center+sz/2,float(t),appearance=[1.,0.] if i==0 else [0.,1.])
                observations.append(obs)
            answers=mapper.process_frame(observations)
            for i,a in enumerate(answers):
                bygt[i].append(a.object_id)
                statuses[a.status]=statuses.get(a.status,0)+1
            last=answers
        stable={i:(len(set(x for x in ids if x is not None))==1 and all(x is not None for x in ids))
                for i,ids in bygt.items()}
        # Cross-assignment after swap is specifically checked; merely staying assigned
        # to the wrong physical object is not counted as a success.
        firstids=[bygt[0][0],bygt[1][0]]
        reentered_correct=sum(bygt[i][-1]==firstids[i] for i in range(2))
        out={'mode':mode,'stable_id_count':sum(stable.values()),
             'correct_reentry_after_swap':reentered_correct,'total_objects':2,
             'tracks_created':len(db.tracks()),'statuses':statuses}
        db.close();return out

def run(output='results/dynamic_ablation'):
    rows=[test_mode(m) for m in ['geometry','appearance','fused','adaptive_reid']]
    out=Path(output);out.mkdir(exist_ok=True,parents=True)
    (out/'moved_objects.json').write_text(json.dumps(rows,indent=2))
    print('MOVABLE_OBJECT_REID',json.dumps(rows),flush=True)
    return rows
if __name__=='__main__':run()
