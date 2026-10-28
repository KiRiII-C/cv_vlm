"""Deterministic end-to-end camera projection -> persistent identity demo."""
import argparse,json,tempfile
import numpy as np
from pathlib import Path
from . import MemoryStore,SemanticMapper,Intrinsics,make_observation


def run(path):
    K=Intrinsics(100,100,16,12,32,24,0.001)
    depth=np.full((24,32),1500,np.uint16)
    mask=np.zeros((24,32),bool);mask[6:18,9:22]=True
    store=MemoryStore(path);engine=SemanticMapper(store)
    ids=[]
    for i,frame in enumerate(['enter','visible','other_room','reenter']):
        if frame=='other_room':continue
        pose=np.eye(4)
        # The same stationary object, from the same global pose; step 2 is after leaving.
        o=make_observation(depth,mask,K,pose,source='demo',frame_id=frame,
                           detection_id='chair',label='chair',timestamp=float(i),
                           appearance=[1,0,0],description='wooden chair')
        if o is None:raise RuntimeError('No measured points')
        ids.append(engine.process_frame([o])[0].__dict__)
    objects=[{'id':t.object_id,'class':t.label,'hits':t.hits,'state':t.state,
              'center_map_m':t.center.tolist()} for t in store.search()]
    result={'decisions':ids,'confirmed_objects':objects,'dedup_proposals':engine.suggest_duplicates()}
    print(json.dumps(result,ensure_ascii=False,indent=2))
    store.close()
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--db',default='semantic_map.sqlite')
    args=ap.parse_args();run(args.db)

if __name__=='__main__':main()
