"""Select a Hypersim scene containing MANY GT physical objects from YOLOE-26 prompts.

Only uses GT labels to select a BENCHMARK SCENE, not to make tracker decisions.
If used for hyperparameter tuning, keep a separate unseen test scene.
"""
import argparse,json,io
from collections import Counter
import numpy as np
from semantic_map.hypersim_range import open_scene,hdf5_array

TARGET={3,4,5,6,7,10,14,25}
def probe(scene,frames=(0,5,10,20,40)):
    remote,z=open_scene(scene)
    try:
        results=[];ids=set();hist=Counter()
        names=set(z.namelist())
        for k in frames:
            p=f'{scene}/images/scene_cam_00_geometry_hdf5/frame.{k:04d}.'
            if p+'semantic_instance.hdf5' not in names:continue
            uid=hdf5_array(z,p+'semantic_instance.hdf5')
            sem=hdf5_array(z,p+'semantic.hdf5')
            eligible=[]
            keys,n=np.unique(uid,return_counts=True)
            for ix,count in zip(keys,n):
                if ix<0 or count<250:continue
                categories=sem[uid==ix]
                categories=categories[categories>=0]
                if not len(categories):continue
                cat=int(Counter(categories.tolist()).most_common(1)[0][0])
                if cat in TARGET:
                    ids.add(int(ix));hist[cat]+=1;eligible.append(int(ix))
            results.append({'frame':k,'target_physical_ids':len(eligible)})
        return {'scene':scene,'sample_frames':results,
                'unique_target_physical_ids':len(ids),
                'target_class_frame_counts':dict(hist),
                'total_target_occurrences':sum(x['target_physical_ids'] for x in results)}
    finally:
        z.close();remote.close()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--scenes',nargs='+',default=['ai_001_002','ai_002_001','ai_003_001','ai_004_001','ai_005_001','ai_006_001','ai_007_001'])
    ap.add_argument('--out',default='results/hypersim_scene_selection.json')
    args=ap.parse_args();results=[]
    for s in args.scenes:
        try:
            r=probe(s)
            print('TARGET_SCENE_PROBE',json.dumps(r),flush=True)
            results.append(r)
        except Exception as exc:
            print('SCENE_UNAVAILABLE',s,str(exc),flush=True)
    results.sort(key=lambda x:x['total_target_occurrences'],reverse=True)
    from pathlib import Path
    p=Path(args.out);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps({'results':results},indent=2))
    if not results:raise RuntimeError('No accessible Hypersim scene')
if __name__=='__main__':main()
