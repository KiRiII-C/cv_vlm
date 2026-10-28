"""Replay fixed YOLOE-26x detections at different confidence thresholds.

The source has actual RGB-D scene detections; physical GT is used for scoring,
not as an input to ObjectMemoryV2.
"""
import argparse,copy,gzip,json
from pathlib import Path
from .memory_v2_replay import replay

def sweep(path,output,thresholds=(.10,.15,.20,.25,.35)):
    with gzip.open(path,'rt',encoding='utf8') as f:
        cache=json.load(f)
    if cache.get('detector')!='yoloe-26x-seg.pt':
        raise ValueError('Incorrect detector provenance, need YOLOE-26x')
    results=[]
    for threshold in thresholds:
        trial=copy.deepcopy(cache)
        for frame in trial['frames']:
            keep=[i for i,o in enumerate(frame['observations'])
                  if o['confidence']>=threshold]
            frame['observations']=[frame['observations'][i] for i in keep]
            frame['physical_gt']=[frame['physical_gt'][i] for i in keep]
        report=replay(trial,'v2')
        gt=report['target_gt']
        results.append(dict(threshold=threshold,
            target_idf1=gt['IDF1'],
            false_merges=gt['false_merge_predicted_uuid'],
            false_splits=gt['false_split_physical_gt'],
            target_idfn=gt['IDFN'],target_idfp=gt['IDFP'],
            predicted_detections=report['input_detections']))
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(dict(scene=cache['scene'],detector=cache['detector'],
        ground_truth_used_only_for_scoring=True,results=results),indent=2))
    print('YOLOE26X_CONFIDENCE_SWEEP',cache['scene'],
          json.dumps(results),flush=True)
    return results

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--cache',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();sweep(a.cache,a.out)
