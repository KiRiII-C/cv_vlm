"""Cross-visit identity consistency audit; conservative diagnostic, never GT IDF1.

For observations at least 30 frames apart, identify highly consistent physical
3D footprints and visual features. Flag different canonical UUIDs, excluding
tracks known to have been simultaneously visible. NO AUTOMATIC MERGING.
"""
from __future__ import annotations
import argparse,gzip,json
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.spatial import cKDTree

def audit(root,output=None,min_gap=30,max_distance=.12,min_cos=.93,min_iou=.45):
    root=Path(root)
    with gzip.open(root/'observation_evidence.json.gz','rt') as f:
        frames=json.load(f)
    records=json.loads((root/'real_frame_associations.json').read_text())
    if len(frames)!=len(records):raise ValueError('Evidence/association frame mismatch')
    grouped=defaultdict(list);coobserved=set()
    for n,(objects,record) in enumerate(zip(frames,records)):
        lookup={a['detection_id']:a['object_id'] for a in record['associations']}
        ids=sorted({v for v in lookup.values() if v})
        for i,a in enumerate(ids):
            for b in ids[i+1:]:coobserved.add((a,b))
        for o in objects:
            uid=lookup.get(o['detection_id'])
            if not uid:continue
            grouped[o['label']].append(
              (n,uid,np.asarray(o['center'],float),np.asarray(o['low'],float),
               np.asarray(o['high'],float),
               np.asarray(o['appearance'],float) if o['appearance'] is not None else None))
    same=0;different=0;tested=0;pairs={}
    for label,rows in grouped.items():
        if len(rows)<2:continue
        tree=cKDTree(np.stack([v[2] for v in rows]))
        for i,j in tree.query_pairs(max_distance):
            a,b=rows[i],rows[j]
            if abs(a[0]-b[0])<min_gap or a[5] is None or b[5] is None:continue
            if a[5].shape!=b[5].shape:continue
            cos=float(a[5]@b[5])
            if cos<min_cos:continue
            inter=np.maximum(0,np.minimum(a[4],b[4])-np.maximum(a[3],b[3])).prod()
            v1=np.maximum(0,a[4]-a[3]).prod();v2=np.maximum(0,b[4]-b[3]).prod()
            iou=float(inter/max(v1+v2-inter,1e-8))
            if iou<min_iou:continue
            tested+=1
            if a[1]==b[1]:
                same+=1;continue
            uid=tuple(sorted([a[1],b[1]]))
            if uid in coobserved:continue
            different+=1
            entry=pairs.setdefault(uid,{'object_ids':list(uid),'category':label,
                                        'evidence_pairs':0,'examples':[]})
            entry['evidence_pairs']+=1
            if len(entry['examples'])<3:entry['examples'].append([a[0],b[0]])
    out={'status':'PROXY_DIAGNOSTIC_NOT_PHYSICAL_GROUND_TRUTH',
         'frames':len(frames),'cross_visit_matched_evidence_pairs':tested,
         'consistent_same_id_pairs':same,
         'different_id_pairs_without_known_covisibility':different,
         'suspected_false_split_track_pairs':sorted(pairs.values(),key=lambda x:-x['evidence_pairs']),
         'constraints':{'gap_frames':min_gap,'center_m':max_distance,'cosine':min_cos,'IoU':min_iou},
         'warning':'Zero flagged pairs does not prove zero duplicated physical objects; tested pairs are selected by an independent geometric+appearance proxy'}
    if output:
        Path(output).parent.mkdir(exist_ok=True,parents=True)
        Path(output).write_text(json.dumps(out,ensure_ascii=False,indent=2))
    print('IDENTITY_AUDIT',json.dumps({k:v for k,v in out.items() if k!='suspected_false_split_track_pairs'},ensure_ascii=False),flush=True)
    return out

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',default='results/tum_real')
    p.add_argument('--out',default='results/tum_real/identity_audit.json')
    args=p.parse_args()
    audit(args.root,args.out)

if __name__=='__main__':main()
