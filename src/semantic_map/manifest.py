"""Offline interchange adapter: model-agnostic RGB-D, masks, embeddings and poses.

This is deliberately NOT a fake detector. Masks/embeddings must be supplied by a
real segmentation/feature-extraction stage, or by benchmark ground truth.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from .geometry import Intrinsics,make_observation
from .storage import MemoryStore
from .engine import SemanticMapper


def run(manifest, db_path, report_path=None):
    try:import cv2
    except ImportError as e:raise RuntimeError('Install package with [rgbd] extra') from e
    manifest=Path(manifest)
    store=MemoryStore(db_path)
    mapper=SemanticMapper(store)
    summary={'frames':0,'detections':0,'confirmed_matches':0,'tentative':0,
             'ambiguous':0,'unassigned_conflict':0,'unresolved':0,'invalid_depth':0,
             'decisions':[]}
    for line in manifest.read_text().splitlines():
        if not line.strip():continue
        f=json.loads(line)
        base=manifest.parent
        k=f['camera']
        intr=Intrinsics(**k)
        depth=cv2.imread(str(base/f['depth_path']),cv2.IMREAD_UNCHANGED)
        if depth is None:raise FileNotFoundError(f['depth_path'])
        T=np.asarray(f['T_map_cam'],np.float64)
        detections=[]
        for d in f['detections']:
            mask=cv2.imread(str(base/d['mask_path']),cv2.IMREAD_UNCHANGED)
            if mask is None:raise FileNotFoundError(d['mask_path'])
            if mask.ndim==3:raise ValueError('Masks must be single-channel')
            o=make_observation(depth,mask>0,intr,T,source=str(f['source']),
                               frame_id=str(f['frame_id']),detection_id=str(d['id']),
                               label=str(d['label']),timestamp=float(f['timestamp']),
                               appearance=d.get('embedding'),
                               description=d.get('description',''),
                               confidence=float(d.get('confidence',1)),
                               map_version=int(f.get('map_version',0)))
            summary['detections']+=1
            if o is not None:detections.append(o)
            else:summary['invalid_depth']+=1
        ds=mapper.process_frame(detections)
        for dec in ds:
            summary['decisions'].append({'source':f['source'],'frame_id':f['frame_id'],**dec.__dict__})
            if dec.status=='confirmed':summary['confirmed_matches']+=1
            if dec.status=='tentative':summary['tentative']+=1
            if dec.status in ('ambiguous','unassigned_conflict','unresolved'):summary[dec.status]+=1
        summary['frames']+=1
    summary['confirmed_objects']=[{'object_id':t.object_id,'label':t.label,
                                   'hits':t.hits,'description':t.description,
                                   'center_map_m':t.center.tolist(),
                                   'box_low':t.low.tolist(),'box_high':t.high.tolist()}
                                  for t in store.search()]
    summary['dedup_proposals']=mapper.suggest_duplicates()
    if report_path:Path(report_path).write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    store.close()
    return summary


def main():
    p=argparse.ArgumentParser(description='Offline RGB-D semantic-memory association')
    p.add_argument('manifest',help='JSONL containing source/frame_id, pose, depth and masks')
    p.add_argument('--db',default='semantic_memory.sqlite')
    p.add_argument('--report',default='semantic_report.json')
    a=p.parse_args()
    summary=run(a.manifest,a.db,a.report)
    print(json.dumps({k:v for k,v in summary.items() if k!='decisions'},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
