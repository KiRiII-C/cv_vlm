"""Explicit, queryable spatial relations over persistent object memory."""
from itertools import combinations
import numpy as np

def compute_relations(tracks,near_threshold=1.75):
    tracks=[t for t in tracks if t.state=='confirmed']
    result=[]
    for a,b in combinations(tracks,2):
        if a.map_id!=b.map_id or a.map_version!=b.map_version:continue
        d=float(np.linalg.norm(a.center[:2]-b.center[:2]))
        if d>near_threshold:continue
        direction=b.center[:2]-a.center[:2]
        result.append({'subject':a.object_id,'predicate':'near','object':b.object_id,'distance_m':round(d,3)})
        if abs(direction[0])>.4:
            result.append({'subject':b.object_id,'predicate':'right_of' if direction[0]>0 else 'left_of',
                           'object':a.object_id,'frame':'map','distance_m':round(d,3)})
        if abs(direction[1])>.4:
            result.append({'subject':b.object_id,'predicate':'above_y' if direction[1]>0 else 'below_y',
                           'object':a.object_id,'frame':'map','distance_m':round(d,3)})
    return result
