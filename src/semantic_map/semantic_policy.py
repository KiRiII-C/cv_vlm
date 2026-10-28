"""Selective view scheduling, preserving evidence before requesting an expensive VLM.

Does not implement a VLM. It determines which object crops should be sent to one.
"""
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class View:
    object_id: str
    frame_id: str
    quality: float
    yaw_rad: float
    timestamp: float

def _delta(a,b):
    return abs(math.atan2(math.sin(a-b),math.cos(a-b)))

def choose_views(candidates,max_views=3,min_angular_separation=.5):
    """Select high-quality, non-redundant views of confirmed objects."""
    if max_views<1:raise ValueError('max_views must be positive')
    groups={}
    for x in candidates:
        if not 0<=x.quality<=1:raise ValueError('quality must be in [0,1]')
        groups.setdefault(x.object_id,[]).append(x)
    selected=[]
    for _,views in groups.items():
        keep=[]
        for v in sorted(views,key=lambda z:(-z.quality,z.frame_id)):
            if len(keep)>=max_views:break
            if all(_delta(v.yaw_rad,k.yaw_rad)>=min_angular_separation for k in keep):keep.append(v)
        selected.extend(keep)
    return selected
