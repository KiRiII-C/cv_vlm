"""Training-free semantic evidence per voxel with uncertainty-aware fusion.

Useful alternative to irreversible one-shot labeling; labels must come from real
perception or from annotated offline data (this module only fuses evidence).
"""
from collections import defaultdict
import math

class VoxelSemanticMemory:
    def __init__(self,voxel_m=.05,min_views=2,margin=.12):
        self.voxel_m=voxel_m;self.min_views=min_views;self.margin=margin
        self.scores=defaultdict(lambda:defaultdict(float))
        self.views=defaultdict(set)
    def observe(self,key,label,confidence,frame_id):
        if not 0<=confidence<=1:raise ValueError('Invalid confidence')
        key=tuple(key)
        if (key,frame_id) in self.views[key]:return
        self.views[key].add((key,frame_id))
        self.scores[key][label]+=float(confidence)
    def query(self,key):
        key=tuple(key)
        c=self.scores.get(key,{})
        if not c:return dict(status='unknown',label=None,posterior=0.,views=0)
        total=sum(c.values());ranked=sorted(c.items(),key=lambda v:-v[1]);top=ranked[0]
        second=ranked[1][1] if len(ranked)>1 else 0
        n=len(self.views[key]);margin=(top[1]-second)/max(total,1e-8)
        status='confirmed' if n>=self.min_views and margin>=self.margin else 'tentative'
        return dict(status=status,label=top[0] if status=='confirmed' else None,
                    posterior=top[1]/max(total,1e-8),views=n,margin=margin)
