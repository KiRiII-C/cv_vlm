"""B2: short-track class uncertainty without a separate neural tracking model.

Tracklet ID persists across synonymous detector labels. The raw detector
confidence increments Dirichlet-style pseudo counts; this is NOT a
calibrated posterior until validated against annotated data. Parent class
gating rejects incompatible categories (chair vs table). Physical UUIDs
remain the independent responsibility of the later long-term memory phase C.
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass,replace
from .temporal_tracker import Detection,TemporalTracker,TrackAssignment


PARENT={
    "office chair":"chair", "desk chair":"chair","dining chair":"chair",
    "computer chair":"chair","swivel chair":"chair",
    "desk":"table","dining table":"table","coffee table":"table",
    "computer monitor":"computer monitor","monitor":"computer monitor",
    "potted plant":"potted plant","couch":"sofa"
}

def class_parent(label:str)->str:
    x=" ".join(str(label).lower().strip().split())
    if not x:raise ValueError("empty semantic category")
    return PARENT.get(x,x)

@dataclass(frozen=True)
class SemanticTrackAssignment:
    detection_id:str
    short_id:str | None
    status:str
    score:float
    raw_label:str
    parent_label:str
    class_probabilities:dict[str,float]

class ClassTolerantTemporalAssociator:
    """One-to-one geometry/appearance tracker + observation-level class evidence."""
    def __init__(self,tracker:TemporalTracker|None=None,alpha=.5):
        if alpha<=0:raise ValueError("alpha must be positive")
        self.tracker=TemporalTracker() if tracker is None else tracker
        self.alpha=float(alpha)
        self._votes:dict[str,dict[str,float]]=defaultdict(dict)
        self._families:dict[str,str]={}

    def update(self,frame:int,detections:list[Detection])->list[SemanticTrackAssignment]:
        canonical=[replace(d,label=class_parent(d.label)) for d in detections]
        result=self.tracker.update(frame,canonical)
        output=[]
        for original,cl,assigned in zip(detections,canonical,result):
            track=assigned.short_id
            posterior={}
            if track:
                if track in self._families and self._families[track]!=cl.label:
                    raise RuntimeError("Identity matched across unrelated class families")
                self._families[track]=cl.label
                evidence=self._votes[track]
                key=" ".join(original.label.lower().strip().split())
                # Fractional pseudo-count is sufficient to prevent one
                # low-confidence label from replacing accumulated evidence.
                evidence[key]=evidence.get(key,0.)+float(original.confidence)
                total=sum(evidence.values())+self.alpha*len(evidence)
                posterior={key:(val+self.alpha)/total for key,val in sorted(evidence.items())}
            output.append(SemanticTrackAssignment(
                detection_id=assigned.detection_id,short_id=track,
                status=assigned.status,score=assigned.score,
                raw_label=original.label,parent_label=cl.label,
                class_probabilities=posterior))
        # Expired tracks cannot be resurrected by this short-term adapter.
        for key in list(self._votes):
            if key not in self.tracker.tracks:
                self._votes.pop(key,None);self._families.pop(key,None)
        return output
