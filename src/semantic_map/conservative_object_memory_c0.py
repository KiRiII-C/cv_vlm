"""C0 conservative observation memory. NOT a validated physical-instance ID tracker.

Each update is a complete frame's visible *detections*, not a declaration that
other map objects ceased to exist.  A 3D geometric match is necessary, but
not sufficient, to re-use a tentative candidate. Conflicting/equal candidates
are left ambiguous rather than merged. GT instance IDs are NEVER accepted.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from collections import Counter
import numpy as np
from scipy.optimize import linear_sum_assignment

@dataclass
class Candidate:
    id: str
    cls: str
    center: np.ndarray
    extent: np.ndarray
    first_frame: int
    last_frame: int
    n: int = 1
    confidence_sum: float = 0.
    feature: np.ndarray | None = None
    short_votes: Counter = field(default_factory=Counter)
    # No ABSENT state: unseen is UNKNOWN until independently verified.
    status: str = "PROVISIONAL"

class ConservativeObjectMemory:
    def __init__(self, *, min_conf=.35, max_distance_m=.42, ambiguity_margin=.075,
                 appearance_weight=.17, short_vote_bonus=.055):
        self.min_conf=float(min_conf)
        self.max_distance_m=float(max_distance_m)
        self.ambiguity_margin=float(ambiguity_margin)
        self.appearance_weight=float(appearance_weight)
        self.short_vote_bonus=float(short_vote_bonus)
        self.objects:dict[str,Candidate]={}
        self.next_id=1
        self.last_frame=-1
        self.count=Counter()
    @staticmethod
    def _feature(row):
        f=row.get("appearance")
        if f is None:return None
        arr=np.asarray(f,dtype=float).ravel()
        if not len(arr) or not np.isfinite(arr).all():return None
        n=np.linalg.norm(arr)
        return arr/n if n>1e-12 else None
    @staticmethod
    def _vec(row,key,other=None):
        value=row.get(key,row.get(other) if other else None)
        x=np.asarray(value,dtype=float)
        if x.shape!=(3,) or not np.isfinite(x).all():
            raise ValueError("Invalid 3D observation field "+key)
        return x
    def _normalize(self,row):
        cls=str(row.get("class",row.get("label",""))).strip().lower()
        if not cls:raise ValueError("Empty class")
        xyz=self._vec(row,"center_map_m","center")
        lo=self._vec(row,"aabb_min_map_m","low") if ("aabb_min_map_m" in row or "low" in row) else xyz-.05
        hi=self._vec(row,"aabb_max_map_m","high") if ("aabb_max_map_m" in row or "high" in row) else xyz+.05
        if np.any(hi<lo):raise ValueError("Invalid AABB")
        conf=float(row.get("confidence",0))
        if not np.isfinite(conf) or not 0<=conf<=1:raise ValueError("Invalid confidence")
        return cls,xyz,np.maximum(hi-lo,.02),conf,self._feature(row),row.get("short_track_id")
    def _score(self,item,c:Candidate,frame):
        cls,xyz,extent,conf,feat,short=item
        if cls!=c.cls:return None
        d=float(np.linalg.norm(xyz-c.center))
        gap=max(1,frame-c.last_frame)
        # Bounded gate: no unbounded drift from temporal absence.
        gate=min(self.max_distance_m,.16+.042*np.sqrt(gap)+.12*float(np.linalg.norm(c.extent)))
        if d>gate:return None
        score=1.-d/max(gate,1e-6)
        esize=float(np.linalg.norm(extent-c.extent))
        score-=.085*min(1.,esize)
        if feat is not None and c.feature is not None and feat.shape==c.feature.shape:
            sim=float(np.clip(np.dot(feat,c.feature),-1,1))
            score+=self.appearance_weight*(sim-.5)
        if short and c.short_votes.get(str(short),0):
            score+=self.short_vote_bonus
        return score
    def update(self,frame:int,observations:list[dict]):
        if frame<=self.last_frame:raise ValueError("Frame indices must be strictly increasing")
        self.last_frame=int(frame)
        items=[self._normalize(row) for row in observations]
        assignment=[{"status":"UNASSIGNED","candidate_id":None} for _ in items]
        candidates=list(self.objects.values())
        scores=np.full((len(items),len(candidates)),-1.e8)
        valid_rows=[]
        for i,item in enumerate(items):
            if item[3]<self.min_conf:
                assignment[i]["status"]="LOW_CONFIDENCE"
                self.count["low_confidence"]+=1
                continue
            valid_rows.append(i)
            for j,c in enumerate(candidates):
                x=self._score(item,c,frame)
                if x is not None:scores[i,j]=x
        ambiguous=set()
        for i in valid_rows:
            found=np.sort(scores[i,scores[i]>-1e7])
            if len(found)>=2 and found[-1]-found[-2]<self.ambiguity_margin:
                ambiguous.add(i)
                self.count["ambiguous"]+=1
                assignment[i]["status"]="AMBIGUOUS"
        used=set()
        if valid_rows and candidates:
            good=[i for i in valid_rows if i not in ambiguous]
            if good:
                cost=-scores[good]
                rr,cc=linear_sum_assignment(cost)
                for a,b in zip(rr,cc):
                    i=good[a]
                    if scores[i,b]<-1e7:continue
                    c=candidates[b]
                    if c.id in used:raise RuntimeError("Duplicate frame assignment")
                    used.add(c.id)
                    cls,xyz,extent,conf,feat,short=items[i]
                    alpha=min(.22,.09+.13*conf)
                    c.center=(1-alpha)*c.center+alpha*xyz
                    c.extent=(1-alpha)*c.extent+alpha*extent
                    if feat is not None:
                        if c.feature is None or c.feature.shape!=feat.shape:c.feature=feat
                        else:
                            x=(1-alpha)*c.feature+alpha*feat
                            c.feature=x/max(np.linalg.norm(x),1.e-12)
                    c.last_frame=frame
                    c.n+=1;c.confidence_sum+=conf
                    if short:c.short_votes[str(short)]+=1
                    c.status="MULTIVIEW_CANDIDATE" if c.n>=3 else "PROVISIONAL"
                    assignment[i]={"status":"MATCHED","candidate_id":c.id}
                    self.count["matched"]+=1
        for i in valid_rows:
            if assignment[i]["status"]!="UNASSIGNED":continue
            # If matching candidate was consumed by another observation,
            # abstain from spawning duplicate evidence for the same geometry.
            if len(candidates) and np.any(scores[i]>-1e7):
                assignment[i]["status"]="AMBIGUOUS"
                self.count["ambiguous"]+=1
                continue
            cls,xyz,extent,conf,feat,short=items[i]
            ident=f"candidate_{self.next_id:06d}"
            self.next_id+=1
            candidate=Candidate(ident,cls,xyz.copy(),extent.copy(),frame,frame,
                confidence_sum=conf,feature=feat)
            if short:candidate.short_votes[str(short)]+=1
            self.objects[ident]=candidate
            assignment[i]={"status":"NEW_PROVISIONAL","candidate_id":ident}
            self.count["new_provisional"]+=1
        self.count["frames"]+=1
        self.count["observations"]+=len(items)
        # Nothing marked disappeared when missing from this frame.
        return assignment
    def snapshot(self):
        return [
            {"candidate_id":c.id,"class":c.cls,"center_map_m":c.center.tolist(),
             "extent_m":c.extent.tolist(),"first_frame":c.first_frame,
             "last_seen_frame":c.last_frame,"observations":c.n,
             "mean_confidence":c.confidence_sum/c.n,
             "status":c.status,"unseen_between_frames_means":"UNKNOWN_NOT_ABSENT",
             "physical_instance_gt_validated":False}
            for c in self.objects.values()]
