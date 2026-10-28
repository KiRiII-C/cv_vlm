"""Offline global data association with persistent IDs and ambiguity handling."""
from __future__ import annotations
from dataclasses import dataclass
from uuid import uuid4
import numpy as np
from scipy.optimize import linear_sum_assignment
from .models import Track
from .storage import MemoryStore
from .association import score_variant,overlap_3d


@dataclass
class Decision:
    detection_id: str
    object_id: str | None
    status: str
    confidence: float


class SemanticMapper:
    def __init__(self,store:MemoryStore,min_hits=2,min_score=.68,ambiguity_margin=.10,association_mode="fused"):
        self.store=store
        self.min_hits=min_hits
        self.min_score=min_score
        self.ambiguity_margin=ambiguity_margin
        self.association_mode=association_mode

    @staticmethod
    def update(t,o,min_hits,relocated=False):
        n=t.hits
        # Avoid moving persistent geometry too aggressively with viewpoint changes.
        if relocated:
            # Track the current pose of a candidate movable instance, not a huge
            # box spanning two locations; observation history remains in SQLite.
            t.center=o.center.copy();t.low=o.low.copy();t.high=o.high.copy()
        else:
            t.center=.75*t.center+.25*o.center
            t.low=np.minimum(t.low,o.low)
            t.high=np.maximum(t.high,o.high)
        if t.appearance is not None and o.appearance is not None:
            z=.85*t.appearance+.15*o.appearance
            t.appearance=z/np.linalg.norm(z)
        elif t.appearance is None:
            t.appearance=o.appearance
        t.hits=n+1
        t.last_seen=max(t.last_seen,o.timestamp)
        if o.description and not t.description:t.description=o.description
        if t.hits>=min_hits:t.state='confirmed'
        return t

    def process_frame(self,observations):
        if not observations:return []
        keys={o.key for o in observations}
        if len(keys)!=len(observations):raise ValueError('Duplicate detection keys within frame')
        frames={(o.source,o.frame_id,o.map_id,o.map_version) for o in observations}
        if len(frames)!=1:raise ValueError('One source, frame and map version per call')
        results=[None]*len(observations)
        candidates=[]
        for i,o in enumerate(observations):
            previous=self.store.seen(o)
            if previous:
                results[i]=Decision(o.detection_id,previous['object_id'],'already_processed',1.0)
            else:candidates.append((i,o))
        if not candidates:return results
        # Within-frame duplicate masks can represent ONE physical object. Keep only
        # very high-overlap, same-class detections with compatible appearance.
        # This does not merge similar but spatially different co-visible objects.
        candidate_by_conf=sorted(candidates,key=lambda a:-a[1].confidence)
        kept=[];suppressed=[]
        for index,o in candidate_by_conf:
            duplicate_of=None
            for kept_index,k in kept:
                iou=overlap_3d(o.low,o.high,k.low,k.high)
                if o.label.strip().lower()!=k.label.strip().lower() or iou<.85:
                    continue
                if np.linalg.norm(o.center-k.center)>.08:
                    continue
                if o.appearance is not None and k.appearance is not None:
                    if o.appearance.shape!=k.appearance.shape or float(o.appearance@k.appearance)<.97:
                        continue
                duplicate_of=kept_index
                break
            if duplicate_of is None:kept.append((index,o))
            else:suppressed.append((index,o,duplicate_of))
        candidates=sorted(kept,key=lambda a:a[0])
        tracks=self.store.tracks()
        scores=np.full((len(candidates),len(tracks)),-1.0)
        for i,(_,o) in enumerate(candidates):
            for j,t in enumerate(tracks):
                s=score_variant(t,o,self.association_mode)
                if s is not None:scores[i,j]=s
        # Global one-to-one Hungarian: same track can be assigned once per frame.
        if scores.size:
            r,c=linear_sum_assignment(-scores)
            proposals={int(i):int(j) for i,j in zip(r,c) if scores[i,j]>=self.min_score}
        else: proposals={}
        used_tracks=set()
        for i,(original_idx,o) in enumerate(candidates):
            ranked=np.sort(scores[i][scores[i]>=self.min_score])[::-1]
            ambiguous=len(ranked)>1 and ranked[0]-ranked[1]<self.ambiguity_margin
            if ambiguous:
                self.store.write_observation(o,None,'ambiguous')
                results[original_idx]=Decision(o.detection_id,None,'ambiguous',float(ranked[0]))
                continue
            if i in proposals and proposals[i] not in used_tracks:
                j=proposals[i];t=tracks[j];used_tracks.add(j)
                t=self.update(t,o,self.min_hits,relocated=(self.association_mode=="adaptive_reid"
                    and o.timestamp-t.last_seen>8 and np.linalg.norm(t.center-o.center)>.45))
                self.store.upsert_track(t)
                self.store.write_observation(o,t.object_id,'matched')
                results[original_idx]=Decision(o.detection_id,t.object_id,
                                               'confirmed' if t.state=='confirmed' else 'tentative',float(scores[i,j]))
            else:
                # A strong match consumed by another detection is NOT a new object.
                top=float(np.max(scores[i])) if len(tracks) else -1
                if top >= self.min_score:
                    self.store.write_observation(o,None,'unassigned_conflict')
                    results[original_idx]=Decision(o.detection_id,None,'unassigned_conflict',top)
                elif top >= .48:
                    # An uncertain possible match is not evidence of a new object.
                    self.store.write_observation(o,None,'unresolved')
                    results[original_idx]=Decision(o.detection_id,None,'unresolved',top)
                else:
                    t=Track(object_id='obj_'+uuid4().hex[:16],label=o.label,center=o.center.copy(),
                            low=o.low.copy(),high=o.high.copy(),appearance=o.appearance,
                            hits=1,first_seen=o.timestamp,last_seen=o.timestamp,
                            description=o.description,map_version=o.map_version,map_id=o.map_id)
                    self.store.upsert_track(t)
                    self.store.write_observation(o,t.object_id,'created_tentative')
                    results[original_idx]=Decision(o.detection_id,t.object_id,'tentative',0.)
        for index,o,original_idx in suppressed:
            primary=results[original_idx]
            self.store.write_observation(o,primary.object_id,'duplicate_detection')
            results[index]=Decision(o.detection_id,primary.object_id,'duplicate_detection',1.0)
        self.store.db.commit()
        return results

    def suggest_duplicates(self,min_iou=.7,min_cos=.96):
        """Offer conservative merge proposals, NEVER merge uncertain identities automatically."""
        tracks=self.store.tracks();out=[]
        for i,a in enumerate(tracks):
            for b in tracks[i+1:]:
                if a.label!=b.label or a.map_version!=b.map_version or a.map_id!=b.map_id:continue
                if self.store.sightings_cooccur(a.object_id,b.object_id):continue
                iou=overlap_3d(a.low,a.high,b.low,b.high)
                if iou<min_iou:continue
                if a.appearance is None or b.appearance is None:continue
                cos=float(a.appearance@b.appearance)
                if cos>=min_cos:out.append((a.object_id,b.object_id,round(iou,3),round(cos,3)))
        return out
