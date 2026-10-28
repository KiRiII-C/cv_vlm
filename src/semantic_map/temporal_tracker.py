"""Two-stage short-term RGB-D instance tracker, independent of persistent map UUIDs.

Architecture: ByteTrack low-confidence rescue + BoT-SORT-style appearance and
camera-motion independent 3D geometry, with uncertainty abstention. This is a
short-term tracklet generator, not a guarantee of long-term physical identity.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import linear_sum_assignment


def _unit(x):
    if x is None:
        return None
    a=np.asarray(x,dtype=float)
    if a.ndim!=1 or not a.size or not np.isfinite(a).all() or np.linalg.norm(a)<1e-9:
        raise ValueError("Invalid appearance feature")
    return a/np.linalg.norm(a)


def box_iou(a,b):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    low=np.maximum(a[:2],b[:2]);hi=np.minimum(a[2:],b[2:])
    overlap=np.prod(np.maximum(0.,hi-low))
    area_a=np.prod(np.maximum(0.,a[2:]-a[:2]))
    area_b=np.prod(np.maximum(0.,b[2:]-b[:2]))
    return float(overlap/max(area_a+area_b-overlap,1e-9))


@dataclass
class Detection:
    detection_id:str
    label:str
    bbox:np.ndarray
    xyz:np.ndarray
    confidence:float
    feature:np.ndarray|None=None
    mask:np.ndarray|None=None

    def __post_init__(self):
        self.bbox=np.asarray(self.bbox,dtype=float)
        self.xyz=np.asarray(self.xyz,dtype=float)
        if (self.bbox.shape!=(4,) or self.xyz.shape!=(3,) or
            not np.isfinite(self.bbox).all() or not np.isfinite(self.xyz).all() or
            np.any(self.bbox[2:]<=self.bbox[:2])):
            raise ValueError("Invalid detection geometry")
        if not (0.<=self.confidence<=1.):
            raise ValueError("Invalid detection confidence")
        if not self.label:
            raise ValueError("Missing category")
        self.feature=_unit(self.feature)
        if self.mask is not None:
            self.mask=np.asarray(self.mask,dtype=bool)
            if self.mask.ndim!=2:raise ValueError("Mask must be HxW")


@dataclass
class Tracklet:
    tid:str
    label:str
    bbox:np.ndarray
    xyz:np.ndarray
    feature:np.ndarray|None
    mask:np.ndarray|None
    first_frame:int
    last_frame:int
    hits:int=1
    velocity:np.ndarray=field(default_factory=lambda:np.zeros(3))

    def predict(self,frame):
        dt=min(4,max(0,frame-self.last_frame))
        return self.xyz+dt*self.velocity


@dataclass(frozen=True)
class TrackAssignment:
    detection_id:str
    short_id:str|None
    status:str
    score:float


class TemporalTracker:
    """1:1 matching over masks, image bounding boxes, XYZ, and appearance.

    No spatial match => do not use appearance alone to merge different,
    simultaneously visible objects. After max_lost frames memory is discarded;
    persistent 3D map association independently resolves long revisits.
    """
    def __init__(self,high_thresh=.27,low_thresh=.10,match_score=.57,
                 max_lost=40,min_hits=2,ambiguity_margin=.065):
        if not (0<=low_thresh<high_thresh<=1):raise ValueError("Invalid confidence thresholds")
        self.high_thresh=high_thresh;self.low_thresh=low_thresh
        self.match_score=match_score;self.max_lost=max_lost
        self.min_hits=min_hits;self.ambiguity_margin=ambiguity_margin
        self.tracks:dict[str,Tracklet]={}
        self._next_id=0
        self.last_frame=-1

    def _score(self,track,d,frame):
        if track.label.lower().strip()!=d.label.lower().strip():
            return -1.
        dt=max(1,frame-track.last_frame)
        old_xyz=track.predict(frame)
        distance=float(np.linalg.norm(old_xyz-d.xyz))
        # Center uncertainty grows with age, but cannot bridge arbitrary room
        # distances based on similarity alone.
        hard_limit=min(.80,.25+.09*np.sqrt(dt))
        if distance>hard_limit:
            return -1.
        xy_iou=box_iou(track.bbox,d.bbox)
        mask_iou=0.
        if track.mask is not None and d.mask is not None and track.mask.shape==d.mask.shape:
            intersect=np.count_nonzero(track.mask & d.mask)
            union=np.count_nonzero(track.mask | d.mask)
            mask_iou=intersect/max(1,union)
        prox=float(np.exp(-.5*(distance/max(.10,hard_limit*.55))**2))
        a=None
        if track.feature is not None and d.feature is not None and len(track.feature)==len(d.feature):
            a=float(max(0.,track.feature@d.feature))
            if a<.32 and distance>.10:return -1.
        # Low 2D IoU is expected for fast camera motion and revisits; map XYZ
        # remains the strongest source, while masks help on adjacent frames.
        score=.58*prox+.13*xy_iou+.09*mask_iou
        score+=.20*a if a is not None else .20*prox
        return float(np.clip(score,0,1))

    def _update(self,track,d,frame):
        dt=max(1,frame-track.last_frame)
        displacement=(d.xyz-track.xyz)/dt
        track.velocity=.75*track.velocity+.25*displacement
        track.xyz=.60*track.xyz+.40*d.xyz
        track.bbox=d.bbox.copy()
        track.mask=None if d.mask is None else d.mask.copy()
        if d.feature is not None:
            if track.feature is None or len(track.feature)!=len(d.feature):
                track.feature=d.feature.copy()
            else:
                track.feature=_unit(.80*track.feature+.20*d.feature)
        track.hits+=1
        track.last_frame=frame

    def _phase(self,frame,indices,dets,available,assignments,threshold):
        if not indices or not available:return
        track_keys=sorted(available)
        scores=np.full((len(indices),len(track_keys)),-1.,dtype=float)
        for i,idx in enumerate(indices):
            for j,k in enumerate(track_keys):
                scores[i,j]=self._score(self.tracks[k],dets[idx],frame)
        rows,cols=linear_sum_assignment(-scores)
        for row,col in zip(rows,cols):
            score=float(scores[row,col])
            if score<threshold:continue
            idx=indices[int(row)];key=track_keys[int(col)]
            ranked=np.sort(scores[row,scores[row]>=threshold])[::-1]
            # Competing candidates => abstain instead of inventing a new ID.
            if len(ranked)>1 and ranked[0]-ranked[1]<self.ambiguity_margin:
                assignments[idx]=TrackAssignment(dets[idx].detection_id,None,"ambiguous",score)
                continue
            self._update(self.tracks[key],dets[idx],frame)
            status="confirmed" if self.tracks[key].hits>=self.min_hits else "tentative"
            assignments[idx]=TrackAssignment(dets[idx].detection_id,key,status,score)
            available.remove(key)

    def update(self,frame:int,detections:list[Detection]):
        if frame<=self.last_frame:raise ValueError("Tracker requires strictly increasing frame index")
        ids=[d.detection_id for d in detections]
        if len(ids)!=len(set(ids)):raise ValueError("Duplicate frame detection ID")
        self.last_frame=frame
        self.tracks={k:t for k,t in self.tracks.items() if frame-t.last_frame<=self.max_lost}
        active=set(self.tracks)
        out=[None]*len(detections)
        high=[i for i,d in enumerate(detections) if d.confidence>=self.high_thresh]
        low=[i for i,d in enumerate(detections) if self.low_thresh<=d.confidence<self.high_thresh]
        self._phase(frame,high,detections,active,out,self.match_score)
        # ByteTrack-inspired second pass: low-score detections may update an
        # existing identity, but NEVER create new permanent or short IDs.
        self._phase(frame,low,detections,active,out,max(self.match_score,.64))
        for i in high:
            if out[i] is not None:continue
            d=detections[i]
            # Suppress uncertain creation when a weak plausible match exists.
            evidence=max((self._score(self.tracks[k],d,frame)
                          for k in active),default=-1.)
            if evidence>=.43:
                out[i]=TrackAssignment(d.detection_id,None,"unresolved",max(0,evidence))
                continue
            self._next_id+=1
            key=f"short_{self._next_id:07d}"
            self.tracks[key]=Tracklet(key,d.label,d.bbox.copy(),d.xyz.copy(),
                 None if d.feature is None else d.feature.copy(),
                 None if d.mask is None else d.mask.copy(),frame,frame)
            status="confirmed" if self.min_hits<=1 else "tentative"
            out[i]=TrackAssignment(d.detection_id,key,status,0.)
        for i,d in enumerate(detections):
            if out[i] is None:out[i]=TrackAssignment(d.detection_id,None,"low_unassigned",0.)
        return out
