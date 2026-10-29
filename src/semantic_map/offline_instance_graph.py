"""Offline class-soft 3D instance graph, no GT inside association.

For a completed RGB-D survey, construct spatial cross-view candidate edges
between object observations and form persistent clusters with a *cannot-link*
constraint on simultaneously visible masks. YOLO class names are soft evidence,
not hard identity gates. The algorithm is detector-/dataset-independent and
accepts only Observation instances, NEVER physical annotation identifiers.

Evaluated with frozen Hypersim GT XYZ observations (mapping pose control);
this is NOT a deployable segmentation-recall repair and not SLAM validated.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid5,NAMESPACE_URL
from collections import defaultdict,Counter
import math,json
import numpy as np
from scipy.spatial import cKDTree
from .models import Observation,Track
from .storage import MemoryStore

@dataclass(frozen=True)
class OfflineAssignment:
    observation_key:tuple
    object_uuid:str
    score:float

class OfflineInstanceGraph:
    def __init__(self,*,radius_m=.65,edge_threshold=.40,max_spread_m=.80,
                 center_scale_m=.50,visual_weight=.15,class_weight=.02,
                 geometry_weight=.10,max_frame_gap=1000000):
        if radius_m<=0 or max_spread_m<=0 or center_scale_m<=0:
            raise ValueError("Positive spatial parameters required")
        weights=(visual_weight,class_weight,geometry_weight)
        if min(weights)<0 or sum(weights)>=1:raise ValueError("Bad feature weights")
        self.radius=float(radius_m)
        self.edge_threshold=float(edge_threshold)
        self.max_spread=float(max_spread_m)
        self.cent_scale=float(center_scale_m)
        self.visual_weight=float(visual_weight)
        self.class_weight=float(class_weight)
        self.geometry_weight=float(geometry_weight)
        self.max_frame_gap=int(max_frame_gap)

    def associate(self,frames):
        """Input: list of (frame_index,list[Observation]).
        Returns deterministic per-observation UUID without using GT label/ID.
        """
        observations=[];indices=[];frame_labels=[]
        keys=set()
        last=-math.inf
        for frame,rows in frames:
            frame=int(frame)
            if frame<=last:raise ValueError("Frame index must increase strictly")
            last=frame
            for o in rows:
                if not isinstance(o,Observation):raise TypeError("Observation required")
                if o.key in keys:raise ValueError("Duplicate observation key")
                keys.add(o.key)
                observations.append(o);indices.append(frame)
                frame_labels.append((o.map_id,o.map_version,o.source,frame))
        n=len(observations)
        if not n:return []
        scenes={(o.map_id,o.map_version) for o in observations}
        if len(scenes)!=1:raise ValueError("One map and SLAM pose version per graph")
        xyz=np.stack([o.center for o in observations])
        low=np.stack([o.low for o in observations])
        high=np.stack([o.high for o in observations])
        label=[o.label for o in observations]
        feats=[o.appearance for o in observations]
        parent=list(range(n));size=[1]*n;members=[{(observations[i].source,indices[i])} for i in range(n)]
        mean=xyz.copy();count=np.ones(n)
        # Candidate generation independent of class, so class flicker can
        # never produce an absolute category reject.
        pairs=cKDTree(xyz).query_pairs(self.radius,output_type="ndarray")
        edges=[]
        for i,j in pairs:
            i=int(i);j=int(j)
            if indices[i]==indices[j] or abs(indices[i]-indices[j])>self.max_frame_gap:
                continue
            gap=np.maximum(0,np.maximum(low[i],low[j])-np.minimum(high[i],high[j]))
            surface_gap=float(np.linalg.norm(gap))
            if surface_gap>.55:continue
            appearance=0.
            if feats[i] is not None and feats[j] is not None:
                if feats[i].shape==feats[j].shape:
                    appearance=float(feats[i]@feats[j])
            dist=float(np.linalg.norm(xyz[i]-xyz[j]))
            if label[i]!=label[j] and appearance<.40 and dist>.20:
                # Class flicker MUST NOT break a geometrically near-identical
                # observation even if no appearance descriptor is available.
                # At longer distances, class disagreement plus inconsistent
                # appearance may veto an otherwise risky merge.
                continue
            score=(1-self.visual_weight-self.class_weight-self.geometry_weight)*math.exp(-dist/self.cent_scale)
            score+=self.geometry_weight*math.exp(-surface_gap/.23)
            score+=self.visual_weight*max(0.,appearance)
            score+=self.class_weight*(label[i]==label[j])
            if score>=self.edge_threshold:edges.append((float(score),i,j))
        # Deterministic, strongest-evidence-first constrained union.
        edges.sort(key=lambda z:(-z[0],z[1],z[2]))
        def root(a):
            while parent[a]!=a:
                parent[a]=parent[parent[a]]
                a=parent[a]
            return a
        for sc,i,j in edges:
            a=root(i);b=root(j)
            if a==b or members[a].intersection(members[b]):continue
            center=(mean[a]*count[a]+mean[b]*count[b])/(count[a]+count[b])
            if np.linalg.norm(center-mean[a])>self.max_spread:continue
            if np.linalg.norm(center-mean[b])>self.max_spread:continue
            if np.linalg.norm(mean[a]-mean[b])>self.max_spread*1.5:continue
            if size[a]<size[b]:a,b=b,a
            parent[b]=a
            size[a]+=size[b]
            members[a].update(members[b])
            mean[a]=center
            count[a]+=count[b]
        groups=defaultdict(list)
        for i in range(n):groups[root(i)].append(i)
        ids={}
        for items in groups.values():
            # Original observation identity is included to prevent collisions
            # across separately recorded maps/scenes; no GT information.
            first=min(items,key=lambda k:(indices[k],str(observations[k].key)))
            key=observations[first].key
            object_id="obj_"+uuid5(NAMESPACE_URL,
                "offline_instance_graph_v1|"+json.dumps(key)).hex[:16]
            for idx in items:ids[idx]=object_id
        return [OfflineAssignment(o.key,ids[i],1.) for i,o in enumerate(observations)]

    def export_sqlite(self,frames,path):
        """Create durable map with observations linked to canonical UUIDs.

        Does not silently modify an existing object map. To integrate with
        other modules, ingest this new SQLite file or copy through application
        import; no mistaken cross-map UUID merging.
        """
        path=Path(path)
        if path.exists():raise FileExistsError(path)
        flat=[o for _,ls in frames for o in ls]
        assigned=self.associate(frames)
        if len(assigned)!=len(flat):raise AssertionError("Assignment index mismatch")
        path.parent.mkdir(parents=True,exist_ok=True)
        db=MemoryStore(path)
        groups=defaultdict(list)
        for o,a in zip(flat,assigned):groups[a.object_uuid].append(o)
        for uid,observations in sorted(groups.items()):
            positions=np.stack([o.center for o in observations])
            lo=np.stack([o.low for o in observations])
            hi=np.stack([o.high for o in observations])
            classes=Counter(o.label for o in observations)
            label=classes.most_common(1)[0][0]
            appearance=next((o.appearance for o in observations if o.appearance is not None),None)
            track=Track(object_id=uid,label=label,center=np.median(positions,axis=0),
                low=np.median(lo,axis=0),high=np.maximum(np.median(hi,axis=0),np.median(lo,axis=0)+.015),
                appearance=appearance,hits=len(observations),
                first_seen=min(o.timestamp for o in observations),
                last_seen=max(o.timestamp for o in observations),
                state="confirmed" if len(observations)>=2 else "tentative",
                map_id=observations[0].map_id,map_version=observations[0].map_version,
                attributes={"offline_instance_graph":True,
                  "semantic_votes":dict(classes),
                  "uncertainty":"unvalidated_physical_identity",
                  "supporting_frame_count":len({o.frame_id for o in observations})})
            db.upsert_track(track)
        for o,a in zip(flat,assigned):
            db.write_observation(o,a.object_uuid,"offline_graph_clustered")
        db.db.commit()
        stats={"objects":len(groups),"observations":len(flat),
               "confirmed":sum(len(v)>=2 for v in groups.values()),
               "status":"OUTPUT_NOT_PHYSICAL_GT_VALIDATED"}
        db.close()
        return stats
