"""Conservative durable relocation hypotheses for moving physical objects.

No appearance-only automatic ID jumps. A physical object UUID can move only
when (1) its long-term appearance is distinctive among class peers,
(2) at least three *independent* new-location observations agree, and
(3) two independent views have verified that its old location is actually
free in a calibrated depth/occupancy map. This module does not raycast;
verification inputs MUST be supplied by a validated map visibility module.
"""
from __future__ import annotations
from dataclasses import dataclass
from .models import Observation
import json
import numpy as np


@dataclass(frozen=True)
class RelocationDecision:
    object_id:str|None
    status:str
    confidence:float
    evidence_count:int


class RelocationMemoryV2:
    def __init__(self,memory,*,min_relocation_views=3,min_free_old_views=2,
                 min_visual=.95,distinctive_margin=.12,max_relocation_m=3.):
        self.memory=memory;self.store=memory.store
        self.min_relocation_views=int(min_relocation_views)
        self.min_free_old_views=int(min_free_old_views)
        self.min_visual=float(min_visual)
        self.distinctive_margin=float(distinctive_margin)
        self.max_relocation_m=float(max_relocation_m)
        self.store.db.executescript('''
        CREATE TABLE IF NOT EXISTS old_region_vacancy_v2 (
            map_id TEXT NOT NULL,object_id TEXT NOT NULL,
            source TEXT NOT NULL,frame_id TEXT NOT NULL,
            independent_ray_validation INTEGER NOT NULL,
            evidence TEXT NOT NULL,
            PRIMARY KEY(map_id,object_id,source,frame_id)
        );
        CREATE TABLE IF NOT EXISTS relocation_hypothesis_v2 (
            map_id TEXT NOT NULL,object_id TEXT NOT NULL,
            source TEXT NOT NULL,frame_id TEXT NOT NULL,
            detection_id TEXT NOT NULL,evidence_json TEXT NOT NULL,
            center TEXT NOT NULL,low TEXT NOT NULL,high TEXT NOT NULL,
            timestamp REAL NOT NULL,score REAL NOT NULL,
            PRIMARY KEY(map_id,object_id,source,frame_id)
        );
        CREATE TABLE IF NOT EXISTS relocation_commit_v2 (
            map_id TEXT NOT NULL,object_id TEXT NOT NULL,
            frame_id TEXT NOT NULL,old_center TEXT NOT NULL,
            new_center TEXT NOT NULL,scene_version INTEGER NOT NULL,
            PRIMARY KEY(map_id,object_id,frame_id)
        );
        ''')
        self.store.db.commit()

    def note_verified_old_region_empty(self,uid,*,map_id='default',source,
                                        frame_id,proof:dict):
        """Record explicit ray-traced free-space evidence, not detector absence.

        proof must contain fields: `origin_map`, `raycast_depth_m`,
        `pose_version`, and `old_region_cleared` == True.
        Actual raytracing is upstream; no success is assumed without proof.
        """
        required={'origin_map','raycast_depth_m','pose_version','old_region_cleared'}
        if not isinstance(proof,dict) or not required.issubset(proof) or proof['old_region_cleared'] is not True:
            raise ValueError('Verified raycast proof is required for old-location vacancy')
        tracks={t.object_id:t for t in self.store.tracks()}
        if uid not in tracks or tracks[uid].map_id!=map_id:raise KeyError(uid)
        if int(proof['pose_version'])!=tracks[uid].map_version:
            raise ValueError('Stale pose graph cannot prove vacancy')
        origin=np.asarray(proof['origin_map'],float)
        if origin.shape!=(3,) or not np.isfinite(origin).all() or not np.isfinite(proof['raycast_depth_m']):
            raise ValueError('Invalid raycast evidence')
        with self.store.db:
            self.store.db.execute('''INSERT OR IGNORE INTO old_region_vacancy_v2
                VALUES(?,?,?,?,?,?)''',(map_id,uid,source,str(frame_id),1,json.dumps(proof)))

    def note_rgbd_old_region(self,uid,depth,K,T_map_cam,*,source,frame_id,map_id='default'):
        """Validate true RGB-D depth rays before registering that an object's old location is free."""
        from .visibility_evidence_v2 import evaluate_visibility
        tracks={t.object_id:t for t in self.store.tracks()}
        if uid not in tracks or tracks[uid].map_id!=map_id:raise KeyError(uid)
        t=tracks[uid]
        visibility=evaluate_visibility(depth,K,T_map_cam,t.low,t.high,pose_version=t.map_version)
        if visibility.proof:
            self.note_verified_old_region_empty(uid,map_id=map_id,
                source=source,frame_id=frame_id,proof=visibility.proof)
        return visibility

    def _old_free_views(self,t):
        rows=self.store.db.execute('''SELECT evidence FROM old_region_vacancy_v2
            WHERE map_id=? AND object_id=? AND independent_ray_validation=1
            ORDER BY source,frame_id''',(t.map_id,t.object_id)).fetchall()
        # Repeated frames from the SAME camera location are correlated; at
        # least 15 cm viewpoint baseline is required per independent proof.
        independent=[]
        for row in rows:
            proof=json.loads(row['evidence'])
            origin=np.asarray(proof['origin_map'],float)
            if all(np.linalg.norm(origin-prev)>=.15 for prev in independent):
                independent.append(origin)
        return len(independent)

    def _best_candidate(self,o):
        tracks=[t for t in self.store.tracks()
                if t.state=='confirmed' and t.map_id==o.map_id
                and t.map_version==o.map_version
                and self.memory._category(t.label)==self.memory._category(o.label)
                and t.last_seen<o.timestamp]
        ranked=[]
        for t in tracks:
            distance=float(np.linalg.norm(t.center-o.center))
            cap=self.memory.DEFAULT_STATIC_REID_RADIUS.get(self.memory._category(t.label),.5)
            if distance<=cap or distance>self.max_relocation_m:continue
            if o.timestamp-t.last_seen<4:continue
            sim=self.memory._gallery_similarity(t,o)
            if sim is None or sim<self.min_visual:continue
            gallery=t.attributes.get('memory_v2',{}).get('prototypes',[])
            if len(gallery)<2:continue
            # An old object simultaneously observed in this frame cannot have
            # moved to a second new location merely because features match.
            covis=self.store.db.execute('''SELECT 1 FROM observations
              WHERE map_id=? AND source=? AND frame_id=? AND object_id=? LIMIT 1''',
              (o.map_id,o.source,o.frame_id,t.object_id)).fetchone()
            if covis:continue
            ranked.append((float(sim),t))
        ranked.sort(key=lambda x:-x[0])
        if not ranked:return None,None
        if len(ranked)>1 and ranked[0][0]-ranked[1][0]<self.distinctive_margin:
            return None,'AMBIGUOUS_SIMILAR_OLD_OBJECTS'
        return ranked[0],None

    def observe_candidate(self,o):
        """Return PENDING or VERIFIED_RELOCATION while preserving original UUID.

        The caller MUST not first write the same detection to regular memory.
        On success the object location changes atomically; geometry rebuild of
        any old surface layer remains the caller's responsibility.
        """
        candidate,reason=self._best_candidate(o)
        if candidate is None:
            return RelocationDecision(None,reason or 'NO_DISTINCTIVE_RELOCATION',0.,0)
        similarity,t=candidate
        past=self.store.db.execute('''SELECT center,source,frame_id FROM relocation_hypothesis_v2
            WHERE map_id=? AND object_id=? ORDER BY timestamp''',(o.map_id,t.object_id)).fetchall()
        if past:
            centroid=np.median(np.asarray([json.loads(row['center']) for row in past]),axis=0)
            if np.linalg.norm(centroid-o.center)>.23:
                # Two incompatible candidate destinations require explicit review.
                return RelocationDecision(None,'CONFLICTING_NEW_LOCATION',similarity,len(past))
        already=self.store.db.execute('''SELECT 1 FROM relocation_hypothesis_v2
            WHERE map_id=? AND object_id=? AND source=? AND frame_id=?''',
            (o.map_id,t.object_id,o.source,o.frame_id)).fetchone()
        if not already:
            self.store.db.execute('''INSERT INTO relocation_hypothesis_v2
                VALUES (?,?,?,?,?,?,?,?,?,?,?)''',
                (o.map_id,t.object_id,o.source,o.frame_id,o.detection_id,
                 self.memory._payload(o),json.dumps(o.center.tolist()),
                 json.dumps(o.low.tolist()),json.dumps(o.high.tolist()),
                 o.timestamp,similarity))
        points=self.store.db.execute('''SELECT * FROM relocation_hypothesis_v2
            WHERE map_id=? AND object_id=? ORDER BY timestamp''',(o.map_id,t.object_id)).fetchall()
        count=len(points);vacancies=self._old_free_views(t)
        if count<self.min_relocation_views or vacancies<self.min_free_old_views:
            self.store.db.commit()
            return RelocationDecision(None,'RELOCATION_PENDING_NEEDS_EVIDENCE',similarity,count)
        centers=np.asarray([json.loads(row['center']) for row in points])
        lows=np.asarray([json.loads(row['low']) for row in points])
        highs=np.asarray([json.loads(row['high']) for row in points])
        new=np.median(centers,axis=0)
        # Critical: old and new geometry must not form a single enormous AABB.
        old=t.center.copy()
        t.center=new;t.low=np.median(lows,axis=0)
        t.high=np.maximum(np.median(highs,axis=0),t.low+.015)
        t.last_seen=max(t.last_seen,o.timestamp)
        history=t.attributes.setdefault('memory_v2',{}).setdefault('relocation_history',[])
        history.append({'old_center':old.tolist(),'new_center':new.tolist(),
                        'timestamp':o.timestamp,'confirming_views':count,'vacancy_views':vacancies})
        with self.store.db:
            # Assign the ORIGINAL physical UUID to each independently verified
            # relocated observation; keep their immutable 3D camera evidence.
            for row in points:
                data=json.loads(row['evidence_json'])
                observed=Observation(source=row['source'],frame_id=row['frame_id'],
                    detection_id=row['detection_id'],label=data['label'],
                    center=data['center'],low=data['low'],high=data['high'],
                    timestamp=data['timestamp'],appearance=data['appearance'],
                    T_map_cam=data.get('T_original_map_cam'),
                    confidence=data['confidence'],map_id=o.map_id,map_version=o.map_version)
                self.memory._write_update(t,observed,float(row['score']))
                self.memory._record(observed,t.object_id,'relocated_association',float(row['score']))
            t.center=new;t.low=np.median(lows,axis=0)
            t.high=np.maximum(np.median(highs,axis=0),t.low+.015)
            self.store.upsert_track(t)
            self.store.db.execute('''INSERT INTO relocation_commit_v2 VALUES (?,?,?,?,?,?)''',
                                  (o.map_id,t.object_id,o.frame_id,
                                   json.dumps(old.tolist()),json.dumps(new.tolist()),o.map_version))
            self.store.db.execute('DELETE FROM relocation_hypothesis_v2 WHERE map_id=? AND object_id=?',
                                  (o.map_id,t.object_id))
            self.store.db.execute('DELETE FROM old_region_vacancy_v2 WHERE map_id=? AND object_id=?',
                                  (o.map_id,t.object_id))
        return RelocationDecision(t.object_id,'RELOCATION_CONFIRMED_NEEDS_SURFACE_REBUILD',similarity,count)
