"""Evidence-sourced object memory with persistent UUIDs, risk-aware association,
view prototypes and atomic global-pose corrections.

The module accepts 3D observations from any detector. GT physical identities
MUST NOT be supplied to this class. It does not claim perfect data association.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict
from uuid import uuid5,NAMESPACE_URL
import json, math
import numpy as np
from scipy.optimize import linear_sum_assignment
from .models import Track
from .association import overlap_3d


@dataclass(frozen=True)
class MemoryDecision:
    detection_id: str
    object_id: str | None
    status: str
    confidence: float


def _normalize(v):
    if v is None:return None
    x=np.asarray(v,dtype=float)
    n=np.linalg.norm(x)
    return x/n if n>1e-8 else None


def _box_corners(lo,hi):
    return np.asarray([[x,y,z] for x in (lo[0],hi[0])
                       for y in (lo[1],hi[1]) for z in (lo[2],hi[2])],float)


def _transform(T,xyz):
    return np.asarray(xyz)@np.asarray(T)[:3,:3].T+np.asarray(T)[:3,3]


def _validate_pose(T):
    a=np.asarray(T,dtype=float)
    if (a.shape!=(4,4) or not np.isfinite(a).all() or
        not np.allclose(a[3],[0,0,0,1],atol=1e-5) or
        not np.allclose(a[:3,:3].T@a[:3,:3],np.eye(3),atol=1e-3) or
        abs(np.linalg.det(a[:3,:3])-1)>1e-3):
        raise ValueError('Invalid SE(3) camera pose')
    return a


class ObjectMemoryV2:
    """Hierarchical instance memory, source evidence and scoped object records.

    - No GT observations or identity labels enter matching.
    - Only unique, confident 3D matches become stable evidence/prototypes.
    - No appearance-only distant merge of lookalike instances.
    - Failed/ambiguous observations stay as unresolved evidence, not fake UUIDs.
    - A new pose graph can rebuild geometry without changing original UUIDs.
    """
    def __init__(self, store, *, min_hits=2, min_score=.65,
                 ambiguous_margin=.075, max_prototypes=6,
                 candidate_floor=.40, visual_weight=.16,
                 conservative=True):
        self.store=store
        self.min_hits=int(min_hits)
        self.min_score=float(min_score)
        self.ambiguous_margin=float(ambiguous_margin)
        self.max_prototypes=int(max_prototypes)
        self.candidate_floor=float(candidate_floor)
        self.visual_weight=float(visual_weight)
        self.conservative=bool(conservative)
        if not (0<self.candidate_floor<self.min_score<1):raise ValueError('Bad matching thresholds')
        self._schema()

    def _schema(self):
        self.store.db.executescript('''
        CREATE TABLE IF NOT EXISTS object_evidence_v2 (
          map_id TEXT NOT NULL, source TEXT NOT NULL, frame_id TEXT NOT NULL,
          detection_id TEXT NOT NULL, map_version INTEGER NOT NULL,
          timestamp REAL NOT NULL, object_id TEXT, status TEXT NOT NULL,
          quality REAL NOT NULL, payload TEXT NOT NULL,
          PRIMARY KEY(map_id,source,frame_id,detection_id)
        );
        CREATE INDEX IF NOT EXISTS idx_v2_evidence_uuid
          ON object_evidence_v2(object_id,map_id,map_version);
        CREATE TABLE IF NOT EXISTS memory_revision_v2 (
          revision_id INTEGER PRIMARY KEY AUTOINCREMENT,
          map_id TEXT NOT NULL, from_version INTEGER NOT NULL,
          to_version INTEGER NOT NULL, corrected_frame_count INTEGER NOT NULL,
          affected_objects INTEGER NOT NULL, reason TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS identity_review_v2 (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          map_id TEXT NOT NULL, source TEXT NOT NULL,
          frame_id TEXT NOT NULL, detection_id TEXT NOT NULL,
          reason TEXT NOT NULL, candidates TEXT NOT NULL,
          resolved INTEGER NOT NULL DEFAULT 0
        );''')
        self.store.db.commit()

    @staticmethod
    def _category(label):
        mapping={'dining table':'table','desk':'table','couch':'sofa',
                 'computer monitor':'tv','monitor':'tv'}
        return mapping.get(label.lower().strip(),label.lower().strip())

    def _gallery_similarity(self,t,obs):
        if obs.appearance is None:return None
        gallery=t.attributes.get('memory_v2',{}).get('prototypes',[])
        sims=[]
        for p in gallery:
            q=np.asarray(p['vector'],float)
            if q.shape==obs.appearance.shape:
                sims.append(float(np.clip(q@obs.appearance,-1,1)))
        if not sims and t.appearance is not None and t.appearance.shape==obs.appearance.shape:
            sims.append(float(t.appearance@obs.appearance))
        if not sims:return None
        # Stable prototypes rewarded, one accidentally contaminated prototype
        # cannot alone overrule spatial evidence.
        return max(sims) if len(sims)==1 else .65*max(sims)+.35*float(np.median(sims))

    def _score(self,t,o):
        if t.map_id!=o.map_id or t.map_version!=o.map_version:return -1.
        if self._category(t.label)!=self._category(o.label):return -1.
        distance=float(np.linalg.norm(t.center-o.center))
        gap=np.maximum(0.,np.maximum(t.low,o.low)-np.minimum(t.high,o.high))
        surface_gap=float(np.linalg.norm(gap))
        size=max(.18,min(1.2,float(np.linalg.norm(t.high-t.low)*.48)))
        # Cropped instance masks may have different centers but their measured
        # 3D surfaces must be close. Lookalike objects 1m apart are rejected.
        if surface_gap>.21 or distance>max(.52,1.55*size):return -1.
        iou=overlap_3d(t.low,t.high,o.low,o.high)
        spatial=.56*math.exp(-surface_gap/.075)+.25*math.exp(-distance/max(.14,.72*size))+.19*iou
        cos=self._gallery_similarity(t,o)
        if cos is not None:
            if cos<.20 and iou<.64:return -1.
            score=(1-self.visual_weight)*spatial+self.visual_weight*max(0.,cos)
        else:score=spatial
        # Low-quality observations may link but cannot contaminate prototypes.
        return float(np.clip(score,0,1))

    def _new_track(self,o):
        uid='obj_'+uuid5(NAMESPACE_URL,'object_memory_v2|'+str(o.key)).hex[:16]
        attrs={'memory_v2':{'prototypes':[],'centers':[], 'semantic_counts':{},
                            'reliable_observations':0,'last_reliable':None,
                            'lifecycle':'tentative','visible_misses':0}}
        t=Track(object_id=uid,label=self._category(o.label),center=o.center.copy(),
                low=o.low.copy(),high=o.high.copy(),appearance=o.appearance.copy() if o.appearance is not None else None,
                hits=0,first_seen=o.timestamp,last_seen=o.timestamp,state='tentative',
                description=o.description,map_version=o.map_version,map_id=o.map_id,attributes=attrs)
        return t

    def _write_update(self,t,o,score,created=False):
        m=t.attributes.setdefault('memory_v2',{'prototypes':[],'centers':[],
             'semantic_counts':{},'reliable_observations':0,'last_reliable':None,
             'lifecycle':'tentative','visible_misses':0})
        quality=float(o.confidence)*(float(score) if not created else .85)
        reliable=bool(o.confidence>=.62 and (created or score>=.72))
        # Preserve location as robust running center, not an ever-growing union
        # of all AABBs (the old implementation makes accidental overlap easier).
        if t.hits:
            ratio=.23 if reliable else .08
            t.center=(1-ratio)*t.center+ratio*o.center
            t.low=(1-ratio)*t.low+ratio*o.low
            t.high=(1-ratio)*t.high+ratio*o.high
        m['centers'].append(o.center.tolist())
        if len(m['centers'])>12:m['centers']=m['centers'][-12:]
        if len(m['centers'])>=5 and reliable:
            recent=np.asarray(m['centers'])
            median=np.median(recent,axis=0)
            t.center=.65*t.center+.35*median
        t.high=np.maximum(t.high,t.low+.015)
        t.hits+=1;t.last_seen=max(t.last_seen,o.timestamp)
        m['semantic_counts'][self._category(o.label)]=m['semantic_counts'].get(self._category(o.label),0)+1
        if reliable:
            m['reliable_observations']+=1;m['last_reliable']=float(o.timestamp)
        if reliable and o.appearance is not None:
            gallery=m['prototypes']
            features=[np.asarray(p['vector'],dtype=float) for p in gallery]
            similarities=[float(a@o.appearance) if a.shape==o.appearance.shape else -1 for a in features]
            if similarities and max(similarities)>.88:
                k=int(np.argmax(similarities));p=gallery[k]
                # Reliability-weighted update of a known, stable view.
                v=_normalize(np.asarray(p['vector'])*.88+o.appearance*.12)
                p['vector']=v.tolist();p['count']+=1;p['quality']=max(p['quality'],quality)
            elif len(gallery)<self.max_prototypes:
                gallery.append({'vector':o.appearance.tolist(),'count':1,'quality':quality})
            else:
                # Replace weakest prototype only for higher-confidence evidence.
                weak=int(np.argmin([p['quality'] for p in gallery]))
                if quality>gallery[weak]['quality']+.13:
                    gallery[weak]={'vector':o.appearance.tolist(),'count':1,'quality':quality}
        if t.appearance is None and o.appearance is not None:t.appearance=o.appearance.copy()
        elif t.appearance is not None and o.appearance is not None and reliable:
            if t.appearance.shape==o.appearance.shape:
                z=_normalize(.92*t.appearance+.08*o.appearance)
                if z is not None:t.appearance=z
        if o.description and not t.description:t.description=o.description
        if t.hits>=self.min_hits:t.state='confirmed'
        m['lifecycle']=t.state
        return quality

    def _payload(self,o):
        data={'label':o.label,'center':o.center.tolist(),'low':o.low.tolist(),
              'high':o.high.tolist(),'confidence':o.confidence,
              'appearance':o.appearance.tolist() if o.appearance is not None else None,
              'description':o.description,'timestamp':o.timestamp}
        if o.T_map_cam is not None:
            T=_validate_pose(o.T_map_cam)
            inv=np.linalg.inv(T)
            data['T_original_map_cam']=T.tolist()
            data['center_cam']=_transform(inv,o.center).tolist()
            data['box_corners_cam']=_transform(inv,_box_corners(o.low,o.high)).tolist()
        return json.dumps(data,ensure_ascii=False)

    def _record(self,o,uid,status,score):
        self.store.write_observation(o,uid,status)
        self.store.db.execute('''INSERT INTO object_evidence_v2
          (map_id,source,frame_id,detection_id,map_version,timestamp,object_id,status,quality,payload)
          VALUES (?,?,?,?,?,?,?,?,?,?)''',
          (o.map_id,o.source,o.frame_id,o.detection_id,o.map_version,o.timestamp,
           uid,status,max(0.,float(score)),self._payload(o)))

    def _review(self,o,reason,tracks_with_scores):
        self.store.db.execute('''INSERT INTO identity_review_v2
          (map_id,source,frame_id,detection_id,reason,candidates)
          VALUES (?,?,?,?,?,?)''',
          (o.map_id,o.source,o.frame_id,o.detection_id,reason,
           json.dumps(tracks_with_scores)))

    def process_frame(self,observations,prior_ids=None):
        if not observations:return []
        prior_ids=prior_ids or {}
        scene={(o.map_id,o.map_version,o.source,o.frame_id) for o in observations}
        if len(scene)!=1:raise ValueError('One map/version/source/frame per update')
        if len({o.key for o in observations})!=len(observations):
            raise ValueError('Duplicate detection ID in frame')
        result=[None]*len(observations)
        fresh=[]
        for i,o in enumerate(observations):
            prior=self.store.seen(o)
            if prior is not None:
                result[i]=MemoryDecision(o.detection_id,prior['object_id'],'already_processed',1.)
            else:fresh.append((i,o))
        if not fresh:return result
        unique=[];duplicates=[]
        for i,o in sorted(fresh,key=lambda item:(-item[1].confidence,item[1].detection_id)):
            dup=next((j for j,k in unique if self._category(o.label)==self._category(k.label)
                     and overlap_3d(o.low,o.high,k.low,k.high)>.93
                     and np.linalg.norm(o.center-k.center)<.065
                     and (o.appearance is None or k.appearance is None or
                          (o.appearance.shape==k.appearance.shape and o.appearance@k.appearance>.96))),None)
            if dup is None:unique.append((i,o))
            else:duplicates.append((i,o,dup))
        unique.sort(key=lambda p:p[0])
        tracks=sorted(self.store.tracks(),key=lambda x:x.object_id)
        scores=np.full((len(unique),len(tracks)),-1.,dtype=float)
        for i,(_,o) in enumerate(unique):
            for j,t in enumerate(tracks):
                s=self._score(t,o)
                if (s>=self.min_score and prior_ids.get(o.detection_id)==t.object_id and
                    t.hits>=3 and 0<=o.timestamp-t.last_seen<=3 and s>=.80):
                    s=min(1.,s+.01)
                scores[i,j]=s
        # Explicit unmatched dummy columns avoid forcing invalid pairings.
        proposals={}
        if len(unique) and len(tracks):
            padded=np.concatenate([scores,np.full((len(unique),len(unique)),self.min_score-1e-4)],axis=1)
            rows,cols=linear_sum_assignment(-padded)
            proposals={int(i):int(j) for i,j in zip(rows,cols)
                       if j<len(tracks) and scores[i,j]>=self.min_score}
        safe={}
        for i,(_,o) in enumerate(unique):
            j=proposals.get(i)
            if j is None:continue
            score=scores[i,j]
            row=np.sort(scores[i,scores[i]>=self.candidate_floor])[::-1]
            col=np.sort(scores[:,j][scores[:,j]>=self.candidate_floor])[::-1]
            row_danger=(len(row)>1 and row[0]-row[1]<self.ambiguous_margin)
            col_danger=(len(col)>1 and col[0]-col[1]<self.ambiguous_margin)
            if row_danger or col_danger:continue
            safe[i]=j
        for i,(index,o) in enumerate(unique):
            if i in safe:
                j=safe[i];t=tracks[j]
                strength=float(scores[i,j]);quality=self._write_update(t,o,strength)
                self.store.upsert_track(t);self._record(o,t.object_id,'matched',quality)
                result[index]=MemoryDecision(o.detection_id,t.object_id,
                    'confirmed' if t.state=='confirmed' else 'tentative',strength)
                continue
            row=scores[i] if len(tracks) else np.empty(0)
            plausible=[(tracks[j].object_id,round(float(z),4)) for j,z in enumerate(row)
                       if z>=self.candidate_floor]
            plausible.sort(key=lambda p:-p[1])
            if plausible:
                status='ambiguous' if row.max()>=self.min_score else 'deferred'
                self._record(o,None,status,float(row.max()))
                self._review(o,status,plausible[:5])
                result[index]=MemoryDecision(o.detection_id,None,status,float(row.max()))
            else:
                t=self._new_track(o)
                quality=self._write_update(t,o,1.,created=True)
                self.store.upsert_track(t);self._record(o,t.object_id,'created_tentative',quality)
                result[index]=MemoryDecision(o.detection_id,t.object_id,'tentative',0.)
        for i,o,target in duplicates:
            original=result[target]
            self._record(o,original.object_id,'duplicate_detection',1.)
            result[i]=MemoryDecision(o.detection_id,original.object_id,'duplicate_detection',1.)
        self.store.db.commit()
        return result

    def query_memory(self,label=None,confirmed_only=True):
        out=[]
        for t in self.store.tracks():
            if label and self._category(label)!=self._category(t.label):continue
            if confirmed_only and t.state!='confirmed':continue
            m=t.attributes.get('memory_v2',{})
            out.append({'uuid':t.object_id,'label':t.label,'center_map':t.center.tolist(),
                        'aabb_low':t.low.tolist(),'aabb_high':t.high.tolist(),
                        'state':t.state,'hits':t.hits,'first_seen':t.first_seen,
                        'last_seen':t.last_seen,'map_id':t.map_id,
                        'map_version':t.map_version,'description':t.description,
                        'appearance_prototypes':len(m.get('prototypes',[])),
                        'reliable_observations':m.get('reliable_observations',0),
                        'class_votes':m.get('semantic_counts',{}),
                        'geometry_status':'observed_not_gt_validated'})
        return sorted(out,key=lambda item:item['uuid'])

    def review_queue(self,map_id=None):
        sql='SELECT * FROM identity_review_v2 WHERE resolved=0'
        rows=self.store.db.execute(sql+(' AND map_id=?' if map_id else ''),
                                   (map_id,) if map_id else ()).fetchall()
        return [dict(row) for row in rows]

    def propose_duplicate_merges(self,min_overlap=.65,min_cos=.96):
        tracks=self.store.tracks();candidates=[]
        for i,a in enumerate(tracks):
            for b in tracks[i+1:]:
                if (a.map_id!=b.map_id or a.map_version!=b.map_version or
                    self._category(a.label)!=self._category(b.label)):continue
                if self.store.sightings_cooccur(a.object_id,b.object_id):continue
                overlap=overlap_3d(a.low,a.high,b.low,b.high)
                if overlap<min_overlap:continue
                if a.appearance is None or b.appearance is None:continue
                cosine=float(a.appearance@b.appearance) if a.appearance.shape==b.appearance.shape else -1.
                if cosine>=min_cos:
                    candidates.append({'first':a.object_id,'second':b.object_id,
                                       'overlap_3d':overlap,'appearance_cosine':cosine,
                                       'action':'MANUAL_REVIEW_NO_AUTOMATIC_MERGE'})
        return candidates

    def nearby_graph(self,radius=2.0,map_id=None):
        tracks=[t for t in self.store.tracks() if t.state=='confirmed' and
                (map_id is None or t.map_id==map_id)]
        edges=[]
        for i,a in enumerate(tracks):
            for b in tracks[i+1:]:
                if a.map_id!=b.map_id or a.map_version!=b.map_version:continue
                d=float(np.linalg.norm(a.center-b.center))
                if d<=radius:
                    edges.append({'source':a.object_id,'target':b.object_id,
                                  'relation':'near','distance_m':round(d,4),
                                  'coordinate_frame':'map'})
        return {'objects':self.query_memory(), 'edges':edges}

    def correct_pose_graph(self,pose_by_frame:dict,*,map_id='default',new_version:int,reason='SLAM loop closure'):
        """Atomically reproject EVERY assigned 3D observation with saved local evidence.

        Reject incomplete camera poses, invalid SE3, missing object evidence, or
        mixed-version objects. UUIDs survive, but this does not re-associate
        incorrect original identities. Caller must re-evaluate identity gates.
        """
        tracks=[t for t in self.store.tracks() if t.map_id==map_id]
        if not tracks:raise ValueError('No objects for map')
        versions={t.map_version for t in tracks}
        if len(versions)!=1:raise ValueError('Mixed old pose versions')
        old_version=next(iter(versions))
        if new_version<=old_version:raise ValueError('Pose version must increment')
        rows=self.store.db.execute('''SELECT * FROM object_evidence_v2
           WHERE map_id=? AND object_id IS NOT NULL AND map_version=?
           ORDER BY timestamp,frame_id,detection_id''',(map_id,old_version)).fetchall()
        if not rows:raise ValueError('Cannot reproject without immutable evidence')
        updated=defaultdict(list)
        transformed_centers={}
        frames=set()
        # Validate every input BEFORE writing to SQLite.
        for r in rows:
            payload=json.loads(r['payload'])
            if 'box_corners_cam' not in payload or 'T_original_map_cam' not in payload:
                raise ValueError('Missing metric camera-frame evidence: atomic rebuild aborted')
            key=(r['source'],r['frame_id'])
            if key not in pose_by_frame:
                raise ValueError(f'Missing optimized pose {key}: atomic rebuild aborted')
            T=_validate_pose(pose_by_frame[key])
            corners=_transform(T,np.asarray(payload['box_corners_cam'],float))
            center=_transform(T,np.asarray(payload['center_cam'],float))
            updated[r['object_id']].append((center,corners.min(axis=0),corners.max(axis=0)))
            transformed_centers[r['map_id'],r['source'],r['frame_id'],r['detection_id']]=center
            frames.add(key)
        used={t.object_id for t in tracks}
        if not used.issubset(updated):
            raise ValueError('Not all active tracks have pose-correctable evidence')
        with self.store.db:
            for t in tracks:
                measurements=updated[t.object_id]
                centers=np.asarray([x[0] for x in measurements])
                t.center=np.median(centers,axis=0)
                t.low=np.median([x[1] for x in measurements],axis=0)
                t.high=np.maximum(np.median([x[2] for x in measurements],axis=0),t.low+.015)
                t.map_version=int(new_version)
                t.attributes.setdefault('memory_v2',{})['centers']=centers[-12:].tolist()
                t.attributes['memory_v2']['pose_revision_required_reassociation']=True
                self.store.upsert_track(t)
            for key,center in transformed_centers.items():
                mid,source,frame,det=key
                self.store.db.execute('''UPDATE observations SET center=?,map_version=?
                    WHERE map_id=? AND source=? AND frame_id=? AND detection_id=?''',
                    (json.dumps(center.tolist()),new_version,mid,source,frame,det))
            self.store.db.execute('''UPDATE object_evidence_v2 SET map_version=?
                WHERE map_id=? AND map_version=?''',(new_version,map_id,old_version))
            self.store.db.execute('''INSERT INTO memory_revision_v2
                (map_id,from_version,to_version,corrected_frame_count,affected_objects,reason)
                VALUES (?,?,?,?,?,?)''',(map_id,old_version,new_version,len(frames),len(tracks),reason))
        return {'status':'REPROJECTED_NEEDS_REASSOCIATION','from_version':old_version,
                'to_version':new_version,'objects_preserved':len(tracks),
                'observation_frames_corrected':len(frames),
                'warning':'Association UUIDs preserved, but identity correctness after loop closure has not been revalidated.'}
