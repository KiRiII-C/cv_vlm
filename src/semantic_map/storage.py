"""Durable SQLite object memory; source observations are immutable and idempotent."""
import json,sqlite3
import numpy as np
from .models import Track


def dumps(v):return json.dumps(v.tolist() if isinstance(v,np.ndarray) else v,ensure_ascii=False)


class MemoryStore:
    def __init__(self,path):
        self.db=sqlite3.connect(str(path))
        self.db.row_factory=sqlite3.Row
        self.db.execute('PRAGMA foreign_keys=ON')
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS objects (
                object_id TEXT PRIMARY KEY, label TEXT NOT NULL,
                center TEXT NOT NULL, low TEXT NOT NULL, high TEXT NOT NULL,
                appearance TEXT, hits INTEGER NOT NULL, first_seen REAL NOT NULL,
                last_seen REAL NOT NULL, state TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '', attributes TEXT NOT NULL DEFAULT '{}',
                map_version INTEGER NOT NULL DEFAULT 0, map_id TEXT NOT NULL DEFAULT 'default'
            );
            CREATE TABLE IF NOT EXISTS observations (
                source TEXT NOT NULL, frame_id TEXT NOT NULL, detection_id TEXT NOT NULL,
                object_id TEXT, timestamp REAL NOT NULL, label TEXT NOT NULL,
                center TEXT NOT NULL, map_version INTEGER NOT NULL,
                association_status TEXT NOT NULL, map_id TEXT NOT NULL DEFAULT 'default',
                UNIQUE(map_id,source,frame_id,detection_id),
                FOREIGN KEY(object_id) REFERENCES objects(object_id)
            );
            CREATE TABLE IF NOT EXISTS aliases (
                deprecated_id TEXT PRIMARY KEY, canonical_id TEXT NOT NULL,
                reason TEXT NOT NULL, created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_objects_label ON objects(label,state);
            CREATE INDEX IF NOT EXISTS idx_obs_object ON observations(object_id);
        ''')
        self.db.commit()

    def close(self):self.db.close()

    def tracks(self):
        out=[]
        for r in self.db.execute("SELECT * FROM objects WHERE state != 'merged'"):
            arr=lambda k: np.asarray(json.loads(r[k]),np.float64) if r[k] is not None else None
            out.append(Track(object_id=r['object_id'],label=r['label'],center=arr('center'),
                             low=arr('low'),high=arr('high'),appearance=arr('appearance'),
                             hits=r['hits'],first_seen=r['first_seen'],last_seen=r['last_seen'],
                             state=r['state'],description=r['description'],
                             map_version=r['map_version'],attributes=json.loads(r['attributes']),map_id=r['map_id']))
        return out

    def upsert_track(self,t):
        self.db.execute('''INSERT INTO objects
        (object_id,label,center,low,high,appearance,hits,first_seen,last_seen,state,description,attributes,map_version,map_id)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(object_id) DO UPDATE SET
          center=excluded.center, low=excluded.low, high=excluded.high,
          appearance=excluded.appearance, hits=excluded.hits, last_seen=excluded.last_seen,
          state=excluded.state,description=excluded.description,
          attributes=excluded.attributes,map_version=excluded.map_version,
          map_id=excluded.map_id''',
        (t.object_id,t.label,dumps(t.center),dumps(t.low),dumps(t.high),
         dumps(t.appearance) if t.appearance is not None else None,t.hits,
         t.first_seen,t.last_seen,t.state,t.description,dumps(t.attributes),t.map_version,t.map_id))

    def seen(self,o):
        r=self.db.execute('SELECT object_id,association_status FROM observations WHERE map_id=? AND source=? AND frame_id=? AND detection_id=?',o.key).fetchone()
        return dict(r) if r else None

    def write_observation(self,o,object_id,status):
        self.db.execute('''INSERT INTO observations
        (source,frame_id,detection_id,object_id,timestamp,label,center,map_version,association_status,map_id)
        VALUES (?,?,?,?,?,?,?,?,?,?)''',(o.source,o.frame_id,o.detection_id,object_id,o.timestamp,o.label,dumps(o.center),o.map_version,status,o.map_id))

    def search(self,label=None,only_confirmed=True):
        ts=self.tracks()
        return [t for t in ts if (not label or label.lower() in (t.label+' '+t.description).lower())
                and (not only_confirmed or t.state=='confirmed')]

    def annotate(self,object_id,description=None,attributes=None):
        row=self.db.execute('SELECT description,attributes FROM objects WHERE object_id=?',(object_id,)).fetchone()
        if row is None:raise KeyError(object_id)
        attrs=json.loads(row['attributes']);attrs.update(attributes or {})
        self.db.execute('UPDATE objects SET description=?,attributes=? WHERE object_id=?',
                        (description if description is not None else row['description'],dumps(attrs),object_id))
        self.db.commit()

    def sightings_cooccur(self,first,second):
        a=self.db.execute('''SELECT 1 FROM observations a JOIN observations b
         ON a.map_id=b.map_id AND a.source=b.source AND a.frame_id=b.frame_id
         WHERE a.object_id=? AND b.object_id=? LIMIT 1''',(first,second)).fetchone()
        return a is not None

    def merge(self,keep,remove,reason,when):
        if keep==remove:return
        a={t.object_id:t for t in self.tracks()}
        if keep not in a or remove not in a:raise KeyError('Unknown track')
        if self.sightings_cooccur(keep,remove):raise ValueError('Cannot merge co-visible objects')
        k,r=a[keep],a[remove]
        if k.label!=r.label or k.map_version!=r.map_version or k.map_id!=r.map_id:raise ValueError('Incompatible tracks')
        total=k.hits+r.hits
        k.center=(k.center*k.hits+r.center*r.hits)/total
        k.low=np.minimum(k.low,r.low);k.high=np.maximum(k.high,r.high)
        k.hits=total;k.last_seen=max(k.last_seen,r.last_seen)
        k.first_seen=min(k.first_seen,r.first_seen)
        k.state='confirmed' if k.state=='confirmed' or r.state=='confirmed' else 'tentative'
        if k.appearance is not None and r.appearance is not None:
            z=k.appearance*k.hits+r.appearance*r.hits;k.appearance=z/np.linalg.norm(z)
        self.upsert_track(k)
        self.db.execute('UPDATE observations SET object_id=? WHERE object_id=?',(keep,remove))
        self.db.execute('DELETE FROM objects WHERE object_id=?',(remove,))
        self.db.execute('INSERT INTO aliases VALUES (?,?,?,?)',(remove,keep,reason,when))
        self.db.commit()

    def resolve(self,object_id):
        visited=set()
        while object_id not in visited:
            visited.add(object_id)
            r=self.db.execute('SELECT canonical_id FROM aliases WHERE deprecated_id=?',(object_id,)).fetchone()
            if not r:return object_id
            object_id=r[0]
        raise ValueError('Alias cycle')
