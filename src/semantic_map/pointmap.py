"""Voxelized observed surface map (NOT free-space occupancy/TSDF/ESDF)."""
import sqlite3
from pathlib import Path
import numpy as np
from .geometry import masked_pointcloud


class SurfaceVoxelMap:
    def __init__(self, db_path, voxel_m=.05):
        self.db=sqlite3.connect(str(db_path))
        self.db.execute('''CREATE TABLE IF NOT EXISTS voxels (
            x INTEGER,y INTEGER,z INTEGER,obs INTEGER NOT NULL,
            PRIMARY KEY(x,y,z))''')
        self.db.execute('''CREATE TABLE IF NOT EXISTS integrated_frames (
            source TEXT,frame_id TEXT,PRIMARY KEY(source,frame_id))''')
        self.db.commit();self.voxel_m=voxel_m

    def add_frame(self,depth,K,T_map_cam,source,frame_id,stride=3):
        if self.db.execute('SELECT 1 FROM integrated_frames WHERE source=? AND frame_id=?',(source,frame_id)).fetchone():
            return 0
        mask=np.zeros_like(depth,dtype=bool);mask[::stride,::stride]=True
        pts=masked_pointcloud(depth,mask,K,T_map_cam)
        vox=np.unique(np.floor(pts/self.voxel_m).astype('int32'),axis=0)
        with self.db:
            self.db.executemany('''INSERT INTO voxels VALUES (?,?,?,1)
                ON CONFLICT(x,y,z) DO UPDATE SET obs=obs+1''',
                [tuple(map(int,row)) for row in vox])
            self.db.execute('INSERT INTO integrated_frames VALUES (?,?)',(source,frame_id))
        return len(vox)

    def export_ply(self,path,min_observations=1):
        rows=self.db.execute('SELECT x,y,z FROM voxels WHERE obs>=?',(min_observations,)).fetchall()
        path=Path(path);path.parent.mkdir(exist_ok=True,parents=True)
        with path.open('w') as f:
            f.write(f'ply\nformat ascii 1.0\nelement vertex {len(rows)}\n')
            f.write('property float x\nproperty float y\nproperty float z\nend_header\n')
            for r in rows:f.write('%.4f %.4f %.4f\n'%tuple((n+.5)*self.voxel_m for n in r))
        return len(rows)

    def close(self):self.db.close()
