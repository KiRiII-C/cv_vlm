"""Revisable per-object voxel evidence from actual RGB-D masks.

This layer is separate from the metric occupancy/navigation map. It is an
identity-indexed surface memory with positive and negative evidence. The
negative-evidence interface is explicit and must only receive points verified
visible from the camera (not merely absent from an unobserved RGB-D frame).
"""
from __future__ import annotations
import hashlib, colorsys, sqlite3, json
from pathlib import Path
import numpy as np


def object_color(uid):
    h=int.from_bytes(hashlib.sha256(str(uid).encode()).digest()[:8],'big')/2**64
    return tuple(int(v*255) for v in colorsys.hsv_to_rgb(h,.78,.96))


class ObjectSurfaceMemoryV2:
    def __init__(self,connection:sqlite3.Connection,voxel_m=.015,minimum_support=2):
        if voxel_m<=0 or minimum_support<1:raise ValueError('Invalid map voxel parameters')
        self.db=connection;self.voxel_m=float(voxel_m)
        self.min_support=int(minimum_support)
        self.db.executescript('''
          CREATE TABLE IF NOT EXISTS object_surface_v2 (
            map_id TEXT NOT NULL,uuid TEXT NOT NULL,
            ix INTEGER NOT NULL,iy INTEGER NOT NULL,iz INTEGER NOT NULL,
            sum_x REAL NOT NULL,sum_y REAL NOT NULL,sum_z REAL NOT NULL,
            sum_r REAL NOT NULL,sum_g REAL NOT NULL,sum_b REAL NOT NULL,
            support INTEGER NOT NULL,negative INTEGER NOT NULL,
            last_evidence REAL NOT NULL,
            PRIMARY KEY(map_id,uuid,ix,iy,iz)
          );
          CREATE INDEX IF NOT EXISTS idx_surface_uuid_v2 ON object_surface_v2(uuid,map_id);
          CREATE TABLE IF NOT EXISTS surface_frame_v2 (
            map_id TEXT NOT NULL,uuid TEXT NOT NULL,source TEXT NOT NULL,
            frame_id TEXT NOT NULL,PRIMARY KEY(map_id,uuid,source,frame_id)
          );''')
        self.db.commit()

    def _indices(self,points):return np.floor(np.asarray(points)/self.voxel_m).astype('int32')

    def integrate_mask(self,uuid,depth,mask,rgb,K,T_map_cam,*,source,frame_id,map_id='default',timestamp=0.,pixel_stride=2,max_depth_m=5.):
        """Fuse observed instance surface pixels; a frame is always idempotent."""
        if pixel_stride<1:raise ValueError('Invalid pixel stride')
        depth=np.asarray(depth);mask=np.asarray(mask,bool);rgb=np.asarray(rgb)
        if mask.shape!=depth.shape or rgb.shape[:2]!=depth.shape or rgb.shape[2]!=3:
            raise ValueError('RGB/mask/depth not pixel-aligned')
        if depth.shape!=(K.height,K.width):raise ValueError('Camera calibration does not match images')
        T=np.asarray(T_map_cam,float)
        if T.shape!=(4,4) or not np.isfinite(T).all():raise ValueError('Invalid T_map_cam')
        existing=self.db.execute('''SELECT 1 FROM surface_frame_v2
            WHERE map_id=? AND uuid=? AND source=? AND frame_id=?''',
            (map_id,uuid,source,str(frame_id))).fetchone()
        if existing:return 0
        sub=np.zeros(mask.shape,bool);sub[::pixel_stride,::pixel_stride]=mask[::pixel_stride,::pixel_stride]
        y,x=np.where(sub)
        if len(x)==0:return 0
        z=depth[y,x].astype(float)*K.depth_unit_m
        good=np.isfinite(z)&(z>.20)&(z<max_depth_m)
        if not np.any(good):return 0
        y=y[good];x=x[good];z=z[good]
        pc=np.column_stack(((x-K.cx)*z/K.fx,(y-K.cy)*z/K.fy,z))
        world=pc@T[:3,:3].T+T[:3,3]
        cols=rgb[y,x].astype(float)
        index=self._indices(world)
        unique,inv=np.unique(index,axis=0,return_inverse=True)
        sumpos=np.column_stack([np.bincount(inv,weights=world[:,c],minlength=len(unique)) for c in range(3)])
        sumrgb=np.column_stack([np.bincount(inv,weights=cols[:,c],minlength=len(unique)) for c in range(3)])
        counts=np.bincount(inv,minlength=len(unique))
        rows=[]
        for (ix,iy,iz),q,clr,cnt in zip(unique,sumpos,sumrgb,counts):
            rows.append((map_id,str(uuid),int(ix),int(iy),int(iz),
                         float(q[0]),float(q[1]),float(q[2]),
                         float(clr[0]),float(clr[1]),float(clr[2]),int(cnt),0,float(timestamp)))
        with self.db:
            self.db.executemany('''INSERT INTO object_surface_v2
               (map_id,uuid,ix,iy,iz,sum_x,sum_y,sum_z,sum_r,sum_g,sum_b,support,negative,last_evidence)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(map_id,uuid,ix,iy,iz) DO UPDATE SET
                 sum_x=sum_x+excluded.sum_x,sum_y=sum_y+excluded.sum_y,sum_z=sum_z+excluded.sum_z,
                 sum_r=sum_r+excluded.sum_r,sum_g=sum_g+excluded.sum_g,sum_b=sum_b+excluded.sum_b,
                 support=support+excluded.support,
                 negative=MAX(0,negative-1),last_evidence=excluded.last_evidence''',rows)
            self.db.execute('INSERT INTO surface_frame_v2 VALUES(?,?,?,?)',
                            (map_id,uuid,source,str(frame_id)))
        return len(world)

    def refute_visible_cells(self,uuid,verified_free_xyz,*,map_id='default',timestamp=0.):
        """Decay ONLY cells whose occupancy was explicitly checked free.

        `verified_free_xyz` must originate from valid ray tracing + pose/TF;
        never call this based on a detector's failure alone.
        """
        pts=np.asarray(verified_free_xyz,float)
        if pts.ndim!=2 or pts.shape[1]!=3 or not np.isfinite(pts).all():
            raise ValueError('Expected N x 3 verified free-space coordinates')
        if len(pts)==0:return 0
        keys=np.unique(self._indices(pts),axis=0)
        with self.db:
            self.db.executemany('''UPDATE object_surface_v2
                SET negative=negative+1,last_evidence=?
                WHERE map_id=? AND uuid=? AND ix=? AND iy=? AND iz=?''',
                [(float(timestamp),map_id,str(uuid),int(i),int(j),int(k)) for i,j,k in keys])
        return len(keys)

    def export(self,path,*,map_id='default',instance_colors=True):
        data=self.db.execute('''SELECT uuid,ix,iy,iz,sum_x,sum_y,sum_z,sum_r,sum_g,sum_b,support,negative
          FROM object_surface_v2 WHERE map_id=? ORDER BY uuid,ix,iy,iz''',(map_id,)).fetchall()
        valid=[r for r in data if r[10]>=self.min_support and r[11]<3]
        uuids=sorted({row[0] for row in valid})
        dictionary={uid:i+1 for i,uid in enumerate(uuids)}
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('w') as f:
            f.write('ply\nformat ascii 1.0\n')
            f.write(f'element vertex {len(valid)}\n')
            f.write('property float x\nproperty float y\nproperty float z\n')
            f.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
            f.write('property int instance_index\nproperty int votes\nend_header\n')
            for row in valid:
                uid,ix,iy,iz,sx,sy,sz,sr,sg,sb,count,neg=row
                color=object_color(uid) if instance_colors else tuple(int(round(v/count)) for v in (sr,sg,sb))
                f.write(f'{sx/count:.5f} {sy/count:.5f} {sz/count:.5f} {color[0]} {color[1]} {color[2]} {dictionary[uid]} {count}\n')
        meta={'map_id':map_id,'voxel_size_m':self.voxel_m,'vertices':len(valid),
              'unique_ids':len(uuids),'instances':{str(dictionary[uid]):{'uuid':uid,'rgb':object_color(uid)} for uid in uuids},
              'negative_policy':'Suppress only after 3 independently verified free-space contradictions',
              'source':'actual observed RGBD mask voxels; GT is not implied'}
        path.with_suffix('.instances.json').write_text(json.dumps(meta,indent=2))
        return meta
