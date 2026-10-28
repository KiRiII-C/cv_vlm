"""Dense globally fused instance point clouds with immutable identity palette.

This module receives already-associated segmentation masks, depth, RGB and
world-frame camera poses. Every point is derived from the *real mask*, not an
instance bounding-box approximation. Voxel cells are tracked separately per
permanent object ID; background remains uncoloured.
"""
from __future__ import annotations
import colorsys
import hashlib
import json
from pathlib import Path
import numpy as np


def palette_rgb(object_id: str) -> tuple[int, int, int]:
    # Deterministic across scenes, restarts and annotation-order changes.
    k = int.from_bytes(hashlib.sha256(object_id.encode()).digest()[:8], 'big')
    h = (k % 65536) / 65536.0
    s = 0.72 + 0.18 * ((k >> 16) % 100) / 100
    v = 0.84 + 0.13 * ((k >> 24) % 100) / 100
    return tuple(int(round(255*x)) for x in colorsys.hsv_to_rgb(h,s,v))


class InstanceCloud:
    def __init__(self, voxel_m: float=0.012, stride: int=2, depth_limit: float=4.0):
        if voxel_m<=0 or stride<1 or depth_limit<=0:raise ValueError('invalid parameters')
        self.voxel_m=float(voxel_m)
        self.stride=int(stride)
        self.depth_limit=float(depth_limit)
        self.rows={}  # (uuid,ix,iy,iz) -> [sum_x,sum_y,sum_z,count]
        self.frame_visits={}

    def add(self, object_id, mask, depth, intrinsics, T_map_cam):
        """Map camera-aligned mask and depth to metric world point coordinates.

        Depth is integer in native sensor units and intrinsics.depth_unit_m
        converts it to meters (TUM: 1/5000; ScanNet: metadata-dependent).
        """
        if not object_id:return 0
        if depth.ndim!=2 or mask.shape!=depth.shape:raise ValueError('Mask/depth shape mismatch')
        if depth.shape!=(intrinsics.height,intrinsics.width):raise ValueError('Camera intrinsics mismatch')
        T=np.asarray(T_map_cam,float)
        if T.shape!=(4,4) or not np.isfinite(T).all():raise ValueError('Invalid camera pose')
        y,x=np.where(mask[::self.stride,::self.stride])
        if not len(x):return 0
        x=x*self.stride;y=y*self.stride
        z=depth[y,x].astype(float)*intrinsics.depth_unit_m
        good=(z>.20)&(z<self.depth_limit)&np.isfinite(z)
        if not np.any(good):return 0
        x=x[good];y=y[good];z=z[good]
        cam=np.column_stack(((x-intrinsics.cx)*z/intrinsics.fx,
                             (y-intrinsics.cy)*z/intrinsics.fy,z))
        xyz=cam@T[:3,:3].T+T[:3,3]
        indices=np.floor(xyz/self.voxel_m).astype('int32')
        unique,inv=np.unique(indices,axis=0,return_inverse=True)
        sums=np.column_stack([np.bincount(inv,weights=xyz[:,j],minlength=len(unique)) for j in range(3)])
        counts=np.bincount(inv,minlength=len(unique))
        for q,sm,cnt in zip(unique,sums,counts):
            key=(str(object_id),int(q[0]),int(q[1]),int(q[2]))
            if key not in self.rows:self.rows[key]=[0.,0.,0.,0]
            entry=self.rows[key]
            for j in range(3):entry[j]+=float(sm[j])
            entry[3]+=int(cnt)
        self.frame_visits[str(object_id)]=self.frame_visits.get(str(object_id),0)+1
        return len(xyz)

    def export(self, output, min_hits=1, track_filter=None):
        """Write machine-readable binary PLY with per-point instance_index.

        The integer index maps to immutable UUID in <name>.instances.json.
        Filters can exclude tentative tracks without recolouring the rest.
        """
        path=Path(output);path.parent.mkdir(parents=True,exist_ok=True)
        wanted=None if track_filter is None else set(track_filter)
        names=sorted({key[0] for key in self.rows if wanted is None or key[0] in wanted})
        numeric={name:i+1 for i,name in enumerate(names)}
        rec=[]
        for (uid,ix,iy,iz),entry in self.rows.items():
            if uid not in numeric or entry[3]<min_hits:continue
            color=palette_rgb(uid)
            rec.append((*[entry[j]/entry[3] for j in range(3)],*color,numeric[uid],entry[3]))
        dtype=np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1'),('instance_index','<u4'),('votes','<u4')])
        points=np.array(rec,dtype=dtype)
        header=('ply\nformat binary_little_endian 1.0\n'
                f'comment voxel_m {self.voxel_m}\n'
                f'element vertex {len(points)}\n'
                'property float x\nproperty float y\nproperty float z\n'
                'property uchar red\nproperty uchar green\nproperty uchar blue\n'
                'property uint instance_index\nproperty uint votes\nend_header\n')
        with path.open('wb') as f:
            f.write(header.encode());f.write(points.tobytes())
        meta={'voxel_m':self.voxel_m,'pixel_stride':self.stride,'exported_points':len(points),
              'unique_ids':len(names),'instances':{str(numeric[uid]):{'uuid':uid,'rgb':palette_rgb(uid),'frames_observed':self.frame_visits.get(uid,0)} for uid in names},
              'coordinate_frame':'map','input':'measured RGB-D points within predicted 2D instance masks; not ground truth'}
        path.with_suffix('.instances.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
        return meta
