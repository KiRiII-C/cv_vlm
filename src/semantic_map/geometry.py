"""Metric backprojection from RGB-D instance masks and calibrated camera poses."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .models import Observation


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    depth_unit_m: float = 0.001

    def __post_init__(self):
        if min(self.fx,self.fy,self.width,self.height,self.depth_unit_m) <= 0:
            raise ValueError("Invalid camera calibration")


def masked_pointcloud(depth, mask, K: Intrinsics, T_map_cam, min_m=0.2, max_m=8.0,
                      max_points=30000):
    """Backproject aligned depth to map frame. T_map_cam maps optical XYZ to map XYZ."""
    depth = np.asarray(depth)
    mask = np.asarray(mask).astype(bool)
    T = np.asarray(T_map_cam, dtype=np.float64)
    if depth.shape != (K.height,K.width) or mask.shape != depth.shape:
        raise ValueError("Depth/mask resolution mismatch")
    if T.shape != (4,4) or not np.isfinite(T).all():
        raise ValueError("Expected 4x4 finite T_map_cam")
    z = depth.astype(np.float64) * K.depth_unit_m
    valid = mask & np.isfinite(z) & (z > min_m) & (z < max_m)
    yy,xx = np.where(valid)
    if len(xx) == 0:
        return np.empty((0,3), np.float64)
    if len(xx) > max_points:
        ix = np.linspace(0,len(xx)-1,max_points).astype(int)
        xx,yy = xx[ix],yy[ix]
    dd = z[yy,xx]
    p_cam = np.column_stack(((xx-K.cx)*dd/K.fx, (yy-K.cy)*dd/K.fy, dd))
    return p_cam @ T[:3,:3].T + T[:3,3]


def make_observation(depth, mask, K: Intrinsics, T_map_cam, *, source, frame_id,
                     detection_id, label, timestamp, appearance=None, description="",
                     confidence=1.0, map_version=0, map_id="default", min_support=25):
    points = masked_pointcloud(depth,mask,K,T_map_cam)
    if len(points)<min_support:
        return None
    # Quantiles suppress stray depth measurements and rough mask boundaries.
    low = np.quantile(points,.05,axis=0)
    high = np.quantile(points,.95,axis=0)
    high = np.maximum(high,low+.015)
    center = np.median(points,axis=0)
    # Retain camera-frame evidence to reproject object coordinates after SLAM
    # pose-graph optimization without inventing a new UUID.
    T=np.asarray(T_map_cam,dtype=float)
    center_cam=(center-T[:3,3])@T[:3,:3]
    return Observation(source=source,frame_id=str(frame_id),detection_id=str(detection_id),
                       label=label,center=center,low=low,high=high,timestamp=float(timestamp),
                       appearance=appearance,description=description,confidence=confidence,
                       map_version=map_version,map_id=map_id,
                       center_cam=center_cam,T_map_cam=T)
