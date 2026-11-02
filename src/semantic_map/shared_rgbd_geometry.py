"""B0/B1: one organized RGB-D geometry frame shared by mapping and object association.

Geometry is computed ONCE in the camera optical frame. Masked 3D points and
reprojected object footprints use this same metric input, not another
depth-to-points pass. A mask ID is observation evidence, not a physical ID.
No GT pose, instance identity, or fabricated depth enters this module.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .geometry import Intrinsics


def checked_pose(T):
    A=np.asarray(T,dtype=np.float64)
    if A.shape!=(4,4) or not np.isfinite(A).all():
        raise ValueError("T_map_camera must be finite 4x4")
    if not np.allclose(A[3],(0,0,0,1),atol=1e-6):
        raise ValueError("Invalid homogeneous transform")
    R=A[:3,:3]
    if not np.allclose(R.T@R,np.eye(3),atol=2e-3) or abs(np.linalg.det(R)-1)>2e-3:
        raise ValueError("Rotation must be rigid SO(3)")
    return A


@dataclass(frozen=True)
class Object3DObservation:
    detection_id:str
    label:str
    confidence:float
    timestamp:float
    frame_index:int
    T_map_camera:np.ndarray
    center_camera:np.ndarray
    center_map:np.ndarray
    aabb_min_map:np.ndarray
    aabb_max_map:np.ndarray
    point_count:int
    mask_pixel_count:int
    valid_depth_ratio:float
    points_camera:np.ndarray
    # This is an observation-specific ID; NOT the persistent object UUID.
    @property
    def observation_id(self):
        return f"{self.frame_index}:{self.detection_id}"


@dataclass(frozen=True)
class ReprojectionEvidence:
    source_points:int
    points_in_view:int
    projected_pixels:int
    geometrically_visible_pixels:int
    depth_consistency_ratio:float | None
    projected_mask_iou:float | None
    occluded_pixels:int
    free_space_conflict_pixels:int


class SharedRGBDFrame:
    """One depth backprojection per RGB-D frame; masks are zero-copy subsets.

    Optional T_map_camera is the optical frame pose after graph correction.
    A frame without valid odometry can still be built for depth safety but
    must not be integrated into a world map.
    """
    def __init__(self,depth,K:Intrinsics,T_map_camera=None, *,
                 depth_min_m=.2,depth_max_m=6.0):
        depth=np.asarray(depth)
        if depth.shape!=(K.height,K.width) or depth.ndim!=2:
            raise ValueError("Depth must be registered to RGB and match intrinsics")
        if depth_min_m<=0 or depth_max_m<=depth_min_m:
            raise ValueError("Invalid depth gate")
        self.K=K
        self.T_map_camera=checked_pose(T_map_camera) if T_map_camera is not None else None
        self.depth_m=depth.astype(np.float32,copy=False)*float(K.depth_unit_m)
        self.valid=np.isfinite(self.depth_m)&(self.depth_m>depth_min_m)&(self.depth_m<depth_max_m)
        y,x=np.indices(depth.shape,dtype=np.float32)
        z=np.where(self.valid,self.depth_m,0.)
        self.points_camera=np.stack(((x-K.cx)*z/K.fx,(y-K.cy)*z/K.fy,z),axis=-1)
        self.points_camera.setflags(write=False)
        self.valid.setflags(write=False)
        self.depth_m.setflags(write=False)

    def sample_mask(self,mask,*,max_points=None):
        mask=np.asarray(mask,dtype=bool)
        if mask.shape!=self.valid.shape:
            raise ValueError("Segmentation mask is not aligned to registered depth")
        y,x=np.nonzero(mask & self.valid)
        if max_points is not None and max_points<1:
            raise ValueError("max_points must be positive")
        if max_points is not None and len(y)>max_points:
            idx=np.linspace(0,len(y)-1,max_points,dtype=np.int64)
            y,x=y[idx],x[idx]
        return self.points_camera[y,x]

    def object_observation(self,mask,*,detection_id,label,confidence,
                           timestamp,frame_index,min_support=25,max_points=30000):
        if self.T_map_camera is None:
            raise ValueError("Cannot create map-aligned object without measured camera pose")
        if not 0<=confidence<=1 or not label:
            raise ValueError("Invalid detection")
        mask=np.asarray(mask,dtype=bool)
        if mask.shape!=self.valid.shape:
            raise ValueError("RGB mask shape mismatch")
        pix=int(np.count_nonzero(mask))
        if not pix:return None
        support=int(np.count_nonzero(mask & self.valid))
        if support<min_support:return None
        pc=self.sample_mask(mask,max_points=max_points)
        T=self.T_map_camera
        pm=pc@T[:3,:3].T+T[:3,3]
        lo=np.quantile(pm,.05,axis=0);hi=np.quantile(pm,.95,axis=0)
        center_cam=np.median(pc,axis=0)
        center_map=center_cam@T[:3,:3].T+T[:3,3]
        return Object3DObservation(
            detection_id=str(detection_id),label=str(label),confidence=float(confidence),
            timestamp=float(timestamp),frame_index=int(frame_index),
            T_map_camera=T.copy(),center_camera=center_cam,center_map=center_map,
            aabb_min_map=lo,aabb_max_map=np.maximum(hi,lo+.015),
            point_count=int(len(pc)),mask_pixel_count=pix,
            valid_depth_ratio=support/pix,points_camera=pc.copy())


def reproject_object(source:Object3DObservation,destination:SharedRGBDFrame,
                     *,target_mask=None,absolute_tol_m=.05,relative_tol=.025):
    """Project OLD measured object points to a NEW RGB-D frame.

    A z-buffer selects the nearest old point for each pixel. Test against
    current observed depth: equal => supported; behind => occluded; in front
    => free-space conflict. Never infer disappearance from unobserved pixels.
    """
    if destination.T_map_camera is None:
        raise ValueError("Destination must have measured pose")
    if absolute_tol_m<=0 or relative_tol<0:
        raise ValueError("Invalid depth tolerance")
    T_old=checked_pose(source.T_map_camera)
    T_new=destination.T_map_camera
    world=source.points_camera@T_old[:3,:3].T+T_old[:3,3]
    cam=(world-T_new[:3,3])@T_new[:3,:3]
    z=cam[:,2]
    safe=z>.01
    u=np.zeros(len(z),dtype=np.int64);v=np.zeros(len(z),dtype=np.int64)
    u[safe]=np.rint(cam[safe,0]*destination.K.fx/z[safe]+destination.K.cx).astype(np.int64)
    v[safe]=np.rint(cam[safe,1]*destination.K.fy/z[safe]+destination.K.cy).astype(np.int64)
    inview=safe&(u>=0)&(u<destination.K.width)&(v>=0)&(v<destination.K.height)
    uv=v[inview]*destination.K.width+u[inview]
    pred=np.full(destination.valid.size,np.inf,dtype=np.float64)
    np.minimum.at(pred,uv,z[inview])
    seen=np.isfinite(pred).reshape(destination.valid.shape)
    observed=seen & destination.valid
    tol=np.maximum(absolute_tol_m,relative_tol*destination.depth_m)
    residual=pred.reshape(destination.valid.shape)-destination.depth_m
    visible=observed&(np.abs(residual)<=tol)
    occluded=observed&(residual>tol)
    conflict=observed&(residual<(-tol))
    ratio=(float(visible.sum())/int(observed.sum())) if observed.any() else None
    iou=None
    if target_mask is not None:
        mask=np.asarray(target_mask,dtype=bool)
        if mask.shape!=destination.valid.shape:
            raise ValueError("Target mask mismatch")
        # The comparison is only in VERIFIED visible pixels; mask absence
        # under occlusion must not count against identity.
        intersection=int((visible & mask).sum())
        union=int((visible | (mask & destination.valid)).sum())
        iou=intersection/union if union else None
    return ReprojectionEvidence(
        source_points=int(len(z)),points_in_view=int(np.count_nonzero(inview)),
        projected_pixels=int(seen.sum()),
        geometrically_visible_pixels=int(visible.sum()),
        depth_consistency_ratio=ratio,projected_mask_iou=iou,
        occluded_pixels=int(occluded.sum()),
        free_space_conflict_pixels=int(conflict.sum()))
