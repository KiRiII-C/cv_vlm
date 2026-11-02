"""B3 z-buffered visible surface footprint from REAL RGB-D masked 3D points.

Pure measured-depth algorithm. Optional small image-space splat supports
regularizing sparse projections; every splatted pixel is RECHECKED against
the destination depth and cannot create observation in missing-depth areas.
Not a full volumetric mesh renderer; compare on physical instance GT before
using this as a hard object-ID gate.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import minimum_filter
from .shared_rgbd_geometry import SharedRGBDFrame, Object3DObservation, checked_pose


@dataclass(frozen=True)
class FootprintEvidence:
    sampled_points:int
    projected_pixels:int
    supported_pixels:int
    occluded_pixels:int
    free_space_conflict_pixels:int
    unobserved_projected_pixels:int
    support_ratio:float|None
    visible_mask_iou:float|None
    precision_on_supported_surface:float|None
    target_recall_on_supported_surface:float|None
    radius_px:int
    status:str


def project_visible_footprint(source:Object3DObservation,
                              destination:SharedRGBDFrame,
                              target_mask=None, *,
                              radius_px:int=0,
                              absolute_tol_m:float=.05,
                              relative_tol:float=.025):
    """Render 3D points in camera pose (SE3), min-depth zbuffer, then test depth.

    The numerator includes only surface pixels whose predicted Z is
    consistent with REAL observed destination Z. Occluded surface pixels
    never become false removal evidence. A low IoU is NOT sufficient
    evidence of a new physical identity.
    """
    if not isinstance(radius_px,int) or not 0<=radius_px<=3:
        raise ValueError("Only radii 0..3 are allowed to prevent geometric overfill")
    if absolute_tol_m<=0 or relative_tol<0:
        raise ValueError("Invalid camera depth uncertainty")
    if destination.T_map_camera is None:
        raise ValueError("Destination global pose unavailable")
    old=checked_pose(source.T_map_camera)
    new=destination.T_map_camera
    xyz=source.points_camera
    world=xyz@old[:3,:3].T+old[:3,3]
    query=(world-new[:3,3])@new[:3,:3]
    z=query[:,2]
    safe=np.isfinite(query).all(axis=1)&(z>.02)
    h,w=destination.depth_m.shape
    u=np.zeros(len(z),dtype=np.int64);v=np.zeros(len(z),dtype=np.int64)
    u[safe]=np.rint(query[safe,0]*destination.K.fx/z[safe]+destination.K.cx).astype(np.int64)
    v[safe]=np.rint(query[safe,1]*destination.K.fy/z[safe]+destination.K.cy).astype(np.int64)
    selected=safe&(u>=0)&(u<w)&(v>=0)&(v<h)
    zbuffer=np.full(h*w,np.inf,dtype=float)
    np.minimum.at(zbuffer,v[selected]*w+u[selected],z[selected])
    depth_pred=zbuffer.reshape(h,w)
    if radius_px:
        # Nearest predicted object surface in a bounded pixel neighborhood.
        # This does not override incoming depth, which is tested below.
        depth_pred=minimum_filter(depth_pred,size=2*radius_px+1,mode="constant",cval=np.inf)
    footprint=np.isfinite(depth_pred)
    observable=footprint&destination.valid
    tolerance=np.maximum(absolute_tol_m,relative_tol*destination.depth_m)
    delta=depth_pred-destination.depth_m
    supported=observable&(np.abs(delta)<=tolerance)
    occluded=observable&(delta>tolerance)
    conflict=observable&(delta<(-tolerance))
    n_obs=int(observable.sum());n_support=int(supported.sum())
    # With no observable predicted pixels, return missing, not a forced zero.
    ratio=n_support/n_obs if n_obs else None
    iou=None;precision=None;recall=None
    if target_mask is not None:
        target=np.asarray(target_mask,dtype=bool)
        if target.shape!=(h,w):
            raise ValueError("Target YOLO mask and depth resolution mismatch")
        target=target&destination.valid
        overlap=int((target&supported).sum())
        area_target=int(target.sum())
        union=n_support+area_target-overlap
        iou=overlap/union if union else None
        precision=overlap/n_support if n_support else None
        recall=overlap/area_target if area_target else None
    return FootprintEvidence(
        sampled_points=len(xyz),projected_pixels=int(footprint.sum()),
        supported_pixels=n_support,occluded_pixels=int(occluded.sum()),
        free_space_conflict_pixels=int(conflict.sum()),
        unobserved_projected_pixels=int((footprint&~destination.valid).sum()),
        support_ratio=ratio,visible_mask_iou=iou,
        precision_on_supported_surface=precision,
        target_recall_on_supported_surface=recall,radius_px=radius_px,
        status="OBSERVED" if n_obs>=8 else "INSUFFICIENT_VIEW")
