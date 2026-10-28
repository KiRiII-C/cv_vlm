"""Fail-closed RGB-D proof of old-world-AABB vacancy, not a detector absence guess."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class VisibilityEvidence:
    state:str
    tested_rays:int
    free_rays:int
    nearer_rays:int
    matched_rays:int
    valid_fraction:float
    proof:dict|None

def evaluate_visibility(depth,K,T_map_cam,low,high,*,pose_version=0,
                        min_rays=6,depth_margin_m=.15,free_fraction=.95):
    depth=np.asarray(depth);low=np.asarray(low,float);high=np.asarray(high,float)
    T=np.asarray(T_map_cam,float)
    if depth.ndim!=2 or depth.shape!=(K.height,K.width):
        raise ValueError('Depth and camera calibration resolution differ')
    if (low.shape!=(3,) or high.shape!=(3,) or np.any(high<=low) or
        not np.isfinite(low).all() or not np.isfinite(high).all() or
        T.shape!=(4,4) or not np.isfinite(T).all() or
        not np.allclose(T[3],[0,0,0,1],atol=1e-5) or
        not np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-3)):
        raise ValueError('Invalid world AABB or camera SE3')
    corners=np.array([[x,y,z] for x in (low[0],high[0])
                      for y in (low[1],high[1]) for z in (low[2],high[2])])
    camera=(corners-T[:3,3])@T[:3,:3]
    if np.any(camera[:,2]<=.1):
        return VisibilityEvidence('UNKNOWN_CAMERA_INSIDE_OR_BEHIND',0,0,0,0,0.,None)
    far_z=float(camera[:,2].max());near_z=float(camera[:,2].min())
    samples=np.stack(np.meshgrid(*[np.linspace(a,b,6) for a,b in zip(low,high)],
                      indexing='ij'),axis=-1).reshape(-1,3)
    pc=(samples-T[:3,3])@T[:3,:3]
    pc=pc[pc[:,2]>.10]
    if not len(pc):return VisibilityEvidence('UNKNOWN_NO_VISIBLE_SAMPLES',0,0,0,0,0.,None)
    u=np.round(K.fx*pc[:,0]/pc[:,2]+K.cx).astype(int)
    v=np.round(K.fy*pc[:,1]/pc[:,2]+K.cy).astype(int)
    ok=(u>=0)&(u<K.width)&(v>=0)&(v<K.height)
    pix=np.unique(np.column_stack((v[ok],u[ok])),axis=0)
    if len(pix)<min_rays:
        return VisibilityEvidence('UNKNOWN_INSUFFICIENT_VIEW',len(pix),0,0,0,0.,None)
    measured=depth[pix[:,0],pix[:,1]].astype(float)*K.depth_unit_m
    measured=measured[np.isfinite(measured)&(measured>.20)&(measured<10.)]
    if len(measured)<min_rays or len(measured)/len(pix)<.8:
        return VisibilityEvidence('UNKNOWN_MISSING_DEPTH',len(measured),0,0,0,
                                  len(measured)/len(pix),None)
    free=int(np.count_nonzero(measured>far_z+depth_margin_m))
    nearer=int(np.count_nonzero(measured<near_z-depth_margin_m))
    match=len(measured)-free-nearer
    if free/len(measured)>=free_fraction and match==0 and nearer==0:
        proof={'origin_map':T[:3,3].tolist(),
               'raycast_depth_m':float(np.median(measured)),
               'pose_version':int(pose_version),'old_region_cleared':True,
               'tested_rays':len(measured),'free_rays':free,
               'min_far_depth_m':far_z,
               'conservative_volumetric_depth_gate':True}
        return VisibilityEvidence('VERIFIED_FREE',len(measured),free,nearer,match,
                                  len(measured)/len(pix),proof)
    state='OCCUPIED_OR_PARTIAL' if match else 'OCCLUDED' if nearer else 'MIXED_UNCERTAIN'
    return VisibilityEvidence(state,len(measured),free,nearer,match,len(measured)/len(pix),None)
