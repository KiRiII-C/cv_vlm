"""C2 measured-depth geometric visibility. No hallucinated physical identity.

An old measured 3D surface is projected using an accepted map->camera pose
into the NEW recorded camera depth. Missing pixels / occlusion != disappearance.
Per-pixel free-space conflict is negative GEOMETRIC evidence, not object UUID
deletion; multiple independent views and reliable relocalization are needed.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation

def evidence(points_map,xyz,quaternion,depth_m,*,K=(517.3,516.5,318.6,255.3),
             pose_xyz_cov=None,min_absolute_tolerance_m=.08):
    pts=np.asarray(points_map,dtype=float)
    pos=np.asarray(xyz,dtype=float)
    quat=np.asarray(quaternion,dtype=float)
    depth=np.asarray(depth_m,dtype=float)
    if pts.ndim!=2 or pts.shape[1]!=3 or depth.ndim!=2 or pos.shape!=(3,) or quat.shape!=(4,):
        raise ValueError("Invalid camera/object dimensions")
    if not (np.isfinite(pts).all() and np.isfinite(pos).all() and np.isfinite(quat).all()):
        raise ValueError("Non-finite points or pose")
    R=Rotation.from_quat(quat/np.linalg.norm(quat)).as_matrix()
    # Map XYZ = R @ camera_XYZ + pos; row-vector inverse is (world-pos) @ R.
    p=(pts-pos)@R
    z=p[:,2]
    h,w=depth.shape
    safe=np.isfinite(z)&(z>.05)
    u=np.zeros(len(z),dtype=np.int64);v=np.zeros(len(z),dtype=np.int64)
    fx,fy,cx,cy=K
    u[safe]=np.rint(p[safe,0]*fx/z[safe]+cx).astype(np.int64)
    v[safe]=np.rint(p[safe,1]*fy/z[safe]+cy).astype(np.int64)
    valid=safe&(u>=0)&(u<w)&(v>=0)&(v<h)
    pred=np.full(h*w,np.inf,dtype=float)
    np.minimum.at(pred,v[valid]*w+u[valid],z[valid])
    pred=pred.reshape(h,w)
    visible=np.isfinite(pred)
    observed=visible&np.isfinite(depth)&(depth>.2)&(depth<6.0)
    sigma=0.
    if pose_xyz_cov is not None:
        variance=np.asarray(pose_xyz_cov,dtype=float)
        if variance.shape!=(3,) or np.any(~np.isfinite(variance)) or np.any(variance<0):
            raise ValueError("Covariance not valid")
        sigma=float(np.sqrt(max(variance)))
    # Expand uncertainty rather than declaring object absence on pose mismatch.
    tolerance=np.maximum(min_absolute_tolerance_m,np.maximum(.035*depth,3*sigma))
    delta=pred-depth
    compatible=observed&(abs(delta)<=tolerance)
    occluded=observed&(delta>tolerance)
    conflict=observed&(delta<(-tolerance))
    n=int(observed.sum())
    outcome={"projected_pixels":int(visible.sum()),"observed_pixels":n,
      "unobserved_pixels":int((visible&~observed).sum()),
      "compatible_pixels":int(compatible.sum()),"occluded_pixels":int(occluded.sum()),
      "free_space_conflict_pixels":int(conflict.sum()),"pose_sigma_max_m":sigma,
      "min_uncertainty_tolerance_m":max(min_absolute_tolerance_m,3*sigma),
      "compatible_fraction":float(compatible.sum()/n) if n else None,
      "occluded_fraction":float(occluded.sum()/n) if n else None,
      "conflict_fraction":float(conflict.sum()/n) if n else None}
    if n<12:
        outcome["state"]="UNKNOWN_NOT_ENOUGH_DEPTH"
    elif outcome["compatible_fraction"]>=.5:
        outcome["state"]="GEOMETRICALLY_COMPATIBLE_NOT_ID_PROOF"
    elif outcome["occluded_fraction"]>=.7:
        outcome["state"]="OCCLUDED_NOT_MISSING"
    elif outcome["conflict_fraction"]>=.7:
        outcome["state"]="FREE_SPACE_CONFLICT_CANDIDATE"
    else:
        outcome["state"]="MIXED_UNCERTAIN"
    return outcome

def independently_separated(poses,*,min_move_m=.12,min_rotation_deg=15):
    """Greedy count independent viewpoint clusters, not raw repeated frames."""
    accepted=[]
    for x in poses:
        pos=np.asarray(x["xyz"],dtype=float)
        rot=Rotation.from_quat(x["quaternion"])
        far=True
        for old in accepted:
            dist=np.linalg.norm(pos-np.asarray(old["xyz"],dtype=float))
            theta=np.degrees((Rotation.from_quat(old["quaternion"]).inv()*rot).magnitude())
            if dist<min_move_m and theta<min_rotation_deg:
                far=False;break
        if far:accepted.append(x)
    return len(accepted)

def summarize_per_hypothesis(states):
    from collections import Counter
    counts=Counter(x["state"] for x in states)
    negatives=[x for x in states if x["state"]=="FREE_SPACE_CONFLICT_CANDIDATE"]
    compatible=counts["GEOMETRICALLY_COMPATIBLE_NOT_ID_PROOF"]
    # A candidate for absence only with 3 distinct viewpoints and not one
    # compatible observation; never automatically modify the old memory state.
    independent=independently_separated(negatives)
    if compatible:
        decision="GEOMETRY_SUPPORTED_AT_SOME_VIEWS_NOT_PHYSICAL_ID"
    elif independent>=3:
        decision="VISIBLY_ABSENT_CANDIDATE_REQUIRES_ACTIVE_RECHECK"
    elif counts["OCCLUDED_NOT_MISSING"]:
        decision="OCCLUDED_OR_UNKNOWN"
    else:
        decision="UNKNOWN_NO_VERIFIED_DISAPPEARANCE"
    return {"states":dict(counts),"independent_conflict_viewpoints":independent,
       "independently_accepted_negative_threshold":3,
       "decision":decision,"object_moved_asserted":False,
       "object_deleted_asserted":False}
