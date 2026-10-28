"""Independent CPU RGB-D odometry (no GT used for pose estimation).

ORB+PnP per synchronized frame; optional verified nonlocal RGB-D loop edges.
Exports camera-to-map transforms for full TSDF rebuild, and ATE/RPE against
TUM reference poses only after the whole estimate is complete.
This is a visual-odometry/pose-graph baseline, NOT a replacement for RTAB-Map.
"""
from __future__ import annotations
import argparse, json, math, time
from pathlib import Path
import cv2, numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.optimize import least_squares
from .tum_yolo_pipeline import synchronized, download_scene

K=np.array([[517.3,0,318.6],[0,516.5,255.3],[0,0,1.]],np.float64)
SCALE=5000.
def read_frame(path_rgb,path_depth,small=1):
    img=cv2.imread(str(path_rgb),cv2.IMREAD_GRAYSCALE)
    dep=cv2.imread(str(path_depth),cv2.IMREAD_UNCHANGED)
    if img is None or dep is None:raise ValueError("RGB-D frame missing")
    if img.shape!=dep.shape:raise ValueError("Unregistered RGB-D input")
    return img,dep

def make_key(img,orb):
    return orb.detectAndCompute(img,None)

def relative_pnp(kp_a,desc_a,depth_a,kp_b,desc_b,matcher,max_err=3.):
    if desc_a is None or desc_b is None:return None
    try:knn=matcher.knnMatch(desc_a,desc_b,k=2)
    except cv2.error:return None
    matches=[m for m,n in knn if m.distance <.73*n.distance]
    if len(matches)<35:return None
    pts_a=np.float32([kp_a[m.queryIdx].pt for m in matches])
    pts_b=np.float32([kp_b[m.trainIdx].pt for m in matches])
    px=np.rint(pts_a).astype(int)
    z=depth_a[np.clip(px[:,1],0,depth_a.shape[0]-1),
              np.clip(px[:,0],0,depth_a.shape[1]-1)].astype(float)/SCALE
    keep=(z>.35)&(z<4.0)
    if keep.sum()<25:return None
    xyz=np.column_stack(((pts_a[keep,0]-K[0,2])*z[keep]/K[0,0],
                         (pts_a[keep,1]-K[1,2])*z[keep]/K[1,1],z[keep]))
    uv=pts_b[keep]
    success,rvec,tvec,inliers=cv2.solvePnPRansac(
        xyz.astype(np.float32),uv.astype(np.float32),K,None,
        iterationsCount=150,reprojectionError=max_err,confidence=.999,
        flags=cv2.SOLVEPNP_EPNP)
    if not success or inliers is None or len(inliers)<18:return None
    inds=inliers[:,0]
    # Reject degenerate projection and implausible frame-to-frame motion.
    try:
        rvec,tvec=cv2.solvePnPRefineLM(xyz[inds],uv[inds],K,None,rvec,tvec)
    except cv2.error:pass
    R,_=cv2.Rodrigues(rvec);T=np.eye(4)
    T[:3,:3]=R;T[:3,3]=tvec.reshape(3)
    rotation=float(np.linalg.norm(Rot.from_matrix(R).as_rotvec()))
    translation=float(np.linalg.norm(T[:3,3]))
    return dict(T=T,inliers=len(inds),matches=len(keep.nonzero()[0]),rot=rotation,trans=translation)

def align_and_metrics(estimated,reference):
    A=np.array([x[:3,3] for x in estimated]);B=np.array([x[:3,3] for x in reference])
    xa=A.mean(0);xb=B.mean(0)
    H=(A-xa).T@(B-xb)/len(A)
    u,s,vt=np.linalg.svd(H)
    fix=np.diag([1.,1.,np.linalg.det(u@vt)])
    R=vt.T@fix@u.T
    t=xb-R@xa
    out=A@R.T+t
    residual=np.linalg.norm(out-B,axis=1)
    rpe=[]
    for i in range(len(estimated)-30):
        da=(estimated[i+30][:3,3]-estimated[i][:3,3])@R.T
        db=reference[i+30][:3,3]-reference[i][:3,3]
        rpe.append(float(np.linalg.norm(da-db)))
    return dict(ATE_RMSE_m=float(np.sqrt(np.mean(residual**2))),
                ATE_median_m=float(np.median(residual)),
                RPE_30frame_RMSE_m=float(np.sqrt(np.mean(np.square(rpe)))) if rpe else None,
                evaluated_frames=len(estimated),metric_scale_preserved=True)

def run(out,dataset='data/tum',max_frames=0,keyframe_every=15):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    frames=synchronized(download_scene(dataset))
    if max_frames:frames=frames[:max_frames]
    if len(frames)<25:raise ValueError('Too few synchronized frames for evaluation')
    orb=cv2.ORB_create(nfeatures=1800,fastThreshold=10)
    matcher=cv2.BFMatcher(cv2.NORM_HAMMING)
    poses=[np.eye(4)]
    history=[];keys=[]
    prev_img,prev_depth=read_frame(frames[0][1],frames[0][2])
    prev_k,prev_d=make_key(prev_img,orb)
    keys.append((0,prev_k,prev_d,prev_depth))
    missing=0
    for i in range(1,len(frames)):
        im,dp=read_frame(frames[i][1],frames[i][2])
        kp,desc=make_key(im,orb)
        rel=relative_pnp(prev_k,prev_d,prev_depth,kp,desc,matcher)
        if rel is None or rel['trans']>.4 or rel['rot']>.38:
            poses.append(poses[-1].copy())
            missing+=1
            history.append(dict(frame=i,status='vo_failed_hold_last_pose'))
        else:
            poses.append(poses[-1]@np.linalg.inv(rel['T']))
            history.append(dict(frame=i,status='estimated',inliers=rel['inliers'],
                                delta_translation_m=rel['trans'],delta_rotation_rad=rel['rot']))
        if i%keyframe_every==0:keys.append((i,kp,desc,dp))
        prev_img,prev_depth,prev_k,prev_d=im,dp,kp,desc
        if i%100==0:print("VO",i,"/",len(frames),"failed",missing,flush=True)
    # Independent nonlocal proposals. Require both visual inliers and 3D geometric consistency.
    # No GT-based gating. We do not claim a loop closure without confirmed constraints.
    loops=[]
    for n,(idx,kp,desc,dep) in enumerate(keys):
        if n<7:continue
        best=None
        for j in range(n-6):
            prev_idx,pk,pd,pdepth=keys[j]
            if np.linalg.norm(poses[idx][:3,3]-poses[prev_idx][:3,3])>1.0:
                continue
            fit=relative_pnp(pk,pd,pdepth,kp,desc,matcher,max_err=2.0)
            if fit is None or fit['inliers']<65 or fit['matches']<90:continue
            predicted=np.linalg.inv(poses[idx])@poses[prev_idx]
            discrepancy=np.linalg.inv(fit['T'])@predicted
            error=float(np.linalg.norm(discrepancy[:3,3]))
            if error>.30:continue
            if best is None or fit['inliers']>best['inliers']:
                best=dict(a=prev_idx,b=idx,T_b_a=fit['T'].tolist(),inliers=fit['inliers'],
                          relative_translation_error=error)
        if best is not None:loops.append(best)
    initial_metrics=align_and_metrics(poses,[f[3] for f in frames])
    # Deliberately never inject ground-truth poses into the generated trajectory.
    generated={"source":"independent_orb_pnp_rgbd_odometry",
               "pose_type":"T_map_cam","poses":[{"timestamp":float(f[0]),"T_map_cam":p.tolist()}
                           for f,p in zip(frames,poses)],
               "quality":{"failed_frame_steps":missing,"failure_fraction":missing/max(1,len(frames)-1),
                          "verified_nonlocal_candidates":len(loops),
                          "odometry_metrics":initial_metrics}}
    (out/"estimated_poses.json").write_text(json.dumps(generated))
    report={"frames":len(frames),"estimated_steps":len(frames)-1-missing,
            "failed_steps":missing,"candidate_loop_closures":len(loops),
            "ATE_RPE_GT_only_for_evaluation":initial_metrics,
            "pose_estimation_status":"CANDIDATE_UNVALIDATED",
            "loop_closure_optimization_status":"NOT_IMPLEMENTED",
            "gt_used_in_estimation":False,
            "loop_proposals":loops[:40],
            "unit":"meters"}
    (out/"vo_evaluation.json").write_text(json.dumps(report,indent=2))
    (out/"frame_tracking.json").write_text(json.dumps(history))
    print("VO_RESULT",json.dumps({k:v for k,v in report.items() if k!="loop_proposals"}),flush=True)
    return report

def main():
    a=argparse.ArgumentParser()
    a.add_argument("--data",default="data/tum")
    a.add_argument("--output",default="results/slam_pose")
    a.add_argument("--max-frames",type=int,default=0)
    c=a.parse_args()
    run(c.output,c.data,c.max_frames)
if __name__=='__main__':main()
