"""Robust SE(3) keyframe pose-graph optimization with verified RGB-D loop edges.

No TUM groundtruth is loaded until AFTER optimization. Post-optimization
trajectory is independently evaluated against GT but never selected using it.
"""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation as Rot,Slerp
from .tum_yolo_pipeline import synchronized,download_scene
from .visual_odometry import align_and_metrics

def from_vec(x):
    T=np.eye(4)
    T[:3,:3]=Rot.from_rotvec(x[3:6]).as_matrix()
    T[:3,3]=x[:3]
    return T
def to_vec(T):
    return np.r_[T[:3,3],Rot.from_matrix(T[:3,:3]).as_rotvec()]

def optimize(poses_json,vo_json,out_dir,dataset='data/tum',key_step=15):
    out_dir=Path(out_dir);out_dir.mkdir(parents=True,exist_ok=True)
    original=json.loads(Path(poses_json).read_text())
    info=json.loads(Path(vo_json).read_text())
    raw=[np.asarray(x['T_map_cam'],np.float64) for x in original['poses']]
    loops=info.get('loop_proposals',[])
    assert len(raw)>=25
    indices=list(range(0,len(raw),key_step))
    if indices[-1]!=len(raw)-1:indices.append(len(raw)-1)
    n=len(indices)
    which={k:i for i,k in enumerate(indices)}
    edges=[]
    for i in range(n-1):
        a=indices[i];b=indices[i+1]
        measurement=np.linalg.inv(raw[b])@raw[a]
        edges.append((i,i+1,measurement,1.0,False))
    for loop in loops:
        a,b=int(loop['a']),int(loop['b'])
        if a not in which or b not in which:continue
        observed=np.asarray(loop['T_b_a'],np.float64)
        if observed.shape!=(4,4):continue
        edges.append((which[a],which[b],observed,2.0,True))
    x0=np.concatenate([to_vec(raw[i]) for i in indices[1:]])
    def unpack(x):
        return [raw[0]]+[from_vec(x[6*(i-1):6*i]) for i in range(1,n)]
    def residual(x):
        poses=unpack(x);values=[]
        for a,b,measured,weight,isloop in edges:
            error=np.linalg.inv(measured)@np.linalg.inv(poses[b])@poses[a]
            values.extend(weight*np.r_[error[:3,3],Rot.from_matrix(error[:3,:3]).as_rotvec()])
        return np.asarray(values)
    # Sparse structure dramatically reduces costs vs dense numerical Jacobian.
    jac=lil_matrix((6*len(edges),6*(n-1)),dtype=int)
    for e,(a,b,m,w,isloop) in enumerate(edges):
        for node in (a,b):
            if node>0:jac[6*e:6*e+6,6*(node-1):6*node]=1
    before=residual(x0)
    optimized=least_squares(residual,x0,jac_sparsity=jac.tocsr(),loss='soft_l1',
                            f_scale=.08,max_nfev=120,ftol=1e-7,xtol=1e-7,gtol=1e-7)
    new_key=unpack(optimized.x)
    # Interpolate world-frame SE(3) corrections, avoid discontinuities at keys.
    corrections=[p@np.linalg.inv(raw[idx]) for idx,p in zip(indices,new_key)]
    newposes=[]
    for k in range(len(raw)):
        j=int(np.searchsorted(indices,k,side='right')-1)
        j=min(j,len(indices)-2)
        a,b=indices[j:j+2]
        alpha=float((k-a)/max(1,b-a))
        t=(1-alpha)*corrections[j][:3,3]+alpha*corrections[j+1][:3,3]
        quats=Rot.from_matrix([corrections[j][:3,:3],corrections[j+1][:3,:3]])
        RR=Slerp([0.,1.],quats)([alpha]).as_matrix()[0]
        c=np.eye(4);c[:3,:3]=RR;c[:3,3]=t
        newposes.append(c@raw[k])
    max_correction=float(max(np.linalg.norm(p[:3,3]-q[:3,3]) for p,q in zip(raw,newposes)))
    # No GT-based choosing. Reject implausibly large optimization regardless of GT.
    numerical_ok=bool(optimized.success and np.isfinite(optimized.x).all()
                      and max_correction<1.0 and np.linalg.norm(residual(optimized.x))<np.linalg.norm(before)+1e-6)
    if not numerical_ok:
        newposes=raw
        result_status='REJECTED_NUMERICAL_OR_LARGE_CORRECTION'
    else:result_status='APPLIED'
    artifact={"source":"rgbd_pnp_verified_loops_robust_posegraph",
              "poses":[{"timestamp":f['timestamp'],"T_map_cam":p.tolist()} for f,p in zip(original['poses'],newposes)],
              "result_status":result_status}
    (out_dir/'optimized_poses.json').write_text(json.dumps(artifact))
    reference=[f[3] for f in synchronized(download_scene(dataset))]
    if len(reference)!=len(raw):
        raise ValueError("GT evaluation sequence length mismatch: no guessing")
    a=align_and_metrics(raw,reference);b=align_and_metrics(newposes,reference)
    summary={"frames":len(raw),"keyframes":len(indices),"nonlocal_loops_verified":sum(e[-1] for e in edges),
             "success_numerical":bool(optimized.success),"optimizer_status":result_status,
             "initial_optimization_norm":float(np.linalg.norm(before)),
             "final_optimization_norm":float(np.linalg.norm(residual(optimized.x))),
             "max_pose_translation_correction_m":max_correction,
             "ATE_RPE_before_gt_eval_only":a,"ATE_RPE_after_gt_eval_only":b,
             "gt_used_in_optimization":False,
             "independent_semantic_id_and_nav2_evaluation":"NOT_TESTED"}
    (out_dir/"posegraph_evaluation.json").write_text(json.dumps(summary,indent=2))
    print("POSE_GRAPH_EVAL",json.dumps(summary),flush=True)
    return summary

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--poses",default="results/vo_mapping/estimated_poses.json")
    p.add_argument("--vo",default="results/vo_mapping/vo_evaluation.json")
    p.add_argument("--out",default="results/vo_mapping")
    p.add_argument("--data",default="data/tum")
    args=p.parse_args()
    optimize(args.poses,args.vo,args.out,args.data)
