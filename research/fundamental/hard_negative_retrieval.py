"""EXP-20261102-02: Hard-negative multi-candidate retrieval of measured RGB-D points.

Physically measured frames with GT-pose correspondences, no synthetic scenes.
Tests structural 3D descriptors against visually confusable nearby points.
The goal is point-level retrieval only, not object-instance reidentification.
"""
import argparse,json,pathlib
import numpy as np
from real_rgbd import load,descriptor
from bonn_extrinsics import camera_pose_from_marker
SEED=20261102
METHODS=("color","dct","depth","spectral","camera_xyz","fusion")
def corrected_frames(root,intrinsics,dataset,n):
    f,total=load(root,intrinsics,n)
    if dataset=="bonn_box":
        new=[]
        for rgb,z,pts,world,R,T,K in f:
            P=np.eye(4);P[:3,:3]=R;P[:3,3]=T
            P=camera_pose_from_marker(P)
            new.append((rgb,z,pts,pts@P[:3,:3].T+P[:3,3],P[:3,:3],P[:3,3],K))
        f=new
    return f,total
def create_queries(f):
    rng=np.random.default_rng(SEED)
    rows=[]
    for i in range(0,len(f)-3,2):
        A=f[i];B=f[i+3];D=B[1];R=B[4];T=B[5];K=B[6]
        pts=A[3];K0=A[6];_,Z,*_=A
        for _ in range(48):
            x=int(rng.integers(9,151));y=int(rng.integers(9,111))
            q=pts[y,x]
            if not np.isfinite(q).all():continue
            pc=(q-T)@R
            if pc[2]<.3:continue
            xx=int(round(K[0]*pc[0]/pc[2]+K[2]))
            yy=int(round(K[1]*pc[1]/pc[2]+K[3]))
            if not(9<=xx<151 and 9<=yy<111) or not(.3<D[yy,xx]<4.5):
                continue
            if abs(D[yy,xx]-pc[2])>.07:continue
            d0=descriptor(A,x,y,"color")
            if d0 is None:continue
            candidates=[]
            # Nearby alternatives in the actual next camera image, outside a 10cm
            # physical neighborhood. Pick visually confusable negatives without
            # consulting the method being compared.
            for _ in range(110):
                nx=xx+int(rng.integers(-28,29));ny=yy+int(rng.integers(-24,25))
                if not (9<=nx<151 and 9<=ny<111):continue
                dn=D[ny,nx]
                if not (.3<dn<4.5):continue
                xyz=B[2][ny,nx]
                if np.linalg.norm(xyz-pc)<.12:continue
                col=descriptor(B,nx,ny,"color")
                if col is None:continue
                candidates.append((float(np.linalg.norm(col-d0)),nx,ny))
            if len(candidates)<4:continue
            candidates.sort()
            # Four closest by color, plus one confirmed same physical point.
            positions=[(xx,yy)]+[(nx,ny) for _,nx,ny in candidates[:4]]
            rows.append((i,A,B,(x,y),positions))
    return rows

def evaluate(rows):
    records={name:[] for name in METHODS}
    for idx,A,B,xy,candidates in rows:
        x,y=xy
        for name in METHODS:
            anchor=descriptor(A,x,y,name)
            if anchor is None:continue
            ds=[descriptor(B,cx,cy,name) for cx,cy in candidates]
            if any(q is None for q in ds):continue
            vals=np.array([np.linalg.norm(anchor-q) for q in ds])
            # All candidates include one true 3D correspondance and four
            # distinct measured locations. Hard negatives selected by color.
            records[name].append((idx,int(np.argmin(vals)==0),float(np.mean(vals[1:]-vals[0]))))
    out={};rng=np.random.default_rng(SEED)
    for name,rec in records.items():
        groups={}
        for idx,correct,margin in rec:groups.setdefault(idx,[]).append((correct,margin))
        if len(rec)<80:raise RuntimeError("Too few hard-negative queries "+name+":"+str(len(rec)))
        frame_ids=list(groups)
        boot=[]
        for _ in range(400):
            sample=rng.choice(frame_ids,len(frame_ids),replace=True)
            items=[x for k in sample for x in groups[k]]
            boot.append(float(np.mean([x[0] for x in items])))
        out[name]={
            "n_queries":len(rec),"n_frame_pairs":len(groups),
            "recall_at_1_of_5":float(np.mean([a for _,a,_ in rec])),
            "mean_positive_vs_negative_margin":float(np.mean([m for _,_,m in rec])),
            "ci95_by_frame_pair":[float(x) for x in np.percentile(boot,[2.5,97.5])],
        }
    return out

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",required=True);p.add_argument("--dataset",required=True)
    p.add_argument("--intrinsics",required=True);p.add_argument("--frames",type=int,default=84)
    p.add_argument("--out",default="results/real_rgbd")
    a=p.parse_args()
    f,total=corrected_frames(a.root,list(map(float,a.intrinsics.split(","))),a.dataset,a.frames)
    queries=create_queries(f)
    scores=evaluate(queries)
    report={"experiment":"EXP-20261102-02","dataset":a.dataset,"real_frames":len(f),
            "raw_sync_frames":total,"n_queries":len(queries),
            "method":"Real-world GT-pose correspondences, true match among four RGB-similar but physically distinct measured 3D negatives. No object-instance labels.",
            "selection_bias":"Negatives chosen to be RGB-close; do not interpret RGB disadvantage as evidence of general 3D superiority.",
            "scores":scores,"seed":SEED}
    path=pathlib.Path(a.out)/a.dataset;path.mkdir(parents=True,exist_ok=True)
    (path/"hard_negative_retrieval.json").write_text(json.dumps(report,indent=2))
    print("HARD_NEGATIVE_RESULT",json.dumps(report,sort_keys=True))
if __name__=="__main__":main()
