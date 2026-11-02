"""EXP-20261102-05. Symmetric hard-negative RGB-D representation tests.

Use ONLY real recorded RGB, measured depth, and pose GT for correspondence oracle.
The exact same positive measurements are tested under four NEGATIVE POLICIES:
local-random, color-hard, geometry-hard, and balanced color+geometry hard.
The goal is to quantify evaluation-induced ranking reversal and minimax robustness.
"""
import argparse, pathlib, json, csv
import numpy as np
from real_rgbd import descriptor, load
from bonn_extrinsics import camera_pose_from_marker

METHODS=('color','dct','depth','spectral','camera_xyz','fusion')
POLICIES=('random','color_hard','geometry_hard','balanced_hard')
SEED=20261102

def corrected(root,dataset,n):
    intrinsic=[542.822841,542.576870,315.593520,237.756098] if dataset=='bonn_box' else [517.3,516.5,318.6,255.3]
    f,total=load(root,intrinsic,n)
    if dataset=='bonn_box':
        out=[]
        for rgb,z,pts,world,R,T,K in f:
            P=np.eye(4);P[:3,:3]=R;P[:3,3]=T
            P=camera_pose_from_marker(P)
            out.append((rgb,z,pts,pts@P[:3,:3].T+P[:3,3],P[:3,:3],P[:3,3],K))
        f=out
    return f,total

def safe_features(frame,x,y):
    try:
        z={m:descriptor(frame,x,y,m) for m in METHODS}
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        return None
    if any(v is None or not np.isfinite(v).all() for v in z.values()):return None
    return z

def collect_queries(f,max_queries):
    rng=np.random.default_rng(SEED)
    collected=[]
    for i in range(0,len(f)-4,2):
        A=f[i];B=f[i+4]
        z=B[1];R=B[4];T=B[5];K=B[6]
        for trial in range(max_queries):
            x=int(rng.integers(9,151));y=int(rng.integers(9,111))
            if not np.isfinite(A[3][y,x]).all():continue
            pt=(A[3][y,x]-T)@R
            if pt[2]<.3 or pt[2]>4.5:continue
            xx=int(round(K[0]*pt[0]/pt[2]+K[2]))
            yy=int(round(K[1]*pt[1]/pt[2]+K[3]))
            if not (9<=xx<151 and 9<=yy<111):continue
            if not (.3<z[yy,xx]<4.5) or abs(z[yy,xx]-pt[2])>.07:continue
            ref=safe_features(A,x,y)
            positive=safe_features(B,xx,yy)
            if ref is None or positive is None:continue
            pool={}
            for k in range(95):
                nx=int(xx+rng.integers(-29,30));ny=int(yy+rng.integers(-25,26))
                if not (9<=nx<151 and 9<=ny<111):continue
                if (nx,ny)==(xx,yy) or not (.3<z[ny,nx]<4.5):continue
                geom=B[2][ny,nx]
                if not np.isfinite(geom).all() or np.linalg.norm(geom-pt)<.14:continue
                if (nx,ny) in pool:continue
                feat=safe_features(B,nx,ny)
                if feat is None:continue
                pool[(nx,ny)]=feat
            if len(pool)<8:continue
            pool_items=list(pool.items())
            # Four disjoint negatives per policy whenever possible. The RGB
            # and depth policies use coarse scalar descriptors; all candidate
            # features are precomputed and no candidate label is guessed.
            dcolor=np.array([np.linalg.norm(ref['color']-p['color']) for pos,p in pool_items])
            dshape=np.array([np.linalg.norm(ref['depth']-p['depth']) for pos,p in pool_items])
            random4=rng.choice(len(pool_items),4,replace=False).tolist()
            hardc=np.argsort(dcolor,kind='stable').tolist()
            hardg=np.argsort(dshape,kind='stable').tolist()
            mix=hardc[:2]+[j for j in hardg if j not in hardc[:2]][:2]
            policy_indices={'random':random4,'color_hard':hardc[:4],
                            'geometry_hard':hardg[:4],'balanced_hard':mix}
            collected.append({'frame':i,'anchor':ref,'positive':positive,
                  'negative':{policy:[pool_items[j][1] for j in inds]
                              for policy,inds in policy_indices.items()}})
    return collected

def summarize(rows):
    result={}
    for m in METHODS:
        report={}
        for p in POLICIES:
            data=[]
            for q in rows:
                feat=q['anchor'][m]
                candidates=[q['positive'][m]]+[c[m] for c in q['negative'][p]]
                # Strict tie handling avoids hidden tie-breaking in favor of index 0.
                dist=np.array([np.linalg.norm(feat-v) for v in candidates])
                hit=float(dist[0]<np.min(dist[1:]))
                data.append((q['frame'],hit,float(dist[1:].min()-dist[0])))
            if len(data)<75:raise RuntimeError(f'Insufficient evaluation for {m}/{p}: {len(data)}')
            groups={}
            for frame,hit,margin in data:groups.setdefault(frame,[]).append(hit)
            rng=np.random.default_rng(SEED)
            ids=np.array(list(groups));bs=[]
            for _ in range(250):
                ss=rng.choice(ids,len(ids),replace=True)
                bs.append(np.mean([v for f in ss for v in groups[int(f)]]))
            report[p]={'rank1':float(np.mean([v[1] for v in data])),
                       'margin':float(np.mean([v[2] for v in data])),
                       'ci95':list(map(float,np.quantile(bs,[.025,.975]))),
                       'n_queries':len(data),'n_view_pairs':len(groups)}
        report['worst_policy']=min(report[p]['rank1'] for p in POLICIES)
        report['mean_policy']=float(np.mean([report[p]['rank1'] for p in POLICIES]))
        result[m]=report
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True);p.add_argument('--dataset',required=True,
                   choices=['tum_xyz','tum_desk','bonn_box'])
    p.add_argument('--n',type=int,default=76)
    p.add_argument('--queries',type=int,default=40)
    p.add_argument('--out',default='results/real_rgbd/modal_hardness')
    args=p.parse_args()
    frames,total=corrected(args.root,args.dataset,args.n)
    q=collect_queries(frames,args.queries)
    if len(q)<75:raise RuntimeError('Not enough physical correspondences, '+str(len(q)))
    result=summarize(q)
    output={'experiment':'EXP-20261102-05','dataset':args.dataset,'sampled_frames':len(frames),
            'synced_recorded_frames':total,'n_query_points':len(q),
            'selection':'One GT+depth-confirmed 3D match, four distinct REAL measured negatives; same observations for all six descriptors.',
            'policies':POLICIES,'methods':METHODS,'results':result,
            'limitations':'Near-camera local negatives, frame-pair bootstrap and GT camera poses. Geometry-hard negatives deliberately disadvantage depth descriptors; no object ID labels. Both TUM fr1 recordings share physical capture site; not two independent buildings.',
            'seed':SEED}
    out=pathlib.Path(args.out);out.mkdir(parents=True,exist_ok=True)
    (out/(args.dataset+'.json')).write_text(json.dumps(output,indent=2))
    with (out/(args.dataset+'.csv')).open('w',newline='') as fp:
        w=csv.writer(fp);w.writerow(['dataset','descriptor','policy','rank1','ci_low','ci_high','n'])
        for m in METHODS:
            for pol in POLICIES:
                row=result[m][pol]
                w.writerow([args.dataset,m,pol,row['rank1'],*row['ci95'],row['n_queries']])
    print('EXP05_RESULT',json.dumps({'dataset':args.dataset,'n_query_points':len(q),
           'results':{m:{p:round(result[m][p]['rank1'],5) for p in POLICIES}|
                      {'worst':round(result[m]['worst_policy'],5)} for m in METHODS}},sort_keys=True))
    print('PASS EXP05',args.dataset)
if __name__=='__main__':main()
