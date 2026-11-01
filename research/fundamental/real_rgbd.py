"""Real RGB-D evaluation: keyframe memory as geometric coverage and 3D descriptor invariance.
Uses original recorded RGB, measured depth, and dataset GT camera poses.
Does NOT claim object-level labels, neural network training or completed navigation.
"""
import argparse, pathlib, csv, json
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
from scipy.fft import dctn
SEED=20261102
def index(path):
    out=[]
    for row in pathlib.Path(path).read_text().splitlines():
        if not row.strip() or row.startswith('#'):continue
        p=row.split()
        if len(p)>=2:
            try:out.append((float(p[0]),p[1:]))
            except ValueError:pass
    return sorted(out)
def nearest(arr,t,limit=.045):
    if not arr:return None
    i=min(range(len(arr)),key=lambda j:abs(arr[j][0]-t))
    return arr[i] if abs(arr[i][0]-t)<limit else None
def load(root,K,n):
    root=pathlib.Path(root);a=index(root/'rgb.txt');b=index(root/'depth.txt');c=index(root/'groundtruth.txt')
    matches=[]
    for t,r in a:
        d=nearest(b,t);p=nearest(c,t)
        if d and p and len(p[1])>=7:
            matches.append((root/r[0],root/d[1][0],np.array(list(map(float,p[1][:7])))))
    if not matches:raise RuntimeError('No matched RGB-D and groundtruth in '+str(root))
    frames=[]
    for j in np.unique(np.round(np.linspace(0,len(matches)-1,min(n,len(matches)))).astype(int)):
        r,d,p=matches[j]
        bgr=cv2.imread(str(r),cv2.IMREAD_COLOR);orig=cv2.imread(str(d),cv2.IMREAD_UNCHANGED)
        if bgr is None or orig is None:continue
        H,W=orig.shape
        rgb=cv2.cvtColor(cv2.resize(bgr,(160,120)),cv2.COLOR_BGR2RGB).astype('f')/255
        z=cv2.resize(orig,(160,120),interpolation=cv2.INTER_NEAREST).astype('f')/5000
        fx,fy,cx,cy=K;K2=np.array([fx*160/W,fy*120/H,cx*160/W,cy*120/H])
        yy,xx=np.mgrid[:120,:160]
        pts=np.stack(((xx-K2[2])*z/K2[0],(yy-K2[3])*z/K2[1],z),-1)
        pts[(z<.3)|(z>4.5)]=np.nan
        R=Rotation.from_quat(p[3:]).as_matrix();T=p[:3];world=pts@R.T+T
        frames.append((rgb,z,pts,world,R,T,K2))
    return frames,len(matches)
def descriptor(frame,x,y,kind):
    rgb,z,xyz,*_=frame
    p=rgb[y-4:y+5,x-4:x+5]
    q=xyz[y-4:y+5,x-4:x+5].reshape(-1,3)
    q=q[np.isfinite(q[:,2])]
    if len(q)<20:return None
    if kind=='color':v=np.r_[p.mean((0,1)),p.std((0,1))]
    elif kind=='dct':v=dctn(p.mean(2),norm='ortho')[:4,:4].ravel()
    elif kind=='depth':v=np.r_[np.log1p(np.maximum(0,np.linalg.eigvalsh(np.cov(q.T)))*100),np.quantile(np.linalg.norm(q-q.mean(0),axis=1),[.1,.5,.9])]
    elif kind=='spectral':
        v=np.histogram(np.linalg.norm(q[::2,None,:]-q[None,::2,:],axis=-1),12,range=(0,.5))[0]
    elif kind=='camera_xyz':v=np.r_[xyz[y,x],q.std(0)]
    elif kind=='fusion':
        c=descriptor(frame,x,y,'color');d=descriptor(frame,x,y,'spectral')
        return np.r_[c,d]/np.sqrt(2)
    else:raise ValueError(kind)
    return v/(np.linalg.norm(v)+1.e-9)
def matching(frames,rng):
    methods=['color','dct','depth','spectral','camera_xyz','fusion']
    hits={m:[0,0] for m in methods}
    for a,b in zip(frames[::3],frames[2::3]):
        world=a[3];R=b[4];T=b[5];K=b[6];z=b[1]
        for y in range(9,110,9):
            for x in range(9,150,9):
                if not np.isfinite(a[2][y,x,2]):continue
                pt=(world[y,x]-T)@R
                if pt[2]<=0:continue
                xx=int(round(K[0]*pt[0]/pt[2]+K[2]));yy=int(round(K[1]*pt[1]/pt[2]+K[3]))
                if not(8<xx<151 and 8<yy<111) or abs(z[yy,xx]-pt[2])>.08:continue
                nx=int(rng.integers(6,154));ny=int(rng.integers(6,114))
                if abs(nx-xx)+abs(ny-yy)<20:continue
                for m in methods:
                    u=descriptor(a,x,y,m);v=descriptor(b,xx,yy,m);w=descriptor(b,nx,ny,m)
                    if u is None or v is None or w is None:continue
                    hits[m][0]+=int(np.linalg.norm(u-v)<np.linalg.norm(u-w));hits[m][1]+=1
    return [dict(method=m,correct=hit,pairs=n,accuracy=round(hit/max(1,n),5)) for m,(hit,n) in hits.items()]
def voxel(f):
    p=f[3].reshape(-1,3);p=p[np.isfinite(p[:,2])][::3]
    return set(map(tuple,np.floor(p/.12).astype(np.int32)))
def memory(frames,rng):
    f=[voxel(q) for q in frames];N=max(1,int(len(f)*.72));cand=f[:N];heldout=set().union(*f[N:]);train=set().union(*cand)
    results=[]
    for k in [4,8,16]:
        if k>N:continue
        for method in ['uniform','random','greedy','diverse']:
            for trial in range(5 if method=='random' else 1):
                if method=='uniform':sel=np.linspace(0,N-1,k).astype(int).tolist()
                elif method=='random':sel=rng.choice(N,k,replace=False).tolist()
                else:
                    sel=[];covered=set()
                    for _ in range(k):
                        rem=set(range(N))-set(sel)
                        if method=='greedy':fun=lambda j:len(cand[j]-covered)
                        else:fun=lambda j:min([1-len(cand[j]&cand[z])/max(1,len(cand[j]|cand[z])) for z in sel] or [1])
                        j=max(rem,key=lambda j:(fun(j),-j));sel.append(j);covered|=cand[j]
                seen=set().union(*(cand[j] for j in sel))
                results.append(dict(method=method,budget=k,trial=trial,
                    seen=len(seen),train_coverage=round(len(seen)/max(1,len(train)),5),
                    heldout_recall=round(len(seen&heldout)/max(1,len(heldout)),5)))
    return results
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True);p.add_argument('--dataset',required=True)
    p.add_argument('--intrinsics',required=True);p.add_argument('--out',default='results/real_rgbd')
    p.add_argument('--n',type=int,default=48);a=p.parse_args()
    rng=np.random.default_rng(SEED)
    f,total=load(a.root,list(map(float,a.intrinsics.split(','))),a.n)
    if a.dataset == 'bonn_box':
        from bonn_extrinsics import camera_pose_from_marker
        corrected=[]
        for rgb,z,pts,world,R,T,K in f:
            pose=np.eye(4);pose[:3,:3]=R;pose[:3,3]=T
            pose=camera_pose_from_marker(pose)
            corrected.append((rgb,z,pts,pts@pose[:3,:3].T+pose[:3,3],pose[:3,:3],pose[:3,3],K))
        f=corrected
    x=matching(f,rng);y=memory(f,rng)
    out=pathlib.Path(a.out)/a.dataset;out.mkdir(parents=True,exist_ok=True)
    for filename,rows in [('descriptors.csv',x),('memory.csv',y)]:
        with (out/filename).open('w',newline='') as h:
            writer=csv.DictWriter(h,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    res={'dataset':a.dataset,'raw_synchronized_frames':total,'frames_used':len(f),'descriptor':x,'memory':y,
         'methodology':'Actual RGB and depth sensor measurements; GT pose used for evaluation of 3D correspondence and geometric memory. No object identity ground truth and no neural network training.','seed':SEED}
    (out/'summary.json').write_text(json.dumps(res,indent=2))
    assert len(f)>=16
    assert max(r['pairs'] for r in x)>20,'Not enough valid reprojection pairs: check calibration and timestamp association'
    print(json.dumps({'dataset':a.dataset,'frames':len(f),'raw_pairs':total,'descriptor':x,
        'memory_budget8':[r for r in y if r['budget']==8 and r['trial']==0]},indent=2))
if __name__=='__main__':main()
