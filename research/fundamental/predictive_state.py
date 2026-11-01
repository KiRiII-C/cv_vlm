"""Chronological evaluation of predictive-state vs frame compression on real RGB-D.
One-step video prediction is a proxy, NOT object memory or navigation success.
"""
import argparse,pathlib,json
import cv2,numpy as np
def read_index(file):
    arr=[]
    for line in pathlib.Path(file).read_text().splitlines():
        if not line.strip() or line.startswith('#'):continue
        x=line.split()
        if len(x)>1:
            try:arr.append((float(x[0]),x[1]))
            except ValueError:pass
    return sorted(arr)
def load(root,n):
    root=pathlib.Path(root);a=read_index(root/'rgb.txt');b=read_index(root/'depth.txt')
    times=np.array([t for t,p in b]);pairs=[]
    for t,p in a:
        j=int(np.argmin(abs(times-t)))
        if abs(times[j]-t)<.035:pairs.append((root/p,root/b[j][1]))
    frames=[]
    for j in np.unique(np.round(np.linspace(0,len(pairs)-1,min(n,len(pairs)))).astype(int)):
        r,d=pairs[j];im=cv2.imread(str(r),0);dep=cv2.imread(str(d),cv2.IMREAD_UNCHANGED)
        if im is None or dep is None:continue
        rgb=cv2.resize(im,(32,24),interpolation=cv2.INTER_AREA).astype('f')/255
        z=cv2.resize(dep,(32,24),interpolation=cv2.INTER_NEAREST).astype('f')/5000
        valid=(z>.3)&(z<5);inv=np.zeros_like(z);inv[valid]=1/np.maximum(z[valid],.3)
        frames.append(np.r_[rgb.ravel(),np.clip(inv/3,0,1).ravel()])
    return np.asarray(frames),len(pairs)
def ridge(Z,Y):
    A=np.c_[Z,np.ones(len(Z))];reg=np.eye(A.shape[1])*.05;reg[-1,-1]=0
    return np.linalg.solve(A.T@A+reg,A.T@Y)
def error(pred,truth):
    a=np.mean(abs(pred[:,:768]-truth[:,:768]))
    valid=truth[:,768:]>0
    b=np.mean(abs(pred[:,768:][valid]-truth[:,768:][valid])) if valid.any() else float('nan')
    return {'gray_mae':round(float(a),6),'invdepth_mae':round(float(b),6)}
def test(A,k):
    n=len(A);ids=np.arange(3,n);split=int(n*.72);train=ids<split;test=ids>=split
    Y=A[3:];X=A[ids-1];H=np.hstack([A[ids-3],A[ids-2],A[ids-1]])
    truth=Y[test];out={}
    out['persistence']=error(X[test],truth)
    out['velocity']=error(np.clip(2*A[ids[test]-1]-A[ids[test]-2],0,1),truth)
    for name,B in [('frame_pca',X),('hankel_3',H)]:
        mean=B[train].mean(0);U,S,V=np.linalg.svd(B[train]-mean,full_matrices=False)
        dim=min(k,len(S),len(U)-2);basis=V[:dim].T
        ztr=(B[train]-mean)@basis;zte=(B[test]-mean)@basis
        W=ridge(ztr,Y[train])
        pred=np.clip(np.c_[zte,np.ones(len(zte))]@W,0,1)
        out[name]=error(pred,truth)
    return out
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--name',required=True)
    p.add_argument('--out',default='results/real_rgbd');p.add_argument('--frames',type=int,default=140);a=p.parse_args()
    A,n=load(a.root,a.frames)
    if len(A)<30:raise RuntimeError('Not enough measured RGB-D frames')
    res={str(k):test(A,k) for k in [4,8,16]}
    out=pathlib.Path(a.out)/a.name;out.mkdir(parents=True,exist_ok=True)
    data={'source':'Real measured RGB-D video','n_frames':len(A),'n_rgbd_pairs':n,'results':res,
          'evaluation':'chronological heldout one-step RGB/inverse-depth MAE; GT pose not used',
          'caveats':'Video proxy only, not an object map, self-localization or navigation benchmark. No semantic object labels.'}
    (out/'predictive_state.json').write_text(json.dumps(data,indent=2));print(json.dumps(data,indent=2))
if __name__=='__main__':main()
