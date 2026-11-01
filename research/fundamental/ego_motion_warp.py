"""Test camera-motion disentanglement with real RGB-D and GT-pose forward warping.
GT pose is oracle here, not the output of a proposed SLAM system.
"""
import argparse,json,pathlib
import cv2,numpy as np
from scipy.spatial.transform import Rotation
def index(path):
    a=[]
    for line in pathlib.Path(path).read_text().splitlines():
        if not line or line.startswith('#'):continue
        t=line.split()
        if len(t)>1:
            try:a.append((float(t[0]),t[1:]))
            except ValueError:pass
    return sorted(a)
def load(root,k,n=56):
    root=pathlib.Path(root);a=index(root/'rgb.txt');b=index(root/'depth.txt');c=index(root/'groundtruth.txt')
    nearest=lambda items,t:min(items,key=lambda e:abs(e[0]-t))
    valid=[]
    for t,filename in a:
        d=nearest(b,t);p=nearest(c,t)
        if abs(d[0]-t)<.045 and abs(p[0]-t)<.045 and len(p[1])>=7:
            valid.append((root/filename[0],root/d[1][0],np.array(list(map(float,p[1][:7])))))
    frames=[]
    for q in np.unique(np.round(np.linspace(0,len(valid)-1,min(n,len(valid)))).astype(int)):
        rp,dp,pose=valid[q];a=cv2.imread(str(rp));dep=cv2.imread(str(dp),cv2.IMREAD_UNCHANGED)
        if a is None or dep is None:continue
        H,W=dep.shape
        rgb=cv2.cvtColor(cv2.resize(a,(160,120)),cv2.COLOR_BGR2RGB).astype('f')/255
        z=cv2.resize(dep,(160,120),interpolation=cv2.INTER_NEAREST).astype('f')/5000
        fx,fy,cx,cy=k;K=np.array([fx*160/W,fy*120/H,cx*160/W,cy*120/H])
        R=Rotation.from_quat(pose[3:]).as_matrix()
        frames.append((rgb,z,K,R,pose[:3]))
    return frames
def compare(a,b):
    rgb,z,K,R,T=a;rgb2,z2,K2,R2,T2=b
    yy,xx=np.mgrid[:120,:160];valid=(z>.3)&(z<4.5)
    x=xx[valid];y=yy[valid];depth=z[valid]
    p=np.stack([(x-K[2])*depth/K[0],(y-K[3])*depth/K[1],depth],-1)
    world=p@R.T+T;tgt=(world-T2)@R2;D=tgt[:,2];ok=D>.3
    x=x[ok];y=y[ok];tgt=tgt[ok];D=D[ok]
    u=np.round(K2[0]*tgt[:,0]/D+K2[2]).astype(int)
    v=np.round(K2[1]*tgt[:,1]/D+K2[3]).astype(int)
    ok=(u>=0)&(u<160)&(v>=0)&(v<120)
    u=u[ok];v=v[ok];D=D[ok];colors=rgb[y[ok],x[ok]]
    if not len(u):return None
    flat=v*160+u;order=np.argsort(D)
    pix,first=np.unique(flat[order],return_index=True);ids=order[first]
    expected_depth=D[ids];expected_rgb=colors[ids]
    observed_depth=z2.ravel()[pix];observed_rgb=rgb2.reshape(-1,3)[pix]
    valid=(observed_depth>.3)&(observed_depth<4.5)
    pix=pix[valid];expected_depth=expected_depth[valid];expected_rgb=expected_rgb[valid]
    observed_depth=observed_depth[valid];observed_rgb=observed_rgb[valid]
    if len(pix)<100:return None
    baseline_depth=z.ravel()[pix];baseline_rgb=rgb.reshape(-1,3)[pix]
    return {'n':len(pix),'overlap_fraction':len(pix)/(160*120),
        'warp_rgb_mae':float(abs(expected_rgb-observed_rgb).mean()),
        'same_pixel_rgb_mae':float(abs(baseline_rgb-observed_rgb).mean()),
        'warp_depth_mae_m':float(abs(expected_depth-observed_depth).mean()),
        'same_pixel_depth_mae_m':float(abs(baseline_depth-observed_depth).mean()),
        'depth_disagreement_over_10cm':float(np.mean(abs(expected_depth-observed_depth)>.1))}
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--name',required=True)
    p.add_argument('--intrinsics',required=True);p.add_argument('--out',default='results/real_rgbd');a=p.parse_args()
    f=load(a.root,list(map(float,a.intrinsics.split(','))))
    records=[r for i in range(len(f)-2) if (r:=compare(f[i],f[i+2])) is not None]
    if len(records)<8:raise RuntimeError('Too few real correspondences')
    weights=np.array([r['n'] for r in records])
    res={k:float(np.average([r[k] for r in records],weights=weights)) for k in records[0] if k!='n'}
    obj={'dataset':a.name,'n_pairs':len(records),'results':res,
       'methodology':'Real RGB-D with oracle dataset poses, nearest-pixel z-buffer forward warp, fair same-pixel baseline on identical observed target pixels',
       'limitations':'GT poses not estimated; sparse overlap; no object labels; not full scene reconstruction.'}
    out=pathlib.Path(a.out)/a.name;out.mkdir(parents=True,exist_ok=True)
    (out/'ego_motion_warp.json').write_text(json.dumps(obj,indent=2))
    print(json.dumps(obj,indent=2))
if __name__=='__main__':main()
