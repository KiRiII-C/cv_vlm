"""1000 reproducible physically grounded RGB-D representation pipelines (CPU).
Ten image operators x ten 3D geometry operators x ten temporal memory operators.
Distinct mathematical compositions; NOT 1000 trained foundation models.
No oracle success claim: synthetic ray-cast observations only.
"""
import csv, json, itertools, os
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.fft import dctn
from scipy.ndimage import sobel
SEED=20261102
IMS=["mean","hist","grid","dct","fft","edge","grad","moment","hue","texture"]
GEO=["centroid","cov","radial","pair","extent","median","fourier","rff","moment","soft"]
MEM=["mean","median","last","first","ema","max","min","medoid","weighted","trim"]
def norm(v):
    v=np.nan_to_num(np.asarray(v,float).ravel())
    return v/max(1e-8,np.linalg.norm(v))
def raycast(objects,view,rng,shift=False):
    H,W=30,42
    x,y=np.meshgrid(np.arange(W),np.arange(H))
    direction=np.stack([(x-W/2)/40,(y-H/2)/38,np.ones_like(x)],-1).reshape(-1,3)
    origin=np.array([view*.15,0.,0.])
    dep=np.full(W*H,9.,float)
    ids=np.full(W*H,-1,int)
    xyz=np.zeros((W*H,3))
    for j,(ctr,r,shape,color) in enumerate(objects):
        c=ctr.copy()
        if shift and j in (0,2):c=objects[2-j][0]
        if shape==0:
            oc=origin-c
            a=np.einsum("ij,ij->i",direction,direction)
            b=2*(direction@oc)
            disc=b*b-4*a*(oc@oc-r*r)
            t=np.full(len(b),np.inf)
            ok=disc>=0
            t[ok]=(-b[ok]-np.sqrt(disc[ok]))/(2*a[ok])
            t[t<0]=np.inf
        else:
            h=np.array([r,r*.8,r*.8])
            safe=np.where(abs(direction)<1.e-10,1.e-10,direction)
            t0=np.minimum((c-h-origin)/safe,(c+h-origin)/safe)
            t1=np.maximum((c-h-origin)/safe,(c+h-origin)/safe)
            near=np.max(t0,axis=1)
            t=np.where((near>0)&(near<=np.min(t1,axis=1)),near,np.inf)
        keep=np.isfinite(t)&(t<dep)
        dep[keep]=t[keep];ids[keep]=j
        xyz[keep]=origin+direction[keep]*t[keep,None]
    rgb=np.full((W*H,3),.07)
    for j,(c,r,shape,col) in enumerate(objects):
        m=ids==j
        rgb[m]=col*(1+.12*np.sin(xyz[m,0]*15+j))
    rgb+=rng.normal(0,.025,rgb.shape)
    dep+=rng.normal(0,.018,dep.shape)
    drop=rng.random(len(dep))<.1
    dep[drop]=np.nan
    return rgb.reshape(H,W,3),dep.reshape(H,W),ids.reshape(H,W),xyz.reshape(H,W,3)
def observe(rgb,depth,labels,xyz,j):
    mask=(labels==j)&np.isfinite(depth)
    if mask.sum()<10:return None
    p=rgb[mask];v=xyz[mask]
    return p,v,mask,rgb
def image(o,name):
    p,v,m,img=o
    if name=="mean":z=p.mean(0)
    elif name=="hist":z=np.concatenate([np.histogram(p[:,c],8,(0,1))[0] for c in range(3)])
    elif name=="moment":z=np.r_[p.mean(0),p.std(0),np.mean((p-p.mean(0))**3,axis=0)]
    elif name=="hue":z=np.histogram(np.argmax(p,axis=1),bins=3,range=(0,3))[0]
    elif name=="grid":
        yy,xx=np.where(m)
        a=(xx-xx.min())/max(1,np.ptp(xx));b=(yy-yy.min())/max(1,np.ptp(yy))
        cell=np.minimum((a*3).astype(int),2)+3*np.minimum((b*3).astype(int),2)
        z=np.concatenate([p[cell==k].mean(0) if (cell==k).any() else np.zeros(3) for k in range(9)])
    else:
        gray=img.mean(2)*m
        if name=="dct":z=dctn(gray,norm="ortho")[:6,:6].ravel()
        elif name=="fft":z=np.abs(np.fft.fft2(gray))[:6,:6].ravel()
        elif name=="edge":z=np.r_[np.mean(abs(sobel(gray,0))[m]),np.mean(abs(sobel(gray,1))[m])]
        elif name=="grad":
            a=sobel(gray,0);b=sobel(gray,1)
            z=np.histogram(np.arctan2(a[m],b[m]),12,(-np.pi,np.pi),weights=np.hypot(a[m],b[m]))[0]
        elif name=="texture":
            z=np.r_[np.std(gray[m]),np.mean(abs(gray-np.roll(gray,1,0))[m]),np.mean(abs(gray-np.roll(gray,1,1))[m])]
        else:raise ValueError(name)
    return norm(z)
def geometry(o,name):
    p,v,m,img=o
    c=v-v.mean(0)
    if name=="centroid":z=v.mean(0)
    elif name=="cov":z=np.r_[v.mean(0),np.cov(c.T).ravel()]
    elif name=="radial":z=np.r_[v.mean(0),np.histogram(np.linalg.norm(c,axis=1),10,(0,1))[0]/len(v)]
    elif name=="pair":
        a=c[np.linspace(0,len(c)-1,min(20,len(c)),dtype=int)]
        z=np.r_[v.mean(0),np.histogram(np.linalg.norm(a[:,None]-a[None,:],axis=-1),10,(0,1.3))[0]]
    elif name=="extent":z=np.r_[v.mean(0),np.ptp(v,axis=0)]
    elif name=="median":z=np.r_[np.median(v,axis=0),np.median(abs(c),axis=0)]
    elif name=="fourier":z=np.r_[v.mean(0),np.cos(c[:,None]*np.arange(1,5)[None,:,None]*8).mean(0).ravel()]
    elif name=="rff":
        W=np.random.default_rng(9).normal(size=(3,16))*8
        z=np.r_[v.mean(0),np.cos(c@W).mean(0)]
    elif name=="moment":z=np.r_[v.mean(0),c.std(0),np.mean(c**3,axis=0)]
    elif name=="soft":
        ctr=np.array(list(itertools.product([-.2,.2],repeat=3)))
        z=np.r_[v.mean(0),np.exp(-np.sum((c[:,None]-ctr[None,:])**2,axis=2)/.07).mean(0)]
    return norm(z)
def memory(v,name):
    a=np.stack(v)
    if name=="mean":z=a.mean(0)
    elif name=="median":z=np.median(a,axis=0)
    elif name=="last":z=a[-1]
    elif name=="first":z=a[0]
    elif name=="ema":z=.25*a[0]+.25*a[1]+.5*a[-1]
    elif name=="max":z=a.max(0)
    elif name=="min":z=a.min(0)
    elif name=="medoid":z=a[np.argmin(np.linalg.norm(a[:,None]-a[None,:],axis=-1).sum(1))]
    elif name=="weighted":z=np.average(a,axis=0,weights=np.arange(1,len(a)+1))
    elif name=="trim":z=np.median(a,axis=0)
    return norm(z)
def get_scene(rng,ood):
    objs=[]
    for j in range(5):
        c=np.array([np.linspace(-1.4,1.4,5)[j],rng.uniform(-.2,.2),rng.uniform(3.2,4.5)])
        col=np.array([.5,.55,.58])+rng.normal(0,.025,3) if ood else rng.uniform(.2,.8,3)
        objs.append((c,rng.uniform(.22,.4),j%2,col))
    refs={j:[] for j in range(5)};query={}
    for t in range(4):
        frame=raycast(objs,t,rng,shift=(ood and t==3))
        for j in range(5):
            obs=observe(*frame,j)
            if obs is not None:
                if t==3:query[j]=obs
                else:refs[j].append(obs)
    ids=[j for j in range(5) if len(refs[j])>=2 and j in query]
    return refs,query,ids
def run(n=40):
    rng=np.random.default_rng(SEED)
    scenes=[get_scene(rng,i>=n//2) for i in range(n)]
    cache=[]
    for refs,query,ids in scenes:
        if len(ids)<3:continue
        im={};geo={}
        for method in IMS:
            im[method]=([[image(obs,method) for obs in refs[j]] for j in ids],np.stack([image(query[j],method) for j in ids]))
        for method in GEO:
            geo[method]=([[geometry(obs,method) for obs in refs[j]] for j in ids],np.stack([geometry(query[j],method) for j in ids]))
        cache.append((im,geo,ids))
    rows=[]
    for i,g,m in itertools.product(IMS,GEO,MEM):
        count=[0,0];correct=[0,0]
        for k,(im,geo,ids) in enumerate(cache):
            u,uq=im[i];v,vq=geo[g]
            p=np.stack([memory(o,m) for o in u]);q=np.stack([memory(o,m) for o in v])
            d=1-.5*uq@p.T-.5*vq@q.T
            rr,cc=linear_sum_assignment(d)
            part=int(k>=len(cache)//2)
            count[part]+=len(rr);correct[part]+=int(np.sum(rr==cc))
        rows.append(dict(image=i,geometry=g,memory=m,normal=round(correct[0]/count[0],5),stress=round(correct[1]/count[1],5)))
    os.makedirs("results/fundamental",exist_ok=True)
    with open("results/fundamental/sweep1000.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    rows.sort(key=lambda x:-x["stress"])
    summary={"n_compositions":len(rows),"n_scenes":len(cache),"top_10_by_stress":rows[:10],"caveat":"Selection by stress is exploratory and optimistic; physical primitives and perfect segmentation; not a real-world benchmark"}
    with open("results/fundamental/summary.json","w") as f:json.dump(summary,f,indent=2)
    print(json.dumps(summary,indent=2))
if __name__=="__main__":run()
