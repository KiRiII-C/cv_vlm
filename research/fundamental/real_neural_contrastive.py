"""Tiny scratch-trained RGB/RGB-D neural encoders on real cross-view matched patches.
Five inductive biases: MLP, RGB CNN, RGB-D CNN, centered-XYZ CNN, patch Transformer.
GT camera poses generate only training/evaluation correspondence labels.
Not foundation-model training; small pretext benchmark, single scene.
"""
import argparse,json,pathlib
import numpy as np,cv2,torch
import torch.nn as nn,torch.nn.functional as F
from scipy.spatial.transform import Rotation
SEED=20261102
np.random.seed(SEED);torch.manual_seed(SEED);torch.set_num_threads(2)
def index(path):
    out=[]
    for line in pathlib.Path(path).read_text().splitlines():
        if line.startswith('#') or not line.strip():continue
        a=line.split()
        if len(a)>1:
            try:out.append((float(a[0]),a[1:]))
            except ValueError:pass
    return sorted(out)
def sequence(root):
    root=pathlib.Path(root);r=index(root/'rgb.txt');d=index(root/'depth.txt');g=index(root/'groundtruth.txt')
    nearest=lambda a,t:min(a,key=lambda p:abs(p[0]-t))
    rows=[]
    for t,p in r:
        dep=nearest(d,t);gt=nearest(g,t)
        if abs(dep[0]-t)<.045 and abs(gt[0]-t)<.045 and len(gt[1])>=7:
            rows.append((root/p[0],root/dep[1][0],np.array(list(map(float,gt[1][:7])))))
    out=[]
    for i in np.unique(np.round(np.linspace(0,len(rows)-1,min(100,len(rows)))).astype(int)):
        rp,dp,pose=rows[i];rgb=cv2.imread(str(rp));dep=cv2.imread(str(dp),-1)
        if rgb is None or dep is None:continue
        h,w=dep.shape;rgb=cv2.cvtColor(cv2.resize(rgb,(160,120)),cv2.COLOR_BGR2RGB).astype('f')/255
        dep=cv2.resize(dep,(160,120),interpolation=cv2.INTER_NEAREST).astype('f')/5000
        K=np.array([517.3*160/w,516.5*120/h,318.6*160/w,255.3*120/h])
        out.append((rgb,dep,K,Rotation.from_quat(pose[3:]).as_matrix(),pose[:3]))
    return out
def crop(frame,x,y):
    rgb,D,K,R,T=frame
    if not(5<=x<155 and 5<=y<115):return None
    im=rgb[y-4:y+5,x-4:x+5];d=D[y-4:y+5,x-4:x+5]
    if np.mean((d>.3)&(d<4.5))<.8 or not(.3<D[y,x]<4.5):return None
    xx,yy=np.meshgrid(np.arange(x-4,x+5),np.arange(y-4,y+5))
    xyz=np.stack(((xx-K[2])*d/K[0],(yy-K[3])*d/K[1],d),-1)
    xyz=np.clip((xyz-xyz[4,4])/.25,-2,2)
    return np.concatenate([im,np.clip(d[...,None]/5,0,1),xyz],axis=-1).transpose(2,0,1).astype('f')
def pairs(frames):
    rng=np.random.default_rng(SEED);X=[];Y=[];times=[]
    for i in range(len(frames)-3):
        a,b=frames[i],frames[i+3];_,depth,K2,R2,T2=b
        _,z,K,R,T=a
        for _ in range(42):
            x=int(rng.integers(6,154));y=int(rng.integers(6,114));Z=z[y,x]
            if not(.3<Z<4.5):continue
            q=np.array([(x-K[2])*Z/K[0],(y-K[3])*Z/K[1],Z])
            q=(q@R.T+T-T2)@R2
            if q[2]<.3:continue
            xx=int(round(K2[0]*q[0]/q[2]+K2[2]));yy=int(round(K2[1]*q[1]/q[2]+K2[3]))
            if not(5<=xx<155 and 5<=yy<115) or abs(depth[yy,xx]-q[2])>.08:continue
            A=crop(a,x,y);B=crop(b,xx,yy)
            if A is None or B is None:continue
            X.append(A);Y.append(B);times.append(i)
    return torch.from_numpy(np.asarray(X)),torch.from_numpy(np.asarray(Y)),np.asarray(times)
class Encoder(nn.Module):
    def __init__(self,kind):
        super().__init__();self.kind=kind
        c=3 if kind=='rgb_cnn' else 6 if kind=='geometry_cnn' else 4
        if kind=='mlp':self.net=nn.Sequential(nn.Flatten(),nn.Linear(c*81,128),nn.GELU(),nn.Linear(128,32))
        elif kind=='patchvit':
            self.proj=nn.Linear(c*9,48);self.pos=nn.Parameter(torch.zeros(1,9,48))
            self.enc=nn.TransformerEncoder(nn.TransformerEncoderLayer(48,4,96,dropout=0,batch_first=True),1)
            self.head=nn.Linear(48,32)
        else:self.net=nn.Sequential(nn.Conv2d(c,24,3,padding=1),nn.GELU(),
                nn.Conv2d(24,32,3,padding=1),nn.GELU(),nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),nn.Linear(32,32))
    def forward(self,x):
        x=x[:,:3] if self.kind=='rgb_cnn' else x[:,:6] if self.kind=='geometry_cnn' else x[:,:4]
        if self.kind=='patchvit':
            tokens=F.unfold(x,3,stride=3).transpose(1,2)
            z=self.head(self.enc(self.proj(tokens)+self.pos).mean(1))
        else:z=self.net(x)
        return F.normalize(z,dim=-1)
def train(X,Y,times,epochs):
    sep=int(.7*(times.max()+1));it=np.where(times<sep)[0];iv=np.where(times>=sep)[0]
    if len(it)<150 or len(iv)<60:raise RuntimeError(f'Insufficient real GT-paired patches: {len(it)}, {len(iv)}')
    rng=np.random.default_rng(SEED);results={}
    for name in ['mlp','rgb_cnn','rgbd_cnn','geometry_cnn','patchvit']:
        torch.manual_seed(SEED);net=Encoder(name);opt=torch.optim.Adam(net.parameters(),lr=.002)
        for epoch in range(epochs):
            order=rng.permutation(it)
            net.train()
            for batch in np.array_split(order,max(1,len(order)//96)):
                if len(batch)<4:continue
                a=net(X[batch]);p=net(Y[batch]);negative=p.roll(1,0)
                loss=F.relu(.25-(a*p).sum(-1)+(a*negative).sum(-1)).mean()
                opt.zero_grad();loss.backward();opt.step()
        net.eval()
        with torch.no_grad():
            a=net(X[iv]);p=net(Y[iv]);neg=p.roll(1,0)
            rank=float(((a*p).sum(-1)>(a*neg).sum(-1)).float().mean())
        results[name]={'ranking_accuracy':round(rank,6),'train':len(it),'test':len(iv),
                       'parameter_count':sum(p.numel() for p in net.parameters()),'epochs':epochs}
    return results
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--out',default='results/real_rgbd/tum_xyz');p.add_argument('--epochs',type=int,default=8);a=p.parse_args()
    frames=sequence(a.root);X,Y,t=pairs(frames)
    print('real measured GT-paired patches:',len(X),flush=True)
    scores=train(X,Y,t,a.epochs)
    report={'dataset':'TUM fr1 xyz real RGB-D','frames':len(frames),'pairs':len(X),'scores':scores,
        'caveats':'Single scene, tiny scratch-trained models, correspondence ranking only, not 3D object ID or foundation model superiority.'}
    out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True)
    (out/'neural_contrastive.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
