"""Full-sequence RGB-D depth completion and posed 3D reconstruction.
Train on ALL TUM fr1_xyz synchronized RGB-D frames and ALL 154 TUM Scribble RGB-D pairs.
Evaluate on ALL synchronized TUM fr1_desk frames; 3D maps use available ground-truth camera poses.
Note: this benchmarks depth inpainting and geometry, NOT semantic mapping or SLAM.
"""
import os, json, time, tarfile, urllib.request, random, math
from pathlib import Path
import numpy as np, cv2, torch, requests
from scipy.spatial.transform import Rotation
from experiment import Net

BASE=Path('full_data'); BASE.mkdir(exist_ok=True)
OUT=Path('full_results'); OUT.mkdir(exist_ok=True)
H,W=96,128
FX,FY,CX,CY=517.3,516.5,318.6,255.3
SEED=20261028
EPOCHS=int(os.getenv('EPOCHS','20'))
STRIDE=int(os.getenv('MAP_STRIDE','3'))
VOXEL=float(os.getenv('VOXEL','0.05'))
MODES=['depth','early','attn_no_geo','geo_attn']
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(3)

def download(url,out):
    out=Path(out)
    if out.exists() and out.stat().st_size>100000:return
    print('DOWNLOAD',url,flush=True)
    tmp=out.with_suffix(out.suffix+'.partial')
    req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0 RGB-D benchmark'})
    with urllib.request.urlopen(req,timeout=180) as resp,tmp.open('wb') as f:
        while True:
            c=resp.read(2**20)
            if not c:break
            f.write(c)
    tmp.replace(out)

def ensure_tum(seq):
    name=f'rgbd_dataset_freiburg1_{seq}'
    archive=BASE/(name+'.tgz')
    if not archive.exists():
        errors=[]
        for host in ('https://vision.in.tum.de','https://webshare.cvg.cit.tum.de','https://cvg.cit.tum.de'):
            try:
                download(f'{host}/rgbd/dataset/freiburg1/{name}.tgz',archive);break
            except Exception as e:
                errors.append(f'{host}:{e}');archive.unlink(missing_ok=True)
        if not archive.exists():raise RuntimeError('Cannot download '+name+': '+str(errors))
    root=BASE/name
    if (root/'groundtruth.txt').exists() and len(list((root/'depth').glob('*.png')))>100:
        return root
    root.mkdir(exist_ok=True)
    print('EXTRACT',name,flush=True)
    with tarfile.open(archive,'r:gz') as t:
        for member in t:
            if not member.isfile():continue
            mp=Path(member.name)
            if mp.is_absolute() or '..' in mp.parts or mp.parts[0]!=name:raise ValueError('Unsafe tar path')
            dest=BASE/mp;dest.parent.mkdir(parents=True,exist_ok=True)
            with t.extractfile(member) as src,dest.open('wb') as dst:
                while True:
                    chunk=src.read(2**20)
                    if not chunk:break
                    dst.write(chunk)
    return root

def index(root,name):
    rows=[]
    for line in (root/name).read_text().splitlines():
        if not line.strip() or line.startswith('#'):continue
        s=line.split()
        try:rows.append((float(s[0]),s[1:]))
        except (ValueError,IndexError):pass
    return rows

def sequence(root):
    rgb=index(root,'rgb.txt'); dep=index(root,'depth.txt'); pose=index(root,'groundtruth.txt')
    dep_ts=np.array([t for t,_ in dep]);pose_ts=np.array([t for t,_ in pose])
    samples=[]
    for ts,paths in rgb:
        j=int(np.searchsorted(dep_ts,ts))
        js=[k for k in [j-1,j] if 0<=k<len(dep_ts)]
        if not js:continue
        j=min(js,key=lambda k:abs(dep_ts[k]-ts))
        if abs(dep_ts[j]-ts)>.025:continue
        k=int(np.searchsorted(pose_ts,ts))
        ks=[i for i in [k-1,k] if 0<=i<len(pose_ts)]
        k=min(ks,key=lambda i:abs(pose_ts[i]-ts)) if ks else -1
        pose_data=None
        if k>=0 and abs(pose_ts[k]-ts)<.03 and len(pose[k][1])>=7:
            p=np.array([float(v) for v in pose[k][1][:3]],np.float32)
            q=np.array([float(v) for v in pose[k][1][3:7]],np.float32)
            pose_data=(p,Rotation.from_quat(q).as_matrix().astype(np.float32))
        samples.append(dict(rgb=root/paths[0],depth=root/dep[j][1][0],id=ts,pose=pose_data,source=root.name))
    print('PAIRS',root.name,'total',len(samples),'posed',sum(s['pose'] is not None for s in samples),flush=True)
    return samples

def scribble_pairs():
    base='https://raw.githubusercontent.com/tum-vision/rgbd_scribble_benchmark/master/LabeledImages/'
    r=requests.get('https://api.github.com/repos/tum-vision/rgbd_scribble_benchmark/contents/LabeledImages',timeout=60)
    r.raise_for_status()
    names={item['name'] for item in r.json()}
    stems=sorted(n[:-len('_image.png')] for n in names if n.endswith('_image.png') and n.replace('_image.png','_depth.png') in names)
    if len(stems)<150:raise ValueError('Scribble index incomplete: '+str(len(stems)))
    root=BASE/'scribble';root.mkdir(exist_ok=True)
    out=[]
    for i,stem in enumerate(stems):
        paths=[]
        for typ in ('image','depth'):
            f=root/f'{stem}_{typ}.png'
            if not f.exists():
                resp=requests.get(base+f.name,timeout=60);resp.raise_for_status();f.write_bytes(resp.content)
            paths.append(f)
        out.append(dict(rgb=paths[0],depth=paths[1],id=stem,pose=None,source='scribble'))
        if (i+1)%40==0:print('SCRIBBLE',i+1,'/',len(stems),flush=True)
    return out

def frame(f):
    im=cv2.imread(str(f['rgb']),cv2.IMREAD_COLOR)
    d=cv2.imread(str(f['depth']),cv2.IMREAD_UNCHANGED)
    if im is None or d is None or d.ndim!=2 or im.shape[:2]!=d.shape:raise ValueError('Invalid aligned RGB-D '+str(f['rgb']))
    rgb=cv2.cvtColor(cv2.resize(im,(W,H),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB)
    dep=d.astype('float32')/(1000 if f['source']=='scribble' else 5000)
    good=((dep>.35)&(dep<4.5)).astype('float32')
    n=cv2.resize(dep*good,(W,H),interpolation=cv2.INTER_AREA)
    b=cv2.resize(good,(W,H),interpolation=cv2.INTER_AREA)
    z=np.where(b>.7,n/np.maximum(b,1e-6),0).astype('float32')
    valid=((b>.7)&(z>.35)&(z<4.5)).astype('float32')
    return dict(rgb=torch.from_numpy(np.ascontiguousarray(rgb.transpose(2,0,1)))[None].float()/255,
                d=torch.from_numpy(z.copy())[None,None],valid=torch.from_numpy(valid.copy())[None,None],
                color=rgb,id=f['id'],pose=f['pose'],source=f['source'])

def make_input(item,seed):
    generator=torch.Generator().manual_seed(seed)
    d=item['d']; valid=item['valid']
    hidden=(torch.rand(d.shape,generator=generator)<.12).float()
    for i in range(3):
        bh=int(torch.randint(8,25,(1,),generator=generator))
        bw=int(torch.randint(10,30,(1,),generator=generator))
        y=int(torch.randint(0,H-bh,(1,),generator=generator))
        x=int(torch.randint(0,W-bw,(1,),generator=generator))
        hidden[:,:,y:y+bh,x:x+bw]=1
    missing=hidden*valid
    observed=(valid-missing).clamp_min(0)
    return item['rgb'],d*observed,observed,missing,d

def measure(sum3,pred,gt,missing):
    err=(pred-gt)*missing
    sum3[0]+=float(np.square(err).sum());sum3[1]+=float(np.abs(err).sum());sum3[2]+=int(missing.sum())

def metrics(v):
    return dict(rmse_m=float(np.sqrt(v[0]/v[2])),mae_m=float(v[1]/v[2]),pixels=int(v[2])) if v[2]>0 else None

class VoxelMap:
    def __init__(self):
        self.gt=set();self.raw=set();self.pred=set();self.nposed=0;self.colors={}
    def integrate(self,item,zraw,zpred,zgt,mask):
        if item['pose'] is None:return
        self.nposed+=1
        Y,X=np.mgrid[0:H:STRIDE,0:W:STRIDE]
        xyz_t,R=item['pose']
        allz={'gt':zgt[::STRIDE,::STRIDE],
              'raw':zraw[::STRIDE,::STRIDE],
              'pred':np.where(mask>.5,zpred,zraw)[::STRIDE,::STRIDE]}
        gtvalid=(allz['gt']>.35)&(allz['gt']<4.5)
        for key,z in allz.items():
            v=(z>.35)&(z<4.5)&gtvalid
            if not v.any():continue
            depth=z[v];x=X[v]*640/W;y=Y[v]*480/H
            pts=np.stack([depth*(x-CX)/FX,depth*(y-CY)/FY,depth],axis=1)
            pts=pts@R.T+xyz_t
            vox=np.unique(np.floor(pts/VOXEL).astype(np.int32),axis=0)
            getattr(self,key).update(map(tuple,vox))
            if key=='pred' and self.nposed%10==0:
                for k in vox[::2]:
                    self.colors.setdefault(tuple(k),[110,155,210])
    def evaluate(self):
        joint=len(self.gt|self.pred); inter=len(self.gt&self.pred)
        raw_inter=len(self.gt&self.raw)
        return dict(voxel_m=VOXEL,reference_voxels=len(self.gt),raw_voxels=len(self.raw),
                    predicted_voxels=len(self.pred),occupied_voxel_iou=inter/max(joint,1),
                    occupied_voxel_precision=inter/max(len(self.pred),1),
                    occupied_voxel_recall=inter/max(len(self.gt),1),
                    raw_iou=raw_inter/max(len(self.gt|self.raw),1),
                    frames_with_reference_pose=self.nposed,
                    spatial_subsampling_px=STRIDE)
    def ply(self,typ,path):
        v=sorted(getattr(self,typ))
        with open(path,'w') as f:
            f.write('ply\nformat ascii 1.0\nelement vertex '+str(len(v))+'\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
            for x,y,z in v:
                px,py,pz=(x+.5)*VOXEL,(y+.5)*VOXEL,(z+.5)*VOXEL
                c=self.colors.get((x,y,z),[160,170,180]) if typ=='pred' else [180,180,180]
                f.write(f'{px:.3f} {py:.3f} {pz:.3f} {c[0]} {c[1]} {c[2]}\n')

def fit(mode,items):
    torch.manual_seed(SEED)
    net=Net(mode);opt=torch.optim.AdamW(net.parameters(),lr=.001)
    history=[];start=time.time()
    for epoch in range(EPOCHS):
        order=np.random.default_rng(SEED+epoch).permutation(len(items))
        losses=[];net.train()
        for j in order:
            rgb,z,obs,missing,gt=make_input(items[int(j)],SEED+epoch*100000+int(j))
            pred=net(rgb,z,obs)
            loss=(torch.abs(pred-gt)*missing).sum()/missing.sum().clamp_min(1)
            opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),5)
            opt.step();losses.append(loss.detach().item())
        history.append(float(np.mean(losses)))
        print('TRAIN',mode,'epoch',epoch+1,'/',EPOCHS,'mae',round(history[-1],4),'s',round(time.time()-start),flush=True)
    torch.save({'state_dict':net.state_dict(),'mode':mode,'epochs':EPOCHS},OUT/(mode+'.pt'))
    return net,history,round(time.time()-start,2)

def test(mode,net,items):
    totals=np.zeros(3);vm=VoxelMap();per=[]
    for i,item in enumerate(items):
        rgb,z,obs,hidden,gt=make_input(item,8000000+i)
        if mode=='raw': pred=z
        elif mode=='telea':
            p=cv2.inpaint(z[0,0].numpy().astype('float32'),(hidden[0,0].numpy()>.5).astype('uint8'),3,cv2.INPAINT_TELEA)
            pred=torch.from_numpy(p)[None,None]
        else:
            net.eval()
            with torch.inference_mode():pred=net(rgb,z,obs)
        pred=pred.clamp(.35,4.5)
        p,g,m=pred[0,0].numpy(),gt[0,0].numpy(),hidden[0,0].numpy()
        measure(totals,p,g,m)
        per_error=np.zeros(3);measure(per_error,p,g,m)
        per.append({'timestamp':str(item['id']),'metrics':metrics(per_error),'pose_found':item['pose'] is not None})
        vm.integrate(item,z[0,0].numpy(),p,g,m)
        if i%100==0:print('TEST',mode,i,'/',len(items),flush=True)
    (OUT/(mode+'_per_frame.json')).write_text(json.dumps(per))
    vm.ply('pred',OUT/(mode+'_map.ply'))
    if mode=='raw':vm.ply('gt',OUT/'reference_map.ply')
    print('EVAL',mode,metrics(totals),vm.evaluate(),flush=True)
    return metrics(totals),vm.evaluate()

def main():
    begin=time.time()
    train_a=sequence(ensure_tum('xyz'))
    test_a=sequence(ensure_tum('desk'))
    train_b=scribble_pairs()
    train=[frame(s) for s in train_a+train_b]
    testset=[frame(s) for s in test_a]
    res={'setup':{'seed':SEED,'epochs':EPOCHS,'input_H_W':[H,W],
                  'train':'ALL Freiburg1 xyz + ALL TUM Scribble RGB-D',
                  'test':'ALL Freiburg1 desk synchronized RGB/depth pairs',
                  'map_poses':'nearest TUM mocap reference within 30 ms',
                  'voxel_m':VOXEL,'depth_scale':'TUM 5000; Scribble 1000',
                  'task':'reconstruct artificially held-out VALID depth pixels; no semantic or SLAM evaluation'},
         'frames':{'train_xyz':len(train_a),'train_scribble':len(train_b),
                   'train_total':len(train),'test_desk':len(test_a),
                   'test_poses':sum(s['pose'] is not None for s in test_a)},
         'models':{}}
    for mode in MODES:
        net,history,seconds=fit(mode,train)
        res['models'][mode]={'train_mae_history_m':history,'training_seconds':seconds,
                             'parameters':sum(p.numel() for p in net.parameters())}
        met,mapping=test(mode,net,testset)
        res['models'][mode].update(depth=met,map=mapping)
        (OUT/'results.json').write_text(json.dumps(res,indent=2))
    for mode in ('raw','telea'):
        met,mapping=test(mode,None,testset)
        res['models'][mode]={'depth':met,'map':mapping}
        (OUT/'results.json').write_text(json.dumps(res,indent=2))
    res['duration_sec']=round(time.time()-begin,2)
    (OUT/'results.json').write_text(json.dumps(res,indent=2))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        names=list(res['models'])
        for metric,name in [('rmse_m','depth_rmse'),('mae_m','depth_mae')]:
            fig,ax=plt.subplots(figsize=(9,4))
            ax.bar(names,[res['models'][m]['depth'][metric] for m in names])
            ax.set_ylabel(metric+' (lower better)');fig.tight_layout()
            fig.savefig(OUT/(name+'.png'),dpi=150);plt.close(fig)
        fig,ax=plt.subplots(figsize=(9,4))
        ax.bar(names,[res['models'][m]['map']['occupied_voxel_iou'] for m in names])
        ax.set_ylabel('3D occupied voxel IoU at 5cm (higher better)')
        fig.tight_layout();fig.savefig(OUT/'map_voxel_iou.png',dpi=150);plt.close(fig)
    except Exception as e:print('PLOT WARNING',repr(e),flush=True)
    print('ALL_DONE',json.dumps({'frames':res['frames'],'duration_sec':res['duration_sec']}),flush=True)

if __name__=='__main__':main()
