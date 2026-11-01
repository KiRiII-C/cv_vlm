"""EXP-20261102-04: stronger real RGB-D representation training with a blind Bonn test.

50 epochs (configurable), contrastive positive pairs + hard measured negative patches,
validation-only checkpoint selection and temporal guard gaps.
Trains tiny-to-medium models FROM SCRATCH, not pretrained visual foundation models.
No Bonn training/selection. GitHub Actions artifacts contain learning curves and weights.
"""
from __future__ import annotations
import argparse,json,pathlib,random,time
import numpy as np,torch
import torch.nn as nn
import torch.nn.functional as F
from real_neural_contrastive import sequence,pairs
from hard_negative_retrieval import corrected_frames,create_queries
from cross_scene_transfer import gather
SEED=20261102
NAMES=('mlp_plus','rgbd_cnn_plus','vit_2layer','vit_4layer','geometry_vit','conv_transformer')

class StrongEncoder(nn.Module):
    def __init__(self,kind):
        super().__init__()
        self.kind=kind
        n_channels=3 if kind=='rgb_cnn_plus' else 4
        if kind=='mlp_plus':
            self.net=nn.Sequential(nn.Flatten(),nn.Linear(4*81,256),nn.LayerNorm(256),nn.GELU(),
                         nn.Linear(256,128),nn.GELU(),nn.Linear(128,64))
        elif kind=='rgbd_cnn_plus':
            self.net=nn.Sequential(nn.Conv2d(4,32,3,padding=1),nn.GELU(),
                         nn.Conv2d(32,64,3,padding=1),nn.GELU(),
                         nn.Conv2d(64,96,3,padding=1),nn.GELU(),
                         nn.AdaptiveAvgPool2d((3,3)),nn.Flatten(),nn.Linear(96*9,64))
        else:
            if kind=='vit_2layer':w,L,heads=96,2,4
            elif kind=='vit_4layer':w,L,heads=128,4,8
            elif kind=='geometry_vit':w,L,heads=96,2,4
            elif kind=='conv_transformer':w,L,heads=96,2,4
            else:raise ValueError('Unsupported model '+kind)
            self.width=w
            if kind=='conv_transformer':
                self.stem=nn.Sequential(nn.Conv2d(4,48,3,padding=1),nn.GELU(),
                                        nn.Conv2d(48,w,3,stride=3),nn.GELU())
            else:self.proj=nn.Linear(4*9,w)
            self.pos=nn.Parameter(torch.zeros(1,10,w))
            nn.init.normal_(self.pos,std=.01)
            self.cls=nn.Parameter(torch.zeros(1,1,w))
            layer=nn.TransformerEncoderLayer(w,heads,4*w,dropout=0.05,
                                             batch_first=True,activation='gelu',norm_first=True)
            self.enc=nn.TransformerEncoder(layer,L,enable_nested_tensor=False)
            self.head=nn.Sequential(nn.LayerNorm(w),nn.Linear(w,64))
    def forward(self,x):
        if self.kind=='geometry_vit':x=x[:,3:7]
        else:x=x[:,:4]
        if self.kind=='mlp_plus' or self.kind=='rgbd_cnn_plus':
            out=self.net(x)
        else:
            if self.kind=='conv_transformer':
                tokens=self.stem(x).flatten(2).transpose(1,2)
            else:tokens=self.proj(F.unfold(x,3,stride=3).transpose(1,2))
            cls=self.cls.expand(tokens.shape[0],-1,-1)
            tokens=torch.cat((cls,tokens),dim=1)+self.pos
            out=self.head(self.enc(tokens)[:,0])
        return F.normalize(out,dim=-1)

def augment(x):
    # Independent, mild photometric perturbation, measured geometry kept unchanged.
    v=x.clone()
    v[:,:3]=(v[:,:3]*torch.empty(len(v),1,1,1).uniform_(.92,1.08)+
              torch.empty(len(v),1,1,1).uniform_(-.035,.035)).clamp(0,1)
    v[:,3:4]=(v[:,3:4]+torch.randn_like(v[:,3:4])*.003).clamp(0,1)
    return v

def rank5(net,queries,device='cpu'):
    X,Y,groups=queries
    net.eval()
    with torch.no_grad():
        scores=[]
        for i in range(0,len(X),128):
            q=net(X[i:i+128])
            C=Y[i:i+128]
            cand=net(C.reshape(-1,7,9,9)).reshape(len(q),5,-1)
            scores.extend(torch.argmax(torch.einsum('bd,bkd->bk',q,cand),dim=1).cpu().numpy()==0)
    return float(np.mean(scores))

def to_data(rows):
    X,Y,groups=gather(rows)
    return (torch.from_numpy(X),torch.from_numpy(Y),groups)

def bootstrap(net,qs,seed):
    X,Y,groups=qs;net.eval();success=[]
    with torch.no_grad():
        for i in range(0,len(X),128):
            q=net(X[i:i+128])
            batch=Y[i:i+128]
            c=net(batch.reshape(-1,7,9,9)).reshape(len(q),5,-1)
            success.extend((torch.argmax(torch.einsum('bd,bkd->bk',q,c),dim=1)==0).numpy().tolist())
    success=np.asarray(success,dtype=float)
    d={k:success[groups==k] for k in np.unique(groups)}
    rng=np.random.default_rng(seed);ids=list(d)
    vals=[]
    for _ in range(300):
        sample=rng.choice(ids,len(ids),replace=True)
        vals.append(np.mean(np.concatenate([d[k] for k in sample])))
    return dict(rank1=float(success.mean()),n_queries=len(success),n_view_pairs=len(ids),
        ci95=[float(q) for q in np.percentile(vals,[2.5,97.5])])

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--tum-root',required=True);p.add_argument('--bonn-root',required=True)
    p.add_argument('--model',choices=NAMES,required=True);p.add_argument('--epochs',type=int,default=48)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--out',default='results/strong_training')
    args=p.parse_args()
    if args.epochs<8:raise ValueError('Need at least 8 epochs for requested stronger training')
    seed=SEED+args.seed;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(2)
    T=sequence(args.tum_root)
    X,Y,times=pairs(T)
    train_ids=np.where(times<58)[0]
    if len(train_ids)<500:raise RuntimeError('Too few real training pairs')
    rawT,_=corrected_frames(args.tum_root,[517.3,516.5,318.6,255.3],'tum_xyz',100)
    rawB,_=corrected_frames(args.bonn_root,[542.822841,542.576870,315.593520,237.756098],'bonn_box',84)
    H=to_data(create_queries(rawT[:58]))
    V=to_data(create_queries(rawT[64:79]))
    HT=to_data(create_queries(rawT[82:]))
    HB=to_data(create_queries(rawB))
    if len(H[0])<150 or len(V[0])<45 or len(HT[0])<45 or len(HB[0])<100:
        raise RuntimeError('Insufficient genuine measured cross-view hard-negative samples')
    model=StrongEncoder(args.model)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0015,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs,eta_min=.00005)
    output_dir=pathlib.Path(args.out)/args.model/f'seed_{args.seed}'
    output_dir.mkdir(parents=True,exist_ok=True)
    best=(-1.,float('inf'));best_epoch=None;curve=[]
    # Equal optimization budget across architectures; no Bonn model selection.
    for epoch in range(args.epochs):
        model.train()
        order=torch.randperm(len(train_ids))
        horder=torch.randperm(len(H[0]))
        losses=[];pair_losses=[];hard_losses=[]
        batches=[a for a in order.split(64) if len(a)>=8]
        for step,inds in enumerate(batches):
            ids=torch.as_tensor(train_ids[inds])
            a=model(augment(X[ids]));b=model(augment(Y[ids]))
            logits=a@b.T/.12
            labels=torch.arange(len(ids))
            lp=(F.cross_entropy(logits,labels)+F.cross_entropy(logits.T,labels))*.5
            hstart=(step*32)%len(horder)
            hi=horder[hstart:hstart+32]
            if len(hi)<8:hi=horder[:32]
            hq=model(augment(H[0][hi]))
            hc=model(augment(H[1][hi].reshape(-1,7,9,9))).reshape(len(hi),5,-1)
            lh=F.cross_entropy(torch.einsum('bd,bkd->bk',hq,hc)/.12,
                               torch.zeros(len(hi),dtype=torch.long))
            loss=lp+.7*lh
            optimizer.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            losses.append(float(loss.item()));pair_losses.append(float(lp.item()));hard_losses.append(float(lh.item()))
        scheduler.step()
        val=rank5(model,V)
        item={'epoch':epoch+1,'loss':float(np.mean(losses)),
              'pair_loss':float(np.mean(pair_losses)),'hard_loss':float(np.mean(hard_losses)),
              'validation_hard_rank1':val,'learning_rate':scheduler.get_last_lr()[0]}
        curve.append(item)
        print('TRAIN_EPOCH',args.model,args.seed,json.dumps(item),flush=True)
        if val>best[0] or (val==best[0] and item['loss']<best[1]):
            best=(val,item['loss']);best_epoch=epoch+1
            torch.save(model.state_dict(),output_dir/'best.pt')
    model.load_state_dict(torch.load(output_dir/'best.pt',weights_only=True,map_location='cpu'))
    report={
        'experiment_id':'EXP-20261102-04',
        'dataset_train':'TUM RGB-D fr1/xyz real measured; train first 58 frames',
        'validation':'TUM frames 64:79, selected best validation checkpoint, no Bonn selection',
        'test':'TUM frames 82:100 and Bonn independent moving_obstructing_box, no Bonn tuning',
        'model':args.model,'seed':args.seed,'epochs_run':args.epochs,
        'parameter_count':sum(p.numel() for p in model.parameters()),
        'training_positive_pairs':len(train_ids),'training_hard_queries':len(H[0]),
        'validation_queries':len(V[0]),'best_epoch':best_epoch,'validation_best':best[0],
        'test_TUM':bootstrap(model,HT,seed+20),
        'test_Bonn':bootstrap(model,HB,seed+30),
        'learning_curve':curve,
        'limitations':'Single train room, two test sequences, CPU scratch training, RGB-selected hard negatives; no 3D object instance GT. Checkpoint selected by TUM validation ONLY.',
    }
    (output_dir/'metrics.json').write_text(json.dumps(report,indent=2))
    print('STRONG_TRAINING_RESULT',json.dumps({k:v for k,v in report.items() if k!='learning_curve'},sort_keys=True),flush=True)
if __name__=='__main__':main()
