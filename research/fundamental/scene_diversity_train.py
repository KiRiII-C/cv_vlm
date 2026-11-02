"""EXP-20261102-07: Does real scene diversity improve transferable RGB-D embeddings?

Balanced 2-arm controlled study:
  A. First 54 frames of TUM fr1/xyz only.
  B. First 54 frames of fr1/xyz PLUS first 54 frames of fr1/desk.
Both use IDENTICAL number of optimizer updates and losses, same architecture,
same seeds and validation protocol. Bonn is entirely blind, without fine-tuning.
"""
import argparse,json,pathlib,random
import numpy as np,torch
import torch.nn.functional as F
from real_neural_contrastive import sequence,pairs
from strong_encoder_training import StrongEncoder,augment,rank5,bootstrap,to_data
from hard_negative_retrieval import corrected_frames,create_queries

SEED=20261102
KINDS=('mlp_plus','vit_4layer','geometry_vit')
ARMS=('xyz_only','xyz_plus_desk')

def make(root,which):
    f=sequence(root)
    X,Y,t=pairs(f)
    mask=t<54
    return X[mask],Y[mask],to_data(create_queries(corrected_frames(root,[517.3,516.5,318.6,255.3],which,100)[0][:54]))

def concat_hard(qa,qb=None):
    if qb is None:return qa
    X=torch.cat([qa[0],qb[0]],dim=0)
    Y=torch.cat([qa[1],qb[1]],dim=0)
    return (X,Y,np.arange(len(X)))

def data_for_test(root,which):
    f,_=corrected_frames(root,[542.822841,542.576870,315.593520,237.756098] if which=='bonn_box' else [517.3,516.5,318.6,255.3],which,84 if which=='bonn_box' else 100)
    return to_data(create_queries(f if which=='bonn_box' else f[82:]))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--xyz-root',required=True);parser.add_argument('--desk-root',required=True);parser.add_argument('--bonn-root',required=True)
    parser.add_argument('--model',choices=KINDS,required=True)
    parser.add_argument('--arm',choices=ARMS,required=True)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--epochs',type=int,default=28)
    parser.add_argument('--out',default='results/real_rgbd/scene_diversity')
    args=parser.parse_args()
    if args.epochs<16:raise ValueError('Require >=16 epochs')
    seed=SEED+args.seed
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(2)
    XA,YA,HA=make(args.xyz_root,'tum_xyz')
    XB,YB,HB=make(args.desk_root,'tum_desk')
    if len(XA)<500 or len(XB)<500:raise RuntimeError('Not enough real measured positive train correspondences')
    # Same update budget for both arms; only variety changes.
    X=torch.cat([XA,XB],dim=0) if args.arm=='xyz_plus_desk' else XA
    Y=torch.cat([YA,YB],dim=0) if args.arm=='xyz_plus_desk' else YA
    H=concat_hard(HA,HB) if args.arm=='xyz_plus_desk' else HA
    val_xyz,_=corrected_frames(args.xyz_root,[517.3,516.5,318.6,255.3],'tum_xyz',100)
    val_desk,_=corrected_frames(args.desk_root,[517.3,516.5,318.6,255.3],'tum_desk',100)
    V=concat_hard(to_data(create_queries(val_xyz[64:79])),to_data(create_queries(val_desk[64:79])))
    test_desk=to_data(create_queries(val_desk[82:]))
    test_bonn=data_for_test(args.bonn_root,'bonn_box')
    if len(H[0])<150 or len(V[0])<65 or len(test_desk[0])<45 or len(test_bonn[0])<100:
        raise RuntimeError('Invalid real measured hard query split')
    model=StrongEncoder(args.model)
    opt=torch.optim.AdamW(model.parameters(),lr=.0015,weight_decay=.01)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,args.epochs,eta_min=.00005)
    out=pathlib.Path(args.out)/args.model/args.arm/('seed_'+str(args.seed))
    out.mkdir(parents=True,exist_ok=True)
    best=-1.;best_epoch=0;metrics=[]
    # Draw exactly 960 training positive pairs per epoch; same 15 batch updates.
    for ep in range(args.epochs):
        model.train()
        inds=torch.randint(0,len(X),(960,))  # enforce exactly 15 minibatches also for one-scene arm
        hindices=torch.randperm(len(H[0]))
        total=0
        for step,idx in enumerate(inds.split(64)):
            j=hindices[(step*24):(step*24+24)]
            if len(j)<8:j=hindices[:24]
            a=model(augment(X[idx]));b=model(augment(Y[idx]))
            scores=a@b.T/.12
            labels=torch.arange(len(idx))
            lp=(F.cross_entropy(scores,labels)+F.cross_entropy(scores.T,labels))*.5
            q=model(augment(H[0][j]))
            c=model(augment(H[1][j].reshape(-1,7,9,9))).reshape(len(j),5,-1)
            hard=F.cross_entropy(torch.einsum('bd,bkd->bk',q,c)/.12,torch.zeros(len(j),dtype=torch.long))
            loss=lp+.7*hard
            opt.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step();total+=float(loss.item())
        schedule.step()
        val=rank5(model,V)
        row={'epoch':ep+1,'loss':total/15,'validation_rank1':val}
        metrics.append(row)
        print('SCENE_DIVERSITY_EPOCH',args.model,args.arm,args.seed,json.dumps(row),flush=True)
        if val>best:
            best=val;best_epoch=ep+1
            torch.save(model.state_dict(),out/'best.pt')
    model.load_state_dict(torch.load(out/'best.pt',weights_only=True,map_location='cpu'))
    result={'experiment':'EXP-20261102-07','model':args.model,'arm':args.arm,'seed':args.seed,
      'epochs':args.epochs,'best_epoch':best_epoch,'validation_rank1':best,
      'same_update_budget':'960 real positive samples/epoch, 15 64-sample updates for both arms',
      'train_pairs_available':len(X),'train_hard_available':len(H[0]),
      'TUM_desk_heldout':bootstrap(model,test_desk,seed+70),
      'Bonn_blind':bootstrap(model,test_bonn,seed+80),
      'curves':metrics,'parameter_count':sum(p.numel() for p in model.parameters()),
      'limitations':'Both TUM sequences are from one laboratory; geographic independence only provided by Bonn. Increased diversity arm also increases total unique observations, but optimizer updates are matched; no full object GT. Camera GT poses generate retrieval labels.'}
    (out/'metrics.json').write_text(json.dumps(result,indent=2))
    print('EXP07_RESULT',json.dumps({k:v for k,v in result.items() if k!='curves'},sort_keys=True),flush=True)
if __name__=='__main__':main()
