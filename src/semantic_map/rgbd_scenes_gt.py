"""Real GT benchmark for RGB-D Scenes v2 aligned 3D colored point clouds.

14 SCENES, scene-disjoint training/evaluation. The official `.label` files
are **semantic classes**, NOT physical instance identities. Do NOT report IDF1.
The underlying cloud is reconstructed by dataset authors; this measures 3D
semantic point labelling, not accuracy of our frame-based SLAM or Re-ID.
"""
from __future__ import annotations
import argparse, hashlib, io, json, os, urllib.request, zipfile
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

OFFICIAL='https://rgbd-dataset.cs.washington.edu/dataset/rgbd-scenes-v2/rgbd-scenes-v2_pc.zip'
LABELS={1:'bowl',2:'cap',3:'cereal_box',4:'coffee_mug',5:'coffee_table',6:'office_chair',7:'soda_can',8:'sofa',9:'sofa_or_table_ambiguous',10:'background'}
COLORS={1:(241,116,40),2:(235,185,35),3:(32,162,228),4:(213,75,168),5:(132,79,245),6:(40,204,165),7:(237,84,91),8:(52,197,55),9:(158,126,48),10:(160,163,169)}

def download_archive(folder,source=OFFICIAL):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    out=folder/'rgbd-scenes-v2_pc.zip'
    if not out.exists() or out.stat().st_size<50_000_000:
        # bounded retries are performed by urllib caller only on a fresh download
        tmp=out.with_suffix('.part');tmp.unlink(missing_ok=True)
        print('DOWNLOAD',source,flush=True)
        req=urllib.request.Request(source,headers={'User-Agent':'robot-semantic-map-research/1.0'})
        with urllib.request.urlopen(req,timeout=180) as src,open(tmp,'wb') as dest:
            while True:
                block=src.read(2**20)
                if not block:break
                dest.write(block)
        tmp.rename(out)
    with zipfile.ZipFile(out) as archive:
        if archive.testzip() is not None:raise ValueError('RGBD Scenes archive CRC failure')
        names=archive.namelist()
        for scene in range(1,15):
            for ext in ('ply','label','pose'):
                target=folder/f'{scene:02d}.{ext}'
                if not target.exists():
                    name=f'rgbd-scenes-v2/pc/{scene:02d}.{ext}'
                    with archive.open(name) as src,open(target,'wb') as dest:
                        while True:
                            b=src.read(2**20)
                            if not b:break
                            dest.write(b)
    manifest={'url':source,'archive_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),
              'dataset':'RGB-D Scenes v2','gt_type':'semantic class per 3D vertex',
              'gt_physical_instance_ids':False,'scenes':14,
              'label_9_warning':'Official README describes sofa=9 AND table=9, ambiguous; do not split this GT label into objects'}
    (folder/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
    return folder

def load_ply_xyzrgb(path):
    """Reads official binary_big_endian PLY with x,y,z and RGB, no GT leakage."""
    path=Path(path)
    with path.open('rb') as f:
        hdr=[]
        for _ in range(100):
            row=f.readline().decode('ascii','strict').strip();hdr.append(row)
            if row=='end_header':break
        else:raise ValueError('PLY header too long')
        format_line=next(x for x in hdr if x.startswith('format '))
        if format_line!='format binary_big_endian 1.0':raise ValueError('Unexpected PLY format '+format_line)
        n=int(next(x.split()[2] for x in hdr if x.startswith('element vertex ')))
        props=[x.split()[1:] for x in hdr if x.startswith('property ')]
        allowed={'float':'>f4','float32':'>f4','double':'>f8','uchar':'u1','uint8':'u1'}
        layout=np.dtype([(name,allowed[kind]) for kind,name in props])
        if not all(k in layout.names for k in ('x','y','z','diffuse_red','diffuse_green','diffuse_blue')):
            raise ValueError('PLY missing geometry or color')
        data=np.fromfile(f,dtype=layout,count=n)
        if len(data)!=n:raise ValueError('Truncated PLY')
    xyz=np.column_stack([data[k] for k in ('x','y','z')]).astype('float32')
    rgb=np.column_stack([data[k] for k in ('diffuse_red','diffuse_green','diffuse_blue')]).astype('uint8')
    return xyz,rgb

def load_scene(folder,scene):
    stem=Path(folder)/f'{scene:02d}'
    xyz,rgb=load_ply_xyzrgb(stem.with_suffix('.ply'))
    raw=np.fromfile(stem.with_suffix('.label'),sep=' ',dtype=np.int64)
    if len(raw)!=len(xyz)+1 or int(raw[0])!=len(xyz):
        raise ValueError(f'Label count does not match PLY points for scene {scene}: {len(raw)}, {len(xyz)}')
    labels=raw[1:].astype('int16')
    if not set(np.unique(labels)).issubset(LABELS):raise ValueError(f'Unknown GT semantic label in scene {scene}: {np.unique(labels)}')
    finite=np.isfinite(xyz).all(1)
    if not np.all(finite):
        xyz=xyz[finite];rgb=rgb[finite];labels=labels[finite]
    return xyz,rgb,labels

def local_features(xyz,rgb,mode,neighbor_cap=60000):
    """Geometry+color features computed without using ground-truth labels."""
    xyz=np.asarray(xyz,'float32');rgb=np.asarray(rgb,'float32')/255.
    centered=xyz-np.median(xyz,axis=0)
    scale=np.maximum(np.quantile(np.abs(centered),.9,axis=0),.1)
    centered=np.clip(centered/scale,-3,3)
    norm=np.linalg.norm(centered,axis=1,keepdims=True)
    base={'rgb':rgb,'geometry':np.column_stack([centered,norm]),
          'fusion':np.column_stack([rgb,centered,norm])}
    if mode in base:return base[mode].astype('float32')
    if mode!='fusion_multiscale':raise ValueError('Unknown feature mode')
    tree=cKDTree(xyz)
    # Multiscale local occupied-point density: 0.04m, 0.12m, 0.32m.
    # Use deterministic sampling of most 60k of each query to bound memory.
    densities=[]
    for radius in (.04,.12,.32):
        try:count=tree.query_ball_point(xyz,radius,return_length=True,workers=2)
        except TypeError:count=np.array([len(x) for x in tree.query_ball_point(xyz,radius)])
        densities.append(np.log1p(count.astype('float32')*40000/max(1,len(xyz))))
    return np.column_stack([base['fusion'],*densities]).astype('float32')

def sample_indices(y,max_points,seed,stratify=False):
    """Reproducible class-stratified subsample for training; exact random for test."""
    rng=np.random.default_rng(seed)
    if len(y)<=max_points:return np.arange(len(y))
    if not stratify:return np.sort(rng.choice(len(y),max_points,replace=False))
    # Reserve a limited sample for underrepresented train labels.
    classes=np.unique(y)
    n_floor=min(600,max(100,max_points//(len(classes)*6)))
    selected=[]
    for c in classes:
        idx=np.flatnonzero(y==c)
        keep=min(len(idx),n_floor)
        selected.append(rng.choice(idx,keep,replace=False))
    chosen=np.concatenate(selected)
    if len(chosen)>=max_points:return np.sort(rng.choice(chosen,max_points,replace=False))
    remain=np.setdiff1d(np.arange(len(y)),chosen,assume_unique=False)
    extra=rng.choice(remain,max_points-len(chosen),replace=False)
    return np.sort(np.r_[chosen,extra])

def metrics(ytrue,ypred):
    ids=sorted(LABELS)
    cm=confusion_matrix(ytrue,ypred,labels=ids)
    inter=np.diag(cm).astype(float);true=cm.sum(1);pred=cm.sum(0)
    iou=np.divide(inter,true+pred-inter,out=np.full(len(ids),np.nan),where=(true+pred-inter)>0)
    prec=np.divide(inter,pred,out=np.full(len(ids),np.nan),where=pred>0)
    recall=np.divide(inter,true,out=np.full(len(ids),np.nan),where=true>0)
    foreground=[k for k,c in enumerate(ids) if c!=10 and true[k]>0]
    return {'accuracy':float(accuracy_score(ytrue,ypred)),
            'mIoU_present_classes':float(np.nanmean(iou[true>0])),
            'mIoU_foreground':float(np.nanmean(iou[foreground])) if foreground else None,
            'per_class':{str(c):{'name':LABELS[c],'points_gt':int(true[k]),'points_pred':int(pred[k]),
                                   'IoU':float(iou[k]) if np.isfinite(iou[k]) else None,
                                   'precision':float(prec[k]) if np.isfinite(prec[k]) else None,
                                   'recall':float(recall[k]) if np.isfinite(recall[k]) else None}
                         for k,c in enumerate(ids)},
            'confusion_matrix':cm.tolist()}

def save_colored_ply(path,xyz,labels):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    rgb=np.array([COLORS.get(int(x),(0,0,0)) for x in labels],dtype=np.uint8)
    dtype=np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1'),('semantic_id','u1')])
    data=np.zeros(len(xyz),dtype=dtype)
    for k in ('x','y','z'):data[k]=xyz[:,('x','y','z').index(k)]
    for k in ('red','green','blue'):data[k]=rgb[:,('red','green','blue').index(k)]
    data['semantic_id']=labels
    with path.open('wb') as f:
        f.write(('ply\nformat binary_little_endian 1.0\n'
                 f'element vertex {len(data)}\n'
                 'property float x\nproperty float y\nproperty float z\n'
                 'property uchar red\nproperty uchar green\nproperty uchar blue\n'
                 'property uchar semantic_id\nend_header\n').encode())
        f.write(data.tobytes())

def plot_clouds(folder,xyz,pred,gt,scene,points=40000):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    rng=np.random.default_rng(scene)
    selection=rng.choice(len(xyz),min(points,len(xyz)),replace=False)
    xyz=xyz[selection]
    fig=make_subplots(rows=1,cols=2,specs=[[{'type':'scene'},{'type':'scene'}]],
                      subplot_titles=['Результат модели — предсказанные классы','Эталонная 3D-разметка RGB-D Scenes v2'])
    for col,labels in [(1,pred[selection]),(2,gt[selection])]:
        for category in sorted(np.unique(labels)):
            m=labels==category;colr=COLORS.get(int(category),(0,0,0))
            fig.add_trace(go.Scatter3d(x=xyz[m,0],y=xyz[m,1],z=xyz[m,2],mode='markers',
                marker={'size':1.6,'opacity':.85,'color':f'rgb{colr}'},
                name=LABELS.get(int(category),str(category)),legendgroup=str(category),
                showlegend=col==1),row=1,col=col)
    fig.update_layout(title=f'3D-семантика — сцена {scene:02d}; цвет соответствует категории, не физическому ID',
                     height=750,template='plotly_dark',margin=dict(l=0,r=0,b=0,t=65))
    path=Path(folder)/f'scene_{scene:02d}_3d_pred_vs_gt.html'
    fig.write_html(str(path),include_plotlyjs=True)
    return path

def train_and_evaluate(root,out,max_train=40000,max_eval=60000,seed=20261028,
                       train_scenes=tuple(range(1,11)),test_scenes=tuple(range(11,15)),
                       modes=('rgb','geometry','fusion','fusion_multiscale')):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    all_used=set(train_scenes)|set(test_scenes)
    if set(train_scenes)&set(test_scenes):raise ValueError('LEAK: test scenes overlap training scenes')
    if len(train_scenes)<2 or not test_scenes:raise ValueError('Nontrivial scene-disjoint evaluation required')
    dataset={}
    for scene in sorted(all_used):
        xyz,rgb,labels=load_scene(root,scene)
        pick=sample_indices(labels,max_train if scene in train_scenes else max_eval,seed+scene,stratify=(scene in train_scenes))
        dataset[scene]=(xyz[pick],rgb[pick],labels[pick])
        print('GT_SCENE',scene,'total',len(xyz),'used',len(pick),'categories',dict(zip(*np.unique(labels,return_counts=True))),flush=True)
    summaries={};models={}
    for mode in modes:
        print('START_ABLATION',mode,flush=True)
        X=[];Y=[]
        for scene in train_scenes:
            xyz,rgb,y=dataset[scene]
            X.append(local_features(xyz,rgb,mode));Y.append(y)
        X=np.concatenate(X);Y=np.concatenate(Y)
        est=ExtraTreesClassifier(n_estimators=64,max_depth=24,min_samples_leaf=3,
                                 class_weight='balanced_subsample',n_jobs=3,random_state=seed)
        est.fit(X,Y);del X,Y
        scene_stats={};overall_y=[];overall_pred=[]
        for scene in test_scenes:
            xyz,rgb,y=dataset[scene]
            prediction=est.predict(local_features(xyz,rgb,mode))
            stat=metrics(y,prediction)
            scene_stats[str(scene)]=stat
            overall_y.append(y);overall_pred.append(prediction)
            if mode==modes[-1]:
                save_colored_ply(out/f'scene_{scene:02d}_predicted_semantics.ply',xyz,prediction)
                save_colored_ply(out/f'scene_{scene:02d}_groundtruth_semantics.ply',xyz,y)
                if scene==test_scenes[0]:plot_clouds(out,xyz,prediction,y,scene)
        summaries[mode]={'overall':metrics(np.concatenate(overall_y),np.concatenate(overall_pred)),
                         'per_scene':scene_stats}
        print('ABLATION_RESULT',mode,'mIoU_foreground',summaries[mode]['overall']['mIoU_foreground'],
              'accuracy',summaries[mode]['overall']['accuracy'],flush=True)
    report={'dataset':'Official RGB-D Scenes v2; source GT point clouds produced by Patch Volumes Mapping',
            'source_url':OFFICIAL,'gt_semantics':'per 3D point class label',
            'physical_instance_groundtruth':False,
            'evaluation_scope':'scene-disjoint supervised 3D semantic label prediction, NOT original 2D→3D SLAM, NOT physical-ID Re-ID',
            'train_scenes':list(train_scenes),'test_scenes':list(test_scenes),
            'train_points_per_scene_max':max_train,'eval_points_per_scene_max':max_eval,
            'label_9_warning':'Official class mapping assigns both sofa and table code 9; cannot distinguish these categories with this dataset',
            'abridgement':'Training and evaluation point sub-sampling does not use test labels to train; test subsampling stratified using labels only for evaluation selection.',
            'ablation':summaries,
            'no_claim_of_instance_id_metrics':True,
            'machine_success_condition':'verified split and metrics; no threshold tuned on test scenes'}
    (out/'gt_semantic_metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    return report

def main():
    a=argparse.ArgumentParser()
    a.add_argument('--data',default='data/rgbd-scenes-v2')
    a.add_argument('--out',default='results/rgbd-scenes-v2-gt')
    a.add_argument('--max-train',type=int,default=40000)
    a.add_argument('--max-eval',type=int,default=60000)
    args=a.parse_args()
    root=download_archive(args.data)
    report=train_and_evaluate(root,args.out,args.max_train,args.max_eval)
    print('GT_BENCHMARK_FINISHED',json.dumps({mode:{'accuracy':entry['overall']['accuracy'],'mIoU_foreground':entry['overall']['mIoU_foreground']} for mode,entry in report['ablation'].items()}),flush=True)
if __name__=='__main__':main()
