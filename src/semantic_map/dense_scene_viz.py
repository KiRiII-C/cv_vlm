"""Visualize actual RGB-D geometry plus per-object 3D points, each UUID a stable color."""
import argparse,json
from pathlib import Path
import numpy as np

def read_colored(path):
    with open(path,'rb') as f:
        for _ in range(128):
            line=f.readline().decode('ascii').strip()
            if line.startswith('element vertex '):n=int(line.split()[-1])
            if line=='end_header':break
        else:raise ValueError('Invalid PLY')
        dt=np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),('r','u1'),('g','u1'),('b','u1'),('instance_index','<u4'),('votes','<u4')])
        data=np.fromfile(f,dtype=dt,count=n)
    if len(data)!=n:raise ValueError('Truncated PLY')
    return data

def visualize(geometry,semantics,out):
    import open3d as o3d
    import plotly.graph_objects as go
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    geo=Path(geometry);sem=Path(semantics);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    cloud=o3d.io.read_point_cloud(str(geo/'tsdf_cloud.ply'))
    bg=np.asarray(cloud.points)
    pts=read_colored(sem/'object_instances_colored.ply')
    info=json.loads((sem/'object_instances_colored.instances.json').read_text())
    report=json.loads((sem/'real_mapping_report.json').read_text())
    records={x['id']:(x['label'],x['hits']) for x in report['objects']}
    if len(bg)<5000 or len(pts)<500:raise ValueError('Dense instance clouds missing')
    rng=np.random.default_rng(20261028)
    b=bg[rng.choice(len(bg),min(len(bg),90000),replace=False)]
    p=pts[rng.choice(len(pts),min(len(pts),150000),replace=False)]
    fig=go.Figure()
    fig.add_trace(go.Scatter3d(x=b[:,0],y=b[:,1],z=b[:,2],name='TSDF background',mode='markers',
                    marker={'size':1,'opacity':.2,'color':'#9ea5b0'},hoverinfo='skip'))
    count=0
    for key,record in info['instances'].items():
        uid=record['uuid'];sel=p['instance_index']==int(key)
        if not sel.any():continue
        count+=1;color=record['rgb'];category,hits=records.get(uid,('unknown',0))
        fig.add_trace(go.Scatter3d(x=p['x'][sel],y=p['y'][sel],z=p['z'][sel],
          name=f'{category} | {uid[-8:]} | {hits} frames',mode='markers',
          marker={'size':2.3,'opacity':.92,'color':f'rgb({color[0]},{color[1]},{color[2]})'},
          hovertemplate=f'{category}<br>ID: {uid}<br>X=%{{x:.2f}} Y=%{{y:.2f}} Z=%{{z:.2f}}<extra></extra>'))
    fig.update_layout(title='Dense RGB-D TSDF and UUID-colored instance point clouds',
        template='plotly_dark',scene={'aspectmode':'data'},height=850,
        margin={'l':0,'r':0,'t':50,'b':0})
    fig.write_html(str(out/'dense_3d_instances.html'),include_plotlyjs=True)
    fig2,ax=plt.subplots(figsize=(11,9))
    ax.scatter(b[:,0],b[:,1],s=.4,c='gray',alpha=.2,rasterized=True)
    cols=np.column_stack([p[k] for k in ('r','g','b')])/255.
    ax.scatter(p['x'],p['y'],s=.7,c=cols,alpha=.85,rasterized=True)
    ax.set(xlabel='X, m',ylabel='Y, m',title='Dense point map; every permanent UUID has one color')
    ax.set_aspect('equal');fig2.savefig(out/'dense_topdown.png',dpi=180,bbox_inches='tight');plt.close(fig2)
    summary={'tsdf_surface_points':len(bg),'colored_foreground_points':len(pts),
             'colored_ids':count,'voxel_m':info['voxel_m'],
             'source':'YOLO masks projected through real TUM depth and VO poses',
             'identity_GT':'NOT_AVAILABLE'}
    (out/'dense_summary.json').write_text(json.dumps(summary,indent=2))
    print('DENSE_OBJECT_MAP',json.dumps(summary),flush=True)
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--geometry',default='results/full_geometry')
    p.add_argument('--semantics',default='results/tum_real')
    p.add_argument('--out',default='results/dense_semantic_viz')
    a=p.parse_args()
    visualize(a.geometry,a.semantics,a.out)
