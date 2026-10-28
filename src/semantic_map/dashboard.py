"""Self-contained offline interactive visualization of the *synthetic* mapping test.

Real-world 3D map not claimed: visualization is of simulated object boxes and
2D occupied cells. Operator commands are grounded to canonical object tracks.
"""
from __future__ import annotations
from pathlib import Path
import json, math
import numpy as np
from .benchmark import run
from .models import Track
from .planner import OccupancyGrid,CommandProcessor
from .scene_graph import compute_relations

def make_world(objects):
    res=.1; w,h=125,90
    a=np.zeros((h,w),dtype=bool)
    a[:2,:]=1;a[-2:,:]=1;a[:,:2]=1;a[:,-2:]=1
    a[2:30,60:63]=1;a[46:-2,60:63]=1
    for o in objects:
        lo=np.asarray(o['low']);hi=np.asarray(o['high'])
        xmin=max(2,int((lo[0]-.05)/res));xmax=min(w-2,int((hi[0]+.05)/res)+1)
        ymin=max(2,int((lo[1]-.05)/res));ymax=min(h-2,int((hi[1]+.05)/res)+1)
        a[ymin:ymax,xmin:xmax]=1
    return OccupancyGrid(a,resolution=res)

def go(output='results/semantic_demo'):
    from matplotlib import pyplot as plt
    from matplotlib.colors import ListedColormap
    import plotly.graph_objects as pg
    out=Path(output);out.mkdir(exist_ok=True,parents=True)
    payload,raw,robot_path=run(out/'tracking',frames=120)
    tracks=[]
    for o in raw:
        tracks.append(Track(object_id=o['object_id'],label=o['label'],center=np.array(o['center']),
                            low=np.array(o['low']),high=np.array(o['high']),appearance=None,
                            hits=o['hits'],first_seen=0,last_seen=119,state=o['state'],
                            description=o['description'],attributes={'color':o['color']}))
    grid=make_world(raw)
    robot=np.array([1.,4.]);yaw=0.
    cp=CommandProcessor(tracks,grid,robot_xy=robot,robot_yaw=yaw,radius=.16)
    queries=[
        'Подъедь к зелёному стулу',
        'Подъедь к зелёному растению',
        'Подъедь к красному стулу',
        'Встань между шкафом и растением',
        'Встань справа от шкафа',
        'Подъедь к фиолетовой лампе',
    ]
    answers=[cp.run(q).__dict__ for q in queries]
    relations=compute_relations(tracks)
    summary={'provenance':'synthetic world and synthetic observations; no real camera benchmark',
             'scene_objects':len(tracks),'confirmed_objects':sum(t.state=='confirmed' for t in tracks),
             'occupancy_resolution_m':grid.res,'robot_xy':robot.tolist(),
             'commands':answers,'scene_graph':relations,'identity_ablation':payload['results']}
    (out/'operator_commands.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    (out/'grid_occupancy.npy').write_bytes(grid.grid.astype('uint8').tobytes())
    cmap=plt.get_cmap('tab20')
    fig,ax=plt.subplots(figsize=(14,9))
    ax.imshow(grid.grid.astype(int),origin='lower',extent=[0,12.5,0,9],cmap=ListedColormap(['#f5f7f9','#4c566a']),alpha=.92)
    for idx,t in enumerate(tracks):
        x,y=t.center[:2];ax.scatter([x],[y],s=110,color=cmap(idx%20),edgecolors='black',zorder=4)
        ax.text(x+.15,y+.10,f'{t.label} #{idx+1}\n{t.description}',fontsize=8,color='#1c2630',weight='bold')
    ax.scatter(*robot,marker='*',s=250,c='#e76f51',edgecolors='black',zorder=8,label='robot')
    styles=['#e63946','#2a9d8f','#7b2cbf','#ff9f1c','#277da1','#bc6c25']
    for idx,cmd in enumerate(answers):
        if cmd['status']=='planned':
            p=np.array(cmd['path_xy'])
            ax.plot(p[:,0],p[:,1],linewidth=2.1,alpha=.86,color=styles[idx],label=f'cmd {idx+1}: {cmd["status"]}')
            ax.scatter([cmd['goal_xy'][0]],[cmd['goal_xy'][1]],s=100,marker='X',color=styles[idx],zorder=8)
    ax.set(xlim=(0,12.5),ylim=(0,9),xlabel='X map [m]',ylabel='Y map [m]',
           title='Object-centric semantic map — controlled SYNTHETIC test (not real RGB-D)')
    ax.legend(loc='upper right',fontsize=9);ax.grid(alpha=.18);fig.tight_layout()
    fig.savefig(out/'map_2d.png',dpi=180);plt.close(fig)
    # 3D interactive object boxes: hover exposes IDs and attributes, no fictional scanned point cloud.
    fig3=pg.Figure()
    for idx,t in enumerate(tracks):
        x,y,z=t.center;lo=t.low;hi=t.high
        corners=np.array([[a,b,c] for a in [lo[0],hi[0]] for b in [lo[1],hi[1]] for c in [lo[2],hi[2]]])
        E=[(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
        ex=[];ey=[];ez=[]
        for i,j in E:
            ex.extend([corners[i,0],corners[j,0],None]);ey.extend([corners[i,1],corners[j,1],None]);ez.extend([corners[i,2],corners[j,2],None])
        fig3.add_trace(pg.Scatter3d(x=ex,y=ey,z=ez,mode='lines',showlegend=False,
                                   line=dict(width=3,color='rgba(45,70,90,.55)'),hoverinfo='skip'))
        fig3.add_trace(pg.Scatter3d(x=[x],y=[y],z=[z],mode='markers+text',
                                   text=[str(idx+1)+' '+t.label],textposition='top center',
                                   marker=dict(size=7,color=idx,colorscale='Viridis',cmin=0,cmax=12),
                                   hovertemplate=f'{t.label}<br>{t.description}<br>ID {t.object_id}<br>hits {t.hits}<extra></extra>',
                                   name=f'{idx+1} {t.label}'))
    fig3.update_layout(title='3D instance memory — synthetic box reconstructions',
                       scene=dict(xaxis_title='X (m)',yaxis_title='Y (m)',zaxis_title='Z (m)',
                                  aspectmode='data'),margin=dict(l=0,r=0,b=0,t=45))
    fig3.write_html(out/'interactive_3d.html',include_plotlyjs=True,full_html=True)
    # Synthetic map object AABB cloud PLY: centers and 8 bounds per instance.
    colors=[];points=[]
    for idx,t in enumerate(tracks):
        coords=np.array([[a,b,c] for a in np.linspace(t.low[0],t.high[0],9)
                         for b in np.linspace(t.low[1],t.high[1],9)
                         for c in [t.low[2],t.high[2]]])
        points.extend(coords.tolist());colors.extend([[int(255*(idx%4)/4),int(255*(idx%3)/3),180]]*len(coords))
    with (out/'synthetic_objects.ply').open('w') as f:
        f.write('ply\nformat ascii 1.0\nelement vertex '+str(len(points))+'\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
        for p,c in zip(points,colors):f.write(' '.join(f'{v:.4f}' for v in p)+' '+' '.join(map(str,c))+'\n')
    print('DASHBOARD',out,'commands',[(x['query'],x['status']) for x in answers],flush=True)
    return summary
if __name__=='__main__':go()
