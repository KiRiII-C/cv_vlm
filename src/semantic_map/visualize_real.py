"""Plot actual observed-surface PLY and confirmed YOLO instance centers.

Do not label it as an occupancy map or as a reconstructed mesh.
"""
import argparse,json
from pathlib import Path
import numpy as np

def render(root='results/tum_real',max_points=60000):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    import plotly.graph_objects as go
    root=Path(root)
    data=json.loads((root/'real_mapping_report.json').read_text())
    with open(root/'observed_surface.ply') as f:
        for line in f:
            if line.strip()=='end_header':break
        pts=np.loadtxt(f,dtype=np.float32,usecols=(0,1,2))
    pts=np.atleast_2d(pts)
    if len(pts)>max_points:pts=pts[np.linspace(0,len(pts)-1,max_points).astype(int)]
    objects=sorted([t for t in data['objects'] if t['state']=='confirmed'],key=lambda o:-o['hits'])
    fig=plt.figure(figsize=(12,7));ax=fig.add_subplot(projection='3d')
    if len(pts):ax.scatter(pts[:,0],pts[:,1],pts[:,2],s=.15,c=pts[:,2],cmap='viridis',alpha=.4)
    for i,o in enumerate(objects):
        c=o['center_xyz_m'];ax.scatter(*c,s=45 if i<12 else 13,c='orangered' if i<12 else 'gray',alpha=.9 if i<12 else .40)
        if i<12:ax.text(*c,o['label']+' '+o['id'][4:8],fontsize=8)
    ax.set(xlabel='X [m]',ylabel='Y [m]',zlabel='Z [m]',
           title='TUM Freiburg1 Desk: real RGB-D observed surfaces + YOLO instance proposals')
    fig.tight_layout();fig.savefig(root/'real_map_3d.png',dpi=160);plt.close(fig)
    fig=go.Figure()
    if len(pts):fig.add_trace(go.Scatter3d(x=pts[:,0],y=pts[:,1],z=pts[:,2],mode='markers',
                            marker=dict(size=1,opacity=.15,color=pts[:,2],colorscale='Viridis'),
                            name='RGB-D observed surface',hoverinfo='skip'))
    for o in objects:
        x,y,z=o['center_xyz_m'];fig.add_trace(go.Scatter3d(x=[x],y=[y],z=[z],mode='markers+text',
                 marker=dict(size=6,color='orangered'),text=[o['label']],name=o['label'],
                 hovertemplate=f"{o['label']}<br>ID: {o['id']}<br>hits: {o['hits']}<extra></extra>"))
    fig.update_layout(title='Real TUM RGB-D sensor points and pretrained YOLO segmentation proposals',
                      scene=dict(aspectmode='data'))
    fig.write_html(root/'real_map_interactive.html',include_plotlyjs=True)
    print('VISUALIZED_REAL',len(pts),'points',len(objects),'objects')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='results/tum_real');a=p.parse_args();render(a.root)
