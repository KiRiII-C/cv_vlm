#!/usr/bin/env python3
"""B6 reproducible dense display of real B5 RGB-D/YOLOE point measurements.

No mesh interpolation, no hallucinated surface points. Each 1.2 cm voxel
stores the centroid of ACTUAL measured YOLO mask points plus vote-weighted
predicted class; disputed class assignments are excluded from rendering.
Optional C0 UUID markers are hypotheses, never physical GT.
"""
from __future__ import annotations
import argparse,json
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import trimesh
import plotly.graph_objects as go

COLORS={'table':'#f97316','chair':'#fb7185','cabinet':'#a78bfa',
 'laptop':'#00d3ff','tv':'#facc15','keyboard':'#34d399',
 'mouse':'#f472b6','cup':'#60a5fa'}

def fuse_measured(xyz,labels,confidence,voxel=.012):
    xyz=np.asarray(xyz,dtype=np.float32)
    labels=np.asarray(labels).astype(str)
    conf=np.asarray(confidence,dtype=np.float32)
    if xyz.ndim!=2 or xyz.shape[1]!=3 or len(xyz)!=len(labels) or len(xyz)!=len(conf):
        raise ValueError("Real B5 XYZ, class and confidence must be aligned")
    if not np.isfinite(xyz).all() or not .003<=voxel<=.1:
        raise ValueError("Invalid metric points/voxel")
    q=np.floor(xyz/voxel).astype(np.int32)
    _,inv=np.unique(q,axis=0,return_inverse=True)
    n=int(inv.max())+1 if len(inv) else 0
    counts=np.bincount(inv,minlength=n).astype(np.int32)
    centers=np.empty((n,3),dtype=np.float32)
    for axis in range(3):
        centers[:,axis]=np.bincount(inv,weights=xyz[:,axis],minlength=n)/counts
    names=np.unique(labels)
    weights=np.stack([np.bincount(inv,weights=np.where(labels==name,conf,0),
                          minlength=n) for name in names],axis=1)
    winner=weights.argmax(axis=1)
    purity=weights[np.arange(n),winner]/np.maximum(weights.sum(axis=1),1.e-9)
    return centers,names[winner],counts,purity

def render(points,labels,purity,base,out):
    fig=plt.figure(figsize=(17.5,10),facecolor="#0b1220",dpi=150)
    ax=fig.add_axes([.025,.19,.59,.68],projection="3d",facecolor="#0b1220")
    xy=fig.add_axes([.66,.56,.29,.30],facecolor="#101a2a")
    xz=fig.add_axes([.66,.18,.29,.30],facecolor="#101a2a")
    ax.scatter(base[:,0],base[:,1],base[:,2],s=.4,c="#9aa5b1",alpha=.12,
               depthshade=False,rasterized=True)
    for sub,dim in ((xy,1),(xz,2)):
        sub.scatter(base[:,0],base[:,dim],s=.3,c="#9aa5b1",alpha=.12,rasterized=True)
    for name in sorted(set(labels)):
        xyz=points[(labels==name)&(purity>=.58)]
        if not len(xyz):continue
        color=COLORS.get(str(name),"#d9e7ec")
        ax.scatter(xyz[:,0],xyz[:,1],xyz[:,2],s=1.6,c=color,
                   alpha=.85,depthshade=False,rasterized=True)
        xy.scatter(xyz[:,0],xyz[:,1],s=1.4,c=color,alpha=.74,rasterized=True)
        xz.scatter(xyz[:,0],xyz[:,2],s=1.4,c=color,alpha=.74,rasterized=True)
    p_lo=np.quantile(points,.002,axis=0);p_hi=np.quantile(points,.998,axis=0)
    mid=(p_lo+p_hi)/2;rad=float(np.max(p_hi-p_lo))/2*1.1
    ax.set_xlim(mid[0]-rad,mid[0]+rad);ax.set_ylim(mid[1]-rad,mid[1]+rad)
    ax.set_zlim(mid[2]-.75*rad,mid[2]+.75*rad)
    ax.set_box_aspect((1,1,.75));ax.view_init(elev=26,azim=-68)
    ax.set_title("3D: измеренные объектные поверхности",color="white",fontsize=14)
    for axis in (ax.xaxis,ax.yaxis,ax.zaxis):
        axis.set_tick_params(colors="#9fb2ca",labelsize=7)
        axis.pane.fill=False
    for sub,dim in ((xy,1),(xz,2)):
        sub.set_title("Проекция X"+("Y" if dim==1 else "Z"),color="white")
        sub.set_xlabel("X, м",color="#9fb2ca");sub.set_ylabel(("Y" if dim==1 else "Z")+", м",color="#9fb2ca")
        sub.tick_params(colors="#9fb2ca");sub.grid(alpha=.14)
    fig.text(.04,.946,"ПЛОТНАЯ СЕМАНТИЧЕСКАЯ КАРТА",color="white",size=21,weight="bold")
    fig.text(.04,.91,"RTAB-Map A6 + YOLOE-26x · реальные RGB-D · взвешенное голосование классов в измеренных пространственных вокселях",
             color="#a8bad0",size=10)
    names=[x for x in sorted(set(labels)) if x in COLORS]
    from matplotlib.lines import Line2D
    handles=[Line2D([],[],color=COLORS[x],marker="o",ls="",markersize=8,label=x) for x in names]
    fig.legend(handles=handles,ncol=4,loc="center",bbox_to_anchor=(.67,.105),
               labelcolor="white",frameon=False)
    fig.text(.04,.046,"Каждая точка — агрегат реально измеренных RGB-D-точек. Класс ≠ подтверждённый физический ID.",
             color="#f6b2a4",size=10)
    fig.savefig(out,facecolor=fig.get_facecolor())
    plt.close(fig)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--cloud",type=Path,required=True)
    p.add_argument("--tsdf",type=Path,required=True)
    p.add_argument("--out",type=Path,default=Path("results/b6"))
    p.add_argument("--voxel",type=float,default=.012)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    data=np.load(a.cloud)
    centers,labels,counts,purity=fuse_measured(
        data["xyz"],data["labels"],data["confidence"],a.voxel)
    base=np.asarray(trimesh.load(a.tsdf,process=False).vertices,dtype=float)
    if len(base)<5000 or len(centers)<30000:raise RuntimeError("Real B5 data incomplete")
    render(centers,labels,purity,base,a.out/"dense_semantic_map.png")
    # Separate an object-class view; no false physical UUID boundary.
    fig=plt.figure(figsize=(17,9),facecolor="#0b1220",dpi=135)
    ax=fig.subplots(2,4,subplot_kw={"projection":"3d"})
    for plot,name in zip(ax.flat,sorted(COLORS)):
        pts=centers[(labels==name)&(purity>=.58)]
        if len(pts):
            plot.scatter(pts[:,0],pts[:,1],pts[:,2],s=1.6,c=COLORS[name],
                         alpha=.86,rasterized=True,depthshade=False)
        plot.set_title(f"{name} · {len(pts):,}",color=COLORS[name],fontsize=13)
        plot.set_axis_off()
    fig.suptitle("ОБЪЕКТНЫЕ ОБЛАКА ПО КАТЕГОРИЯМ (не physical-ID)",color="white",fontsize=19)
    fig.tight_layout()
    fig.savefig(a.out/"object_clouds_by_class.png",facecolor=fig.get_facecolor())
    plt.close(fig)
    plot=go.Figure()
    plot.add_trace(go.Scatter3d(x=base[:,0],y=base[:,1],z=base[:,2],mode="markers",
        name="RTAB-Map TSDF",marker={"size":1,"color":"#a4adba","opacity":.12},visible="legendonly"))
    for name in sorted(COLORS):
        pts=centers[(labels==name)&(purity>=.58)]
        if not len(pts):continue
        if len(pts)>20000:pts=pts[np.linspace(0,len(pts)-1,20000,dtype=int)]
        plot.add_trace(go.Scatter3d(x=pts[:,0],y=pts[:,1],z=pts[:,2],name=name,
            mode="markers",marker={"size":1.8,"color":COLORS[name],"opacity":.85}))
    plot.update_layout(template="plotly_dark",scene={"aspectmode":"data"},
       title="B6: измеренные семантические RGB-D облака YOLOE",
       margin={"l":0,"r":0,"t":55,"b":0})
    plot.write_html(a.out/"interactive_dense_semantic_map.html",
                    include_plotlyjs=True,full_html=True)
    np.savez_compressed(a.out/"semantic_voxels_measured.npz",
       xyz=centers,labels=labels,support=counts,purity=purity)
    rgba=np.tile(np.array([128,139,155,255],dtype=np.uint8),(len(centers),1))
    for name,color in COLORS.items():
        rgb=np.asarray(matplotlib.colors.to_rgb(color))*255
        rgba[labels==name,:3]=np.asarray(rgb,dtype=np.uint8)
    trimesh.points.PointCloud(centers,colors=rgba).export(a.out/"semantic_voxels_colored.ply")
    meta={"source_measured_YOLO_XYZ":len(data["xyz"]),
          "source_A6_TSDF_points":len(base),"voxel_size_m":a.voxel,
          "observed_voxels":len(centers),
          "no_synthetic_or_interpolated_points":True,
          "class_votes_are_not_physical_UUIDs":True,
          "navigation_ready":False,
          "class_voxels":dict(Counter(str(q) for q in labels))}
    (a.out/"b6_report.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2)+"\n")
    print("B6_DENSE_MEASURED_MAP",json.dumps(meta,ensure_ascii=False),flush=True)

if __name__=="__main__":main()
