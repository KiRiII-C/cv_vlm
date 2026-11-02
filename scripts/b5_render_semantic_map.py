#!/usr/bin/env python3
"""B5 actual semantic map visualization: RTAB-Map A6 TSDF + real YOLOE RGBD.

Every displayed point is reconstructed from recorded TUM depth. Grey base:
A6 full 594-pose RTAB-Map optimized TSDF point cloud. Class colours:
real depth pixels inside predicted YOLOE instance masks at 119 of 595
real RGB-D timestamps. NO label is accepted as a verified physical ID.
"""
from __future__ import annotations
import argparse, json
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import trimesh

COLORS={
 "chair":"#f77f00","table":"#e63946","sofa":"#a855f7","cabinet":"#2a9d8f",
 "bookshelf":"#9b5de5","bed":"#e76f51","laptop":"#16a6e0",
 "computer monitor":"#2563eb","monitor":"#2563eb","tv":"#2563eb",
 "keyboard":"#14b8a6","mouse":"#f4a261","cup":"#ffcf33",
 "potted plant":"#3eaf59"}
LABELS={"chair":"Стул","table":"Стол","sofa":"Диван","cabinet":"Шкаф",
 "bookshelf":"Стеллаж","bed":"Кровать","laptop":"Ноутбук",
 "computer monitor":"Монитор","monitor":"Монитор","tv":"Экран",
 "keyboard":"Клавиатура","mouse":"Мышь","cup":"Чашка",
 "potted plant":"Растение"}

def robust_sample(xyz,scale=.055,cap=28000):
    if len(xyz)==0:return np.zeros(0,dtype=int)
    grid=np.floor(np.asarray(xyz,dtype=float)/scale).astype(np.int64)
    _,idx=np.unique(grid,axis=0,return_index=True)
    idx=np.sort(idx)
    if len(idx)>cap:idx=idx[np.linspace(0,len(idx)-1,cap,dtype=int)]
    return idx

def draw(args):
    original=trimesh.load(args.geometry,process=False)
    xyz=np.asarray(original.vertices)
    if xyz.ndim!=2 or xyz.shape[1]!=3 or len(xyz)<5000:
        raise ValueError("A6 TSDF cloud is missing / too small")
    data=np.load(args.objects)
    sample=np.asarray(data["xyz"],dtype=float)
    classes=np.asarray(data["labels"]).astype(str)
    confidence=np.asarray(data["confidence"],dtype=float)
    if len(sample)!=len(classes) or len(sample)!=len(confidence):
        raise ValueError("Object point/label indexing mismatch")
    if len(sample)<10000:
        raise ValueError("Expected actual real YOLOE RGB-D object points")
    events=[json.loads(v) for v in args.observations.read_text().splitlines() if v.strip()]
    report=json.loads(args.report.read_text())
    assert report["counts"]["rgbd_frames_processed"]==595
    assert report["counts"]["detector_keyframes"]==119
    assert report["semantic_map_export"]["object_observations"]==len(events)
    assert not report["semantic_map_export"]["physical_instances_verified"]
    assert report["ground_truth_used_for_pose"] is False

    fig=plt.figure(figsize=(18.5,10.4),facecolor="#0e1623",dpi=145)
    ax=fig.add_axes([.025,.14,.61,.74],projection="3d",facecolor="#0e1623")
    ax2=fig.add_axes([.67,.23,.30,.59],facecolor="#152034")
    fg="#e2e8f0";sub="#a7b4c8"
    plot_bg=xyz[robust_sample(xyz,.028,20000)]
    ax.scatter(plot_bg[:,0],plot_bg[:,1],plot_bg[:,2],
               s=1.1,c="#8793a7",alpha=.13,rasterized=True,depthshade=False)
    ax2.scatter(plot_bg[:,0],plot_bg[:,1],s=.45,c="#a3afbd",alpha=.14,rasterized=True)

    bounds_min=np.percentile(xyz,.2,axis=0)-.40
    bounds_max=np.percentile(xyz,99.8,axis=0)+.40
    used=Counter()
    for label in sorted(set(classes),key=lambda c:-(classes==c).sum()):
        keep=(classes==label)
        points=sample[keep]
        valid=np.isfinite(points).all(axis=1)&np.all(points>=bounds_min-.8,axis=1)&np.all(points<=bounds_max+.8,axis=1)
        points=points[valid]
        if len(points)<10:continue
        ind=robust_sample(points,.045,3000)
        points=points[ind]
        used[label]=int(np.sum(keep))
        color=COLORS.get(label,"#b7ed4b")
        ax.scatter(points[:,0],points[:,1],points[:,2],c=color,
                   s=3.4,alpha=.90,rasterized=True,depthshade=False)
        ax2.scatter(points[:,0],points[:,1],s=3.2,c=color,
                    alpha=.74,rasterized=True)

    # Annotate distinct class observation clusters, not invented physical IDs.
    # Class labels may correspond to multiple different objects.
    for label in sorted(used,key=lambda c:-used[c])[:10]:
        centers=np.array([e["center_map_m"] for e in events if e["class"]==label
                          and np.asarray(e["center_map_m"]).shape==(3,)])
        if len(centers)<2:continue
        # Median of genuine observed object positions, not a claimed UUID.
        c=np.median(centers,axis=0)
        color=COLORS.get(label,"#b7ed4b")
        mark=LABELS.get(label,label)
        annotation=ax2.annotate(mark,xy=(c[0],c[1]),xytext=(5,5),
                              textcoords="offset points",color=color,
                              fontsize=8.5,weight="bold")
        annotation.set_path_effects([pe.withStroke(linewidth=2.5,foreground="#0e1623")])

    center=(bounds_max+bounds_min)/2
    spans=bounds_max-bounds_min
    radius=max(spans)/2
    ax.set_xlim(center[0]-radius,center[0]+radius)
    ax.set_ylim(center[1]-radius,center[1]+radius)
    ax.set_zlim(max(0,center[2]-radius*.62),center[2]+radius*.62)
    ax.set_box_aspect((1,1,.65),zoom=.9)
    ax.view_init(elev=26,azim=-66)
    ax.set_xlabel("X, м",color=sub,labelpad=1)
    ax.set_ylabel("Y, м",color=sub,labelpad=1)
    ax.set_zlabel("Z, м",color=sub,labelpad=1)
    for axis in (ax.xaxis,ax.yaxis,ax.zaxis):
        axis.set_tick_params(colors=sub,labelsize=8)
        axis.line.set_color("#53627b")
        axis.pane.fill=False
        axis.pane.set_edgecolor("#1b2a40")
    ax.set_title("3D: TSDF-поверхность и обнаруженные объекты",color=fg,fontsize=14,pad=20)
    ax2.set_title("Проекция XY — наблюдения по классам",color=fg,fontsize=12,pad=12)
    ax2.set_aspect("equal",adjustable="datalim")
    ax2.set_xlabel("X, м",color=sub)
    ax2.set_ylabel("Y, м",color=sub)
    ax2.tick_params(colors=sub,labelsize=8)
    for sp in ax2.spines.values():sp.set_color("#455773")
    ax2.grid(alpha=.15,color="#91a5c2")
    ax2.set_xlim(bounds_min[0],bounds_max[0])
    ax2.set_ylim(bounds_min[1],bounds_max[1])

    fig.text(.04,.95,"СЕМАНТИЧЕСКАЯ КАРТА  •  TUM Freiburg1 desk",
             color="#ffffff",size=19,weight="bold")
    fig.text(.04,.915,
             "Реальные RGB-D | RTAB-Map A6, оптимизированные позы | YOLOE-26x-seg | без GT при построении",
             color=sub,size=10.5)
    keys=[k for k in sorted(used,key=lambda c:-used[c]) if used[k]>=1]
    import matplotlib.lines as mlines
    handles=[mlines.Line2D([],[],color=COLORS.get(k,"#b7ed4b"),
                   marker="o",ls="",markersize=7,label=LABELS.get(k,k))
             for k in keys[:10]]
    ax2.legend(handles=handles,loc="lower center",bbox_to_anchor=(.50,-.31),
               ncol=3,fontsize=8.5,facecolor="#18263c",edgecolor="#455773",
               labelcolor=fg,framealpha=.95)
    fig.text(.04,.062,
             f"TSDF: {len(xyz):,} точек  •  RGB-D: 595/595  •  YOLOE: 119 ключевых кадров  •  3D-наблюдения: {len(events):,}  •  точек объекта: {len(sample):,}",
             color=fg,size=10.5)
    fig.text(.04,.032,
             "Цвет = предсказанный класс YOLOE, не подтверждённый физический ID. Карта не валидирована для автономной навигации.",
             color="#efac8b",size=10)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.out,dpi=145,facecolor=fig.get_facecolor())
    plt.close(fig)
    # Export actual combined geometry with class colors to open in CloudCompare.
    rgb_base=np.tile(np.array([[128,137,151,255]],dtype=np.uint8),(len(xyz),1))
    rgb_sem=np.zeros((len(sample),4),dtype=np.uint8)
    for label in set(classes):
        color=matplotlib.colors.to_rgb(COLORS.get(label,"#b7ed4b"))
        rgb_sem[classes==label,:3]=np.array(color)*255
    rgb_sem[:,3]=255
    merged_xyz=np.concatenate([xyz.astype(np.float32),sample.astype(np.float32)])
    merged_colors=np.concatenate([rgb_base,rgb_sem])
    pcl=trimesh.points.PointCloud(merged_xyz,colors=merged_colors)
    ply=args.out.with_name("semantic_map_colored.ply")
    pcl.export(ply)
    info={"source_geometry":str(args.geometry),"source_observations":str(args.observations),
      "rendered_png":str(args.out),"colored_ply":str(ply),
      "tsdf_points":len(xyz),"semantic_mask_xyz_points":len(sample),
      "mask_observations":len(events),
      "class_observation_counts":dict(Counter(e["class"] for e in events)),
      "not_physical_instance_count":True,
      "not_navigable_map":True,
      "no_GT_used_for_semantics":True,
      "YOLOE_keyframes":119,"rgbd_frames":595}
    args.out.with_name("semantic_map_metadata.json").write_text(json.dumps(info,indent=2,ensure_ascii=False)+"\n")
    print("B5_MAP_RENDERED",json.dumps(info,ensure_ascii=False),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--geometry",type=Path,required=True)
    p.add_argument("--objects",type=Path,required=True)
    p.add_argument("--observations",type=Path,required=True)
    p.add_argument("--report",type=Path,required=True)
    p.add_argument("--out",type=Path,default=Path("results/b5/semantic_map_highlighted.png"))
    draw(p.parse_args())
