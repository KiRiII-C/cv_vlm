#!/usr/bin/env python3
"""C1: real 1.8M measured B7 points -> existing physical-UUID HYPOTHESIS voxels.

Flattened B7 NPZ is ordered by B7 mask observations. For each observation,
the exact number of measured points is min(3000,mask_pixel_count); validate
the sum AND exact per-event YOLO class before slicing. All C0 ambiguous,
deferred and quality-rejected evidence remains unassigned: NEVER choose a
physical instance by proximity alone. No GT physical labels in association.
"""
from __future__ import annotations
import argparse,json,shutil
from collections import Counter
from pathlib import Path
import sqlite3
import numpy as np
import plotly.graph_objects as go
from semantic_map.object_surface_memory_v2 import ObjectSurfaceMemoryV2,object_color

MAP_ID="A6_RTABMAP_FULL_TUM_DESK"
SOURCE="TUM_fr1_desk_B7_YOLOE26x"

def read_jsonl(path):
    return [json.loads(v) for v in Path(path).read_text().splitlines() if v.strip()]

def assign(measured_npz,source_events,c0_events,connection,output,*,max_per_mask=3000):
    raw=np.load(measured_npz)
    xyz=np.asarray(raw["xyz"])
    labels=np.asarray(raw["labels"]).astype(str)
    conf=np.asarray(raw["confidence"])
    if len(xyz)!=len(labels) or len(xyz)!=len(conf) or len(xyz)<700000:
        raise RuntimeError("Not the complete dense B7 real source point cloud")
    if len(source_events)!=871 or len(c0_events)!=871:
        raise RuntimeError("B7 and C0 must refer to the same full real detections")
    audit={(int(x["frame_index"]),str(x["detection_id"])):x for x in c0_events}
    if len(audit)!=len(c0_events):raise RuntimeError("Duplicate C0 evidence key")
    tracks={row["object_id"]:row for row in connection.execute(
        "SELECT object_id,label,state,map_version,map_id FROM objects WHERE map_id=?",(MAP_ID,))}
    confirmed={uid for uid,row in tracks.items() if row["state"]=="confirmed" and row["map_version"]==0}
    if not confirmed:raise RuntimeError("No repeated-confirmed C0 hypotheses")
    memory=ObjectSurfaceMemoryV2(connection,voxel_m=.015,minimum_support=2)
    counts=Counter()
    pos=0
    for event in source_events:
        frame=int(event["frame_index"])
        det=str(event["detection_id"])
        label=str(event["class"])
        n=min(int(event["mask_pixel_count"]),max_per_mask)
        end=pos+n
        if n<1 or end>len(xyz) or not np.all(labels[pos:end]==label):
            raise RuntimeError(f"B7 flattened mask points/label order broken {frame}/{det}")
        points=xyz[pos:end]
        pos=end
        counts["all_real_mask_observations"]+=1
        counts["all_source_points"]+=len(points)
        status=audit.get((frame,det))
        if status is None:raise RuntimeError("Missing C0 UUID audit for genuine B7 mask")
        uid=status.get("object_uuid")
        if not uid or uid not in confirmed or status["status"] not in ("tentative","confirmed"):
            counts["unassigned_or_unconfirmed_observations"]+=1
            counts["unassigned_measured_points"]+=len(points)
            continue
        if str(tracks[uid]["label"])!=label:
            raise RuntimeError("C1 source class differs from C0 identity evidence")
        inserted=memory.integrate_world_points(
            uid,points,source=SOURCE,frame_id=str(frame),
            map_id=MAP_ID,pose_version=0,timestamp=float(event["timestamp_tum"]))
        if inserted<=0:raise RuntimeError("Duplicate confirmed observation in one map frame")
        counts["confirmed_hypothesis_observations_fused"]+=1
        counts["measured_points_fused_into_UUID_hypotheses"]+=inserted
    if pos!=len(xyz):raise RuntimeError("Unconsumed actual B7 source RGB-D points")
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    exported=memory.export(output/"hypothesis_surfaces.ply",
                           map_id=MAP_ID,instance_colors=True,confirmed_only=True)
    rows=connection.execute('''SELECT uuid,sum_x,sum_y,sum_z,support,negative
              FROM object_surface_v2 WHERE map_id=? AND support>=2 AND negative<3''',
              (MAP_ID,)).fetchall()
    clouds={}
    for uid,x,y,z,support,negative in rows:
        if uid not in confirmed:continue
        if uid not in clouds:clouds[uid]=[]
        clouds[uid].append([x/support,y/support,z/support,support])
    fig=go.Figure()
    for uid,pts in sorted(clouds.items(),key=lambda item:-len(item[1])):
        pts=np.asarray(pts,dtype=np.float32)
        if len(pts)>20000:pts=pts[np.linspace(0,len(pts)-1,20000,dtype=int)]
        r,g,b=object_color(uid)
        fig.add_trace(go.Scatter3d(
           x=pts[:,0],y=pts[:,1],z=pts[:,2],mode="markers",
           name=str(tracks[uid]["label"])+" · "+uid[4:10],
           marker={"size":2,"color":f"rgb({r},{g},{b})","opacity":.86}))
    fig.update_layout(template="plotly_dark",scene={"aspectmode":"data"},
        title="C1: RGB-D облака по гипотезам ObjectMemoryV2 (не ground truth)",
        legend={"title":"Гипотезы UUID: включение/скрытие"},
        margin={"l":0,"r":0,"t":60,"b":0})
    fig.write_html(output/"hypothesis_clouds_interactive.html",
                   include_plotlyjs=True,full_html=True)
    report={
      "experiment":"C1_ACTUAL_B7_RGBD_TO_C0_UUID_SURFACE_FUSION",
      "source":"B7_full_595_TUM_RGBD_YOLOE26X_3000_REAL_POINTS_PER_MASK",
      "original_rgbd_frames":595,"YOLOE_keyframes":119,
      "real_source_semantic_points":len(xyz),
      "original_B7_observations":len(source_events),
      "preexisting_confirmed_hypotheses":len(confirmed),
      "observed_UUID_hypotheses_exported":exported["unique_ids"],
      "exported_UUID_voxels_min_2_support":exported["vertices"],
      "counts":dict(counts),
      "pose_graph_version":0,"metric_voxel_size_m":.015,
      "no_interpolated_points":True,
      "ambiguity_does_not_force_UUID":True,
      "uuid_physical_gt_accuracy":"NOT_EVALUATED",
      "object_movement_detection":"NOT_TESTED",
      "safety_and_nav2_ready":False,
      "note":"Class color memory and physical UUID are different; these are only repeated-observation hypotheses."}
    (output/"c1_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print("C1_REAL_INSTANCE_SURFACE",json.dumps(report,ensure_ascii=False),flush=True)
    return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--points",type=Path,required=True)
    p.add_argument("--source-events",type=Path,required=True)
    p.add_argument("--audit",type=Path,required=True)
    p.add_argument("--c0-db",type=Path,required=True)
    p.add_argument("--out",type=Path,default=Path("results/c1"))
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    dest=args.out/"object_memory_with_surfaces.sqlite"
    if dest.exists():dest.unlink()
    shutil.copyfile(args.c0_db,dest)
    db=sqlite3.connect(dest)
    db.row_factory=sqlite3.Row
    try:
        assign(args.points,read_jsonl(args.source_events),
               read_jsonl(args.audit),db,args.out)
    finally:
        db.close()

if __name__=="__main__":main()
