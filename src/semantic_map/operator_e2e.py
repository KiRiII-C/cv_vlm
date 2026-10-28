"""Grounded operator commands against real RGB-D map and persistent object IDs.

This is a strict offline operator-goal test. It does not claim that an A* route
is robot motion, nor that inferred floor / unknown occupancy are navigation safe.
"""
from __future__ import annotations
import argparse,json,re,math
from pathlib import Path
import numpy as np
from .storage import MemoryStore
from .planner import OccupancyGrid,CommandProcessor,GoalResult
from .nav2_bridge import goal_as_dict

def _yaml_scalar(path,name):
    import ast
    for line in Path(path).read_text().splitlines():
        if line.startswith(name+":"):
            value=line.split(":",1)[1].strip()
            try:return json.loads(value)
            except json.JSONDecodeError:return ast.literal_eval(value)
    raise ValueError("Missing "+name)

def load_grid(root):
    root=Path(root)
    values=np.load(root/"occupancy.npy",allow_pickle=False)
    if not set(np.unique(values)).issubset({-1,0,100}):
        raise ValueError("Unexpected occupancy representation")
    res=float(_yaml_scalar(root/"occupancy.yaml","resolution"))
    origin=_yaml_scalar(root/"occupancy.yaml","origin")
    # Explicitly prevent unknown territory from being regarded as traversable.
    return OccupancyGrid(values!=0,resolution=res,origin=origin[:2]),values

def select_start(grid:OccupancyGrid,values,preferred):
    free=np.argwhere(values==0)
    if not len(free):return None
    # This is SIMULATED initial 2D base pose; camera is handheld, not robot.
    options=[grid.point(row) for row in free]
    allowed=[p for p in options if not grid.inflated(.15)[grid.cell(p)]]
    if not allowed:return None
    return list(min(allowed,key=lambda p:np.linalg.norm(np.asarray(p)-preferred)))

def nearest_candidate_label(t):
    label=t.label.lower()
    return {"dining table":"table","tv":"tv","keyboard":"keyboard","laptop":"laptop"}.get(label,label)

def run(memory_dir,geom_dir,output_dir,gravity_dir=None):
    from matplotlib import pyplot as plt
    memroot=Path(memory_dir);geomroot=Path(geom_dir);out=Path(output_dir)
    out.mkdir(parents=True,exist_ok=True)
    report=json.loads((geomroot/"mapping_report.json").read_text())
    navroot=Path(gravity_dir) if gravity_dir else geomroot
    gravity=None;Tnav=np.eye(4)
    if gravity_dir:
        gravity=json.loads((navroot/"gravity_report.json").read_text())
        if gravity["floor_detection"]!="PLAUSIBLE_PLANE":
            raise RuntimeError("Cannot plan on nonvalidated floor estimate")
        Tnav=np.asarray(gravity["T_aligned_map"],np.float64)
    grid,values=load_grid(navroot)
    poses=json.loads((geomroot/"poses_used.json").read_text())["poses"]
    camera_world=np.array(poses[0]["T_map_cam"])[:3,3]
    camera_xy=(Tnav[:3,:3]@camera_world+Tnav[:3,3])[:2]
    start=select_start(grid,values,camera_xy)
    store=MemoryStore(memroot/"semantic_memory.sqlite")
    tracks=[t for t in store.tracks() if t.state=="confirmed" and t.hits>=8]
    for t in tracks:
        t.label=nearest_candidate_label(t)
        if gravity_dir:
            # Convert entire object cuboid to plane-aligned coordinates.
            low,high=t.low.copy(),t.high.copy()
            corners=np.asarray([[x,y,z] for x in (low[0],high[0])
                                for y in (low[1],high[1]) for z in (low[2],high[2])])
            transformed=corners@Tnav[:3,:3].T+Tnav[:3,3]
            t.center=t.center@Tnav[:3,:3].T+Tnav[:3,3]
            t.low=transformed.min(0);t.high=transformed.max(0)
    processor=CommandProcessor(tracks,grid,robot_xy=start or [0,0],radius=.15) if start is not None else None
    distinct=sorted({t.label for t in tracks})
    query=[
       "найди объект keyboard",
       "подъедь к экрану tv",
       "подъедь к столу",
       "встань между шкафом и стулом"
    ]
    # Identity-specific commands must never silently bind to a similar object.
    query += ["подъедь к "+t.object_id for t in sorted(tracks,key=lambda x:-x.hits)[:3]]
    answer=[]
    for q in query:
        match=re.search(r"obj_[0-9a-f]{16}",q)
        selected=[t for t in tracks if t.object_id==match.group()] if match else []
        if match:
            if not selected:
                d=GoalResult(q,"not_found",[],None,[],"No such confirmed object")
            elif processor is None:
                d=GoalResult(q,"unreachable",[selected[0].object_id],None,[],"No free collision-safe starting cell")
            else:
                # Only pass the selected physical instance to language-to-goal planner.
                single=CommandProcessor(selected,grid,robot_xy=start,radius=.15)
                d=single.run("подъедь к "+selected[0].label)
        elif processor is None:
            d=GoalResult(q,"unreachable",[],None,[],"No free collision-safe starting cell")
        else:
            d=processor.run(q)
        entry={"query":q,"status":d.status,"target_ids":d.target_ids,
               "goal_xy_m":d.goal_xy,"path_xy":d.path_xy,
               "explanation":d.explanation,"source":"offline_astar_not_nav2"}
        if d.status=="planned":
            if bool(report.get("occupancy_2d",{}).get("nav_ready")) and gravity_dir is None:
                entry["nav2_goal_candidate_not_executed"]=goal_as_dict(d)
            else:
                # A diagnostic (uncalibrated) occupancy projection is NEVER a
                # verified Nav2 costmap. Keep path visible, block dispatch.
                entry["status"]="route_candidate_unvalidated_map"
                entry["explanation"] += "; маршрут только диагностический: высота пола не калибрована"
        answer.append(entry)
    payload={
        "data_source":"actual TUM fr1_desk RGB-D map and YOLO observations",
        "pose_source":report.get("camera_pose_source"),
        "frames_integrated":report.get("frames_integrated"),
        "identity_verified_with_GT":False,
        "robot_actuation":"NOT_EXECUTED",
        "navigation_map_ready":False if gravity_dir else bool(report.get("occupancy_2d",{}).get("nav_ready")),
        "gravity_aligned_offline_map":bool(gravity_dir),
        "gravity_planar_evidence":gravity["candidate_status"] if gravity else None,
        "unknown_cells_blocked":True,
        "simulated_robot_xy_m":start,
        "confirmed_tracks_used":len(tracks),
        "classes":distinct,
        "queries":answer
    }
    (out/"operator_e2e.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2))
    fig,ax=plt.subplots(figsize=(11,8))
    from matplotlib.colors import ListedColormap
    tri=np.where(values==100,2,np.where(values==0,1,0))
    ax.imshow(tri,origin="lower",cmap=ListedColormap(["#999999","#f8f8f8","#252525"]),
              vmin=-.5,vmax=2.5,interpolation="nearest",
              extent=[grid.origin[0],grid.origin[0]+values.shape[1]*grid.res,
                      grid.origin[1],grid.origin[1]+values.shape[0]*grid.res])
    for t in tracks:
        xy=t.center[:2]
        ax.scatter(*xy,s=18,c="tab:orange",alpha=.8)
        if t.hits>25:ax.annotate(t.object_id[-5:],xy,fontsize=6)
    if start is not None:
        ax.scatter(*start,c="tab:green",s=80,marker="*",label="Simulated start")
    for k,a in enumerate(answer):
        if a["status"] in ("planned","route_candidate_unvalidated_map"):
            route=np.array(a["path_xy"])
            ax.plot(route[:,0],route[:,1],lw=1.8,label=f"route{k}")
    ax.set(xlabel="World map X (m)",ylabel="World map Y (m)",
           title="Real RGB-D evidence + object tracks (unverified ID), offline goal proposals")
    ax.set_aspect('equal')
    ax.legend(loc="upper right",fontsize=7)
    fig.tight_layout();fig.savefig(out/"operator_map.png",dpi=170);plt.close(fig)
    store.close()
    print("OPERATOR_E2E",json.dumps({k:v for k,v in payload.items() if k!="queries"}),flush=True)
    for q in answer:print("OPERATOR_GOAL",q["query"],q["status"],"path",len(q["path_xy"]),flush=True)
    return payload

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--memory",default="results/tum_real")
    p.add_argument("--geometry",default="results/full_geometry")
    p.add_argument("--output",default="results/operator_e2e")
    p.add_argument("--gravity",default=None)
    a=p.parse_args()
    run(a.memory,a.geometry,a.output,a.gravity)
if __name__=='__main__':main()
