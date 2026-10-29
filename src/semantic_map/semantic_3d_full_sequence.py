"""Full 100-frame, GT-verified *predicted* dense semantic -> 3D instances.

YOLOE26x is retained as primary configured model elsewhere; this script
evaluates a second class-agnostic 3D instance proposal route, as the requested
85% physical-IDF1 cannot be reached with current sparse YOLOE26x masks.
No GT semantic categories, instance labels, masks or GT identity are used
inside model, 3D clustering or identity association; GT XYZ is used as a
controlled metric 3D mapping-pose source as in previous benchmarks.
"""
from __future__ import annotations
import argparse,csv,gzip,io,json,time
from pathlib import Path
from collections import Counter
import cv2,numpy as np
from .hypersim_range import open_scene,hdf5_array
from .hypersim_full_sequences import discover
from .ade20k_semantic_pilot import canonical_class
from .semantic_3d_instance_proposals import instance_proposals
from .yoloe26x_factorial import target_instances,match_masks,observation
from .offline_graph_benchmark import benchmark
from .offline_instance_graph import OfflineInstanceGraph
from .memory_v2_replay import replay
from .hypersim_full_sequence_eval import gap_stats,_evolving_scores

def run(scene,out,*,eps=.12,imgsz=640,min_area=220,include_v2=True):
    from ultralytics import YOLO
    model="yolo26m-sem-ade20k.pt"
    detector=YOLO(model)
    mapping={int(i):canonical_class(v) for i,v in detector.names.items()
             if canonical_class(v)}
    root=Path(out);root.mkdir(parents=True,exist_ok=True)
    remote,z=open_scene(scene)
    rows=[];frames=[];first=0;last=99;missing_3d=0
    try:
        meta=discover(z,scene)
        if "cam_00" not in meta:raise ValueError("Missing camera")
        listed=meta["cam_00"]["frames_complete"]
        if listed!=list(range(100)) or meta["cam_00"]["missing_modalities"]:
            raise ValueError("Must include 100/100 annotated frames; no subsampling")
        scale=float(dict(csv.reader(io.StringIO(
            z.read(f"{scene}/_detail/metadata_scene.csv").decode())))["meters_per_asset_unit"])
        for n in listed:
            prefix=f"{scene}/images/scene_cam_00_geometry_hdf5/frame.{n:04d}."
            jpg=f"{scene}/images/scene_cam_00_final_preview/frame.{n:04d}.tonemap.jpg"
            gt=hdf5_array(z,prefix+"semantic_instance.hdf5").astype("int32")
            gt_cls=hdf5_array(z,prefix+"semantic.hdf5").astype("int32")
            world=hdf5_array(z,prefix+"position.hdf5").astype("float32")*scale
            rgb=cv2.imdecode(np.frombuffer(z.read(jpg),"uint8"),cv2.IMREAD_COLOR)
            if rgb is None or not(gt.shape==gt_cls.shape==rgb.shape[:2]==world.shape[:2]):
                raise ValueError("RGB, depth XYZ and physical GT misregistered")
            t=time.monotonic()
            result=detector.predict(rgb,imgsz=imgsz,device="cpu",verbose=False)[0]
            if result.semantic_mask is None:raise RuntimeError("No predicted semantic class map")
            dense=result.semantic_mask.data.detach().cpu().numpy().astype("int32")
            if dense.shape!=gt.shape:
                dense=cv2.resize(dense.astype("float32"),
                  (gt.shape[1],gt.shape[0]),interpolation=cv2.INTER_NEAREST).astype("int32")
            proposals=instance_proposals(dense,world,mapping,eps=eps,
                        subsample=3,min_support=8,min_area=min_area)
            labels=[v[0] for v in proposals]
            masks=[v[1] for v in proposals]
            targets=target_instances(gt,gt_cls,limit=100000)
            assignment,ious,_=match_masks(masks,gt,targets,threshold=.10)
            observations=[];physical=[]
            for j,(label,mask) in enumerate(proposals):
                # Confidence of semantic map isn't instance confidence. Set
                # pseudo measurement confidence .65, no GT calibration.
                o=observation(world,rgb,mask,n,"sem3d_"+str(j),
                              label,.65,scene)
                if o is None:
                    missing_3d+=1
                    continue
                observations.append(o)
                # The following GT physical ID is a scoring-only sidecar,
                # NEVER read by replay() or graph.associate().
                physical.append(int(targets[assignment[j]][0])
                                if j in assignment else None)
            frames.append(dict(frame=int(n),target_visible=[int(v[0]) for v in targets],
                 all_gt_visible=[int(v[0]) for v in targets],
                 observations=observations,physical_gt=physical))
            rows.append(dict(frame=int(n),GT_visible=len(targets),
                model_pred_instances=len(masks),
                inferred_3d_observations=len(observations),
                IoU25_tp=sum(ious[i,j]>=.25 for i,j in assignment.items()),
                IoU50_tp=sum(ious[i,j]>=.5 for i,j in assignment.items()),
                IoU75_tp=sum(ious[i,j]>=.75 for i,j in assignment.items()),
                class_correct_match=sum(labels[i]==targets[j][1]
                    for i,j in assignment.items()),
                time_seconds=time.monotonic()-t))
            if n%10==0 or n==99:
                print("SEMANTIC_3D_FULL_PROGRESS",scene,n+1,"/100",
                  "GT",len(targets),"mask_proposals",len(masks),
                  "tp50",rows[-1]["IoU50_tp"],
                  "sec",round(rows[-1]["time_seconds"],2),flush=True)
    finally:z.close();remote.close()
    cache=dict(scene=scene,detector=model,mode="predicted_yolo_masks",
       frames=frames,gt_labels_are_scoring_only=True,used_GT_world_XYZ=True,
       meta={"cluster_eps_m":eps,"semantic_imgsz":imgsz,
                 "class_agnostic_3d_instance_splitting":True})
    with gzip.open(root/"semantic_3d_observations_and_GT_scoring.json.gz","wt",encoding="utf8") as f:
        json.dump(cache,f)
    # Offline map: class-soft global instance graph. No physical IDs provided
    # to association, only their independent scoring sidecar afterward.
    g=benchmark(cache,OfflineInstanceGraph())
    results={"offline_graph":g}
    if include_v2:
        r=replay(cache,"v2")
        diag=gap_stats(cache,r["timeline"])["gaps"]
        results["ObjectMemoryV2"]={
          "target_gt":r["target_gt"],
          "reidentification":diag,
          "time_windows":_evolving_scores(cache,r["timeline"])}
    total_gt=sum(x["GT_visible"] for x in rows)
    report=dict(status="MEASURED_NOT_AUTO_ACCEPTED",
       scene=scene,frames=100,full_trajectory=True,
       model=model,primary_model_elsewhere="yoloe-26x-seg.pt",
       clustering_eps_m=eps,semantic_imgsz=imgsz,
       GT_instance_identity_as_algorithm_input=False,
       GT_world_XYZ_used=True,SLAM="NOT_TESTED",
       physical_gt_appearances=total_gt,
       physical_unique_GT_IDs=len({k for f in frames for k in f["target_visible"]}),
       predicted_3d_instance_observations=sum(len(f["observations"]) for f in frames),
       unavailable_3d_proposals=missing_3d,
       mask_recall25=sum(x["IoU25_tp"] for x in rows)/max(1,total_gt),
       mask_recall50=sum(x["IoU50_tp"] for x in rows)/max(1,total_gt),
       mask_recall75=sum(x["IoU75_tp"] for x in rows)/max(1,total_gt),
       model_seconds=sum(x["time_seconds"] for x in rows),
       results=results,per_frame=rows,
       caveats=["Dense predicted semantic map is not GT and 3D splitting sees no physical instance IDs",
        "Official GT XYZ is used for 3D points; not a test of noisy depth/SLAM",
        "Physical GT only used for scoring and IoU diagnostic alignment",
        "IoU >= 0.1 observation-level assignment not official HOTA/TrackEval IDF1",
        "ADE20K model pretrained; only classes intersecting required 12 prompts are used"])
    (root/"semantic_3d_full_GT_report.json").write_text(
       json.dumps(report,ensure_ascii=False,indent=2,
        default=lambda a:a.item() if isinstance(a,np.generic)
            else a.tolist() if isinstance(a,np.ndarray) else str(a)))
    print("SEMANTIC3D_FULL_100FRAME_IDF1",scene,
      "recall50",report["mask_recall50"],
      "graph",g["target_gt"]["IDF1"],
      "v2",results.get("ObjectMemoryV2",{}).get("target_gt",{}).get("IDF1"),
      flush=True)
    return report

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--scene",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--eps",type=float,default=.12)
    p.add_argument("--imgsz",type=int,default=640)
    p.add_argument("--skip-v2",action="store_true")
    a=p.parse_args()
    run(a.scene,a.out,eps=a.eps,imgsz=a.imgsz,include_v2=not a.skip_v2)
