#!/usr/bin/env python3
"""B4: real YOLOE-26x masks, all 100 Hypersim frames per scene, physical GT.

Hypothesis: current-depth visible 3D-surface mask correspondence adds identity
discriminability beyond 3D centroid + HSV. Groundtruth instance ids are NEVER
used to generate pair candidates or scores; only label each candidate AFTER
all observations and scores are generated.

Important CONTROLLED ORACLE: Hypersim's world XYZ is used for 3D positions
in BOTH baselines (no RTAB-Map); this tests the association mechanism, NOT
real-world SLAM accuracy. Source and target masks are actual YOLOE outputs.
World-coordinate nearest-surface assignment avoids needing a camera model
but is NOT the perspective z-buffer algorithm from B3: evaluate separately.
"""
from __future__ import annotations
import argparse,csv,io,json,time
from pathlib import Path
import cv2,numpy as np
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score,average_precision_score
from semantic_map.hypersim_range import open_scene,hdf5_array
from semantic_map.hypersim_full_sequences import discover
from semantic_map.yoloe26x_factorial import target_instances,match_masks,observation
from semantic_map.yoloe26_detector import build_detector,class_list,canonical
from semantic_map.reid_features import hsv_spatial

def norm(x):
    a=np.asarray(x,dtype=float)
    n=np.linalg.norm(a)
    return a/max(n,1.e-12)

def prepare_observations(predictor,rgb,gt,sem,world,frame,scene):
    found=predictor.predict(rgb,conf=.10,device="cpu",imgsz=640)
    masks=[];labels=[];confidence=[]
    if found.masks is not None and found.boxes is not None:
        for raw in found.masks.data.cpu().numpy():
            if raw.shape!=gt.shape:
                raw=cv2.resize(raw.astype("float32"),(gt.shape[1],gt.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
            masks.append(raw>.5)
        labels=[predictor.class_name(x) for x in found.boxes.cls.cpu().numpy()]
        confidence=[float(x) for x in found.boxes.conf.cpu().numpy()]
    if len(masks)!=len(labels) or len(labels)!=len(confidence):
        raise RuntimeError("YOLO real mask/label count mismatch")
    targets=target_instances(gt,sem,limit=1_000_000)
    matching,ious,_=match_masks(masks,gt,targets,threshold=.10)
    valid_world=np.isfinite(world).all(axis=2)
    # Query nearest currently *measured* world surface (pixel decimation 2),
    # but NEVER physical instance GT labels to build this index.
    valid_grid=np.zeros_like(valid_world,dtype=bool)
    valid_grid[::2,::2]=True
    ys,xs=np.nonzero(valid_world&valid_grid)
    cloud=world[ys,xs].astype(np.float64)
    tree=cKDTree(cloud) if len(cloud)>100 else None
    objects=[]
    for i,(mask,label,conf) in enumerate(zip(masks,labels,confidence)):
        if int(mask.sum())<250:continue
        obs=observation(world,rgb,mask,frame,f"det_{i}",label,conf,scene)
        if obs is None:continue
        yy,xx=np.nonzero(mask&valid_world)
        if not len(xx):continue
        idx=np.linspace(0,len(xx)-1,min(12000,len(xx)),dtype=int)
        surf=world[yy[idx],xx[idx]].astype(np.float64)
        feat=hsv_spatial(rgb,mask)
        identity=int(targets[matching[i]][0]) if i in matching else None
        objects.append({"label":canonical(label),"xyz":np.asarray(obs["center"],dtype=float),
          "feature":norm(feat) if feat is not None else None,
          "surface":surf,"mask":mask,"gt_scoring_only":identity,
          "confidence":conf})
    return objects,tree,(ys,xs),len(masks),len(targets)

def projected_surface_support(old,current,tree,pixels,shape,tolerance=.04):
    # Search for reobserved 3D surface with official world XYZ in BOTH frames.
    # Unlike a point-to-image perspective renderer, this is a 3D NN control
    # that is only available because Hypersim gives correct world XYZ.
    if tree is None or not len(old["surface"]):return None
    dist,index=tree.query(old["surface"],k=1,distance_upper_bound=tolerance,workers=1)
    ok=np.isfinite(dist)&(index<len(pixels[0]))
    if int(ok.sum())<8:return None
    predicted=np.zeros(shape,dtype=bool)
    y,x=pixels
    predicted[y[index[ok]],x[index[ok]]]=True
    target=current["mask"]
    # Require at least 8 projected surface samples; missing pixels do not
    # prove disappearance and contribute no false-negative identity penalty.
    inter=int((predicted&target).sum())
    union=int((predicted|target).sum())
    return inter/union if union else None

def evaluate(scores,truth):
    if not truth or len(set(truth))<2:return {"valid":False,"pairs":len(truth)}
    truth=np.asarray(truth,dtype=bool)
    scores=np.asarray(scores,dtype=float)
    return {"valid":True,"pairs":len(scores),"positives":int(truth.sum()),
       "negatives":int((~truth).sum()),
       "roc_auc":float(roc_auc_score(truth,scores)),
       "average_precision":float(average_precision_score(truth,scores)),
       "mean_positive_score":float(scores[truth].mean()),
       "mean_negative_score":float(scores[~truth].mean())}

def run(scene,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    model="yoloe-26x-seg.pt"
    predictor=build_detector(model,class_list("config/yoloe26_classes.json"))
    remote,z=open_scene(scene)
    pairs=[];prev=[];perframe=[];counts={"total_keyframes":0,"mask_observations":0,
        "all_same_class_pairs":0,"same_ID_pairs":0,"distinct_ID_pairs":0,
        "candidate_projective_evidence":0,"no_projective_evidence":0}
    try:
        metadata=discover(z,scene).get("cam_00")
        if metadata is None:raise RuntimeError("Missing annotated camera")
        frames=metadata["frames_complete"]
        if frames!=list(range(100)) or metadata["missing_modalities"] or metadata["missing_frame_indices"]:
            raise RuntimeError("B4 demands the entire 100/100 sequence")
        scale=float(dict(csv.reader(io.StringIO(z.read(
            f"{scene}/_detail/metadata_scene.csv").decode("utf8"))))["meters_per_asset_unit"])
        for t in frames:
            prefix=f"{scene}/images/scene_cam_00_geometry_hdf5/frame.{t:04d}."
            pic=f"{scene}/images/scene_cam_00_final_preview/frame.{t:04d}.tonemap.jpg"
            gt=hdf5_array(z,prefix+"semantic_instance.hdf5").astype("int32")
            sem=hdf5_array(z,prefix+"semantic.hdf5").astype("int32")
            world=hdf5_array(z,prefix+"position.hdf5").astype("float32")*scale
            rgb=cv2.imdecode(np.frombuffer(z.read(pic),"uint8"),cv2.IMREAD_COLOR)
            if not (rgb is not None and gt.shape==sem.shape==world.shape[:2]==rgb.shape[:2]):
                raise RuntimeError("Alignment error on real Hypersim modalities")
            start=time.monotonic()
            curr,tree,pixels,num_masks,num_target=prepare_observations(
                predictor,rgb,gt,sem,world,t,scene)
            counts["total_keyframes"]+=1
            counts["mask_observations"]+=len(curr)
            for old in prev:
                for now in curr:
                    if old["label"]!=now["label"]:continue
                    if old["gt_scoring_only"] is None or now["gt_scoring_only"] is None:
                        # GT decides whether to SCORE, never to shortlist / score.
                        continue
                    same=old["gt_scoring_only"]==now["gt_scoring_only"]
                    d=float(np.linalg.norm(old["xyz"]-now["xyz"]))
                    base=float(np.exp(-.5*(d/.30)**2))
                    if old["feature"] is not None and now["feature"] is not None:
                        app=float(np.clip(old["feature"]@now["feature"],0,1))
                        base=.78*base+.22*app
                    projected=projected_surface_support(old,now,tree,pixels,gt.shape)
                    # Insufficient observed surface: abstain from adding
                    # projection, NOT manufacture zero confidence.
                    joint=.75*base+.25*projected if projected is not None else base
                    pairs.append({"source_frame":t-1,"target_frame":t,
                      "same_physical_GT_scoring_only":same,"baseline_score":base,
                      "joint_score":joint,"projection_iou":projected,
                      "separation_m":d,
                      "hard_negative_distance_le_0_8m":d<=.8})
                    counts["all_same_class_pairs"]+=1
                    counts["same_ID_pairs" if same else "distinct_ID_pairs"]+=1
                    counts["candidate_projective_evidence" if projected is not None
                         else "no_projective_evidence"]+=1
            perframe.append({"frame":t,"all_gt_target_instances":num_target,
               "all_YOLOE_masks":num_masks,"usable_real_predicted_masks":len(curr),
               "seconds_inference_and_association":time.monotonic()-start})
            prev=curr
            if t%10==0 or t==99:
                print("B4_FRAME",scene,t+1,"/",100,"pairs",len(pairs),flush=True)
    finally:
        z.close();remote.close()
    if counts["total_keyframes"]!=100:raise RuntimeError("Frame completeness failure")
    def subset_filter(item,scope):
        return scope=="all_same_class" or item["hard_negative_distance_le_0_8m"]
    metrics={}
    for scope in ("all_same_class","near_0_8m"):
        chosen=[p for p in pairs if subset_filter(p,scope)]
        labels=[p["same_physical_GT_scoring_only"] for p in chosen]
        metrics[scope]={
           "baseline_3D_HSV":evaluate([p["baseline_score"] for p in chosen],labels),
           "joint_surface_3D_HSV":evaluate([p["joint_score"] for p in chosen],labels),
           "projected_overlap_only_on_observable":[
                 evaluate([p["projection_iou"] for p in chosen if p["projection_iou"] is not None],
                          [p["same_physical_GT_scoring_only"] for p in chosen if p["projection_iou"] is not None])][0]}
    result={"experiment":"B4_GT_PHYSICAL_ID_PAIR_RANKING","scene":scene,
      "data":"ALL_100_OFFICIAL_HYPERSIM_REAL_YOLOE26X_MASKS",
      "source_gt_instance_ids_used_only_for_scoring":True,
      "GT_XYZ_USED_FOR_BOTH_ASSOCIATION_METHODS":True,
      "RTABMAP_NOT_TESTED":True,
      "this_is_not_full_HOTA_or_tracking_IDF1":True,
      "method":"nearest_world_surface_reobservation_geometry_oracle_not_B3_SE3_projection",
      "counts":counts,"metrics":metrics,"per_frame":perframe,
      "warning":"True-world XYZ is oracle geometry in both arms. Mask miss rate limits identity, "
                "and pair AUC is NOT track-level stability or calibrated probability."}
    (out/"b4_pairwise_report.json").write_text(json.dumps(result,indent=2)+"\n")
    # Full candidate pair provenance: GT physical ID only as evaluation label.
    with (out/"b4_pairwise_scores.jsonl").open("w") as handle:
        for p in pairs:handle.write(json.dumps(p)+"\n")
    print("B4_PHYSICAL_GT_RESULTS",scene,json.dumps({
       k:{v:{metric:d[metric] for metric in ("pairs","positives","negatives","roc_auc","average_precision")}
            if d.get("valid") else d for v,d in rec.items()}
          for k,rec in metrics.items()}),flush=True)
    return result

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--scene",choices=["ai_002_001","ai_005_001","ai_017_001"],required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    run(args.scene,args.out)
