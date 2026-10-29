"""Independent pretrained closed-set YOLO26x fallback to YOLOE26x 12 prompts.

Not a switch away from YOLOE26x as primary. Tests whether independent
closed-set COCO segmentation improves poor YOLOE26x recall on 10 of the 12
requested classes. Only allowlisted class outputs are used. Physical GT
labels are for scoring after inference; they are NOT fed into filtering,
deduplication, matching or detector.

Uses GT Hypersim XYZ for 3D, so cannot claim SLAM-tested navigation.
"""
from __future__ import annotations
import argparse,csv,gzip,io,json,time
from collections import defaultdict
from pathlib import Path
import cv2,numpy as np
from .hypersim_range import open_scene,hdf5_array
from .yoloe26x_factorial import target_instances,match_masks,observation
from .memory_v2_replay import replay
from .offline_graph_benchmark import benchmark
from .offline_instance_graph import OfflineInstanceGraph

ALLOW={
    "chair":"chair","couch":"sofa","dining table":"table","bed":"bed",
    "tv":"tv","laptop":"laptop","mouse":"mouse","keyboard":"keyboard",
    "cup":"cup","potted plant":"potted plant"
}
def load(path):
    with gzip.open(path,"rt",encoding="utf8") as f:return json.load(f)

def process(scene,raw_path,output,model="yolo26x-seg.pt",conf=.01,imgsz=960):
    from ultralytics import YOLO
    if model!="yolo26x-seg.pt":raise ValueError("Explicit YOLO26x-seg COCO fallback model required")
    raw=load(raw_path)
    if raw["scene"]!=scene or raw["detector"]!="yoloe-26x-seg.pt" or len(raw["frames"])!=100:
        raise ValueError("Primary detector must be full YOLOE26x 100-frame cached run")
    co=YOLO(model)
    # COCO category indices are run-time validated, not guessed.
    raw_names={int(k):v for k,v in co.names.items()}
    classids=[k for k,v in raw_names.items() if v in ALLOW]
    assert len(classids)==10,raw_names
    remote,z=open_scene(scene)
    aux_frames=[];fused_frames=[];per_frame=[]
    try:
        metadata=dict(csv.reader(io.StringIO(z.read(f"{scene}/_detail/metadata_scene.csv").decode())))
        scale=float(metadata["meters_per_asset_unit"])
        for i,primary in enumerate(raw["frames"]):
            frame=int(primary["frame"])
            if frame!=i:raise ValueError("Non-contiguous frames!")
            prefix=f"{scene}/images/scene_cam_00_geometry_hdf5/frame.{frame:04d}."
            jpg=f"{scene}/images/scene_cam_00_final_preview/frame.{frame:04d}.tonemap.jpg"
            gt=hdf5_array(z,prefix+"semantic_instance.hdf5").astype("int32")
            sem=hdf5_array(z,prefix+"semantic.hdf5").astype("int32")
            xyz=hdf5_array(z,prefix+"position.hdf5").astype("float32")*scale
            im=cv2.imdecode(np.frombuffer(z.read(jpg),"uint8"),cv2.IMREAD_COLOR)
            if im is None or xyz.shape[:2]!=gt.shape or im.shape[:2]!=gt.shape:
                raise ValueError("Misregistered annotated RGBD")
            targets=target_instances(gt,sem,limit=1_000_000)
            began=time.monotonic()
            result=co.predict(im,conf=conf,imgsz=imgsz,device="cpu",
                classes=classids,retina_masks=True,verbose=False,agnostic_nms=True,
                max_det=500)[0]
            dt=time.monotonic()-began
            masks=[];classes=[];confidences=[]
            if result.masks is not None:
                masks=[cv2.resize(m.astype("float32"),(gt.shape[1],gt.shape[0]),
                        interpolation=cv2.INTER_NEAREST)>.5 if m.shape!=gt.shape
                        else m>.5 for m in result.masks.data.cpu().numpy()]
                classes=[ALLOW[raw_names[int(i)]] for i in result.boxes.cls.int().cpu().tolist()]
                confidences=[float(p) for p in result.boxes.conf.cpu().tolist()]
            alignment,ious,_=match_masks(masks,gt,targets,threshold=.10)
            existing=set(g for g in primary["physical_gt"] if g is not None)
            augmented=[];gtlabels=[]
            for j,(mask,label,score) in enumerate(zip(masks,classes,confidences)):
                if mask.sum()<192:continue
                o=observation(xyz,im,mask,frame,"aux_coco_"+str(j),label,score,scene)
                if o is None:continue
                augmented.append(o)
                gt_id=targets[alignment[j]][0] if j in alignment else None
                gtlabels.append(gt_id if gt_id not in existing else None)
            # Scores below are attached for independent eval only. NEVER use
            # identity labels to decide whether to keep/dismiss a mask.
            fields={k:primary[k] for k in ("frame","target_visible","all_gt_visible")}
            aux_frames.append({**fields,"observations":augmented,"physical_gt":gtlabels})
            fused_frames.append({**fields,
                "observations":primary["observations"]+augmented,
                "physical_gt":primary["physical_gt"]+gtlabels})
            per_frame.append(dict(frame=frame,aux_detections=len(augmented),
              detected_new_GT_ids=len(set(x for x in gtlabels if x is not None)),
              total_target_GT=len(targets),
              inference_seconds=dt,
              coco_mask_tp50=sum(ious[k,v]>=.5 for k,v in alignment.items())))
            if i%10==0 or i==99:
                print("AUX_COCO_FULL_FRAME",scene,i+1,"/",100,
                   "aux",len(augmented),"new_gt_scoring",sum(g is not None for g in gtlabels),
                   "sec",round(dt,2),flush=True)
    finally:z.close();remote.close()
    results={}
    for label,frames in [("primary",raw["frames"]),("auxiliary",aux_frames),("fusion",fused_frames)]:
        cached=dict(scene=scene,mode="predicted_yolo_masks",
            detector="yoloe-26x-seg.pt" if label!="auxiliary" else model,
            gt_labels_are_scoring_only=True,frames=frames)
        result=replay(cached,"v2")
        g=benchmark(cached,OfflineInstanceGraph())
        results[label]={"v2":result["target_gt"],"graph":g["target_gt"],
           "raw_detections":sum(len(f["observations"]) for f in frames)}
        if label=="fusion":
            out=Path(output)
            out.mkdir(parents=True,exist_ok=True)
            with gzip.open(out/"fused_observations_and_GT_scoring.json.gz","wt",encoding="utf8") as f:json.dump(cached,f)
        print("COCO_FALLBACK_REAL_IDF1",scene,label,
            "V2",result["target_gt"]["IDF1"],"GRAPH",g["target_gt"]["IDF1"],flush=True)
    report={"scene":scene,"frames":100,
        "primary":"yoloe-26x-seg.pt with fixed 12 text prompts",
        "secondary":model+" with 10 overlapping COCO classes only",
        "secondary_COCO_allowlist":ALLOW,
        "forbidden_GT_identity_as_model_input":True,
        "GT_world_XYZ_used":True,
        "estimated_SLAM":"NOT_TESTED",
        "confidence":conf,"imgsz":imgsz,"per_frame":per_frame,
        "results":results,
        "caveats":["10/12 fallback classes only; cabinet/bookshelf not covered",
          "GT is for scoring only, not fusion filtering",
          "IoU<0.10 is unmatched and counts against physical IDF1",
          "Additional model requires additional VRAM/time or offline sequential inference"]}
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    (out/"coco_fallback_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    return report

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--scene",required=True)
    p.add_argument("--primary",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--model",default="yolo26x-seg.pt")
    p.add_argument("--conf",type=float,default=.01)
    p.add_argument("--imgsz",type=int,default=960)
    args=p.parse_args()
    process(args.scene,args.primary,args.out,model=args.model,conf=args.conf,imgsz=args.imgsz)
