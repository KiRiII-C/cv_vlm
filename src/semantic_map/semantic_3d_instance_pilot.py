"""Measured ADE20K semantic pixels -> 3D instance proposals: isolated mask GT.

GT instance IDs serve ONLY to score proposals. No GT masks/classes are
given to DBSCAN, semantic model, or any other predicted observation.
"""
import argparse,io,json,csv,time
from pathlib import Path
import cv2,numpy as np
from collections import Counter
from .ade20k_semantic_pilot import canonical_class
from .semantic_3d_instance_proposals import instance_proposals
from .hypersim_range import open_scene,hdf5_array
from .yoloe26x_factorial import target_instances,match_masks

def evaluate(scene,output,*,model="yolo26m-sem-ade20k.pt",
             nframes=15,eps_values=(.06,.10,.14)):
    from ultralytics import YOLO
    model_obj=YOLO(model)
    names={int(i):v for i,v in model_obj.names.items()}
    catids={i:canonical_class(name) for i,name in names.items() if canonical_class(name)}
    remote,z=open_scene(scene)
    results={str(eps):[] for eps in eps_values}
    scale=float(dict(csv.reader(io.StringIO(
        z.read(f"{scene}/_detail/metadata_scene.csv").decode())))["meters_per_asset_unit"])
    try:
        for f in range(nframes):
            pre=f"{scene}/images/scene_cam_00_geometry_hdf5/frame.{f:04d}."
            jpg=f"{scene}/images/scene_cam_00_final_preview/frame.{f:04d}.tonemap.jpg"
            inst=hdf5_array(z,pre+"semantic_instance.hdf5").astype("int32")
            cls=hdf5_array(z,pre+"semantic.hdf5").astype("int32")
            xyz=hdf5_array(z,pre+"position.hdf5").astype("float32")*scale
            rgb=cv2.imdecode(np.frombuffer(z.read(jpg),"uint8"),cv2.IMREAD_COLOR)
            got=model_obj.predict(rgb,imgsz=640,device="cpu",verbose=False)[0]
            dense=got.semantic_mask.data.cpu().numpy().astype("int32")
            if dense.shape!=inst.shape:
                dense=cv2.resize(dense.astype("float32"),(inst.shape[1],inst.shape[0]),
                                 interpolation=cv2.INTER_NEAREST).astype("int32")
            targets=target_instances(inst,cls,limit=100000)
            for eps in eps_values:
                begin=time.monotonic()
                proposals=instance_proposals(dense,xyz,catids,eps=eps,
                    min_support=8,subsample=3,min_area=220)
                masks=[m for _,m in proposals];labels=[label for label,_ in proposals]
                matched,IoU,_=match_masks(masks,inst,targets,threshold=.10)
                tp25=tp50=tp75=class_hits=0
                for i,j in matched.items():
                    tp25+=IoU[i,j]>=.25
                    tp50+=IoU[i,j]>=.50
                    tp75+=IoU[i,j]>=.75
                    class_hits+=labels[i]==targets[j][1]
                results[str(eps)].append({
                  "frame":f,"GT_objects":len(targets),
                  "mask_proposals":len(proposals),"GT_matched_IoU10":len(matched),
                  "GT_matched_IoU25":tp25,
                  "GT_matched_IoU50":tp50,
                  "GT_matched_IoU75":tp75,
                  "class_correct_matched":class_hits,
                  "seconds_split":time.monotonic()-begin,
                })
                print("SEM3D_SPLIT_FRAME",scene,f,"eps",eps,"gt",len(targets),
                      "proposals",len(proposals),"tp50",tp50,flush=True)
    finally:z.close();remote.close()
    summary={}
    for k,records in results.items():
        gts=sum(x["GT_objects"] for x in records)
        summary[k]={"GT_occurrences":gts,
          "model_proposals":sum(x["mask_proposals"] for x in records),
          "IoU10_recall":sum(x["GT_matched_IoU10"] for x in records)/max(1,gts),
          "IoU25_recall":sum(x["GT_matched_IoU25"] for x in records)/max(1,gts),
          "IoU50_recall":sum(x["GT_matched_IoU50"] for x in records)/max(1,gts),
          "IoU75_recall":sum(x["GT_matched_IoU75"] for x in records)/max(1,gts),
          "runtime_split_sec":sum(x["seconds_split"] for x in records),
          "model_is_not_oracle":True}
    report={"scene":scene,"frames":nframes,"semantic_source":model,
          "method":"ADE20K dense classes -> 2D connected components -> 3D density clusters -> backproject masks",
          "GT_instance_ID_as_segmentation_input":False,
          "GT_xyz_for_3D_Geometry":True,"full_memory_IDF1":"NOT_TESTED_BY_THIS_PILOT",
          "results":summary,"per_frame":results}
    p=Path(output);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(report,indent=2,ensure_ascii=False,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist() if isinstance(x,np.ndarray) else str(x)))
    print("SEMANTIC_3D_SPLIT_RESULTS",scene,json.dumps(summary),flush=True)
    return report
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--scene",default="ai_005_001")
    p.add_argument("--out",default="results/semantic_3d_instance_pilot.json")
    p.add_argument("--max-frames",type=int,default=15)
    a=p.parse_args();evaluate(a.scene,a.out,nframes=a.max_frames)
