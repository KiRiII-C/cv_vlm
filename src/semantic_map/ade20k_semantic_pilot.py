"""Probe independent ADE20K dense semantic model as object-recall signal.

This *does not* claim instance IDs or 85% IDF1: semantic map merges
adjacent same-class objects. Quantify whether detector-free class pixels
exist for missed YOLOE26x chairs before investing in 3D instance proposals.
"""
import argparse,csv,io,json,time
from pathlib import Path
import cv2,numpy as np
from .hypersim_range import open_scene,hdf5_array
from .yoloe26x_factorial import target_instances,NYU

MAP_NAMES={
 "chair":"chair","armchair":"chair",
 "table":"table","desk":"table",
 "sofa":"sofa","couch":"sofa","bed":"bed",
 "cabinet":"cabinet","cupboard":"cabinet",
 "bookcase":"bookshelf","bookshelf":"bookshelf",
 "screen":"tv","television receiver":"tv","monitor":"tv",
 "plant":"potted plant","pot":"potted plant"
}
def canonical_class(s):
    label=str(s).lower().strip()
    for token in label.replace("; ",",").split(","):
        label2=token.strip()
        if label2 in MAP_NAMES:return MAP_NAMES[label2]
    return None
def run(scene,output,frames=range(0,30),modelname="yolo26m-sem-ade20k.pt",imgsz=640):
    from ultralytics import YOLO
    mdl=YOLO(modelname)
    names={int(i):v for i,v in mdl.names.items()}
    mapping={i:canonical_class(v) for i,v in names.items() if canonical_class(v)}
    assert "chair" in set(mapping.values()),"ADE20K model lacks chair semantic class"
    print("ADE20K_AVAILABLE_TARGET_CLASSES",mapping,flush=True)
    remote,z=open_scene(scene);observations=[];all_targets=0
    total_gt_pixels=0;correct_pixels=0
    try:
        for frame in frames:
            pre=f"{scene}/images/scene_cam_00_geometry_hdf5/frame.{frame:04d}."
            jpg=f"{scene}/images/scene_cam_00_final_preview/frame.{frame:04d}.tonemap.jpg"
            gt=hdf5_array(z,pre+"semantic_instance.hdf5").astype("int32")
            sem=hdf5_array(z,pre+"semantic.hdf5").astype("int32")
            rgb=cv2.imdecode(np.frombuffer(z.read(jpg),"uint8"),cv2.IMREAD_COLOR)
            begin=time.monotonic()
            result=mdl.predict(rgb,device="cpu",imgsz=imgsz,verbose=False)[0]
            if result.semantic_mask is None:raise RuntimeError("ADE20K model did not produce a dense semantic map")
            dense=result.semantic_mask.data.detach().cpu().numpy()
            if dense.shape!=gt.shape:
                dense=cv2.resize(dense.astype("float32"),(gt.shape[1],gt.shape[0]),
                                 interpolation=cv2.INTER_NEAREST).astype("int32")
            targets=target_instances(gt,sem,limit=1_000_000)
            for uid,label,pix in targets:
                correct=np.zeros_like(gt,bool)
                for k,semantic_label in mapping.items():
                    if semantic_label==label:correct|=(dense==k)
                n=int(np.count_nonzero(correct&(gt==uid)))
                frac=n/max(1,int(pix))
                total_gt_pixels+=pix;correct_pixels+=n;all_targets+=1
                observations.append(dict(frame=int(frame),GT_instance=int(uid),
                   GT_class=label,GT_pixels=int(pix),correct_pixels=n,
                   correct_pixel_fraction=frac))
            if frame%5==0:print("ADE20K_SEMANTIC_PROGRESS",scene,frame,
              "objects",len(targets),"seconds",round(time.monotonic()-begin,2),flush=True)
    finally:z.close();remote.close()
    fractions=[x["correct_pixel_fraction"] for x in observations]
    report=dict(scene=scene,source=modelname,imgsz=imgsz,
        frames=len(list(frames)),unique_GT_objects=len({o["GT_instance"] for o in observations}),
        gt_object_appearances=all_targets,
        gt_pixel_class_recall=correct_pixels/max(1,total_gt_pixels),
        GT_appearances_semantic_coverage_10pct=sum(v>=.10 for v in fractions)/max(1,len(fractions)),
        GT_appearances_semantic_coverage_30pct=sum(v>=.30 for v in fractions)/max(1,len(fractions)),
        GT_appearances_semantic_coverage_50pct=sum(v>=.50 for v in fractions)/max(1,len(fractions)),
        names=mapping,per_GT_instance=observations,
        GT_as_model_input=False,
        limitation="Semantic pixels only. Cannot directly convert to instance detections without class-agnostic 3D clustering.")
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2))
    print("ADE20K_SEMANTIC_RECALL_RESULT",scene,json.dumps({k:v for k,v in report.items() if k.startswith("GT_appearances") or k.startswith("gt_pixel")}),flush=True)
    return report

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--scene",default="ai_005_001")
    p.add_argument("--out",default="results/ade20k_semantic_probe.json")
    p.add_argument("--max-frames",type=int,default=30)
    p.add_argument("--model",default="yolo26m-sem-ade20k.pt")
    p.add_argument("--imgsz",type=int,default=640)
    a=p.parse_args()
    run(a.scene,a.out,frames=range(a.max_frames),modelname=a.model,imgsz=a.imgsz)
