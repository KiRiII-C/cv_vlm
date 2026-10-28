"""Prompted YOLOE-26 instance-segmentation adapter with a bounded category set.

Text prompts are fixed once per run. Does not silently fall back to YOLO11 or
prompt-free YOLOE. First run requires external weights+MobileCLIP text encoder.
"""
from __future__ import annotations
from pathlib import Path
import json

DEFAULT_CLASSES=(
    "chair","table","sofa","cabinet","bookshelf","bed",
    "laptop","computer monitor","keyboard","mouse","cup","potted plant",
)
MODEL_SET={
    "yoloe-26n-seg.pt","yoloe-26s-seg.pt","yoloe-26m-seg.pt",
    "yoloe-26l-seg.pt","yoloe-26x-seg.pt",
}
# Fixed canonical labels maintain one semantic namespace in SQLite / Nav2.
ALIASES={"computer monitor":"tv","monitor":"tv","potted plant":"potted plant",
         "desk":"table","dining table":"table","couch":"sofa"}

def canonical(label):
    label=str(label).strip().lower()
    return ALIASES.get(label,label)

def class_list(path=None):
    if path:
        data=json.loads(Path(path).read_text(encoding="utf-8"))
        labels=data["classes"]
    else:
        labels=list(DEFAULT_CLASSES)
    if not isinstance(labels,list) or not 7<=len(labels)<=15:
        raise ValueError("Require 7-15 text-prompted object classes")
    if any(not isinstance(c,str) or not c.strip() for c in labels):
        raise ValueError("Every category must be a nonempty text prompt")
    names=[c.strip().lower() for c in labels]
    if len(set(names))!=len(names):raise ValueError("Duplicate category prompts")
    # A model must not be asked to segment competing aliases of one class.
    if len({canonical(c) for c in names})!=len(names):
        raise ValueError("Prompts map to ambiguous duplicate canonical classes")
    return names

class YOLOE26Segmenter:
    def __init__(self,weights="yoloe-26s-seg.pt",classes=None):
        if Path(weights).name not in MODEL_SET:
            raise ValueError("YOLOE-26 instance-seg checkpoint required (-seg.pt, not -seg-pf.pt)")
        self.classes=list(DEFAULT_CLASSES) if classes is None else list(classes)
        if not 7<=len(self.classes)<=15 or len(set(self.classes))!=len(self.classes):
            raise ValueError("7-15 unique text categories required")
        from ultralytics import YOLOE
        self.model=YOLOE(weights)
        self.model.set_classes(self.classes)
        self.weights=weights
        self.names={i:canonical(v) for i,v in enumerate(self.classes)}

    def predict(self,rgb,*,conf=.23,device="cpu",imgsz=640):
        results=self.model.predict(
            source=rgb,conf=float(conf),device=device,imgsz=int(imgsz),
            retina_masks=True,verbose=False,agnostic_nms=True,nms=True,
        )
        if len(results)!=1:raise RuntimeError("Expected one-frame YOLOE result")
        return results[0]

    def class_name(self,idx):
        if int(idx) not in self.names:raise ValueError(f"YOLOE returned unconfigured class {idx}")
        return self.names[int(idx)]

    def save_prompt_embeddings(self,path):
        self.model.save_prompt_embeddings(str(path))

def build_detector(weights,classes,*,legacy=False):
    """Legacy YOLO is allowed only when explicitly selected for comparison."""
    if Path(weights).name in MODEL_SET:
        return YOLOE26Segmenter(weights,classes)
    if not legacy:raise ValueError("Legacy model requires explicit legacy=True")
    from ultralytics import YOLO
    detector=YOLO(weights)
    class Legacy:
        names=detector.names
        def predict(self,rgb,*,conf=.23,device="cpu",imgsz=640):
            return detector.predict(source=rgb,conf=conf,device=device,imgsz=imgsz,
                                    retina_masks=True,verbose=False)[0]
        def class_name(self,idx):return canonical(self.names[int(idx)])
    return Legacy()
