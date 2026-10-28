"""Optional CPU image-language semantic enrichment of stable object hypotheses.

Runs on selected cropped real RGB views, not on every video frame. VLM text is
unverified evidence and NEVER changes physical object IDs or geometry.
"""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np,torch
from PIL import Image
from .storage import MemoryStore

CLASS_SYNONYMS={
  "chair":("chair","seat","stool"),"tv":("tv","monitor","screen","television","display"),
  "book":("book",),"keyboard":("keyboard",),"laptop":("laptop","notebook"),
  "mouse":("mouse",),"cup":("cup","mug"),"dining table":("table","desk"),
  "bottle":("bottle",),"remote":("remote",),"bed":("bed",),"cell phone":("phone","mobile")
}

def class_compatible(label,description):
    terms=CLASS_SYNONYMS.get(label.lower(),(label.lower(),))
    return any(word in description.lower() for word in terms)

def enrich(root,out,max_objects=8,model_name="HuggingFaceTB/SmolVLM-256M-Instruct"):
    from transformers import AutoProcessor, AutoModelForImageTextToText
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    db=MemoryStore(root/"semantic_memory.sqlite")
    tracks=sorted([t for t in db.tracks() if t.state=="confirmed" and t.hits>=10],
                  key=lambda t:-t.hits)[:max_objects]
    db.close()
    selected=[]
    for t in tracks:
        pics=sorted((root/"object_views").glob(t.object_id+"_*.jpg"))
        # Multiple independent views: higher pixel area first, rather than
        # accidentally picking the earliest occluded/truncated crop.
        if pics:selected.append((t,sorted(pics,key=lambda p:p.stat().st_size,reverse=True)[:2]))
    if not selected:
        raise RuntimeError("No real object view crops: missing perception evidence")
    torch.set_num_threads(3)
    print("VLM_LOAD",model_name,"objects",len(selected),flush=True)
    processor=AutoProcessor.from_pretrained(model_name)
    model=AutoModelForImageTextToText.from_pretrained(model_name,dtype=torch.float32,
                                                     attn_implementation="eager").eval().to("cpu")
    # Small CPU-friendly crop with one visual patch, no interactive cloud API.
    try:
        processor.image_processor.size={"longest_edge":512}
        processor.image_processor.max_image_size={"longest_edge":512}
    except Exception:
        pass
    descriptions=[]
    for track,img_paths in selected:
        views=[]
        for img_path in img_paths:
            image=Image.open(img_path).convert("RGB")
            image.thumbnail((448,448))
            msg=[{"role":"user","content":[
                {"type":"image"},
                {"type":"text","text":
                 "Describe the main physical object in this image in one short factual sentence. "
                 "Give color and visible distinguishing features only when clear. "
                 "If the object is ambiguous or the crop too poor, say uncertain."}]}]
            prompt=processor.apply_chat_template(msg,add_generation_prompt=True)
            inputs=processor(text=prompt,images=[image],return_tensors="pt")
            started=time.time()
            with torch.inference_mode():
                ids=model.generate(**inputs,max_new_tokens=48,do_sample=False)
            cut=int(inputs["input_ids"].shape[-1])
            response=processor.batch_decode(ids[:,cut:],skip_special_tokens=True)[0].strip()
            compatible=class_compatible(track.label,response)
            views.append({"image":str(img_path.relative_to(root)),
                "vlm_text":response,"time_sec":round(time.time()-started,3),
                "class_lexically_compatible":bool(compatible)})
        # This is a lexical agreement guard, NOT correctness validation.
        selected_view=next((v for v in views if v["class_lexically_compatible"]),views[0])
        compatible=bool(selected_view["class_lexically_compatible"])
        record={"object_id":track.object_id,"detector_label":track.label,
                "observation_count":track.hits,
                "image":selected_view["image"],
                "vlm_text":selected_view["vlm_text"],
                "candidate_views":views,
                "time_sec":round(sum(v["time_sec"] for v in views),3),
                "validation_status":"UNVERIFIED_CONSISTENT_VLM_DESCRIPTION" if compatible else "REJECTED_SEMANTIC_DISAGREEMENT",
                "identity_was_not_modified":True}
        descriptions.append(record)
        print("VLM_OBJECT",json.dumps(record,ensure_ascii=False),flush=True)
    accepted=[x for x in descriptions if x["validation_status"]=="UNVERIFIED_CONSISTENT_VLM_DESCRIPTION"]
    result={"model":model_name,"objects_considered":len(tracks),
            "objects_described":len(descriptions),"consistent_descriptions":len(accepted),
            "quarantined_descriptions":len(descriptions)-len(accepted),
            "semantic_accuracy_against_GT":"NOT_EVALUATED",
            "identity_changes":0,"descriptions":descriptions}
    # Persist VLM text as non-authoritative attributes; never rewrite geometry or ID.
    store=MemoryStore(root/"semantic_memory.sqlite")
    for entry in accepted:
        store.annotate(entry["object_id"],attributes={
            "vlm_description_unverified":entry["vlm_text"],
            "vlm_model":model_name,
            "vlm_validation":"UNVERIFIED"})
    store.close()
    (out/"vlm_semantics.json").write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print("VLM_DONE",len(descriptions),flush=True)
    return result

if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("--root",default="results/tum_real")
    a.add_argument("--out",default="results/vlm_semantics")
    a.add_argument("--max-objects",type=int,default=8)
    args=a.parse_args()
    enrich(args.root,args.out,args.max_objects)
