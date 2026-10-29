"""Discover complete Hypersim camera trajectories from official scene ZIP metadata."""
import argparse,json,re
from pathlib import Path
from .hypersim_range import open_scene

def discover(archive, scene):
    groups={}
    for name in archive.namelist():
        m=re.fullmatch(re.escape(scene)+r"/images/scene_(cam_\d\d)_(final_preview|geometry_hdf5)/frame\.(\d+)\.(tonemap\.jpg|semantic_instance\.hdf5|semantic\.hdf5|position\.hdf5)",name)
        if not m:continue
        camera,source,idx,kind=m.groups()
        token=(source,kind)
        groups.setdefault(camera,{}).setdefault(token,set()).add(int(idx))
    report={}
    expected={("final_preview","tonemap.jpg"),("geometry_hdf5","semantic_instance.hdf5"),
              ("geometry_hdf5","semantic.hdf5"),("geometry_hdf5","position.hdf5")}
    for camera,streams in sorted(groups.items()):
        complete=set.intersection(*(streams.get(k,set()) for k in expected)) if all(k in streams for k in expected) else set()
        anyframes=set.union(*streams.values()) if streams else set()
        contiguous=set(range(min(anyframes),max(anyframes)+1)) if anyframes else set()
        report[camera]={"frames_complete":sorted(complete),
                        "first":min(complete) if complete else None,
                        "last":max(complete) if complete else None,
                        "complete_count":len(complete),
                        "union_count":len(anyframes),
                        "missing_modalities":sorted(anyframes-complete),
                        "missing_frame_indices":sorted(contiguous-anyframes),
                        "per_stream_counts":{str(k):len(v) for k,v in streams.items()}}
    return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--scenes",nargs="+",default=["ai_002_001","ai_005_001","ai_017_001"])
    p.add_argument("--output",default="results/hypersim_full_sequences_inventory.json")
    a=p.parse_args()
    result={"source":"Official Hypersim scene ZIP archives",
            "scope":"Every complete RGB/semantic/physical-instance/XYZ frame for every camera trajectory within selected scenes",
            "scenes":{}}
    for scene in a.scenes:
        stream,z=open_scene(scene)
        try:
            cams=discover(z,scene);result["scenes"][scene]=cams
            print("FULL_SEQUENCE_INVENTORY",scene,json.dumps({k:{kk:vv for kk,vv in x.items() if kk!="frames_complete"} for k,x in cams.items()}),flush=True)
            if not cams:raise RuntimeError(f"No camera streams in {scene}")
        finally:z.close();stream.close()
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2))
    return result
if __name__=="__main__":main()
