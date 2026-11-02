#!/usr/bin/env python3
"""A7: download a wholly independent TUM sequence and replay ALL RGB-D pairs.

Map session: freiburg1_desk. Query session: freiburg1_desk2. Same desks, new
recording. No RGB-D query frame is drawn from the mapping source. No global
GT poses are passed to SLAM and no synthetic movement/teleport is introduced.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import time
import urllib.request

from semantic_map.tum_yolo_pipeline import parse
from tum_full_replay_a4 import associate_full_rgb_depth, replay

SCENE = "rgbd_dataset_freiburg1_desk2"
URL = "https://webshare.cvg.cit.tum.de/rgbd/dataset/freiburg1/" + SCENE + ".tgz"

def download_desk2(cache: Path):
    cache.mkdir(parents=True, exist_ok=True)
    folder = cache / SCENE
    if (folder/"rgb.txt").is_file() and (folder/"depth.txt").is_file() and (folder/"groundtruth.txt").is_file():
        return folder
    archive = cache / (SCENE+".tgz")
    if not archive.exists() or archive.stat().st_size<20_000_000:
        for attempt in range(3):
            try:
                print("A7_DOWNLOAD_ATTEMPT",attempt+1,URL,flush=True)
                req=urllib.request.Request(URL,headers={"User-Agent":"cv-vlm-A7-research/1.0"})
                with urllib.request.urlopen(req,timeout=120) as response, (cache/"desk2.tmp").open("wb") as out:
                    import shutil
                    shutil.copyfileobj(response,out)
                if (cache/"desk2.tmp").stat().st_size<20_000_000:
                    raise RuntimeError("Unexpectedly small TUM archive")
                (cache/"desk2.tmp").replace(archive)
                break
            except Exception as e:
                if (cache/"desk2.tmp").exists():(cache/"desk2.tmp").unlink()
                if attempt==2:raise
                time.sleep(3*(attempt+1))
    with tarfile.open(archive,"r:gz") as tar:
        members=tar.getmembers()
        for member in members:
            target=Path(member.name)
            if (target.is_absolute() or ".." in target.parts or
                not member.name.startswith(SCENE+"/") or
                not (member.isfile() or member.isdir())):
                raise ValueError("Unsafe archive member: "+member.name)
        tar.extractall(cache,members=members,filter="data")
    for name in ("rgb.txt","depth.txt","groundtruth.txt"):
        if not (folder/name).is_file():
            raise RuntimeError("Incomplete official TUM archive: "+str(folder/name))
    return folder

def independent_pairs(root: Path):
    rgb=parse(root/"rgb.txt")
    depth=parse(root/"depth.txt")
    assignments,total_dt=associate_full_rgb_depth(rgb,depth,.03)
    if not assignments:raise RuntimeError("No independent RGB-D pairs")
    records=[(float(rgb[i][0]),root/rgb[i][1][0],
              root/depth[j][1][0],float(depth[j][0])) for i,j in assignments]
    if len({str(x[1]) for x in records})!=len(records) or len({str(x[2]) for x in records})!=len(records):
        raise RuntimeError("Duplicate RGB or depth in independent query")
    import bisect
    ds=[float(x[0]) for x in depth]
    no_candidate=0
    for t,_ in rgb:
        at=bisect.bisect_left(ds,t)
        if not any(0<=j<len(ds) and abs(ds[j]-t)<=.03 for j in (at-1,at)):
            no_candidate+=1
    stats={"source_dataset":"TUM_freiburg1_desk2_INDEPENDENT_QUERY",
           "source_rgb_frames":len(rgb),"source_depth_frames":len(depth),
           "synchronized_rgbd_pairs":len(records),
           "rgb_without_nearby_depth":no_candidate,
           "rgb_excluded_after_one_to_one_matching":len(rgb)-no_candidate-len(records),
           "unpaired_depth":len(depth)-len(records),
           "pairing_tolerance_s":.03,"matching_method":"monotonic_global_maximum_cardinality_minimum_total_timestamp_error",
           "total_rgb_depth_abs_dt_s":total_dt,"all_paired_frames_selected":True,
           "frame_stride":1,"max_frames":None,
           "groundtruth_access_during_pairing":False,
           "independent_of_mapping_recording":True,
           "mapping_scene":"rgbd_dataset_freiburg1_desk",
           "query_scene":SCENE,
           "no_frame_subsampling":True}
    return records,stats

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",type=Path,default=Path("data/tum_a7"))
    p.add_argument("--output",type=Path,default=Path("results/a7/full_desk2.json"))
    p.add_argument("--manifest-only",action="store_true")
    p.add_argument("--max-rate-hz",type=float,default=15.)
    p.add_argument("--ack-timeout-s",type=float,default=5.)
    args=p.parse_args()
    folder=download_desk2(args.data)
    pairs,meta=independent_pairs(folder)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    (args.output.parent/"independent_query_manifest.json").write_text(json.dumps(meta,indent=2)+"\n")
    print("A7_INDEPENDENT_TUM_DATASET",json.dumps(meta),flush=True)
    if args.manifest_only:return
    # The A4 replay routine's post-hoc GT evaluator uses the queried folder,
    # but query GT is NOT read by the SLAM pipeline.
    replay(args,pairs,meta)
    out=json.loads(args.output.read_text())
    out["source_dataset"]="TUM_freiburg1_desk2_INDEPENDENT_QUERY"
    out["independent_of_mapping_recording"]=True
    out["GT_used_as_input"]=False
    args.output.write_text(json.dumps(out,indent=2)+"\n")

if __name__=="__main__":
    main()
