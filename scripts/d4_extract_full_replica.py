#!/usr/bin/env python3
"""Selectively fetch ORIGINAL full NICE-SLAM Replica office0 source via HTTP ranges.

Unlike D3, this extracts frames from the original 2000-frame trajectory,
not a sparsely resampled 100-frame third-party subset. The 12.4 GB source ZIP
is *never* downloaded in full. Replica research-only license applies.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

URL="https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip"


def indices():
    mapping=set(range(0,2000,4))
    # YOLOE 200 multi-view source frames at original temporal spacing
    segmented=set(range(0,2000,10))
    # Depth-only independent validation: no overlap with integration frames
    withheld=set(range(2,2000,80))
    assert not (mapping & withheld)
    return sorted(mapping|segmented|withheld),mapping,segmented,withheld


def main():
    from remotezip import RemoteZip
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    dest=a.out
    dest.mkdir(parents=True,exist_ok=True)
    frames,geometry,yolo,withheld=indices()
    # 500 integration frames, 200 source model frames, 25 HELDOUT.
    assert (len(geometry),len(yolo),len(withheld))==(500,200,25)
    targets=[
       "Replica/cam_params.json",
       "Replica/office0/traj.txt",
       "Replica/office0_mesh.ply"
    ]
    for idx in frames:
        targets.extend((
          f"Replica/office0/results/frame{idx:06d}.jpg",
          f"Replica/office0/results/depth{idx:06d}.png"
        ))
    with RemoteZip(URL,initial_buffer_size=1048576,timeout=180) as remote:
        manifest={f.filename:f for f in remote.infolist()}
        if len([k for k in manifest if k.startswith(
          "Replica/office0/results/depth") and k.endswith(".png")])!=2000:
            raise RuntimeError("Official archive lacks complete 2000-frame depth source")
        for i,name in enumerate(targets):
            if name not in manifest:
                raise RuntimeError(f"Missing original source: {name}")
            path=dest/name
            path.parent.mkdir(parents=True,exist_ok=True)
            payload=remote.read(name)
            if len(payload)!=manifest[name].file_size:
                raise RuntimeError(f"Partial remote extraction {name}")
            path.write_bytes(payload)
            if i%100==0:
                print("D4_OFFICIAL_REPLICA_SOURCE_PROGRESS",
                      i,len(targets),name,flush=True)
    stats={
     "source_original":URL,
     "source_research_only_license":True,
     "source_total_RBGD_available":2000,
     "source_retrieval":"remote_zip_HTTP_range_original_entries",
     "integration_original_indices":sorted(geometry),
     "YOLOE_original_indices":sorted(yolo),
     "independent_depth_validation_indices":sorted(withheld),
     "original_office0_GT_mesh_local_only":True,
     "file_total_retrieved":len(targets)
    }
    (dest/"d4_provenance.json").write_text(json.dumps(stats,indent=2)+"\n")
    for kind,frame_set in (("integration",geometry),("yolo",yolo),("heldout",withheld)):
        print("D4_ORIGINAL_FRAME_SET",kind,len(frame_set),
              min(frame_set),max(frame_set),flush=True)


if __name__=="__main__":
    main()
