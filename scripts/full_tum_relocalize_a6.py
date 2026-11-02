#!/usr/bin/env python3
"""A6-L: repeat ALL paired RGB-D frames from a different starting viewpoint.

This is an intentionally favourable SAME-SEQUENCE re-observation smoke test.
No initial_pose or /initialpose is published. A map DB from a complete prior
mapping pass is required. RGBD/StartAtOrigin=true prevents using the saved
last localization pose as an initialization hint.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
from tum_full_replay_a4 import full_rgbd_pairs,replay

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",type=Path,default=Path("data/tum"))
    p.add_argument("--output",type=Path,default=Path("results/a6loc/full_revisit.json"))
    p.add_argument("--cyclic-start",type=int,default=297)
    p.add_argument("--max-rate-hz",type=float,default=15.)
    p.add_argument("--ack-timeout-s",type=float,default=5.)
    args=p.parse_args()
    pairs,meta=full_rgbd_pairs(args.data)
    if not 0<=args.cyclic_start<len(pairs):
        p.error("cyclic-start out of range")
    if args.cyclic_start==0:
        p.error("test must start at different viewpoint")
    first_original_idx=args.cyclic_start
    pairs=pairs[args.cyclic_start:]+pairs[:args.cyclic_start]
    meta["relocalization_full_dataset"]=True
    meta["cyclic_start_original_index"]=first_original_idx
    meta["cyclic_wrap_creates_physical_teleport"]=True
    meta["no_manual_initial_pose"]=True
    meta["all_pairs_replayed_exactly_once"]=True
    args.output.parent.mkdir(parents=True,exist_ok=True)
    (args.output.parent/"revisit_manifest.json").write_text(json.dumps(meta,indent=2)+"\n")
    print("A6_RELOCALIZATION_INPUT",json.dumps(meta),flush=True)
    replay(args,pairs,meta)

if __name__=="__main__":
    main()
