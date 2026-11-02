#!/usr/bin/env python3
"""C0: replay REAL B5 TUM 3D observations and three REAL 100-frame Hypersim caches.

No GT identity used in matching. Physical GT IDs are scored AFTER assignments.
Do not call candidate IDs verified instance UUIDs. No disappearance from absence.
"""
import argparse,gzip,json
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment
from semantic_map.conservative_object_memory_c0 import ConservativeObjectMemory

SCENES=("ai_002_001","ai_005_001","ai_017_001")

def identity_score(predictions,gt_total):
    """Global one-to-one matching IDTP/IDFP/IDFN on every original GT frame."""
    counts=Counter()
    gt_to_pred=defaultdict(set); pred_to_gt=defaultdict(set)
    count_pred=0
    for physical,candidate in predictions:
        if candidate is None:continue
        count_pred+=1
        if physical is None:continue
        counts[(int(physical),str(candidate))]+=1
        gt_to_pred[int(physical)].add(str(candidate))
        pred_to_gt[str(candidate)].add(int(physical))
    gt=sorted(gt_to_pred)
    pred=sorted(pred_to_gt)
    hits=0
    if gt and pred:
        mat=np.zeros((len(gt),len(pred)),dtype=np.int64)
        gi={v:i for i,v in enumerate(gt)}
        pi={v:i for i,v in enumerate(pred)}
        for (g,p),k in counts.items():mat[gi[g],pi[p]]=k
        ix,jx=linear_sum_assignment(-mat)
        hits=int(mat[ix,jx].sum())
    fp=count_pred-hits
    fn=gt_total-hits
    return {"IDTP":hits,"IDFP":fp,"IDFN":fn,
        "IDF1":2*hits/max(1,2*hits+fp+fn),
        "candidate_merge_with_multiple_physical_GT":sum(len(v)>1 for v in pred_to_gt.values()),
        "physical_GT_fragmented_across_candidates":sum(len(v)>1 for v in gt_to_pred.values()),
        "gt_total_visible_frame_instances":gt_total,
        "prediction_with_candidate_id":count_pred}

def evaluate_hypersim(root,scene,use_appearance):
    p=root/f"full_sequence_{scene}"/"raw_observations_and_GT_scoring.json.gz"
    if not p.exists():p=root/"results"/f"full_sequence_{scene}"/"raw_observations_and_GT_scoring.json.gz"
    if not p.exists():raise FileNotFoundError(p)
    with gzip.open(p,"rt") as handle:data=json.load(handle)
    assert data["detector"]=="yoloe-26x-seg.pt"
    assert len(data["frames"])==100
    assert [int(f["frame"]) for f in data["frames"]]==list(range(100))
    memory=ConservativeObjectMemory(appearance_weight=.17 if use_appearance else 0.)
    scored=[]
    total_visible=0
    number_observations=0
    for f in data["frames"]:
        frame=int(f["frame"])
        obs=f["observations"]
        gt=f["physical_gt"]
        target=set(int(v) for v in f["target_visible"])
        if len(obs)!=len(gt):raise RuntimeError("Hypersim GT not aligned to observations")
        for o in obs:
            if "physical_gt" in o:raise RuntimeError("GT physical ID leaked into tracker observation")
        decisions=memory.update(frame,obs)
        for label,decision in zip(gt,decisions):
            physical=(int(label) if label is not None and int(label) in target else None)
            scored.append((physical,decision["candidate_id"]))
        total_visible+=len(target)
        number_observations+=len(obs)
    result=identity_score(scored,total_visible)
    result.update({"scene":scene,"all_original_frames":100,
        "real_frozen_YOLOE26x_observations":number_observations,
        "candidate_count":len(memory.objects),
        "association_counts":dict(memory.count),
        "input_geometry":"OFFICIAL_HYPERSIM_3D_XYZ_ORACLE_NOT_RTABMAP",
        "physical_GT_used_for_assignment":False,
        "physical_GT_scoring_only":True,
        "appearance_used":use_appearance})
    return result

def evaluate_tum(path):
    events=[json.loads(row) for row in Path(path).read_text().splitlines() if row.strip()]
    if len(events)!=871:raise RuntimeError(f"Unexpected B5 evidence count {len(events)}")
    frames=defaultdict(list)
    for row in events:
        k=int(row["frame_index"])
        if not 0<=k<595:raise RuntimeError("Frame outside complete TUM desk")
        frames[k].append(row)
    memory=ConservativeObjectMemory()
    status=Counter()
    record=[]
    for frame in range(595):
        assignments=memory.update(frame,frames.get(frame,[]))
        for orig,item in zip(frames.get(frame,[]),assignments):
            status[item["status"]]+=1
            record.append({"frame":frame,"detection_id":orig["detection_id"],
                           "semantic_class":orig["class"],
                           "candidate_id":item["candidate_id"],"status":item["status"],
                           "physical_ID_validated":False})
    return memory,record,{"source":"TUM_freiburg1_desk_real_B5_YOLOE26x",
        "real_rgbd_frames_accounted":595,
        "recorded_YOLOE_keyframes":len(frames),
        "real_3d_observations":len(events),
        "observation_statuses":dict(status),
        "candidate_count":len(memory.objects),
        "multiple_view_candidate_count":sum(c.n>=3 for c in memory.objects.values()),
        "physical_ID_count":"UNKNOWN",
        "unknown_unobserved_objects_are_never_auto_deleted":True}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--tum",type=Path,default=Path("results/b5/b5_object_observations.jsonl"))
    parser.add_argument("--hypersim",type=Path,default=Path("results/c0_inputs"))
    parser.add_argument("--output",type=Path,default=Path("results/c0"))
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    memory,events,tum=evaluate_tum(args.tum)
    scores={}
    for scene in SCENES:
        scores[scene]={}
        for name,use_appearance in (("3d_only",False),("3d_plus_hsv",True)):
            scores[scene][name]=evaluate_hypersim(args.hypersim,scene,use_appearance)
    pooled={}
    for setting in ("3d_only","3d_plus_hsv"):
        metrics=Counter()
        for scene in SCENES:
            row=scores[scene][setting]
            metrics.update({k:row[k] for k in ("IDTP","IDFP","IDFN")})
        pooled[setting]={"IDTP":metrics["IDTP"],"IDFP":metrics["IDFP"],"IDFN":metrics["IDFN"],
                         "IDF1":2*metrics["IDTP"]/max(1,2*metrics["IDTP"]+metrics["IDFP"]+metrics["IDFN"])}
    output={"experiment":"C0_CONSERVATIVE_OBJECT_OBSERVATION_MEMORY",
        "tum":tum,"hypersim_full_300_physical_GT":scores,
        "pooled_physical_IDF1":pooled,
        "physical_UUID_validation":"NOT_ACCEPTED",
        "object_moved_detection":"NOT_TESTED",
        "object_disappeared_detection":"NOT_TESTED",
        "camera_pose_errors_on_hypersim":"NOT_TESTED_ORACLE_XYZ",
        "limitation":"C0 candidate landmarks, not physical object IDs or dynamic scene inference"}
    (args.output/"c0_report.json").write_text(json.dumps(output,indent=2,ensure_ascii=False)+"\n")
    with (args.output/"c0_observation_assignments.jsonl").open("w") as f:
        for x in events:f.write(json.dumps(x,ensure_ascii=False)+"\n")
    (args.output/"c0_candidate_landmarks.json").write_text(json.dumps(memory.snapshot(),indent=2,ensure_ascii=False)+"\n")
    print("C0_REAL_DATA_RESULT",json.dumps({"tum":tum,"pooled":pooled},ensure_ascii=False),flush=True)
if __name__=="__main__":main()
