"""C4 — cross-session multi-view consensus for *geometric* object UUID candidates.

Inputs are real C2 measured-depth view evidence and real C3 independent YOLOE
detections. Neither input contains cross-session physical instance GT. This
module NEVER creates, renames, deletes or confirms physical UUIDs, and never
treats lack of a YOLO detection as evidence that an object disappeared.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import numpy as np


def unit_quat(q):
    x = np.asarray(q, dtype=float)
    if x.shape != (4,) or not np.isfinite(x).all() or np.linalg.norm(x) < 1e-8:
        raise ValueError("Invalid accepted pose quaternion")
    return x / np.linalg.norm(x)


def is_independent_view(a, b, min_translation_m=.12, min_rotation_deg=15.):
    """A view is new when camera translation OR camera orientation changes."""
    d = np.linalg.norm(np.asarray(a["xyz"], float) - np.asarray(b["xyz"], float))
    angle = math.degrees(2 * math.acos(min(1., abs(float(np.dot(
        unit_quat(a["quaternion"]), unit_quat(b["quaternion"])))))))
    return bool(d >= min_translation_m or angle >= min_rotation_deg)


def independent_representatives(rows):
    unique = []
    for item in sorted(rows, key=lambda x: x["frame"]):
        if all(is_independent_view(item, previous) for previous in unique):
            unique.append(item)
    return unique


def read_jsonlines(path):
    with Path(path).open() as f:
        return [json.loads(s) for s in f if s.strip()]


def audit(c2_report, c2_rows, c3_report, c3_rows):
    if c2_report.get("hypothesis_count") != 15 or c3_report.get("old_C1_UUID_hypotheses") != 15:
        raise ValueError("Source 15-hypothesis provenance mismatch")
    if not c3_report.get("no_existing_C1_SQLite_changes") or not c3_report.get("no_new_physical_UUID_asserted"):
        raise ValueError("C3 source violates immutable object-memory contract")
    if c2_report.get("old_memory_mutated") is not False:
        raise ValueError("C2 source modified original physical ID memory")
    if c2_report.get("accepted_relocalization_poses_evaluated") != c3_report.get("actual_YOLOE26x_inference_frames"):
        raise ValueError("C2/C3 accepted source-pose counts differ")
    if c2_report.get("real_independent_depth_frames_processed_for_source_validation") != 639:
        raise ValueError("C2 did not verify all real desk2 depth measurements")
    if c3_report.get("all_independent_RGBD_source_frames_accounted") != 639:
        raise ValueError("C3 did not account for full desk2 source")
    ids = {x["uuid_hypothesis"] for x in c2_report["hypotheses"]}
    if len(ids) != 15:
        raise ValueError("Duplicate or missing C1 saved UUID hypothesis")
    evidence = {}
    accepted_frames = set()
    frame_pose = {}
    for x in c2_rows:
        frame = int(x["frame"])
        uid = str(x["source_uuid_hypothesis"])
        if uid not in ids:
            raise ValueError("Evidence refers to unknown C1 UUID")
        key = (frame, uid)
        if key in evidence:
            raise ValueError("Duplicated C2 evidence for a UUID/frame")
        evidence[key] = x
        accepted_frames.add(frame)
        position, quaternion = np.asarray(x["xyz"], float), unit_quat(x["quaternion"])
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("Invalid C2 map-relative position")
        if frame in frame_pose:
            old = frame_pose[frame]
            if not np.allclose(old["xyz"], position, atol=1e-8) or (
                    abs(float(np.dot(unit_quat(old["quaternion"]), quaternion))) < 1.-1e-8):
                raise ValueError("Conflicting accepted camera pose in one C2 frame")
        else:
            frame_pose[frame] = {"frame": frame, "xyz": position.tolist(),
                                 "quaternion": quaternion.tolist()}
    if len(accepted_frames) != c3_report["actual_YOLOE26x_inference_frames"]:
        raise ValueError("C2 actual accepted pose count mismatched C3")
    if len(evidence) != len(ids) * len(accepted_frames):
        raise ValueError("Missing C2 map-frame geometry evidence")
    if len(c3_rows) != c3_report["counts"]["valid_new_3D_observations"]:
        raise ValueError("Missing C3 YOLOE 3D observations")

    all_matches = defaultdict(list)
    per_frame_candidates = set()
    nonmatched = []
    for row in c3_rows:
        frame = int(row["frame_index"])
        if frame not in accepted_frames:
            raise ValueError("C3 observation without accepted RTAB-Map localization pose")
        cls = row.get("class")
        center = np.asarray(row["center_map_m"], float)
        if center.shape != (3,) or not np.isfinite(center).all() or not cls:
            raise ValueError("C3 invalid real object geometry")
        status = row["status"]
        uid = row.get("old_uuid_candidate")
        if status == "GEOMETRIC_MATCH_CANDIDATE_NOT_PHYSICAL_ID":
            if uid not in ids or row.get("geometry_score") is None:
                raise ValueError("Matched candidate has no source or score")
            s = float(row["geometry_score"])
            if not .48 <= s <= 1.0000001:
                raise ValueError("C3 geometry score out of documented range")
            if (frame, uid) in per_frame_candidates:
                raise ValueError("C3 attempts to associate several objects to same UUID in one frame")
            per_frame_candidates.add((frame, uid))
            combined = dict(row)
            combined["view"] = frame_pose[frame]
            combined["depth_state"] = evidence[(frame, uid)]["state"]
            all_matches[uid].append(combined)
        else:
            if uid is not None:
                raise ValueError("Unmatched or ambiguous C3 observation has forced UUID")
            nonmatched.append(row)

    hypotheses = []
    state_counts = Counter()
    for uid in sorted(ids):
        obs = sorted(all_matches[uid], key=lambda r: r["frame_index"])
        viewpoints = independent_representatives([
            frame_pose[r["frame_index"]] for r in obs])
        depth_ok = [r for r in obs if r["depth_state"] == "GEOMETRICALLY_COMPATIBLE_NOT_ID_PROOF"]
        depth_conflict = [r for r in obs if r["depth_state"] == "FREE_SPACE_CONFLICT_CANDIDATE"]
        centers = np.asarray([r["center_map_m"] for r in obs], float).reshape(-1, 3)
        # This measures detection-center dispersion, not absolute 3D position error.
        med = np.median(centers, axis=0) if len(obs) else None
        deviations = np.linalg.norm(centers - med, axis=1) if len(obs) else np.asarray([])
        spread = float(np.median(deviations)) if len(obs) else None
        scores = [float(r["geometry_score"]) for r in obs]
        median_score = float(np.median(scores)) if scores else None
        # Absence without detection is never a negative; conflicts require review.
        if not obs:
            state = "UNOBSERVED_OR_UNMATCHED_UNKNOWN_NOT_ABSENT"
        elif depth_conflict:
            state = "REVIEW_DEPTH_GEOMETRY_CONTRADICTION"
        elif spread is not None and spread > .30:
            state = "REVIEW_MULTIVIEW_CENTER_INCONSISTENCY"
        elif len(viewpoints) >= 3 and len(depth_ok) >= 2 and median_score >= .60:
            state = "MULTIVIEW_GEOMETRY_SUPPORTED_NOT_PHYSICAL_ID"
        else:
            state = "PROVISIONAL_NOT_ENOUGH_INDEPENDENT_EVIDENCE"
        state_counts[state] += 1
        hypotheses.append({
            "uuid_hypothesis": uid,
            "status": state,
            "C3_matched_real_observations": len(obs),
            "matched_frame_indices": [int(r["frame_index"]) for r in obs],
            "independent_viewpoints": len(viewpoints),
            "median_geometry_score": median_score,
            "median_center_deviation_m": spread,
            "same_frame_depth_compatible_observations": len(depth_ok),
            "same_frame_depth_conflicts": len(depth_conflict),
            "physical_id_verified": False,
            "new_uuid_committed": False,
            "object_moved_or_deleted": False,
        })
    if sum(x["C3_matched_real_observations"] for x in hypotheses) + len(nonmatched) != len(c3_rows):
        raise RuntimeError("C3 observation conservation violated")
    out = {
        "experiment": "C4_CROSS_SESSION_MULTIVIEW_GEOMETRIC_CONSENSUS",
        "source_runs": {"C2": 41489070935, "C3": 41489734662},
        "actual_independent_RGBD_depth_frames": 639,
        "accepted_localization_frames": len(accepted_frames),
        "real_YOLOE_3D_observations": len(c3_rows),
        "geometrically_matched_observations": sum(len(v) for v in all_matches.values()),
        "unmatched_or_ambiguous_observations": len(nonmatched),
        "saved_old_uuid_hypotheses": len(ids),
        "hypothesis_status_counts": dict(state_counts),
        "hypotheses": hypotheses,
        "no_mutation_of_C1_SQLite": True,
        "no_new_physical_identity_asserted": True,
        "no_unseen_object_deletion": True,
        "physical_reidentification_IDF1": "NOT_EVALUATED_NO_COMMON_INSTANCE_GT",
        "limitation": ("C3 geometry-only matching, no saved appearance embedding; "
          "A8 provides just 25 accepted poses in 639 query depth frames; "
          "camera pose uncertainty not calibrated; nearby observations "
          "and same-class repeats can share surfaces. Multiview support "
          "is geometric consistency, not independent proof of physical ID.")
    }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--c2", type=Path, default=Path("results/c2"))
    p.add_argument("--c3", type=Path, default=Path("results/c3"))
    p.add_argument("--out", type=Path, default=Path("results/c4"))
    args = p.parse_args()
    c2 = json.loads((args.c2 / "c2_cross_session_visibility.json").read_text())
    c3 = json.loads((args.c3 / "c3_report.json").read_text())
    result = audit(c2, read_jsonlines(args.c2 / "c2_frame_uuid_evidence.jsonl"),
                   c3, read_jsonlines(args.c3 / "c3_independent_object_candidates.jsonl"))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "c4_conservative_multiview_report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    with (args.out / "c4_uuid_decisions.jsonl").open("w") as f:
        for row in result["hypotheses"]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("C4_REAL_MULTIVIEW_CONSENSUS", json.dumps({
        "accepted_localization_frames": result["accepted_localization_frames"],
        "real_YOLOE_3D_observations": result["real_YOLOE_3D_observations"],
        "geometrically_matched_observations": result["geometrically_matched_observations"],
        "unmatched_or_ambiguous_observations": result["unmatched_or_ambiguous_observations"],
        "hypothesis_status_counts": result["hypothesis_status_counts"],
        "physical_reidentification_IDF1": result["physical_reidentification_IDF1"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
