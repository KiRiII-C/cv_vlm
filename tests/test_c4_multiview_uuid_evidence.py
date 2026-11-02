"""C4 unit tests: toy fixtures validate safety, NEVER represent sensor benchmarks."""
from __future__ import annotations
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from c4_multiview_uuid_evidence import audit, is_independent_view


def fixture():
    uids = [f"uuid-{i:02}" for i in range(15)]
    frames = [0, 2, 4, 6, 8]
    evidence = []
    for k, frame in enumerate(frames):
        for uid in uids:
            evidence.append({"frame": frame, "source_uuid_hypothesis": uid,
                "xyz": [0.2*k,0,0], "quaternion":[0,0,0,1],
                "state":"GEOMETRICALLY_COMPATIBLE_NOT_ID_PROOF"})
    c2 = {"hypothesis_count":15, "old_memory_mutated":False,
          "accepted_relocalization_poses_evaluated":5,
          "real_independent_depth_frames_processed_for_source_validation":639,
          "hypotheses":[{"uuid_hypothesis": uid} for uid in uids]}
    obs = [{"frame_index":frame, "class":"chair",
            "center_map_m":[0.03 + .01*k, 0., 1.],
            "status":"GEOMETRIC_MATCH_CANDIDATE_NOT_PHYSICAL_ID",
            "old_uuid_candidate":"uuid-00", "geometry_score":.81}
           for k,frame in enumerate(frames[:3])]
    c3 = {"old_C1_UUID_hypotheses":15,
        "no_existing_C1_SQLite_changes":True,
        "no_new_physical_UUID_asserted":True,
        "actual_YOLOE26x_inference_frames":5,
        "all_independent_RGBD_source_frames_accounted":639,
        "counts":{"valid_new_3D_observations":len(obs)}}
    return c2,evidence,c3,obs


class ConsensusTests(unittest.TestCase):
    def test_multiple_independent_valid_frames_support_geometry_only(self):
        a,b,c,d=fixture()
        result=audit(a,b,c,d)
        m=next(x for x in result["hypotheses"] if x["uuid_hypothesis"]=="uuid-00")
        self.assertEqual(m["status"],"MULTIVIEW_GEOMETRY_SUPPORTED_NOT_PHYSICAL_ID")
        self.assertEqual(m["independent_viewpoints"],3)
        self.assertFalse(m["physical_id_verified"])
        self.assertTrue(result["no_mutation_of_C1_SQLite"])

    def test_other_saved_objects_are_unknown_not_disappeared(self):
        a,b,c,d=fixture()
        r=audit(a,b,c,d)
        missing=[x for x in r["hypotheses"] if x["C3_matched_real_observations"]==0]
        self.assertEqual(len(missing),14)
        self.assertTrue(all(x["status"]=="UNOBSERVED_OR_UNMATCHED_UNKNOWN_NOT_ABSENT" for x in missing))

    def test_depth_contradiction_requires_review(self):
        a,b,c,d=fixture()
        next(x for x in b if x["frame"]==2 and x["source_uuid_hypothesis"]=="uuid-00")[
            "state"]="FREE_SPACE_CONFLICT_CANDIDATE"
        result=audit(a,b,c,d)
        self.assertEqual(result["hypotheses"][0]["status"],"REVIEW_DEPTH_GEOMETRY_CONTRADICTION")

    def test_duplicate_candidate_same_frame_is_rejected(self):
        a,b,c,d=fixture()
        d.append(dict(d[0]))
        c["counts"]["valid_new_3D_observations"]=len(d)
        with self.assertRaisesRegex(ValueError,"same UUID"):
            audit(a,b,c,d)

    def test_without_accepted_global_pose_is_rejected(self):
        a,b,c,d=fixture()
        d[0]["frame_index"]=123
        with self.assertRaisesRegex(ValueError,"without accepted"):
            audit(a,b,c,d)

    def test_c2_evidence_incomplete_is_rejected(self):
        a,b,c,d=fixture()
        b.pop()
        with self.assertRaisesRegex(ValueError,"Missing C2"):
            audit(a,b,c,d)

    def test_not_three_independent_views(self):
        a,b,c,d=fixture()
        for row in b:
            row["xyz"]=[0,0,0]
        r=audit(a,b,c,d)
        self.assertEqual(r["hypotheses"][0]["status"],
                         "PROVISIONAL_NOT_ENOUGH_INDEPENDENT_EVIDENCE")

    def test_genuine_non_rotation_quaternion_logic(self):
        a={"xyz":[0,0,0],"quaternion":[0,0,0,1]}
        b={"xyz":[0,0,0],"quaternion":[0,0,0,-1]}
        self.assertFalse(is_independent_view(a,b))
        c={"xyz":[.13,0,0],"quaternion":[0,0,0,1]}
        self.assertTrue(is_independent_view(a,c))


if __name__ == "__main__":
    unittest.main()
