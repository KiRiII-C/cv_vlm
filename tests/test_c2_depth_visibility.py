import numpy as np
import pytest
from semantic_map.c2_depth_visibility import evidence,independently_separated,summarize_per_hypothesis

def scene(d):
    K=(100.,100.,16.,16.)
    pts=np.array([[(u-16)*2/100,(v-16)*2/100,2.] for u in range(10,20) for v in range(10,20)])
    depth=np.full((32,32),d,dtype=np.float32)
    return evidence(pts,[0,0,0],[0,0,0,1],depth,K=K,pose_xyz_cov=[.0001]*3)

def test_same_surface_is_compatible_but_not_id_proof():
    out=scene(2)
    assert out["state"]=="GEOMETRICALLY_COMPATIBLE_NOT_ID_PROOF"
    assert out["compatible_pixels"]>=80

def test_behind_current_foreground_is_occluded_not_absent():
    out=scene(1)
    assert out["state"]=="OCCLUDED_NOT_MISSING"
    assert out["free_space_conflict_pixels"]==0

def test_object_surface_in_front_of_current_depth_is_candidate_only():
    out=scene(3)
    assert out["state"]=="FREE_SPACE_CONFLICT_CANDIDATE"
    assert out["occluded_pixels"]==0

def test_depth_missing_does_not_prove_disappearance():
    out=scene(0)
    assert out["state"]=="UNKNOWN_NOT_ENOUGH_DEPTH"
    assert out["free_space_conflict_pixels"]==0

def test_covariance_inflates_depth_gate_conservatively():
    out=evidence(
       np.array([[0,0,2]]*20,dtype=float),
       [0,0,0],[0,0,0,1],
       np.full((32,32),2.17),
       K=(100,100,16,16),pose_xyz_cov=[.01]*3)
    assert out["min_uncertainty_tolerance_m"]>=.3-1e-9
    assert out["state"]=="UNKNOWN_NOT_ENOUGH_DEPTH"

def test_three_similar_viewpoints_do_not_validate_absence():
    statuses=[{"state":"FREE_SPACE_CONFLICT_CANDIDATE",
               "xyz":[.01*k,0,0],"quaternion":[0,0,0,1]} for k in range(3)]
    assert independently_separated(statuses)==1
    decision=summarize_per_hypothesis(statuses)
    assert decision["decision"]=="UNKNOWN_NO_VERIFIED_DISAPPEARANCE"
    assert decision["object_deleted_asserted"] is False

def test_three_separate_views_only_raise_review_candidate():
    statuses=[{"state":"FREE_SPACE_CONFLICT_CANDIDATE",
               "xyz":[.2*k,0,0],"quaternion":[0,0,0,1]} for k in range(3)]
    dec=summarize_per_hypothesis(statuses)
    assert dec["decision"]=="VISIBLY_ABSENT_CANDIDATE_REQUIRES_ACTIVE_RECHECK"
    assert dec["object_deleted_asserted"] is False
    assert dec["object_moved_asserted"] is False

def test_invalid_pose_covariance_rejected():
    with pytest.raises(ValueError,match="Covariance"):
        evidence(np.array([[0,0,2.]]),[0,0,0],[0,0,0,1],
            np.ones((32,32)),K=(100,100,16,16),pose_xyz_cov=[-1,0,0])
