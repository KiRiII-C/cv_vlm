from semantic_map.identity_failure_attribution import classify_switch

def test_continuous_observation_uuid_error_is_not_out_of_view():
    assert classify_switch({"frame":5},{"frame":6},set(range(5,7)),{5,6})=="direct_consecutive_frames"

def test_true_out_of_view_split():
    assert classify_switch({"frame":3},{"frame":9},{3,4,9},{3,9})=="true_out_of_view"

def test_detector_missing_same_gt_still_visible():
    assert classify_switch({"frame":3},{"frame":9},set(range(3,10)),{3,9})=="still_visible_detector_gap"

def test_gt_matched_but_uuid_unassigned_during_gap():
    assert classify_switch({"frame":3},{"frame":9},set(range(3,10)),
                           {3,4,5,6,7,8,9})=="still_visible_but_unassigned_uuid"

def test_multiple_gt_matched_masks_same_frame():
    assert classify_switch({"frame":5},{"frame":5},{5},{5})=="multiple_masks_same_frame"

def test_real_out_of_view_overrides_detector_gap():
    assert classify_switch({"frame":3},{"frame":9},{3,4,7,8,9},{3,4,9})=="true_out_of_view"
