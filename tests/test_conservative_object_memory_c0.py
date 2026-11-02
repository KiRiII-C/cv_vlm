import pytest
from semantic_map.conservative_object_memory_c0 import ConservativeObjectMemory

def obs(label,x,conf=.9):
    return {"class":label,"center_map_m":[x,0,1],
        "aabb_min_map_m":[x-.04,-.04,.96],
        "aabb_max_map_m":[x+.04,.04,1.04],
        "confidence":conf}

def test_same_object_reobserved_becomes_candidate_not_verified_id():
    m=ConservativeObjectMemory()
    a=m.update(0,[obs("chair",0)])[0]
    b=m.update(1,[obs("chair",.02)])[0]
    c=m.update(2,[obs("chair",.03)])[0]
    assert a["candidate_id"]==b["candidate_id"]==c["candidate_id"]
    state=m.snapshot()[0]
    assert state["status"]=="MULTIVIEW_CANDIDATE"
    assert state["physical_instance_gt_validated"] is False

def test_unseen_object_does_not_disappear():
    m=ConservativeObjectMemory()
    m.update(0,[obs("cup",0)])
    for i in range(1,50):m.update(i,[])
    assert m.snapshot()[0]["status"]=="PROVISIONAL"
    assert m.snapshot()[0]["unseen_between_frames_means"]=="UNKNOWN_NOT_ABSENT"

def test_simultaneous_detections_cannot_share_candidate():
    m=ConservativeObjectMemory(ambiguity_margin=0.01)
    m.update(0,[obs("chair",0)])
    got=m.update(1,[obs("chair",0),obs("chair",.005)])
    ids=[x["candidate_id"] for x in got if x["candidate_id"]]
    assert len(ids)==len(set(ids))
    assert any(x["status"]=="AMBIGUOUS" for x in got)

def test_unrelated_class_not_merged_despite_same_xyz():
    m=ConservativeObjectMemory()
    a=m.update(0,[obs("chair",0)])[0]
    b=m.update(1,[obs("table",0)])[0]
    assert a["candidate_id"]!=b["candidate_id"]

def test_ambiguous_candidates_abstain():
    m=ConservativeObjectMemory(ambiguity_margin=.1)
    m.update(0,[obs("chair",-.12),obs("chair",.12)])
    ids=list(m.objects)
    result=m.update(1,[obs("chair",0)])
    assert result[0]["status"]=="AMBIGUOUS"
    assert result[0]["candidate_id"] is None
    assert list(m.objects)==ids

def test_low_confidence_no_new_candidate():
    m=ConservativeObjectMemory()
    got=m.update(0,[obs("cup",0,.1)])
    assert got[0]["status"]=="LOW_CONFIDENCE"
    assert len(m.objects)==0

def test_monotonic_frame_order_required():
    m=ConservativeObjectMemory()
    m.update(1,[])
    with pytest.raises(ValueError,match="increasing"):m.update(1,[])

def test_reject_invalid_center():
    m=ConservativeObjectMemory()
    with pytest.raises(ValueError,match="Invalid 3D"):
        m.update(0,[{"class":"chair","center":[float("nan"),0,0],
                     "confidence":.9}])
