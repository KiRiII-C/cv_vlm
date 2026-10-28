import numpy as np
from semantic_map.temporal_tracker import Detection,TemporalTracker
from semantic_map.yoloe26_detector import class_list,canonical,MODEL_SET
from semantic_map.reid_features import hsv_spatial

def det(name,xyz,score=.90,label='chair',box=(10,10,40,40),descriptor=None):
    m=np.zeros((60,80),bool);m[10:30,10:30]=True
    return Detection(name,label,np.asarray(box),np.asarray(xyz),score,
                     np.asarray(descriptor if descriptor is not None else [1.,0.,0.]),m)

def test_reentry_same_id_after_4_missing_frames():
    tracker=TemporalTracker(max_lost=12)
    a=tracker.update(0,[det("x",[0,0,1])])[0]
    tracker.update(1,[det("x",[.02,0,1])])
    for t in range(2,6):assert tracker.update(t,[])==[]
    b=tracker.update(6,[det("x",[.01,0,1])])[0]
    assert b.short_id==a.short_id
    assert b.status=="confirmed"

def test_low_confidence_rescue_does_not_create_id():
    tr=TemporalTracker(high_thresh=.3,low_thresh=.10)
    first=tr.update(0,[det("x",[0,0,1])])[0].short_id
    recovered=tr.update(1,[det("a",[0,0,1],score=.15)])[0]
    assert recovered.short_id==first
    assert recovered.status=="confirmed"
    new_low=tr.update(2,[det("b",[2,0,1],score=.15)])[0]
    assert new_low.short_id is None and new_low.status=="low_unassigned"

def test_symmetric_identical_objects_do_not_match_by_appearance_alone():
    t=TemporalTracker()
    ids=t.update(0,[det("a",[0,0,1]),det("b",[1.5,0,1])])
    assert ids[0].short_id!=ids[1].short_id
    flipped=t.update(1,[det("a",[1.49,0,1]),det("b",[.01,0,1])])
    assert flipped[0].short_id==ids[1].short_id
    assert flipped[1].short_id==ids[0].short_id

def test_short_tracker_ttl_not_a_global_uuid():
    t=TemporalTracker(max_lost=2)
    a=t.update(0,[det("a",[0,0,1])])[0]
    t.update(1,[]);t.update(2,[])
    b=t.update(3,[det("b",[0,0,1])])[0]
    assert b.short_id!=a.short_id

def test_class_gate_must_not_match_different_classes():
    t=TemporalTracker()
    a=t.update(0,[det("a",[0,0,1])])[0]
    b=t.update(1,[det("b",[0,0,1],label="table")])[0]
    assert b.short_id!=a.short_id

def test_enforce_unique_ids_and_time_monotonicity():
    t=TemporalTracker()
    import pytest
    with pytest.raises(ValueError,match="Duplicate"):
        t.update(0,[det("a",[0,0,1]),det("a",[1,0,1])])
    assert t.update(0,[])==[]
    with pytest.raises(ValueError,match="strictly"):
        t.update(0,[])

def test_mask_hsv_spatial_feature_reproducible():
    import cv2
    im=np.zeros((60,80,3),dtype=np.uint8)
    im[10:30,10:30]=[30,80,250]
    mask=np.zeros((60,80),dtype=bool);mask[10:30,10:30]=True
    a=np.asarray(hsv_spatial(im,mask))
    b=np.asarray(hsv_spatial(im,mask))
    assert np.allclose(a,b)
    assert np.isclose(np.linalg.norm(a),1,atol=1e-5)
    assert len(a)==256

def test_12_class_config_and_validation(tmp_path):
    p=tmp_path/"names.json"
    import json,pytest
    p.write_text(json.dumps({"classes":["chair","table","sofa","cabinet","bookshelf","bed","laptop","computer monitor","keyboard","mouse","cup","potted plant"]}))
    assert len(class_list(p))==12
    assert "yoloe-26s-seg.pt" in MODEL_SET
    assert canonical("computer monitor")=="tv"
    p.write_text(json.dumps({"classes":["chair"]}))
    with pytest.raises(ValueError):class_list(p)
    p.write_text(json.dumps({"classes":["chair"]*8}))
    with pytest.raises(ValueError):class_list(p)
