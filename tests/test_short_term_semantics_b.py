import numpy as np
from semantic_map.temporal_tracker import Detection,TemporalTracker
from semantic_map.short_term_semantics_b import ClassTolerantTemporalAssociator,class_parent


def det(name,label,xyz,conf=.9):
    return Detection(name,label,np.array([1.,1.,15.,15.]),np.array(xyz),conf,
                     np.array([1.,0.,0.]))


def test_alias_class_change_keeps_short_track_id():
    t=ClassTolerantTemporalAssociator()
    x=t.update(0,[det("a","chair",[0,0,2])])[0]
    y=t.update(1,[det("b","office chair",[0.01,0,2])])[0]
    assert x.short_id==y.short_id
    assert y.status=="confirmed"
    assert y.parent_label=="chair"
    assert y.class_probabilities["chair"]>0 and y.class_probabilities["office chair"]>0
    assert abs(sum(y.class_probabilities.values())-1)<1e-10


def test_unrelated_categories_do_not_collapse():
    t=ClassTolerantTemporalAssociator()
    a=t.update(0,[det("a","chair",[0,0,2])])[0]
    b=t.update(1,[det("b","table",[0,0,2])])[0]
    assert a.short_id!=b.short_id


def test_two_simultaneous_chairs_remain_two_tracklets():
    t=ClassTolerantTemporalAssociator()
    first=t.update(0,[det("a","chair",[0,0,2]),det("b","office chair",[1,0,2])])
    second=t.update(1,[det("a2","office chair",[0.01,0,2]),det("b2","chair",[1.01,0,2])])
    assert first[0].short_id==second[0].short_id
    assert first[1].short_id==second[1].short_id
    assert first[0].short_id!=first[1].short_id


def test_low_confidence_cannot_create_track():
    t=ClassTolerantTemporalAssociator(TemporalTracker(high_thresh=.4,low_thresh=.1))
    assert t.update(0,[det("a","desk chair",[0,0,2],.2)])[0].short_id is None


def test_no_global_uuid_claim_and_expiry():
    t=ClassTolerantTemporalAssociator(TemporalTracker(max_lost=1))
    old=t.update(0,[det("a","chair",[0,0,2])])[0]
    t.update(1,[])
    now=t.update(2,[det("b","chair",[0,0,2])])[0]
    assert old.short_id!=now.short_id
    assert old.short_id not in t._votes


def test_family_vocab_is_explicit_and_finite():
    assert class_parent(" OFFICE   CHAIR ")=="chair"
    assert class_parent("desk")=="table"
    assert class_parent("table")!="chair"
