import numpy as np
from semantic_map.models import Track, Observation
from semantic_map.storage import MemoryStore
from semantic_map.offline_reconcile import proposal,reconcile

def track(uid,shift=0.0,appearance=None,category="chair",hits=10):
    c=np.array([1.0+shift,2.,1.])
    return Track(uid,category,c,c-.2,c+.2,
                 np.array([1.,0.,0.]) if appearance is None else appearance,
                 hits,0.,20.,"confirmed",map_id="real-scene")

def test_fragment_reentry_merges_and_aliases(tmp_path):
    db=MemoryStore(tmp_path/"map.sqlite")
    a=track("obj_aaaaaaaaaaaaaaaa",hits=20)
    b=track("obj_bbbbbbbbbbbbbbbb",shift=.02,hits=5)
    db.upsert_track(a);db.upsert_track(b);db.db.commit();db.close()
    outcome=reconcile(tmp_path/"map.sqlite",tmp_path/"out.json")
    assert outcome["pre_track_count"]==2
    assert outcome["post_track_count"]==1
    assert len(outcome["reconciliations"])==1
    db=MemoryStore(tmp_path/"map.sqlite")
    assert db.resolve(b.object_id)==a.object_id
    assert db.tracks()[0].hits==25
    db.close()

def test_distinct_simultaneous_lookalikes_never_merge(tmp_path):
    db=MemoryStore(tmp_path/"map.sqlite")
    a=track("obj_aaaaaaaaaaaaaaaa")
    b=track("obj_bbbbbbbbbbbbbbbb",shift=.01)
    db.upsert_track(a);db.upsert_track(b)
    for det,t in enumerate((a,b)):
        obs=Observation(source="rgbd",frame_id="frame_10",detection_id=str(det),
                        label="chair",center=t.center,low=t.low,high=t.high,
                        timestamp=10.,appearance=t.appearance,map_id="real-scene")
        db.write_observation(obs,t.object_id,"confirmed")
    db.db.commit();db.close()
    outcome=reconcile(tmp_path/"map.sqlite",tmp_path/"out.json")
    assert outcome["pre_track_count"]==outcome["post_track_count"]==2

def test_cross_version_rejected(tmp_path):
    db=MemoryStore(tmp_path/"map.sqlite")
    a=track("obj_aaaaaaaaaaaaaaaa")
    b=track("obj_bbbbbbbbbbbbbbbb")
    b.map_version=1
    db.upsert_track(a);db.upsert_track(b);db.db.commit();db.close()
    o=reconcile(tmp_path/"map.sqlite")
    assert o["post_track_count"]==2
