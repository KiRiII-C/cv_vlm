import numpy as np
from semantic_map.models import Observation
from semantic_map.storage import MemoryStore
from semantic_map.engine import SemanticMapper

def make_observation(frame):
    return Observation(source="rgbd",frame_id=frame,detection_id="det0",
                       label="chair",center=np.array([1.,2.,.8]),
                       low=np.array([.8,1.8,.5]),
                       high=np.array([1.2,2.2,1.1]),
                       timestamp=float(frame),appearance=np.array([1.,0.,0.]),
                       map_id="test_map")

def test_repeatable_first_observation_id_across_independent_clean_rebuilds(tmp_path):
    ids=[]
    for name in ("first.sqlite","rebuild.sqlite"):
        db=MemoryStore(tmp_path/name)
        o=make_observation("10")
        decisions=SemanticMapper(db).process_frame([o])
        ids.append(decisions[0].object_id)
        assert db.tracks()[0].object_id==ids[-1]
        db.close()
    assert ids[0].startswith("obj_")
    assert ids[0]==ids[1]

def test_independent_map_ids_do_not_collide(tmp_path):
    store=MemoryStore(tmp_path/"maps.sqlite")
    a=make_observation("10")
    b=make_observation("10");b.map_id="other_map"
    m=SemanticMapper(store)
    ida=m.process_frame([a])[0].object_id
    idb=m.process_frame([b])[0].object_id
    assert ida != idb
    store.close()
