import numpy as np
import sqlite3
from semantic_map.offline_instance_graph import OfflineInstanceGraph
from semantic_map.models import Observation

def obs(frame,detection,xyz,label="chair",feature=None):
    c=np.asarray(xyz,float)
    return Observation(source="test",frame_id=str(frame),detection_id=str(detection),
        center=c,low=c-.12,high=c+.12,label=label,timestamp=float(frame),
        appearance=np.asarray(feature,float) if feature is not None else None,
        confidence=.9,map_id="sceneA")

def test_class_flicker_same_physical_object_same_global_uuid():
    scene=[(0,[obs(0,0,(0,0,0),"sofa")]),
           (1,[obs(1,0,(.05,0,0),"chair")]),
           (2,[obs(2,0,(.08,0,0),"sofa")])]
    ids=OfflineInstanceGraph().associate(scene)
    assert ids[0].object_uuid==ids[1].object_uuid==ids[2].object_uuid

def test_two_covisible_nearby_chairs_never_cluster_together():
    rows=[(0,[obs(0,0,(0,0,0)),obs(0,1,(.15,0,0))]),
          (1,[obs(1,0,(0.02,0,0)),obs(1,1,(.17,0,0))])]
    ids=OfflineInstanceGraph().associate(rows)
    assert ids[0].object_uuid!=ids[1].object_uuid
    assert len(set(a.object_uuid for a in ids))>=2

def test_deterministic_uuid_and_no_gt_api():
    data=[(0,[obs(0,0,(0,0,0))]),(1,[obs(1,0,(0.01,0,0))])]
    m=OfflineInstanceGraph()
    a=m.associate(data);b=m.associate(data)
    assert [x.object_uuid for x in a]==[x.object_uuid for x in b]
    assert all(x.object_uuid.startswith("obj_") for x in a)

def test_sqlite_export_and_evidence(tmp_path):
    rows=[(0,[obs(0,0,(0,0,0))]),(1,[obs(1,0,(0.04,0,0))])]
    p=tmp_path/"memory.sqlite"
    summary=OfflineInstanceGraph().export_sqlite(rows,p)
    assert summary["objects"]==1
    with sqlite3.connect(p) as c:
        assert c.execute("select count(*) from objects").fetchone()[0]==1
        assert c.execute("select count(*) from observations").fetchone()[0]==2
        assert c.execute("select count(distinct object_id) from observations").fetchone()[0]==1

def test_duplicate_detection_identity_rejected():
    data=[(0,[obs(0,0,(0,0,0)),obs(0,0,(0,0,0))])]
    import pytest
    with pytest.raises(ValueError,match="Duplicate"):
        OfflineInstanceGraph().associate(data)
