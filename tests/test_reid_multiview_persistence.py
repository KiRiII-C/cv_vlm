import numpy as np
from semantic_map.models import Observation
from semantic_map.engine import SemanticMapper
from semantic_map.storage import MemoryStore

def obs(frame,det,center,feature=(1,0,0),label="chair",map_version=0):
    p=np.asarray(center,dtype=float)
    return Observation(source="camera",frame_id=str(frame),detection_id=str(det),
        label=label,center=p,low=p-.20,high=p+.20,timestamp=float(frame),
        appearance=np.asarray(feature,dtype=float),map_version=map_version,
        confidence=.8)

def test_multiview_gallery_survives_database_restart(tmp_path):
    path=tmp_path/"map.sqlite"
    db=MemoryStore(path)
    engine=SemanticMapper(db,association_mode="multiview")
    a=engine.process_frame([obs(0,"A",[0,0,1])])[0].object_id
    b=engine.process_frame([obs(1,"A",[.01,0,1],feature=(0.6,0.8,0))])[0]
    assert b.object_id==a
    gallery=db.tracks()[0].attributes.get("visual_gallery",[])
    assert len(gallery)==2
    db.close()
    second=MemoryStore(path)
    tracks=second.tracks()
    assert tracks[0].object_id==a
    assert len(tracks[0].attributes["visual_gallery"])==2
    second.close()

def test_short_term_hint_cannot_override_different_3d_object(tmp_path):
    db=MemoryStore(tmp_path/"map.sqlite")
    mapper=SemanticMapper(db,association_mode="multiview")
    first=mapper.process_frame([obs(0,"A",[0,0,1])])[0].object_id
    other=mapper.process_frame([obs(1,"B",[3,0,1])],
                                prior_ids={"B":first})[0]
    assert other.object_id!=first
    assert other.status=="tentative"
    db.close()

def test_hint_not_allowed_across_pose_graph_map_versions(tmp_path):
    db=MemoryStore(tmp_path/"map.sqlite")
    mapper=SemanticMapper(db,association_mode="multiview")
    first=mapper.process_frame([obs(0,"A",[0,0,1],map_version=0)])[0].object_id
    later=mapper.process_frame([obs(1,"B",[0,0,1],map_version=1)],
        prior_ids={"B":first})[0]
    assert later.object_id!=first
    db.close()
