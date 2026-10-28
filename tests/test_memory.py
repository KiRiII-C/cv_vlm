import numpy as np
import pytest
from semantic_map import Observation, MemoryStore, SemanticMapper


def obs(frame, det='chair', label='chair', x=1.0,app=None,source='bag1',version=0):
    return Observation(source=source,frame_id=str(frame),detection_id=det,
        label=label,center=[x,0,1.5],low=[x-.2,-.2,1.3],high=[x+.2,.2,1.7],
        timestamp=float(frame),appearance=app,map_version=version)


def test_reidentify_after_disappearance_and_restart(tmp_path):
    path=tmp_path/'memory.sqlite'
    db=MemoryStore(path);mapper=SemanticMapper(db)
    first=mapper.process_frame([obs(1,app=[1,0,0])])[0]
    assert first.status=='tentative'
    mapper.process_frame([obs(2,x=1.03,app=[.98,.02,0])])
    assert len(db.search())==1
    original=db.search()[0].object_id
    assert original==first.object_id
    db.close()  # simulate process/container restart
    db=MemoryStore(path);mapper=SemanticMapper(db)
    # Observe other physical objects and leave the room in between
    mapper.process_frame([obs(100,det='table',label='table',x=3,app=[0,1,0])])
    returned=mapper.process_frame([obs(200,x=.98,app=[.99,.01,0])])[0]
    assert returned.object_id==original
    assert returned.status=='confirmed'
    assert db.search(label='chair')[0].hits==3
    assert len(db.search(label='chair'))==1
    db.close()


def test_nearby_similar_class_different_appearance_remain_distinct(tmp_path):
    db=MemoryStore(tmp_path/'test.sqlite');m=SemanticMapper(db)
    two=m.process_frame([obs(1,det='left',x=1,app=[1,0,0]),
                         obs(1,det='right',x=1.25,app=[0,1,0])])
    assert two[0].object_id!=two[1].object_id
    again=m.process_frame([obs(2,det='right',x=1.26,app=[0,1,0]),
                           obs(2,det='left',x=1.01,app=[1,0,0])])
    assert again[0].object_id==two[1].object_id
    assert again[1].object_id==two[0].object_id
    assert len(db.search(only_confirmed=False))==2
    db.close()


def test_idempotent_frame_replay(tmp_path):
    db=MemoryStore(tmp_path/'test.sqlite');m=SemanticMapper(db)
    first=m.process_frame([obs(1)])[0]
    repeated=m.process_frame([obs(1)])[0]
    assert repeated.status=='already_processed'
    assert repeated.object_id==first.object_id
    assert db.tracks()[0].hits==1
    db.close()


def test_reject_different_pose_version_and_class(tmp_path):
    db=MemoryStore(tmp_path/'test.sqlite');m=SemanticMapper(db)
    a=m.process_frame([obs(1)])[0]
    b=m.process_frame([obs(2,label='table')])[0]
    c=m.process_frame([obs(3,version=1)])[0]
    assert a.object_id!=b.object_id!=c.object_id
    db.close()


def test_block_merge_when_covisible(tmp_path):
    db=MemoryStore(tmp_path/'test.sqlite');m=SemanticMapper(db)
    a,b=m.process_frame([obs(1,det='a',app=[1,0]),obs(1,det='b',x=1.35,app=[0,1])])
    with pytest.raises(ValueError):db.merge(a.object_id,b.object_id,'manual',3)
    db.close()


def test_semantic_memory_annotation_survives_restart(tmp_path):
    path=tmp_path/'memory.sqlite'
    db=MemoryStore(path);m=SemanticMapper(db)
    a=m.process_frame([obs(1,app=[1,0,0])])[0]
    m.process_frame([obs(2,app=[1,0,0])])
    db.annotate(a.object_id,'red chair beside desk',{'color':'red','source':'VLM','confidence':0.83})
    db.close()
    db=MemoryStore(path)
    items=db.search('red chair')
    assert len(items)==1 and items[0].attributes['color']=='red'
    db.close()


def test_conflict_same_object_twice_in_one_frame(tmp_path):
    db=MemoryStore(tmp_path/'test.sqlite');m=SemanticMapper(db)
    first=m.process_frame([obs(1,app=[1,0])])[0]
    result=m.process_frame([obs(2,det='a',app=[1,0]),obs(2,det='b',app=[1,0])])
    ids=[v.object_id for v in result if v.object_id==first.object_id]
    assert len(ids)==1
    assert len(db.search())==1
    assert set(v.status for v in result)=={'confirmed','unassigned_conflict'}
    db.close()


def test_ambiguous_hypothesis_is_not_duplicate(tmp_path):
    db=MemoryStore(tmp_path/'test.sqlite');m=SemanticMapper(db,ambiguity_margin=.2)
    a,b=m.process_frame([obs(1,det='a',x=.85,app=[1,0]),
                         obs(1,det='b',x=1.15,app=[1,0])])
    decision=m.process_frame([obs(2,x=1.0,app=[1,0])])[0]
    assert decision.status=='ambiguous' and decision.object_id is None
    assert len(db.search(only_confirmed=False))==2
    db.close()
