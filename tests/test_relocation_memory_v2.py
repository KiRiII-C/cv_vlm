import numpy as np
import pytest
from semantic_map.storage import MemoryStore
from semantic_map.models import Observation
from semantic_map.object_memory_v2 import ObjectMemoryV2
from semantic_map.relocation_memory_v2 import RelocationMemoryV2


def observation(frame,x,feature,who='A'):
    c=np.asarray([x,0.,1.],float)
    return Observation(source='cam',frame_id=str(frame),detection_id=who,
          label='chair',center=c,low=c-.14,high=c+.14,
          timestamp=float(frame),appearance=feature,confidence=.96)


def make(tmp_path):
    db=MemoryStore(tmp_path/'db.sqlite')
    mem=ObjectMemoryV2(db)
    r=RelocationMemoryV2(mem,min_visual=.94)
    id=mem.process_frame([observation(0,0,[1.,0,0])])[0].object_id
    mem.process_frame([observation(1,.01,[.8,.6,0])])
    assert len(mem.store.tracks()[0].attributes['memory_v2']['prototypes'])==2
    return db,mem,r,id


def proof(frame):return {'origin_map':[float(frame),0.,1.],'raycast_depth_m':2.5,
       'pose_version':0,'old_region_cleared':True}


def test_moving_object_hypothesis_requires_3_views_and_2_vacancy_rays(tmp_path):
    db,mem,r,uid=make(tmp_path)
    new=[observation(frame,1.3,[.97,.243,0]) for frame in (10,11,12)]
    assert r.observe_candidate(new[0]).status=='RELOCATION_PENDING_NEEDS_EVIDENCE'
    assert r.observe_candidate(new[1]).status=='RELOCATION_PENDING_NEEDS_EVIDENCE'
    assert r.observe_candidate(new[2]).status=='RELOCATION_PENDING_NEEDS_EVIDENCE'
    assert np.linalg.norm(mem.store.tracks()[0].center)<1.3
    r.note_verified_old_region_empty(uid,source='depth',frame_id='20',proof=proof(20))
    r.note_verified_old_region_empty(uid,source='depth',frame_id='21',proof=proof(21))
    result=r.observe_candidate(new[2])
    assert result.object_id==uid
    assert result.status=='RELOCATION_CONFIRMED_NEEDS_SURFACE_REBUILD'
    t=mem.store.tracks()[0]
    assert np.isclose(t.center[0],1.3)
    assert t.hits==5
    assert len(t.attributes['memory_v2']['relocation_history'])==1
    assert db.db.execute("SELECT count(*) FROM object_evidence_v2 WHERE object_id=?",(uid,)).fetchone()[0]==5
    assert mem.process_frame([new[2]])[0].status=='already_processed'
    db.close()


def test_duplicate_or_unverified_vacancy_cannot_relocate(tmp_path):
    db,mem,r,uid=make(tmp_path)
    with pytest.raises(ValueError):r.note_verified_old_region_empty(uid,source='cam',frame_id='1',proof={'old_region_cleared':True})
    with pytest.raises(ValueError):r.note_verified_old_region_empty(uid,source='cam',frame_id='1',proof=dict(proof(1),pose_version=999))
    for f in (10,11,12):
        decision=r.observe_candidate(observation(f,1.3,[.97,.243,0]))
        assert decision.object_id is None
    assert mem.store.tracks()[0].center[0]<.1
    db.close()


def test_far_identical_chairs_no_pure_appearance_relocation(tmp_path):
    db,mem,r,uid=make(tmp_path)
    other=mem.process_frame([observation(3,2.5,[1.,0.,0.],who='B')])[0].object_id
    mem.process_frame([observation(4,2.51,[.8,.6,0.],who='B')])
    assert other!=uid
    candidate=r.observe_candidate(observation(12,1.3,[.97,.243,0]))
    assert candidate.object_id is None
    assert candidate.status in ('AMBIGUOUS_SIMILAR_OLD_OBJECTS','NO_DISTINCTIVE_RELOCATION')
    db.close()

def test_same_camera_view_cannot_fake_two_independent_vacancies(tmp_path):
    db,mem,r,uid=make(tmp_path)
    for frame in (10,11,12):
        r.observe_candidate(observation(frame,1.3,[.97,.243,0]))
    r.note_verified_old_region_empty(uid,source='depth',frame_id='20',proof=proof(20))
    r.note_verified_old_region_empty(uid,source='depth',frame_id='21',proof=dict(proof(20)))
    assert r._old_free_views(mem.store.tracks()[0])==1
    assert r.observe_candidate(observation(12,1.3,[.97,.243,0])).object_id is None
    db.close()
