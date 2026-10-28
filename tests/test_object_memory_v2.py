from __future__ import annotations
import json
import numpy as np
import pytest
from semantic_map.models import Observation
from semantic_map.storage import MemoryStore
from semantic_map.object_memory_v2 import ObjectMemoryV2


def make(frame, center, name='chair', who='a', feature=(1.,0.,0.),
         conf=.94, pose=None, version=0, source='sensor'):
    c=np.array(center,float)
    return Observation(source=source,frame_id=str(frame),detection_id=str(who),
        label=name,center=c,low=c-.19,high=c+.19,
        timestamp=float(frame),appearance=np.array(feature,float),
        T_map_cam=None if pose is None else np.asarray(pose,float),
        confidence=conf,map_version=version)


def memory(tmp_path, **kw):
    store=MemoryStore(tmp_path/'memory.sqlite')
    return ObjectMemoryV2(store,**kw)


def test_id_survives_long_absence_and_db_reopen(tmp_path):
    mem=memory(tmp_path)
    a=mem.process_frame([make(0,[0,0,1])])[0]
    assert a.object_id is not None
    assert mem.process_frame([make(1,[.02,0,1])])[0].object_id==a.object_id
    mem.store.close()
    mem=memory(tmp_path)
    b=mem.process_frame([make(200,[.015,.005,1])])[0]
    assert b.object_id==a.object_id
    assert mem.query_memory()[0]['hits']==3
    assert mem.query_memory()[0]['last_seen']==200
    assert mem.process_frame([make(200,[.015,.005,1])])[0].status=='already_processed'
    assert mem.query_memory()[0]['hits']==3
    mem.store.close()


def test_two_identical_appearance_instances_remain_separate(tmp_path):
    mem=memory(tmp_path)
    a=make(0,[0,0,1],who='A');b=make(0,[1.4,0,1],who='B')
    uids=[d.object_id for d in mem.process_frame([a,b])]
    assert len(set(uids))==2
    revisit=mem.process_frame([make(20,[1.41,0,1],who='B'),make(20,[.01,0,1],who='A')])
    assert [x.object_id for x in revisit]==[uids[1],uids[0]]
    assert mem.propose_duplicate_merges()==[]
    mem.store.close()


def test_duplicate_mask_in_same_frame_does_not_create_uuid(tmp_path):
    mem=memory(tmp_path)
    a=make(0,[0,0,1],who='A');b=make(0,[.001,0,1],who='B')
    result=mem.process_frame([a,b])
    assert result[0].object_id==result[1].object_id
    assert any(x.status=='duplicate_detection' for x in result)
    assert len(mem.store.tracks())==1
    mem.store.close()


def test_ambiguous_not_randomly_assigned_or_new_id(tmp_path):
    mem=memory(tmp_path)
    a=mem.process_frame([make(0,[0,0,1],who='a'),make(0,[.55,0,1],who='b')])
    assert len({x.object_id for x in a})==2
    c=mem.process_frame([make(1,[.275,0,1],who='new')])[0]
    assert c.object_id is None
    assert c.status in ('ambiguous','deferred')
    assert len(mem.review_queue())==1
    assert len(mem.store.tracks())==2
    mem.store.close()


def test_outlier_low_confidence_does_not_contaminate_prototypes(tmp_path):
    mem=memory(tmp_path)
    uid=mem.process_frame([make(0,[0,0,1])])[0].object_id
    mem.process_frame([make(1,[.01,0,1])])
    before=mem.store.tracks()[0].attributes['memory_v2']['prototypes']
    assert len(before)==1
    mem.process_frame([make(2,[.02,0,1],feature=(.4,.916515,0),conf=.12)])
    after=mem.store.tracks()[0].attributes['memory_v2']['prototypes']
    assert after==before
    assert mem.store.tracks()[0].object_id==uid
    mem.store.close()


def test_pose_correction_preserves_uuid_and_atomicity(tmp_path):
    mem=memory(tmp_path)
    T=np.eye(4)
    first=mem.process_frame([make(0,[0,0,1],pose=T)])[0]
    mem.process_frame([make(1,[.02,0,1],pose=T)])
    new_T=np.eye(4);new_T[0,3]=2.
    with pytest.raises(ValueError,match='Missing optimized pose'):
        mem.correct_pose_graph({('sensor','0'):new_T},new_version=1)
    assert mem.store.tracks()[0].map_version==0
    assert mem.store.tracks()[0].center[0]<1.
    report=mem.correct_pose_graph({('sensor','0'):new_T,('sensor','1'):new_T},new_version=1)
    assert report['status']=='REPROJECTED_NEEDS_REASSOCIATION'
    assert report['objects_preserved']==1
    t=mem.store.tracks()[0]
    assert t.object_id==first.object_id and t.map_version==1
    assert np.isclose(t.center[0],2.01)
    assert mem.store.db.execute('select count(*) from memory_revision_v2').fetchone()[0]==1
    mem.store.close()


def test_pose_correction_missing_camera_evidence_refused(tmp_path):
    mem=memory(tmp_path)
    mem.process_frame([make(0,[0,0,1])])
    with pytest.raises(ValueError,match='Missing metric camera-frame evidence'):
        mem.correct_pose_graph({('sensor','0'):np.eye(4)},new_version=1)
    assert mem.store.tracks()[0].map_version==0
    mem.store.close()


def test_class_gate_and_map_version(tmp_path):
    mem=memory(tmp_path)
    old=mem.process_frame([make(0,[0,0,1])])[0].object_id
    new_label=mem.process_frame([make(1,[0,0,1],name='table')])[0].object_id
    new_map=mem.process_frame([make(2,[0,0,1],version=1)])[0].object_id
    assert len({old,new_label,new_map})==3
    mem.store.close()


def test_scene_graph_and_query(tmp_path):
    mem=memory(tmp_path)
    mem.process_frame([make(0,[0,0,1],who='a'),make(0,[1.3,0,1],who='b')])
    mem.process_frame([make(1,[.01,0,1],who='a'),make(1,[1.31,0,1],who='b')])
    graph=mem.nearby_graph(radius=2.)
    assert len(graph['objects'])==2
    assert len(graph['edges'])==1
    mem.store.close()


def test_elongated_false_mask_does_not_move_chair_to_neighbor(tmp_path):
    mem=memory(tmp_path)
    def obs(frame,c,ident):
        o=make(frame,c,who=ident,feature=(1,0,0),conf=.90)
        o.low=np.asarray(c)-np.array([2.,.1,.20])
        o.high=np.asarray(c)+np.array([.15,3.5,.20])
        return o
    uid=mem.process_frame([obs(0,[0,0,1],'first')])[0].object_id
    other=mem.process_frame([obs(1,[0,1.1,1],'second')])[0]
    assert other.object_id!=uid
    assert other.status in ('tentative','deferred','ambiguous')
    mem.store.close()


def test_cup_near_neighbor_not_same_instance(tmp_path):
    mem=memory(tmp_path)
    uid=mem.process_frame([make(0,[0,0,1],name='cup',who='a')])[0].object_id
    candidate=mem.process_frame([make(1,[0,.24,1],name='cup',who='b')])[0]
    assert candidate.object_id!=uid
    mem.store.close()
