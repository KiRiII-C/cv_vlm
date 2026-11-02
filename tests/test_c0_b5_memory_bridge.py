import importlib.util
import json
from pathlib import Path
import numpy as np

from semantic_map.storage import MemoryStore

def load_script():
    p=Path(__file__).resolve().parents[1]/"scripts"/"c0_replay_real_b5_memory.py"
    spec=importlib.util.spec_from_file_location("c0_replay",p)
    obj=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj

def row(frame,det="a",label="chair",x=0.,quality=.91):
    return {
        "frame_index":frame,"timestamp_tum":float(frame),
        "detection_id":det,"class":label,"confidence":quality,
        "center_map_m":[x,0.,2.],
        "aabb_min_map_m":[x-.15,-.15,1.85],
        "aabb_max_map_m":[x+.15,.15,2.15],
        "valid_depth_ratio":.98,"depth_support_pixels":250,
        "short_track_id":"unverified_shorttrack",
    }

def test_real_b5_bridge_evidence_is_idempotent_without_disappearance(tmp_path):
    mod=load_script()
    rows=[row(0,x=0),row(1,x=.01),row(2,x=.005),
          row(4,det="far",x=2.2),row(5,det="rejected",x=3.,quality=.11)]
    transforms={i:np.eye(4) for i in [0,1,2,4,5]}
    p=tmp_path/"memory.sqlite"
    a=mod.replay(rows,transforms,p)
    assert a["original_observations"]==5
    assert a["counts"]["DEFERRED_LOW_QUALITY"]==1
    assert a["observations_persisted"]==4
    assert a["absence_of_detection_does_not_mark_deleted"] is True
    tracks=MemoryStore(p)
    original={t.object_id for t in tracks.tracks()}
    assert len(original)==2
    assert len([t for t in tracks.tracks() if t.state=="confirmed"])==1
    tracks.close()
    b=mod.replay(rows,transforms,p)
    assert b["counts"]["already_processed"]==4
    tracks=MemoryStore(p)
    assert {t.object_id for t in tracks.tracks()}==original
    assert tracks.db.execute("select count(*) from observations").fetchone()[0]==4
    tracks.close()

def test_B5_observations_preserve_measured_pose_and_no_false_embedding():
    mod=load_script()
    row0=row(10,x=1.0)
    T=np.eye(4);T[0,3]=.2
    obs=mod.convert(row0,T)
    assert obs.appearance is None
    assert obs.map_id==mod.MAP_ID
    np.testing.assert_allclose(obs.center,[1.,0.,2.])
    np.testing.assert_allclose(obs.T_map_cam,T)

def test_source_rejects_GT_pose_and_duplicates(tmp_path):
    mod=load_script()
    inp=tmp_path/"rows.jsonl"
    inp.write_text(json.dumps(row(0))+"\n"+json.dumps(row(0))+"\n")
    poses=tmp_path/"poses.json"
    poses.write_text(json.dumps({"GT_used_as_input":False,"poses":[
        {"index":0,"T_map_cam":np.eye(4).tolist()}]}))
    try:
        mod.read_inputs(inp,poses)
    except RuntimeError as e:
        assert "Duplicate" in str(e)
    else:
        raise AssertionError("Expected duplicate evidence rejection")
    poses.write_text(json.dumps({"GT_used_as_input":True,"poses":[]}))
    try:
        mod.read_inputs(inp,poses)
    except RuntimeError as e:
        assert "GT-derived" in str(e)
    else:
        raise AssertionError("GT must never be supplied to mapping")
