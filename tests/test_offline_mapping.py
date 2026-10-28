import json
import numpy as np
import pytest
from semantic_map.offline_mapping import EvidenceGrid,validate_pose,pose_override

def test_ray_carving_unknown_free_occupied():
    o=EvidenceGrid(.1)
    for _ in range(3):
        o.update_ray([0,0,0],[1,0,.5])
    assert o.logodds[o.index([1,0,.5])] > .85
    assert o.logodds[o.index([.3,0,.15])] < -.65
    assert o.summary()["occupied_voxels"]>=1

def test_reject_pose_not_se3():
    T=np.eye(4)
    validate_pose(T)
    T[0,0]=1.1
    with pytest.raises(ValueError,match="rotation"):validate_pose(T)

def test_corrected_pose_required_no_stale_reference_fallback(tmp_path):
    T=np.eye(4);T[0,3]=2.
    path=tmp_path/"poses.json"
    path.write_text(json.dumps({"source":"rtabmap","poses":[{"timestamp":1.,"T_map_cam":T.tolist()}]}))
    pairs=[(1.,"rgb.png","d.png",np.eye(4))]
    new,source=pose_override(pairs,path)
    assert source=="rtabmap"
    assert new[0][-1][0,3]==2.
    with pytest.raises(ValueError,match="Missing optimized pose"):
        pose_override([(3.,"a","b",np.eye(4))],path)

def test_unknown_floor_cannot_be_nav_ready(tmp_path):
    o=EvidenceGrid(.1)
    for _ in range(3):o.update_ray([0,0,0],[1,0,.5])
    info=o.export_2d(tmp_path)
    assert not info["nav_ready"]
    assert info["occupied_cells"]>0
    assert (tmp_path/"occupancy.yaml").exists()
    assert (tmp_path/"occupancy.pgm").exists()

def test_calibrated_floor_flag(tmp_path):
    o=EvidenceGrid(.1)
    for _ in range(3):o.update_ray([0,0,0],[1,0,.5])
    info=o.export_2d(tmp_path,floor_z=0.)
    assert info["nav_ready"]
    assert info["occupied_cells"]>0
