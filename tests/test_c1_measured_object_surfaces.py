import sqlite3
import numpy as np
import pytest
from semantic_map.object_surface_memory_v2 import ObjectSurfaceMemoryV2

def test_measured_xyz_is_idempotent_and_colour_not_fabricated(tmp_path):
    db=sqlite3.connect(tmp_path/"test.sqlite")
    memory=ObjectSurfaceMemoryV2(db,voxel_m=.02,minimum_support=1)
    points=np.array([[.001,.002,1.],[.004,.002,1.],[.04,.0,1.]],float)
    assert memory.integrate_world_points("hypothesis_uuid",points,source="real_rgbd",
               frame_id="11",map_id="scene",timestamp=11,pose_version=0)==3
    assert memory.integrate_world_points("hypothesis_uuid",points,source="real_rgbd",
               frame_id="11",map_id="scene",timestamp=11,pose_version=0)==0
    exported=memory.export(tmp_path/"surface.ply",map_id="scene")
    assert exported["unique_ids"]==1
    assert exported["vertices"]==2
    assert db.execute("select sum(support) from object_surface_v2").fetchone()[0]==3
    # Not an RGB input; stored measured RGB channel sums must stay zero.
    assert db.execute("select sum(sum_r+sum_g+sum_b) from object_surface_v2").fetchone()[0]==0
    db.close()

def test_pose_version_conflicts_cannot_silently_merge(tmp_path):
    db=sqlite3.connect(tmp_path/"test2.sqlite")
    mem=ObjectSurfaceMemoryV2(db,minimum_support=1)
    points=np.asarray([[0,0,1],[0,.02,1]],float)
    assert mem.integrate_world_points("a",points,source="s",frame_id="1",
             map_id="room",pose_version=0)==2
    with pytest.raises(RuntimeError,match="another pose-graph version"):
        mem.integrate_world_points("a",points,source="s",frame_id="2",
             map_id="room",pose_version=1)
    mem.reset_for_pose_rebuild(map_id="room")
    assert mem.integrate_world_points("a",points,source="s",frame_id="1",
             map_id="room",pose_version=1)==2
    db.close()

def test_invalid_or_unmeasured_geometry_refused(tmp_path):
    db=sqlite3.connect(tmp_path/"test3.sqlite")
    mem=ObjectSurfaceMemoryV2(db)
    for invalid in (np.ones((3,2)),np.array([[float("nan"),1.,1.]])):
        with pytest.raises(ValueError,match="finite Nx3"):
            mem.integrate_world_points("a",invalid,source="s",frame_id="1")
    with pytest.raises(ValueError,match="RGB"):
        mem.integrate_world_points("a",np.array([[0,0,1.]]),
               measured_rgb=np.array([[500.,0.,0.]]),source="s",frame_id="1")
    db.close()
