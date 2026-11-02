import importlib.util
from pathlib import Path
import numpy as np
import pytest

def module():
    path=Path(__file__).resolve().parents[1]/"scripts"/"b6_render_dense_semantic_map.py"
    spec=importlib.util.spec_from_file_location("b6_dense",path)
    obj=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj

def test_voxel_centers_are_measured_point_means_not_interpolated_geometry():
    x=module()
    xyz=np.array([[.001,.001,1.001],[.002,.002,1.002],[.1,.2,1.1]],dtype=float)
    labels=["chair","chair","table"]
    cent,cat,n,p=x.fuse_measured(xyz,labels,[.9,.8,.8],.01)
    assert len(cent)==2
    assert sorted(n.tolist())==[1,2]
    assert set(cat)=={"chair","table"}
    chair=cent[cat=="chair"][0]
    np.testing.assert_allclose(chair,np.mean(xyz[:2],axis=0),atol=2e-7)

def test_mixed_class_voxels_get_explicit_ambiguity_score():
    x=module()
    xyz=np.array([[.001,.001,1.],[.002,.001,1.]])
    centers,cat,n,p=x.fuse_measured(xyz,["chair","table"],[.9,.9],.012)
    assert len(centers)==1
    assert p[0]==pytest.approx(.5)
    assert p[0]<.58

def test_bad_depth_or_mismatched_observations_rejected():
    x=module()
    with pytest.raises(ValueError):
        x.fuse_measured(np.ones((4,3)),["chair"],[.8])
    with pytest.raises(ValueError):
        x.fuse_measured(np.array([[float("nan"),0,1]]),["chair"],[.8])
