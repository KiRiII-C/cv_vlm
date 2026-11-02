import numpy as np
import pytest
from semantic_map.geometry import Intrinsics
from semantic_map.shared_rgbd_geometry import SharedRGBDFrame
from semantic_map.projective_association_b3 import project_visible_footprint


def example():
    K=Intrinsics(100,100,16,12,32,24,.001)
    depth=np.full((24,32),2000,np.uint16)
    mask=np.zeros((24,32),bool)
    mask[3:21,4:27]=True
    source=SharedRGBDFrame(depth,K,np.eye(4))
    return K,depth,mask,source


def test_full_dense_identity_footprint_is_perfect():
    K,d,mask,f=example()
    o=f.object_observation(mask,detection_id="a",label="chair",
        confidence=1.,timestamp=0.,frame_index=0,max_points=10000)
    r=project_visible_footprint(o,f,target_mask=mask,radius_px=0)
    assert r.status=="OBSERVED"
    assert r.visible_mask_iou==pytest.approx(1)
    assert r.supported_pixels==mask.sum()
    assert r.free_space_conflict_pixels==0


def test_sparse_splat_increases_coverage_not_sensor_evidence():
    K,d,mask,f=example()
    o=f.object_observation(mask,detection_id="a",label="chair",
        confidence=1.,timestamp=0.,frame_index=0,max_points=120)
    a=project_visible_footprint(o,f,target_mask=mask,radius_px=0)
    b=project_visible_footprint(o,f,target_mask=mask,radius_px=1)
    assert b.visible_mask_iou>a.visible_mask_iou
    assert b.supported_pixels>a.supported_pixels


def test_foreground_occluder_must_not_be_free_space_conflict():
    K,d,mask,f=example()
    o=f.object_observation(mask,detection_id="a",label="chair",
        confidence=1.,timestamp=0.,frame_index=0,max_points=400)
    changed=d.copy()
    changed[mask]=1000
    current=SharedRGBDFrame(changed,K,np.eye(4))
    out=project_visible_footprint(o,current,target_mask=np.zeros_like(mask))
    assert out.occluded_pixels>0
    assert out.free_space_conflict_pixels==0


def test_no_depth_is_unknown_not_disappearance():
    K,d,mask,f=example()
    o=f.object_observation(mask,detection_id="a",label="chair",
        confidence=1.,timestamp=0.,frame_index=0,max_points=400)
    changed=d.copy()
    changed[mask]=0
    out=project_visible_footprint(o,SharedRGBDFrame(changed,K,np.eye(4)),
        target_mask=np.zeros_like(mask))
    assert out.status=="INSUFFICIENT_VIEW"
    assert out.visible_mask_iou is None or out.visible_mask_iou==0.
    assert out.free_space_conflict_pixels==0


def test_real_free_space_conflict_when_object_removed():
    K,d,mask,f=example()
    o=f.object_observation(mask,detection_id="a",label="chair",
        confidence=1.,timestamp=0.,frame_index=0,max_points=400)
    changed=d.copy()
    changed[mask]=3000
    out=project_visible_footprint(o,SharedRGBDFrame(changed,K,np.eye(4)))
    assert out.free_space_conflict_pixels>0
    assert out.occluded_pixels==0


def test_missing_pose_or_invalid_radius_rejected():
    K,d,mask,f=example()
    o=f.object_observation(mask,detection_id="a",label="chair",
        confidence=1.,timestamp=0.,frame_index=0)
    with pytest.raises(ValueError,match="Destination"):
        project_visible_footprint(o,SharedRGBDFrame(d,K))
    with pytest.raises(ValueError,match="radii"):
        project_visible_footprint(o,f,radius_px=5)
