import numpy as np
import pytest
from semantic_map.geometry import Intrinsics,masked_pointcloud
from semantic_map.shared_rgbd_geometry import SharedRGBDFrame,reproject_object


def fixture():
    K=Intrinsics(100,100,16,12,32,24,.001)
    depth=np.full((24,32),2000,dtype=np.uint16)
    return K,depth


def test_shared_matches_legacy_mask_backprojection():
    K,d=fixture()
    mask=np.zeros(d.shape,bool);mask[3:21,3:27]=True
    T=np.eye(4);T[:3,3]=[.1,.2,.3]
    shared=SharedRGBDFrame(d,K,T)
    a=shared.sample_mask(mask)@T[:3,:3].T+T[:3,3]
    b=masked_pointcloud(d,mask,K,T,max_points=5000)
    np.testing.assert_allclose(a,b,atol=1e-6)


def test_invalid_depth_stays_unobserved():
    K,d=fixture()
    d[3:8,:]=0
    mask=np.ones(d.shape,bool)
    frm=SharedRGBDFrame(d,K)
    assert not np.any(frm.valid[3:8])
    assert len(frm.sample_mask(mask))==(24*32-5*32)
    with pytest.raises(ValueError,match="measured camera pose"):
        frm.object_observation(mask,detection_id="a",label="chair",
                               confidence=.9,timestamp=0.,frame_index=0)


def test_reprojection_identity_and_depth_conflict():
    K,d=fixture()
    mask=np.zeros(d.shape,bool);mask[3:21,3:27]=True
    frm=SharedRGBDFrame(d,K,np.eye(4))
    obj=frm.object_observation(mask,detection_id="mask1",label="chair",
            confidence=.8,timestamp=1,frame_index=1)
    assert obj.observation_id=="1:mask1"
    assert obj.point_count>25
    result=reproject_object(obj,frm,target_mask=mask)
    assert result.depth_consistency_ratio==1.
    assert result.projected_mask_iou==1.
    assert result.free_space_conflict_pixels==0
    d2=d.copy()
    d2[mask]=2600
    second=SharedRGBDFrame(d2,K,np.eye(4))
    result2=reproject_object(obj,second,target_mask=mask)
    assert result2.depth_consistency_ratio==0
    assert result2.free_space_conflict_pixels>0


def test_occlusion_does_not_prove_disappearance():
    K,d=fixture()
    mask=np.zeros(d.shape,bool);mask[3:21,3:27]=True
    obj=SharedRGBDFrame(d,K,np.eye(4)).object_observation(
            mask,detection_id="a",label="chair",confidence=.9,timestamp=1,frame_index=1)
    new=d.copy();new[mask]=1100
    current=SharedRGBDFrame(new,K,np.eye(4))
    r=reproject_object(obj,current,target_mask=np.zeros(mask.shape,bool))
    assert r.occluded_pixels>0 and r.free_space_conflict_pixels==0
    assert r.projected_mask_iou is None or r.projected_mask_iou==0.


def test_nonrigid_pose_is_rejected():
    K,d=fixture()
    T=np.eye(4);T[0,0]=2.
    with pytest.raises(ValueError,match="SO"):
        SharedRGBDFrame(d,K,T)
