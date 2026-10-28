import numpy as np
import pytest
from semantic_map import Intrinsics,masked_pointcloud,make_observation
from semantic_map.pointmap import SurfaceVoxelMap


def test_metric_backprojection_and_pose():
    K=Intrinsics(fx=100,fy=100,cx=2,cy=2,width=5,height=5,depth_unit_m=.001)
    depth=np.full((5,5),2000,np.uint16)
    mask=np.zeros((5,5),bool);mask[2,2]=True
    T=np.eye(4);T[:3,3]=[1,3,0]
    p=masked_pointcloud(depth,mask,K,T)
    np.testing.assert_allclose(p,[[1,3,2]])


def test_observation_from_rgbd_instance_mask():
    K=Intrinsics(100,100,16,12,32,24)
    depth=np.full((24,32),1500,np.uint16)
    mask=np.zeros((24,32),bool);mask[4:19,5:25]=True
    o=make_observation(depth,mask,K,np.eye(4),source='s',frame_id='1',
                       detection_id='chair1',label='chair',timestamp=1.,appearance=[.5,.5])
    assert o is not None and abs(o.center[2]-1.5)<1e-4
    assert o.high[0]>o.low[0]


def test_invalid_depth_skipped():
    K=Intrinsics(100,100,16,12,32,24)
    depth=np.zeros((24,32),np.uint16)
    mask=np.ones_like(depth,bool)
    assert make_observation(depth,mask,K,np.eye(4),source='s',frame_id='1',
          detection_id='1',label='chair',timestamp=0) is None


def test_surface_map_idempotence(tmp_path):
    path=tmp_path/'vox.sqlite'
    K=Intrinsics(100,100,16,12,32,24)
    depth=np.full((24,32),2000,np.uint16)
    v=SurfaceVoxelMap(path)
    n=v.add_frame(depth,K,np.eye(4),'bag','1')
    assert n>0 and v.add_frame(depth,K,np.eye(4),'bag','1')==0
    count=v.export_ply(tmp_path/'cloud.ply')
    assert count>0
    v.close()
