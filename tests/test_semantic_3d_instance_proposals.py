import numpy as np
from semantic_map.semantic_3d_instance_proposals import instance_proposals

def test_3d_separates_touching_identical_class_pixels():
    h,w=54,72
    ids=np.full((h,w),255,dtype="int32")
    ids[6:48,6:66]=19
    yy,xx=np.mgrid[0:h,0:w]
    xyz=np.stack([xx*.014,yy*.014,np.zeros_like(xx,float)],axis=2)
    xyz[:,36:,0]+=2.
    masks=instance_proposals(ids,xyz,{19:"chair"},eps=.16,
                             subsample=3,min_support=4,min_area=35)
    assert len(masks)>=2
    assert all(m.shape==ids.shape for _,m in masks)

def test_ade20k_class_aliases_merge_into_one_canonical_semantic_region():
    h,w=54,54
    ids=np.full((h,w),255,dtype="int32")
    ids[7:26,7:26]=19
    ids[26:45,7:26]=30
    yy,xx=np.mgrid[0:h,0:w]
    xyz=np.stack([xx*.014,yy*.014,np.zeros_like(xx,float)],axis=2)
    masks=instance_proposals(ids,xyz,{19:"chair",30:"chair"},
                             eps=.16,subsample=3,min_support=4,min_area=35)
    assert len(masks)>=1
    assert all(label=="chair" for label,_ in masks)

def test_empty_classes_return_no_physical_instances():
    ids=np.zeros((12,12),dtype="uint8")
    xyz=np.zeros((12,12,3),float)
    assert instance_proposals(ids,xyz,{19:"chair"})==[]
