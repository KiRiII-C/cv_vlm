import numpy as np
from dataclasses import dataclass
from semantic_map.visibility_evidence_v2 import evaluate_visibility

@dataclass
class K:
    fx:float=100.;fy:float=100.;cx:float=80.;cy:float=60.
    width:int=160;height:int=120;depth_unit_m:float=.001

L=np.array([-.2,-.2,1.5]);H=np.array([.2,.2,1.9]);T=np.eye(4)

def test_empty_region_requires_measured_depth_behind_entire_box():
    depth=np.full((120,160),2600,np.uint16)
    x=evaluate_visibility(depth,K(),T,L,H)
    assert x.state=='VERIFIED_FREE'
    assert x.tested_rays>=6 and x.free_rays==x.tested_rays
    assert x.proof['old_region_cleared'] and x.proof['pose_version']==0

def test_detector_absence_cannot_prove_region_empty_if_occluded():
    depth=np.full((120,160),1000,np.uint16)
    x=evaluate_visibility(depth,K(),T,L,H)
    assert x.state=='OCCLUDED' and x.proof is None

def test_old_object_actually_present_is_not_empty():
    depth=np.full((120,160),1650,np.uint16)
    x=evaluate_visibility(depth,K(),T,L,H)
    assert x.state=='OCCUPIED_OR_PARTIAL' and x.proof is None

def test_invalid_depth_unknown_not_free():
    depth=np.zeros((120,160),np.uint16)
    x=evaluate_visibility(depth,K(),T,L,H)
    assert x.proof is None and x.state=='UNKNOWN_MISSING_DEPTH'

def test_pose_outside_view_unknown():
    t=np.eye(4);t[0,3]=10
    x=evaluate_visibility(np.full((120,160),3000,np.uint16),K(),t,L,H)
    assert x.proof is None and x.state=='UNKNOWN_INSUFFICIENT_VIEW'

def test_invalid_pose_rejected():
    t=np.eye(4);t[0,0]=1.3
    import pytest
    with pytest.raises(ValueError):
        evaluate_visibility(np.full((120,160),3000,np.uint16),K(),t,L,H)
