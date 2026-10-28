import numpy as np
from semantic_map.yoloe26x_factorial import match_masks,target_instances,NYU

def make_gt():
    gt=np.full((40,60),-1,np.int32)
    sem=np.zeros((40,60),np.int32)
    gt[2:22,2:22]=101
    gt[2:22,35:55]=102
    sem[gt==101]=5 # chair
    sem[gt==102]=7 # table
    return gt,sem

def test_gt_12_class_target_labels():
    gt,sem=make_gt()
    t=target_instances(gt,sem)
    assert len(t)==2
    assert sorted(a[1] for a in t)==['chair','table']

def test_hungarian_one_to_one_gt_alignment():
    gt,sem=make_gt()
    t=target_instances(gt,sem)
    first=gt==101
    second=gt==102
    pairs,ious,inter=match_masks([first,second],gt,t)
    assert len(pairs)==2
    assert np.isclose(ious[0,pairs[0]],1)
    assert np.isclose(ious[1,pairs[1]],1)
    assert pairs[0]!=pairs[1]

def test_missing_detector_mask_has_no_fake_true_positive():
    gt,sem=make_gt()
    t=target_instances(gt,sem)
    pairs,ious,_=match_masks([gt==101],gt,t)
    assert len(pairs)==1
    assert sum(ious[i,j]>=.5 for i,j in pairs.items())==1
    assert len(t)-len(pairs)==1

def test_duplicate_masks_do_not_get_two_gt_matches():
    gt,sem=make_gt()
    t=target_instances(gt,sem)
    p=gt==101
    pairs,ious,_=match_masks([p,p],gt,t)
    assert len(pairs)==1

def test_outside_gt_mask_is_not_matched():
    gt,sem=make_gt()
    t=target_instances(gt,sem)
    bad=np.zeros_like(gt,bool)
    bad[25:38,20:35]=1
    pairs,ious,_=match_masks([bad],gt,t)
    assert not pairs

def test_gt_target_semantics_are_explicit():
    assert NYU=={3:'cabinet',4:'bed',5:'chair',6:'sofa',7:'table',
                 10:'bookshelf',14:'table',25:'tv'}
