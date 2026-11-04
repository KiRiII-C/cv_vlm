"""D5: deterministic toy math tests only, never sensor benchmark results."""
from __future__ import annotations
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from d5_official_instance_gt_audit import evaluate_instance_regions,area3,VARIANTS


class D5InstanceMathTests(unittest.TestCase):
    def test_perfect_two_gt_instances_and_two_patches(self):
        gt=np.repeat([0,1],60)
        pred=np.repeat([1,2],60)
        r=evaluate_instance_regions(gt,pred,np.ones(120),gmin=.01,pmin=.01)
        self.assertEqual(r["TP"],2)
        self.assertEqual(r["FP"],0)
        self.assertEqual(r["FN"],0)
        self.assertAlmostEqual(r["physical_instance_PQ_at_IoU50"],1)
        self.assertEqual(r["GT_original_instance_ids_considered"],[0,1])

    def test_one_region_merges_two_physical_gt_instances(self):
        gt=np.repeat([0,1],60)
        pred=np.ones(120,np.int32)
        r=evaluate_instance_regions(gt,pred,np.ones(120),gmin=.01,pmin=.01)
        self.assertLess(r["physical_instance_PQ_at_IoU50"],.40)
        self.assertEqual(r["proposed_regions_with_min_observed_area"],1)
        self.assertEqual(r["mean_GT_instances_contributing_at_least_10_percent_to_a_proposal"],2)

    def test_split_gt_instances_is_penalized(self):
        gt=np.repeat([0,1],60)
        pred=np.concatenate([np.ones(30),np.ones(30)*2,
                             np.ones(30)*3,np.ones(30)*4]).astype(np.int32)
        r=evaluate_instance_regions(gt,pred,np.ones(120),gmin=.01,pmin=.01)
        self.assertGreater(r["mean_proposal_regions_covering_GT_at_least_10_percent"],1)
        self.assertGreater(r["FP"],0)
        self.assertLess(r["physical_instance_PQ_at_IoU50"],1)

    def test_abstention_is_not_mistaken_for_object(self):
        gt=np.repeat([0,1],60)
        pred=np.zeros(120,dtype=np.int32)
        r=evaluate_instance_regions(gt,pred,np.ones(120),gmin=.01,pmin=.01)
        self.assertEqual(r["TP"],0)
        self.assertEqual(r["FN"],2)
        self.assertEqual(r["proposed_regions_with_min_observed_area"],0)
        self.assertEqual(r["physical_instance_PQ_at_IoU50"],0)
        self.assertEqual(r["abstained_area_fraction_of_GT_matched"],1)

    def test_unknown_gt_does_not_form_false_physical_instance(self):
        gt=np.concatenate((np.full(50,-1),np.zeros(50,dtype=int)))
        p=np.concatenate((np.full(50,1),np.full(50,2)))
        r=evaluate_instance_regions(gt,p,np.ones(100),gmin=.01,pmin=.01)
        self.assertEqual(r["independent_GT_objects_with_min_observed_area"],1)
        self.assertEqual(r["proposed_regions_with_min_observed_area"],1)

    def test_sample_triangle_area_with_no_hallucinated_geometry(self):
        v=np.array([[0,0,0],[1,0,0],[0,1,0]],np.float32)
        f=np.array([[0,1,2]],np.int32)
        self.assertAlmostEqual(float(area3(v,f)[0]),.5)

    def test_predeclared_only_three_class_agnostic_arms(self):
        self.assertEqual(set(VARIANTS),{
            "published_D4","finer_normal_color","moderate_normal_color"})
        self.assertEqual(VARIANTS["published_D4"],None)

if __name__=="__main__":
    unittest.main()
