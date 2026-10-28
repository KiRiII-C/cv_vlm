import unittest
from semantic_map.hypersim_instance_eval import evaluate_identity

class HypersimPhysicalIdentityMetricTests(unittest.TestCase):
    def test_stable_gt_uuid_is_perfect(self):
        o=[(1,'a'),(2,'b'),(1,'a'),(2,'b'),(1,'a')]
        r=evaluate_identity(o,5)
        self.assertAlmostEqual(r['IDF1_observation_level'],1.)
        self.assertEqual(r['false_split_gt_objects'],0)
        self.assertEqual(r['false_merge_predicted_ids'],0)

    def test_revisit_split_penalized(self):
        r=evaluate_identity([(1,'a'),(1,'a'),(1,'c'),(1,'c')],4)
        self.assertAlmostEqual(r['IDF1_observation_level'],0.5)
        self.assertEqual(r['false_split_gt_objects'],1)

    def test_same_uuid_wrong_physical_objects_penalized(self):
        r=evaluate_identity([(1,'a'),(2,'a')],2)
        self.assertAlmostEqual(r['IDF1_observation_level'],0.5)
        self.assertEqual(r['false_merge_predicted_ids'],1)
