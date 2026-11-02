"""Unit tests for EXP-20261102-08 design; synthetic tensors ONLY test code invariants.
They are never counted as experimental sensor measurements or scientific results.
"""
from __future__ import annotations
import pathlib, sys, unittest

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'research' / 'fundamental'))
from matched_unique_train import ARMS, matched_indices, sample_positives


class MatchedSupportTests(unittest.TestCase):
    def setUp(self):
        self.xyz = (torch.arange(800*7*9*9, dtype=torch.float32).reshape(800,7,9,9),
                    torch.zeros(800,7,9,9))
        self.desk = (torch.ones(800,7,9,9), torch.ones(800,7,9,9))

    def test_exact_cardinality_and_sources(self):
        for arm in ARMS:
            with self.subTest(arm=arm):
                X,Y,counts,digest=sample_positives(self.xyz,self.desk,arm,20261102)
                n=int(arm.split('_')[-1])
                expected=(n//2,n//2) if arm.startswith('mixed') else (n,0)
                self.assertEqual((counts['xyz'],counts['desk']),expected)
                self.assertEqual(len(X),n)
                self.assertEqual(len(Y),n)
                self.assertEqual(len(digest),64)

    def test_matched_subsets_are_nested_per_seed(self):
        for seed in [20261102,20261103,20261104]:
            x256,d0=matched_indices(800,800,'xyz_256',seed)
            x512,d1=matched_indices(800,800,'xyz_512',seed)
            mix256,desk256=matched_indices(800,800,'mixed_256',seed)
            mix512,desk512=matched_indices(800,800,'mixed_512',seed)
            np.testing.assert_array_equal(x256,x512[:256])
            np.testing.assert_array_equal(mix256,mix512[:128])
            np.testing.assert_array_equal(mix512,x256)
            np.testing.assert_array_equal(desk256,desk512[:128])
            self.assertEqual(len(np.unique(x512)),512)
            self.assertEqual(len(np.unique(desk512)),256)
            self.assertEqual(len(d0),0)
            self.assertEqual(len(d1),0)

    def test_reproducibility_and_rejection_of_missing_samples(self):
        a=matched_indices(800,800,'mixed_512',20261102)
        b=matched_indices(800,800,'mixed_512',20261102)
        np.testing.assert_array_equal(a[0],b[0])
        np.testing.assert_array_equal(a[1],b[1])
        with self.assertRaises(ValueError):
            matched_indices(255,800,'xyz_256',20261102)
        with self.assertRaises(ValueError):
            matched_indices(800,200,'mixed_512',20261102)
        with self.assertRaises(ValueError):
            matched_indices(800,800,'unknown_256',20261102)


if __name__ == '__main__':
    unittest.main()
