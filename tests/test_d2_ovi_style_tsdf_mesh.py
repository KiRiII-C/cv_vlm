"""D2 semantic mesh safeguards; arrays are synthetic fixtures, not RGB-D results."""
from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from d2_ovi_style_tsdf_mesh import label_mesh_vertices, centered_limits


class OverlayTests(unittest.TestCase):
    def setUp(self):
        self.palette=np.array([[250,10,0],[0,25,250]],np.uint8)
        self.points=np.array([[0,0,1],[.002,0,1],[0,.003,1],
                              [.004,.002,1]],np.float32)
        self.colors=np.tile(self.palette[0],(4,1))
        self.verts=np.array([[.001,.001,1.],[.4,.4,1.]],np.float32)

    def test_verified_nearby_support_only(self):
        x=label_mesh_vertices(self.verts,self.points,self.colors,self.palette)
        self.assertEqual(x.tolist(),[1,0])

    def test_unknown_unobserved_surface(self):
        out=label_mesh_vertices(self.verts,np.empty((0,3),np.float32),
             np.empty((0,3),np.uint8),self.palette)
        self.assertTrue(np.all(out==0))

    def test_single_neighbor_is_not_physical_support(self):
        x=label_mesh_vertices(np.array([[.01,0,1]],np.float32),
             self.points[:1],self.colors[:1],self.palette)
        self.assertEqual(x.tolist(),[0])

    def test_conflicting_classes_abstain(self):
        colors=self.colors.copy()
        colors[:2]=self.palette[1]
        x=label_mesh_vertices(self.verts[:1],self.points,colors,self.palette)
        self.assertEqual(x.tolist(),[0])

    def test_unknown_color_rejected_as_provenance_violation(self):
        with self.assertRaisesRegex(ValueError,"palette"):
            label_mesh_vertices(self.verts,self.points,
                np.full((4,3),[128,128,128],np.uint8),self.palette)

    def test_int32_palette_no_overflow(self):
        test=np.array([[1.,0,1.]],np.float32)
        p=np.array([[1.,0,1.],[1.001,0,1.],[1.,.001,1.]],np.float32)
        colors=np.tile(np.array([255,255,255],np.uint8),(3,1))
        palette=np.array([[0,0,0],[255,255,255]],np.uint8)
        out=label_mesh_vertices(test,p,colors,palette)
        self.assertEqual(out.tolist(),[2])

    def test_robust_focus_excludes_extreme_outlier(self):
        coords=np.column_stack((np.arange(100)/100.,np.zeros(100),np.zeros(100)))
        coords=np.concatenate((coords,[[100000.,0.,0.]]))
        ctr,size=centered_limits(coords)
        self.assertLess(ctr[0],1.)
        self.assertLess(size[0],3.)

    def test_invalid_geometry_rejected(self):
        bad=self.verts.copy();bad[0,0]=np.nan
        with self.assertRaises(ValueError):
            label_mesh_vertices(bad,self.points,self.colors,self.palette)


if __name__=="__main__":
    unittest.main()
