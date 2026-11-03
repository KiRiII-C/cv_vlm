"""D1 protocol unit tests use toy arrays ONLY; sensor results come from CI datasets."""
from __future__ import annotations
import os
import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from d1_multidataset_measured_maps import (
    DATASETS, BONN_EXTRINSIC, camera_transform, measured_xyz,
    apply_pose, aggregate_voxels, write_binary_ply, matched_sensor_rows,
)


class RealMapInvariantTests(unittest.TestCase):
    def test_all_supported_datasets_have_valid_camera_calibrations(self):
        self.assertEqual(set(DATASETS),{"tum_xyz","tum_desk2","bonn_box"})
        for row in DATASETS.values():
            self.assertTrue(all(np.isfinite(row["K"])))
            self.assertGreater(row["K"][0],0)
            self.assertGreater(row["K"][1],0)
            self.assertEqual(row["depth_scale"],5000.)

    def test_tum_gt_pose_transform_no_extra_marker(self):
        T=camera_transform([1,2,3,0,0,0,1],"tum_xyz")
        np.testing.assert_allclose(T[:3,:3],np.eye(3),atol=1e-6)
        np.testing.assert_allclose(apply_pose(np.array([[0,0,2.]],np.float32),T),
                                   [[1,2,5]],atol=1e-5)

    def test_bonn_marker_extrinsics_rotation_is_rigid(self):
        T=camera_transform([0,0,0,0,0,0,1],"bonn_box")
        np.testing.assert_allclose(T,BONN_EXTRINSIC,atol=1e-6)
        np.testing.assert_allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-5)

    def test_never_interpolates_missing_depth(self):
        d=np.array([[1.,0.,2.,np.nan],[0.,3.,0.,0.],[1.,1.,1.,1.]],dtype=float)
        xyz,yy,xx=measured_xyz(d,(100,100,0,0),step=1)
        self.assertEqual(len(xyz),7)
        self.assertTrue(np.all((d[yy,xx]>=.25)&(d[yy,xx]<=5.)))

    def test_voxel_averages_measured_positions_and_rgb(self):
        xyz=np.array([[.001,0,1],[.006,0,1],[.2,0,1]],np.float32)
        rgb=np.array([[10,20,30],[30,40,50],[100,0,0]],np.uint8)
        out,col,count=aggregate_voxels(xyz,rgb,.01)
        self.assertEqual(len(out),2)
        self.assertEqual(sorted(count.tolist()),[1,2])
        i=np.flatnonzero(count==2)[0]
        np.testing.assert_allclose(out[i],[.0035,0,1],atol=1e-5)
        np.testing.assert_array_equal(col[i],[20,30,40])

    def test_binary_ply_data_header_and_empty_semantics(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"cloud.ply"
            xyz=np.empty((0,3),np.float32)
            rgb=np.empty((0,3),np.uint8)
            write_binary_ply(p,xyz,rgb)
            data=p.read_bytes()
            self.assertIn(b"element vertex 0",data)
            self.assertIn(b"format binary_little_endian",data)
            self.assertEqual(data.count(b"end_header"),1)

    def test_manifest_matches_tum_timestamp_triplets(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            (p/"rgb.txt").write_text("# rgb\n0.000 rgb/a.png\n0.100 rgb/b.png\n")
            (p/"depth.txt").write_text("0.003 depth/a.png\n0.101 depth/b.png\n")
            (p/"groundtruth.txt").write_text(
                "0.004 0 0 0 0 0 0 1\n0.102 1 0 0 0 0 0 1\n")
            seq,meta=matched_sensor_rows(p)
            self.assertEqual(len(seq),2)
            self.assertEqual(meta["synchronized_matched"],2)
            self.assertAlmostEqual(seq[1]["pose"][0],1.)

    def test_out_of_tolerance_does_not_invent_pose(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            (p/"rgb.txt").write_text("0.000 rgb/a.png\n")
            (p/"depth.txt").write_text("0.100 depth/a.png\n")
            (p/"groundtruth.txt").write_text("0.000 0 0 0 0 0 0 1\n")
            seq,meta=matched_sensor_rows(p)
            self.assertEqual(len(seq),0)


if __name__=="__main__":
    unittest.main()
