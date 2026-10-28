import sys, tempfile, unittest, pathlib
import numpy as np
from semantic_map.rgbd_scenes_gt import sample_indices,metrics,local_features,save_colored_ply,train_and_evaluate,load_ply_xyzrgb,class_union_localization
from semantic_map.instance_cloud import InstanceCloud,palette_rgb
class TestGt(unittest.TestCase):
    def test_no_leak(self):
        with self.assertRaises(ValueError):train_and_evaluate('/nope','/tmp/nope',train_scenes=(1,2),test_scenes=(2,3))
    def test_3d_class_union_localization_without_claiming_instance_gt(self):
        xyz=np.array([[0.,0.,0.],[1,1,1],[.1,.1,.1],[1.1,1.1,1.1]]*40,dtype='float32')
        truth=np.array([1,1,1,1]*40)
        pred=truth.copy()
        result=class_union_localization(xyz,truth,pred,min_support=10)
        self.assertAlmostEqual(result['1']['centroid_distance_m'],0.,places=6)
        self.assertAlmostEqual(result['1']['class_union_aabb_3d_iou'],1.,places=6)
    def test_exact_iou(self):
        r=metrics(np.array([1,1,2,2,10]),np.array([1,2,2,2,10]))
        self.assertAlmostEqual(r['per_class']['1']['IoU'],.5)
        self.assertAlmostEqual(r['per_class']['2']['IoU'],2/3)
    def test_eval_sampling_no_labels_leak(self):
        n=20000;y=np.ones(n,dtype='int16');y[-3:]=2
        a=sample_indices(y,4000,22,stratify=False)
        b=sample_indices(np.full(n,10),4000,22,stratify=False)
        np.testing.assert_array_equal(a,b)
    def test_palette_stable(self):
        self.assertEqual(palette_rgb('uuid-a'),palette_rgb('uuid-a'))
        self.assertNotEqual(palette_rgb('uuid-a'),palette_rgb('uuid-b'))
    def test_world_project_and_unique_id(self):
        class K:height=3;width=4;cx=1;cy=1;fx=100;fy=100;depth_unit_m=.001
        c=InstanceCloud(.01,stride=1)
        m=np.ones((3,4),bool);d=np.full((3,4),1000,'uint16')
        T=np.eye(4);T[0,3]=3
        self.assertEqual(c.add('obj_a',m,d,K(),T),12)
        self.assertEqual(c.add('obj_b',m,d,K(),T),12)
        with tempfile.TemporaryDirectory() as td:
            meta=c.export(pathlib.Path(td)/'cloud.ply')
            self.assertEqual(meta['unique_ids'],2)
            self.assertEqual(len(meta['instances']),2)
    def test_output_ply_counts(self):
        with tempfile.TemporaryDirectory() as td:
            p=pathlib.Path(td)/'x.ply'
            save_colored_ply(p,np.zeros((2,3),'float32'),np.array([1,10]))
            self.assertIn(b'element vertex 2',p.read_bytes())
if __name__=='__main__': unittest.main()
