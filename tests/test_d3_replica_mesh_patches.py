"""D3 deterministic software tests (toy shapes only; science uses Replica office0)."""
import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from d3_replica_office0_mesh_patches import (
    mask_to_measured_world,segment_surface_patches,transfer_semantic,
)


class TestD3Invariants(unittest.TestCase):
    def test_measured_pixels_only_world_coordinates(self):
        y,x=np.mgrid[:16,:16]
        d=np.ones((16,16),dtype=np.float32)*2
        valid=np.ones((16,16),dtype=bool)
        mask=np.ones((16,16),dtype=bool)
        valid[0,:]=False
        T=np.eye(4)
        T[0,3]=1.5
        K=np.array([[200,0,7.5],[0,200,7.5],[0,0,1]],float)
        pts=mask_to_measured_world(mask,d,valid,K,T)
        self.assertEqual(len(pts),240)
        self.assertTrue(np.all(pts[:,0]>1.3))
        self.assertTrue(np.allclose(pts[:,2],2))

    def test_no_invention_when_depth_missing(self):
        mask=np.ones((20,20),bool);z=np.zeros((20,20),float)
        out=mask_to_measured_world(mask,z,np.zeros_like(mask),np.eye(3),np.eye(4))
        self.assertEqual(out.shape,(0,3))

    def test_validated_depth_background_contamination_removed(self):
        z=np.ones((15,15),dtype=np.float32)
        z[0,0]=5
        mask=np.ones_like(z,dtype=bool)
        out=mask_to_measured_world(mask,z,mask,np.array([[100,0,7],[0,100,7],[0,0,1.]]),np.eye(4))
        self.assertTrue(np.all(out[:,2]<2))

    def test_independent_of_any_semantic_label(self):
        # Connected coplanar 3x3 quad triangles form one continuous patch.
        vertices=np.array([[i,j,0] for j in range(4) for i in range(4)],np.float32)
        faces=[]
        for y in range(3):
            for x in range(3):
                a=y*4+x;b=a+1;c=a+4;d=c+1
                faces.extend([[a,b,c],[b,d,c]])
        # 18 faces; artificially set test min threshold to 5.
        faces=np.asarray(faces,np.int32)
        # Add a duplicate disconnected group to meet function min 100 faces.
        mesh_v=[];mesh_f=[]
        for k in range(6):
            mesh_v.append(vertices+np.array([k*20.,0,0]))
            mesh_f.append(faces+k*len(vertices))
        V=np.concatenate(mesh_v);F=np.concatenate(mesh_f)
        colors=np.ones_like(V)*.7
        regions,meta=segment_surface_patches(V,F,colors,min_faces=10,max_regions=10)
        self.assertEqual(len(regions),108)
        self.assertEqual(len(set(regions)),6)
        self.assertEqual(meta["displayed_geometric_regions"],6)
        self.assertTrue(meta["class_agnostic"])
        self.assertFalse(meta["physical_instance_ids_claimed"])

    def test_semantic_support_requires_consensus(self):
        ref=np.array([[0,0,1],[.001,0,1],[0,.001,1],[.002,0,1],[0,.002,1]],np.float32)
        classes=np.array([2,2,2,1,1],np.int16)
        query=np.array([[.0001,.0001,1],[1,1,1]],np.float32)
        out=transfer_semantic(query,ref,classes,k=5,purity=.7)
        self.assertEqual(out.tolist(),[0,0])
        classes[:4]=2
        out=transfer_semantic(query,ref,classes,k=5,purity=.7)
        self.assertEqual(out.tolist(),[2,0])

    def test_pose_transform_is_not_slam(self):
        # Proves only matrix convention used for original rendered Replica.
        Twc=np.eye(4);Twc[0,3]=5
        Pw=np.array([[0,0,1.]])
        Tcw=np.linalg.inv(Twc)
        pcam=(Pw@Twc[:3,:3].T+Twc[:3,3])
        np.testing.assert_allclose(pcam,[5,0,1])
        recovered=pcam@Tcw[:3,:3].T+Tcw[:3,3]
        np.testing.assert_allclose(recovered,Pw)


if __name__=="__main__":
    unittest.main()
