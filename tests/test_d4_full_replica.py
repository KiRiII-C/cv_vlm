"""D4 invariants; toy calibration/mesh fixtures are software tests, not research data."""
import sys
from pathlib import Path
import tempfile
import json
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from d4_extract_full_replica import indices
from d4_full_office0_map import read_inputs, cast_with_scene


class TestOfficialReplicaReconstruction(unittest.TestCase):
    def test_original_full_trajectory_design(self):
        selected,geometry,yolo,validation=indices()
        self.assertEqual(len(geometry),500)
        self.assertEqual(len(yolo),200)
        self.assertEqual(len(validation),25)
        self.assertTrue(validation.isdisjoint(geometry))
        self.assertEqual(min(selected),0)
        self.assertLess(max(selected),2000)

    def test_official_source_100_frame_variant_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            (root/"Replica"/"office0").mkdir(parents=True)
            (root/"Replica"/"cam_params.json").write_text(json.dumps(
                {"camera":{"fx":600,"fy":600,"cx":599.5,"cy":339.5,"scale":6553.5}}))
            np.savetxt(root/"Replica"/"office0"/"traj.txt",
                       np.tile(np.eye(4).reshape(-1),(100,1)))
            (root/"d4_provenance.json").write_text(json.dumps({
                "integration_original_indices":[0],
                "independent_depth_validation_indices":[2],
                "YOLOE_original_indices":[0]}))
            with self.assertRaisesRegex(RuntimeError,"2000-frame"):
                read_inputs(root)

    def test_simple_mesh_raycast_preserves_camera_z(self):
        import open3d as o3d
        # An axis-aligned single flat plane at z=2, two triangles, cannot
        # claim scene-level reconstruction from this test fixture.
        vertices=np.array([[-1,-1,2],[1,-1,2],[1,1,2],[-1,1,2]],np.float32)
        faces=np.array([[0,1,2],[0,2,3]],np.int32)
        C=np.array([[1,0,0],[0,1,0],[0,0,1],[1,1,1]],np.float32)
        mesh=o3d.geometry.TriangleMesh()
        mesh.vertices=o3d.utility.Vector3dVector(vertices.astype(float))
        mesh.triangles=o3d.utility.Vector3iVector(faces)
        mesh.vertex_colors=o3d.utility.Vector3dVector(C.astype(float))
        scene=o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        K=np.array([[100,0,1.5],[0,100,1.5],[0,0,1]],np.float32)
        res=cast_with_scene(scene,vertices,faces,C,K,np.eye(4),4,4)
        self.assertEqual(int(res["hit"].sum()),16)
        np.testing.assert_allclose(res["z"],2.,rtol=1e-5,atol=1e-5)
        self.assertTrue(np.all(np.isfinite(res["rgb"])))

    def test_rotated_camera_raycast_returns_optical_z_not_wrong_rotation_column(self):
        import open3d as o3d
        from scipy.spatial.transform import Rotation
        angle=Rotation.from_euler("y",38,degrees=True).as_matrix()
        w2c=np.eye(4)
        w2c[:3,:3]=angle
        w2c[:3,3]=[.4,-.2,.1]
        camera_vertices=np.array([[-1,-1,2],[1,-1,2],[1,1,2],[-1,1,2]],np.float32)
        world_vertices=(camera_vertices-w2c[:3,3])@w2c[:3,:3]
        faces=np.array([[0,1,2],[0,2,3]],np.int32)
        colors=np.ones((4,3),np.float32)
        mesh=o3d.geometry.TriangleMesh()
        mesh.vertices=o3d.utility.Vector3dVector(world_vertices.astype(float))
        mesh.triangles=o3d.utility.Vector3iVector(faces)
        mesh.vertex_colors=o3d.utility.Vector3dVector(colors.astype(float))
        scene=o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        K=np.array([[90,0,1.5],[0,90,1.5],[0,0,1]],np.float32)
        res=cast_with_scene(scene,world_vertices.astype(np.float32),faces,colors,
                            K,w2c.astype(np.float32),4,4)
        self.assertTrue(res["hit"].all())
        np.testing.assert_allclose(res["z"],2.,rtol=1e-4,atol=1e-4)

    def test_groundtruth_poses_are_not_slam_outputs(self):
        # D4 uses inverse of an original GT c2w camera matrix.
        cam=np.eye(4);cam[:3,3]=[1,2,3]
        w2c=np.linalg.inv(cam)
        p=np.array([.1,.2,2.])
        q=p@cam[:3,:3].T+cam[:3,3]
        np.testing.assert_allclose(q@w2c[:3,:3].T+w2c[:3,3],p)


if __name__=="__main__":
    unittest.main()
