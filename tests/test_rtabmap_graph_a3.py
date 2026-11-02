"""Structural tests for A3 math, graph identity and timestamp matching.

These are unit tests only. Real dataset quality is assessed by A3 CI.
"""
from __future__ import annotations
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from rtabmap_graph_reconstruct_a3 import (aligned_errors, graph_topology,
                                           match_keyframes, transform)


class TestA3(unittest.TestCase):
    def test_pose_transform(self):
        R=Rotation.from_euler("xyz",[.1,.2,.3]).as_quat().tolist()
        T=transform({"xyz":[1,2,3],"xyzw":R})
        self.assertTrue(np.allclose(T[:3,3],[1,2,3]))
        self.assertAlmostEqual(np.linalg.det(T[:3,:3]),1.)
        with self.assertRaises(ValueError):
            transform({"xyz":[0,0,0],"xyzw":[0,0,0,0]})

    def test_aligned_ate_is_se3_not_sim3(self):
        x=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],dtype=float)
        R=Rotation.from_euler("z",.5).as_matrix()
        y=x@R.T + np.array([4,2,-8])
        report=aligned_errors(x,y)
        self.assertLess(report["rmse_m"],1e-10)
        # Scale cannot be removed by a rigid alignment.
        scaled=aligned_errors(2*x,y)
        self.assertGreater(scaled["rmse_m"],.2)
        self.assertEqual(scaled["sample_count"],4)

    def test_matching_excludes_intermediate_negative_weight_nodes(self):
        with tempfile.TemporaryDirectory() as folder:
            db=Path(folder)/"db.sqlite"
            con=sqlite3.connect(db)
            con.execute("CREATE TABLE Node (id INTEGER PRIMARY KEY, stamp FLOAT, weight INTEGER)")
            published=[{"ros_time":1200.0+.5*i,
                        "source_tum_timestamp":1000.0+.1*i} for i in range(50)]
            poses={}
            for j in range(20):
                node_id=j+1
                pose={"xyz":[float(j)/20,0.,0.],"xyzw":[0.,0.,0.,1.]}
                poses[str(node_id)]=pose
                con.execute("INSERT INTO Node VALUES (?,?,?)",
                            (node_id,published[j*2]["ros_time"],-9 if j==4 else 0))
            con.commit();con.close()
            graph={
                "GT_used_as_input":False,
                "unoptimized":{"frame_id":"map","poses":poses,"links":[]},
                "optimized":{"frame_id":"map","poses":poses,"links":[]},
            }
            result=match_keyframes(graph,db,published,min_nodes=18)
            self.assertEqual(len(result),19)
            self.assertNotIn(5,{m["node_id"] for m in result})
            self.assertEqual(result[0]["published_index"],0)
            self.assertEqual(result[-1]["published_index"],38)
            self.assertTrue(all(m["stamp_error_s"]<1e-6 for m in result))

    def test_graph_topology_counts_cycles_and_components(self):
        matches=[{"node_id":i} for i in range(1,5)]
        graph={"links":[
            {"from_id":1,"to_id":2,"type":0},
            {"from_id":2,"to_id":3,"type":0},
            {"from_id":3,"to_id":4,"type":0},
            {"from_id":1,"to_id":4,"type":1}]}
        result=graph_topology(graph,matches)
        self.assertEqual(result["connected_components"],1)
        self.assertEqual(result["link_types"]["global_closure"],1)
        self.assertEqual(result["graph_links_among_matched_nodes"],4)


if __name__=="__main__":
    unittest.main()
