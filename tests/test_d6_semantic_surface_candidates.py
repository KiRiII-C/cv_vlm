"""D6 geometry/semantic invariants tested on toy arrays, not scientific datasets."""
from __future__ import annotations
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from d6_semantic_seeded_surface_candidates import (
    triangle_support,expand_near_surface,segment_class_surface,
)


def grid(nx,ny):
    v=np.asarray([[x*.05,y*.05,1] for y in range(ny+1)
                  for x in range(nx+1)],np.float32)
    f=[]
    for y in range(ny):
        for x in range(nx):
            a=y*(nx+1)+x;c=a+nx+1
            f.extend(((a,a+1,c),(a+1,c+1,c)))
    return v,np.asarray(f,np.int32)


class TestD6(unittest.TestCase):
    def test_two_agree_and_no_conflicting_class_only(self):
        v,f=grid(1,1)
        c=np.asarray([1,1,0,0],np.int16)
        out,area,n,center,_=triangle_support(v,f,c)
        self.assertEqual(out[0],1)
        self.assertEqual(out[1],0)
        self.assertTrue((area>0).all())

    def test_conflicting_observation_not_forced(self):
        v,f=grid(1,1)
        c=np.asarray([1,1,2,0],np.int16)
        lab,*_=triangle_support(v,f,c)
        self.assertEqual(lab[0],0)

    def test_growth_preserves_all_hard_original_category_seeds(self):
        n=120
        cls=np.zeros(n,dtype=np.int16)
        cls[:60]=1
        cls[60:80]=2
        pos=np.zeros((n,3),np.float32)
        pos[:,:2]=np.asarray([[.003*(i%15),.003*(i//15)] for i in range(n)])
        normal=np.tile(np.array([0,0,1.],np.float32),(n,1))
        rgb=np.tile(np.array([.5,.5,.5],np.float32),(n,1))
        expanded,_=expand_near_surface(cls,pos,normal,rgb)
        self.assertTrue(np.array_equal(expanded[:80],cls[:80]))

    def test_high_color_difference_stops_hallucinated_semantics(self):
        cls=np.zeros(120,dtype=np.int16);cls[:100]=1
        center=np.column_stack((np.arange(120)*.0003,np.zeros(120),np.zeros(120))).astype(np.float32)
        norm=np.tile(np.array([0,0,1.],np.float32),(120,1))
        rgb=np.ones((120,3),np.float32)*.8
        rgb[100:]=.1
        lab,inferred=expand_near_surface(cls,center,norm,rgb)
        self.assertEqual(int(inferred.sum()),0)
        self.assertTrue(np.all(lab[100:]==0))

    def test_two_disconnected_same_category_surfaces_remain_separate(self):
        v,f=grid(8,8)
        v2=v+np.array([5,0,0],np.float32)
        v=np.concatenate((v,v2))
        f=np.concatenate((f,f+len(v)//2))
        normals=np.tile([0,0,1.],(len(f),1))
        areas=np.ones(len(f),np.float32)*.1
        cls=np.ones(len(f),np.int16)
        clusters,summary=segment_class_surface(v,f,cls,areas,normals,
             min_faces=10,min_area_m2=.1)
        ids=np.unique(clusters[clusters>0])
        self.assertEqual(len(ids),2)
        self.assertFalse(summary["physical_instances_verified"])

    def test_two_categories_sharing_triangles_never_merge(self):
        v,f=grid(8,8)
        cats=np.ones(len(f),np.int16)
        cats[len(f)//2:]=2
        norm=np.tile([0,0,1.],(len(f),1))
        ids,_=segment_class_surface(v,f,cats,np.ones(len(f),np.float32),norm,
           min_faces=2,min_area_m2=.01)
        self.assertEqual(len(np.unique(ids[ids>0])),2)
        self.assertNotEqual(ids[10],ids[-10])

if __name__=="__main__":
    unittest.main()
