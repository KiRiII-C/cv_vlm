import sqlite3
import numpy as np
import pytest
from semantic_map.object_surface_memory_v2 import ObjectSurfaceMemoryV2,object_color
from dataclasses import dataclass

@dataclass
class K:
 fx:float=100.;fy:float=100.;cx:float=4.;cy:float=4.
 width:int=8;height:int=8;depth_unit_m:float=.001

def fixture():
 depth=np.full((8,8),1000,dtype=np.uint16)
 mask=np.zeros((8,8),bool);mask[1:7,1:7]=True
 rgb=np.zeros((8,8,3),dtype=np.uint8);rgb[:]=[30,100,200]
 return depth,mask,rgb

def test_observed_surface_idempotent_colored_ply_and_persistence(tmp_path):
 db=sqlite3.connect(str(tmp_path/'surface.sqlite'))
 surface=ObjectSurfaceMemoryV2(db,.02,minimum_support=1)
 dep,mask,rgb=fixture()
 n=surface.integrate_mask('uuid_chair',dep,mask,rgb,K(),np.eye(4),source='rgb',frame_id='f0',pixel_stride=1)
 assert n==36
 assert surface.integrate_mask('uuid_chair',dep,mask,rgb,K(),np.eye(4),source='rgb',frame_id='f0',pixel_stride=1)==0
 meta=surface.export(tmp_path/'map.ply')
 assert meta['vertices']>0 and meta['unique_ids']==1
 assert meta['instances']['1']['uuid']=='uuid_chair'
 assert len(object_color('uuid_chair'))==3
 db.close()
 db=sqlite3.connect(str(tmp_path/'surface.sqlite'))
 assert ObjectSurfaceMemoryV2(db,.02,1).export(tmp_path/'reload.ply')['vertices']==meta['vertices']
 db.close()

def test_verified_negative_evidence_then_actual_reobservation(tmp_path):
 db=sqlite3.connect(str(tmp_path/'s.db'))
 store=ObjectSurfaceMemoryV2(db,.02,1)
 dep,mask,rgb=fixture()
 store.integrate_mask('a',dep,mask,rgb,K(),np.eye(4),source='s',frame_id='0',pixel_stride=1)
 before=store.export(tmp_path/'before.ply')['vertices']
 points=np.asarray([[0,0,1],[.01,0,1],[0,.01,1]],float)
 for k in range(3):store.refute_visible_cells('a',points,timestamp=float(k))
 after=store.export(tmp_path/'after.ply')['vertices']
 assert after<before
 store.integrate_mask('a',dep,mask,rgb,K(),np.eye(4),source='s',frame_id='1',pixel_stride=1)
 assert store.export(tmp_path/'again.ply')['vertices']>=after
 db.close()

def test_bad_calibration_and_depth_rejected(tmp_path):
 db=sqlite3.connect(str(tmp_path/'s.db'))
 store=ObjectSurfaceMemoryV2(db,.02,1)
 dep,mask,rgb=fixture()
 with pytest.raises(ValueError):store.integrate_mask('a',dep,mask[:5],rgb,K(),np.eye(4),source='s',frame_id='0')
 with pytest.raises(ValueError):store.refute_visible_cells('a',np.ones((3,2)))
 db.close()
