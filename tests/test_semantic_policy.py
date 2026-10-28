from semantic_map.semantic_policy import View,choose_views
from semantic_map.voxel_votes import VoxelSemanticMemory
from semantic_map.benchmark_extensions import run

def test_view_selection():
    a=[View('a',f'f{i}',1-i*.02,0.1*i,i) for i in range(10)]
    selected=choose_views(a,max_views=3,min_angular_separation=.15)
    assert len(selected)<=3
    assert len(set(x.frame_id for x in selected))==len(selected)

def test_voxel_voting_duplicate_frame_ignored():
    mem=VoxelSemanticMemory(min_views=2)
    mem.observe((1,0,0),'chair',.8,'frame0')
    mem.observe((1,0,0),'table',.7,'frame0')
    assert mem.query((1,0,0))['views']==1
    mem.observe((1,0,0),'chair',.8,'frame1')
    assert mem.query((1,0,0))['label']=='chair'

def test_semantic_ablation(tmp_path):
    a=run(tmp_path)
    assert a['five_view_vote_accuracy']>=a['single_view_accuracy']
    assert a['vlm_queries_after_selection']<=60
