import numpy as np
from semantic_map.models import Track
from semantic_map.planner import OccupancyGrid,CommandProcessor

def t(uid,label):
    c=np.array([2.,2.,1.])
    return Track(uid,label,c,c-.1,c+.1,np.array([1.,0.]),5,0.,1.,"confirmed")

def test_semantic_query_does_not_match_unrelated_coco_objects():
    grid=OccupancyGrid(np.zeros((50,50),dtype=bool),resolution=.1)
    keyboard=t('obj_aaaaaaaaaaaaaaaa','keyboard')
    book=t('obj_bbbbbbbbbbbbbbbb','book')
    processor=CommandProcessor([keyboard,book],grid,robot_xy=[.5,.5])
    assert [v.object_id for v in processor._matches('найди keyboard')]==[keyboard.object_id]
    assert [v.object_id for v in processor._matches('найди клавиатуру')]==[keyboard.object_id]
    assert [v.object_id for v in processor._matches('подъедь к книге')]==[book.object_id]
    assert processor._matches('неизвестный космический объект')==[]
    assert [v.object_id for v in processor._matches('подъедь к obj_bbbbbbbbbbbbbbbb')]==[book.object_id]

def test_ambiguous_similar_objects_fail_closed():
    grid=OccupancyGrid(np.zeros((50,50),dtype=bool),resolution=.1)
    p=CommandProcessor([t('obj_aaaaaaaaaaaaaaaa','book'),t('obj_bbbbbbbbbbbbbbbb','book')],
                       grid,robot_xy=[.5,.5])
    result=p.run('подъедь к книге')
    assert result.status=='ambiguous'
    assert result.path_xy==[]
