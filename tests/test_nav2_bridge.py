import pytest
from semantic_map.planner import GoalResult
from semantic_map.nav2_bridge import goal_as_dict,send_nav2_goal

def test_valid_goal_not_misreported_as_execution():
    g=GoalResult('go','planned',['obj1'],[2.,3.],[[1.,1.],[2.,3.]],'safe')
    out=goal_as_dict(g)
    assert out['pose']['position']=={'x':2.,'y':3.,'z':0.}
    assert out['header']['frame_id']=='map'
    with pytest.raises(RuntimeError,match='ROS2 Jazzy'):
        send_nav2_goal(g)

def test_ambiguous_goal_rejected():
    with pytest.raises(ValueError):
        goal_as_dict(GoalResult('go','ambiguous',['1','2'],None,[],'not unique'))
