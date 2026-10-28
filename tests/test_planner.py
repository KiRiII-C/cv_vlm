import numpy as np
from semantic_map.models import Track
from semantic_map.planner import OccupancyGrid,CommandProcessor
from semantic_map.scene_graph import compute_relations

def tr(oid,label,x,y,color):
    p=np.array([x,y,.6]);return Track(oid,label,p,p-.25,p+.25,None,3,0,4,'confirmed',color+' '+label,0,{'color':color})

def test_goals_and_ambiguity():
    g=OccupancyGrid(np.zeros((90,130),dtype=bool),.1)
    t=[tr('a','chair',2,2,'red'),tr('b','chair',6,2,'red'),tr('c','table',4,5,'brown'),tr('d','plant',8,7,'green')]
    p=CommandProcessor(t,g,robot_xy=(1,1))
    assert p.run('подъедь к красному стулу').status=='ambiguous'
    answer=p.run('подъедь к зелёному растению')
    assert answer.status=='planned' and len(answer.path_xy)>2
    assert p.run('встань между столом и растением').status=='planned'
    assert p.run('подъедь к фиолетовой лампе').status=='not_found'
    assert len(compute_relations(t,near_threshold=20))>=1

def test_obstacles_and_rejected_goals():
    a=np.zeros((50,50),bool);a[:,25]=True
    g=OccupancyGrid(a,.1)
    assert g.astar([.5,.5],[4.,.5])==[]
    p=CommandProcessor([tr('a','chair',4,2,'red')],g,robot_xy=(.5,.5))
    assert p.run('подъедь к красному стулу').status=='unreachable'
