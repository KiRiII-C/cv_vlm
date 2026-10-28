"""Deterministic Russian/English language-to-goal and collision-checked 2D A*.

Goal proposal is not a robot actuation command. Nav2 accepts goals only after
localization, map-frame consistency, costmap validation and lifecycle checks.
"""
from __future__ import annotations
from dataclasses import dataclass
from heapq import heappush, heappop
import math, re
import numpy as np

@dataclass
class GoalResult:
    query: str
    status: str
    target_ids: list
    goal_xy: list | None
    path_xy: list
    explanation: str
    source: str = 'offline_astar_not_nav2'

class OccupancyGrid:
    def __init__(self, occupied, resolution=0.10, origin=(0.,0.)):
        a=np.asarray(occupied,dtype=bool)
        if a.ndim!=2 or not all(n>2 for n in a.shape):raise ValueError('Invalid occupancy grid')
        self.grid=a;self.res=float(resolution);self.origin=np.array(origin,dtype=float)
        if self.res<=0:raise ValueError('Invalid resolution')
    def cell(self,xy):
        ix=np.floor((np.asarray(xy)[:2]-self.origin)/self.res).astype(int)
        return int(ix[1]),int(ix[0])
    def point(self,rc):
        r,c=rc;return [float(self.origin[0]+(c+.5)*self.res),float(self.origin[1]+(r+.5)*self.res)]
    def inflated(self,radius_m):
        from scipy.ndimage import binary_dilation
        k=int(math.ceil(radius_m/self.res));
        if k<=0:return self.grid.copy()
        yy,xx=np.mgrid[-k:k+1,-k:k+1]
        return binary_dilation(self.grid,structure=(xx*xx+yy*yy)<= (radius_m/self.res)**2 +1e-9)
    def astar(self,start,goal,radius=.18):
        blocked=self.inflated(radius);s=self.cell(start);g=self.cell(goal)
        h,w=blocked.shape
        if any(r<0 or c<0 or r>=h or c>=w for r,c in (s,g)):return []
        if blocked[s] or blocked[g]:return []
        dist={s:0.};parent={};pq=[(math.dist(s,g),0.,s)]
        neighbors=[(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
        while pq:
            _,cost,cur=heappop(pq)
            if cost>dist[cur]+1e-8:continue
            if cur==g:
                path=[cur]
                while path[-1] in parent:path.append(parent[path[-1]])
                return [self.point(c) for c in reversed(path)]
            r,c=cur
            for dr,dc in neighbors:
                nr,nc=r+dr,c+dc
                if not (0<=nr<h and 0<=nc<w) or blocked[nr,nc]:continue
                if dr and dc and (blocked[r+dr,c] or blocked[r,c+dc]):continue
                nxt=(nr,nc);ncost=cost+math.hypot(dr,dc)
                if ncost<dist.get(nxt,1e50):
                    dist[nxt]=ncost;parent[nxt]=cur
                    heappush(pq,(ncost+math.dist(nxt,g),ncost,nxt))
        return []

def _object_name(t):
    return (t.label+' '+t.description+' '+' '.join(str(x) for x in t.attributes.values())).lower()

class CommandProcessor:
    COLORS={'красн':'red','син':'blue','зел':'green','желт':'yellow','бел':'white','черн':'black',
            'red':'red','blue':'blue','green':'green','yellow':'yellow','фиолет':'purple','purple':'purple'}
    SYN={'стул':'chair','стуль':'chair','chair':'chair','кресл':'chair',
         'стол':'table','table':'table','desk':'table','dining table':'table',
         'диван':'sofa','sofa':'sofa','couch':'sofa',
         'ламп':'lamp','lamp':'lamp','шкаф':'cabinet','cabinet':'cabinet',
         'растен':'plant','plant':'plant','двер':'door','door':'door',
         'клавиатур':'keyboard','keyboard':'keyboard',
         'монитор':'tv','телевизор':'tv','экран':'tv','screen':'tv','tv':'tv',
         'книг':'book','book':'book',
         'ноутбук':'laptop','laptop':'laptop',
         'мышк':'mouse','mouse':'mouse',
         'чашк':'cup','кружк':'cup','cup':'cup',
         'бутылк':'bottle','bottle':'bottle',
         'телефон':'cell phone','phone':'cell phone',
         'пульт':'remote','remote':'remote'}
    def __init__(self,tracks,grid:OccupancyGrid,robot_xy=(.5,.5),robot_yaw=0.,radius=.18):
        self.tracks=[t for t in tracks if t.state=='confirmed']
        self.grid=grid;self.robot_xy=np.asarray(robot_xy,dtype=float)
        self.yaw=float(robot_yaw);self.radius=radius
    def _matches(self,q):
        q=q.lower()
        uuid_match=re.search(r'obj_[a-f0-9]{16}',q)
        if uuid_match:
            return [t for t in self.tracks if t.object_id.lower()==uuid_match.group()]
        cats={v for k,v in self.SYN.items() if k in q}
        colors={v for k,v in self.COLORS.items() if k in q}
        # Reject unrecognized object references rather than silently search every object.
        if not cats and not colors:return []
        return [t for t in self.tracks if (not cats or t.label.lower() in cats)
                and (not colors or any(color in _object_name(t) for color in colors))]
    def _candidates(self,xy,rmin=.6,rmax=1.1):
        for r in np.arange(rmin,rmax+.01,.15):
            for a in np.linspace(-math.pi,math.pi,32,endpoint=False):
                yield np.array(xy[:2])+r*np.array([math.cos(a),math.sin(a)])
    def _pick(self,point):
        options=[]
        for xy in self._candidates(point):
            path=self.grid.astar(self.robot_xy,xy,self.radius)
            if path:options.append((sum(math.dist(a,b) for a,b in zip(path,path[1:])),xy,path))
        return min(options,key=lambda x:x[0]) if options else None
    def run(self,query):
        q=query.strip().lower()
        if not q:return GoalResult(query,'unresolved',[],None,[],'Пустой запрос')
        if 'между' in q or 'between' in q:
            names=self._matches(q)
            if len(names)!=2:
                return GoalResult(query,'unresolved',[t.object_id for t in names],None,[],
                                  'Для команды «между» требуются ровно два однозначно выбранных объекта')
            midpoint=(names[0].center[:2]+names[1].center[:2])/2
            path=self.grid.astar(self.robot_xy,midpoint,self.radius)
            if not path:return GoalResult(query,'unreachable',[t.object_id for t in names],midpoint.tolist(),[],
                                           'Середина между объектами недоступна')
            return GoalResult(query,'planned',[t.object_id for t in names],midpoint.tolist(),path,
                              'Геометрическая середина двух объектов, проверена на карте занятости')
        names=self._matches(q)
        if len(names)==0:return GoalResult(query,'not_found',[],None,[],'В памяти нет подтверждённого объекта, подходящего под запрос')
        if len(names)>1:
            return GoalResult(query,'ambiguous',[t.object_id for t in names],None,[],
                              'Найдено несколько объектов. Нужны цвет, номер ID или пространственное уточнение')
        target=names[0];pt=target.center[:2]
        if any(p in q for p in ('справа от','right of','слева от','left of')):
            right=np.array([math.sin(self.yaw),-math.cos(self.yaw)])
            if 'слева' in q or 'left of' in q:right=-right
            p=pt+right*(.55+.5*float(np.max(target.high[:2]-target.low[:2])))
            path=self.grid.astar(self.robot_xy,p,self.radius)
            if not path:return GoalResult(query,'unreachable',[target.object_id],p.tolist(),[],
                                          'Справа/слева определено в системе heading робота, но целевая точка недоступна')
            return GoalResult(query,'planned',[target.object_id],p.tolist(),path,
                              'Направление справа/слева задано по yaw робота; маршрут проверен A*')
        candidate=self._pick(pt)
        if candidate is None:return GoalResult(query,'unreachable',[target.object_id],None,[],
                                                'Нет доступного подхода с заданным радиусом безопасности')
        _,xy,path=candidate
        return GoalResult(query,'planned',[target.object_id],xy.tolist(),path,
                          'Выбрана достижимая точка подхода к экземпляру объекта')
