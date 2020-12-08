import heapq,re,time,enum,math
import numpy as np
import matplotlib.pyplot as plt

def read_maze(filename):
    with open(filename,'r',encoding='utf-8') as f:
        maze=[]
        for row,line in enumerate(f.readlines()):
            line=line.strip()
            s=line.find('S')
            if (-1!=s):
                start=(row,s)
            e=line.find('E')
            if (-1!=e):
                end=(row,e)
            maze.append(line)
        return maze, start, end

class node:
    """
    定义搜索的节点类
    """
    def __init__(self,pos,cost=0,estimate=0,ancestor=None):
        self.pos=pos
        self.cost=cost      
        self.estimate=estimate
        self.ancestor=ancestor
    def __lt__(self, rhs):
        if self.cost + self.estimate == rhs.cost + rhs.estimate:
            return self.estimate<rhs.estimate
        else:
            return self.cost+self.estimate<rhs.cost+rhs.estimate
class direction(enum.Enum):
    """
    可以移动的四个方向
    """
    UP=(0,1)
    DOWN=(0,-1)
    RIGHT=(1,0)
    LEFT=(-1,0)
def move(pos,direction):
    """
    移动的结果
    """
    return pos[0]+direction.value[0],pos[1]+direction.value[1]

def Manhattan(curr,end):
    """
    曼哈顿距离
    """
    return abs(curr[0]-end[0])+abs(curr[1]-end[1])
def Chebyshev(curr,end):
    """
    切比雪夫距离
    """
    return max(abs(curr[0]-end[0]),abs(curr[1]-end[1]))
def Euler(curr,end):
    """
    平方
    """
    return math.sqrt((curr[0]-end[0])**2+(curr[1]-end[1])**2)
def Quadratic(curr,end):
    """
    平方
    """
    return (curr[0]-end[0])**2+(curr[1]-end[1])**2
heuristic_methods={'Manhattan':Manhattan,'Chebyshev':Chebyshev,'Euler':Euler,'Quadratic':Quadratic}

def isValid(maze,pos):
    """
    判断是否为可行位置
    """
    x,y=pos
    if x<0 or x>=len(maze) or y<0 or y>=len(maze[0]) or maze[x][y] == '1':
        return False
    else:
        return True

def getRoute(node):
    ret=[]
    while(not node is None):
        ret.append(node)
        node=node.ancestor
    return ret            


def UCS(maze,start,end):
    spatial_complexity=0
    time_complexity=0

    frontier = []
    visited = [[0] * len(maze[0]) for _ in range(len(maze))] 
    visited_list=[]
    heapq.heappush(frontier,node(start))
    while True:
        if len(frontier)==0:
            return False
            
        curr = heapq.heappop(frontier)
        visited_list.append(curr)
        if curr.pos==end:
            print("空间复杂度:{}, 时间复杂度:{}".format(spatial_complexity,time_complexity))
            return [obj.pos for obj in getRoute(curr)],visited,visited_list
        visited[curr.pos[0]][curr.pos[1]]=1
        time_complexity+=1
        # 遍历可能的动作
        for d in direction:
            InFrontier=False
            new_pos=move(curr.pos,d)
            # 节点在frontier中，若cost更小，更新frontier
            for Node in frontier:
                if Node.pos == new_pos:
                    InFrontier=True
                    if Node.cost>curr.cost+1:
                        Node.cost=curr.cost+1
                        Node.ancestor=curr
                        # heapq.heapify(frontier)
                        break
            # 节点不在frontier中，未被访问过，则放入状态空间(堆)中                    
            if isValid(maze,new_pos) and (not InFrontier) and visited[new_pos[0]][new_pos[1]]==0:
                heapq.heappush(frontier,node(new_pos,cost=curr.cost+1,ancestor=curr))
                spatial_complexity=max(spatial_complexity,len(frontier))
            
def A_star(maze, start, end,heuristic=Manhattan):
    spatial_complexity=0
    time_complexity=0

    frontier = []
    visited = [[0] * len(maze[0]) for _ in range(len(maze))] 
    visited_list=[]
    heapq.heappush(frontier,node(start,estimate=heuristic(start,end)))
    while True:
        if len(frontier)==0:
            return False

        # for n in frontier:
        #     print(n.cost,n.estimate,n.cost+n.estimate)
        # print('')            
        curr = heapq.heappop(frontier)
        # print('选出: ',curr.cost,curr.estimate,curr.cost+curr.estimate)
        
        visited_list.append(curr)
        if curr.pos==end:
            print("空间复杂度:{}, 时间复杂度:{}".format(spatial_complexity,time_complexity))
            return [obj.pos for obj in getRoute(curr)],visited,visited_list
        visited[curr.pos[0]][curr.pos[1]]=1
        time_complexity+=1
        # 遍历可能的动作
        for d in direction:
            InFrontier=False
            new_pos=move(curr.pos,d)
            # 节点在frontier中，若cost更小，更新frontier
            for Node in frontier:
                if Node.pos == new_pos:
                    InFrontier=True
                    if Node.cost>curr.cost+1:
                        Node.cost=curr.cost+1
                        Node.ancestor=curr
                        # heapq.heapify(frontier)
                        break
            # 节点不在frontier中，未被访问过，则放入状态空间(堆)中                    
            if (not InFrontier) and isValid(maze,new_pos)  and visited[new_pos[0]][new_pos[1]]==0:
                heapq.heappush(frontier,node(new_pos,cost=curr.cost+1,estimate=heuristic(new_pos,end),ancestor=curr))
                spatial_complexity=max(spatial_complexity,len(frontier))
            
def getCoord(Map,mark):
    X=[]
    Y=[]
    for i in range(len(Map)):
        for j in range(len(Map[0])):
            if Map[i][j] in mark:
                X.append(i)
                Y.append(j)
    return X,Y
def draw_map(Map,method,color,marker):
    X,Y=Map
    method(Y,-np.array(X),c=color,marker=marker)