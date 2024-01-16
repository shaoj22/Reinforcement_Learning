# MCRMFS启发式算法的主程序
# 2022.10.16
import numpy as np
import copy
from MCRMFS_Instance import Instance_mini
import matplotlib.pyplot as plt
from MCRMFS_tools import Draw_Opt

# 定义系统环境的类，用于记录每个T对应的系统环境
problem = Draw_Opt(Instance_mini)
def Draw(system_Environment_List):
    fig = plt.figure()
    for t in range(len(system_Environment_List)):
        plt.clf()
        ax = plt.subplot(111)
        problem._draw_t(ax, system_Environment_List[t].p, system_Environment_List[t].s, system_Environment_List[t].h, system_Environment_List[t].e, system_Environment_List[t].g)
        plt.draw()
        plt.pause(2)
    plt.clf()

class system_Env(object):
    def __init__(self):
        self.W = 5
        self.L = 5
        self.m = 3
        self.n = 3

        self.picking_Stations = []
        self.target_Shelves = []
        self.empty_Position = []
        self.sub_Target_Shelves = []

        self.AGV_number = 2
        self.AGV_num = [0] * self.AGV_number
        self.AGV_Coordinates = []
        self.AGV_Present_Mission = []
        self.AGV_Present_Mission_Destination = []
        self.AGV_Time_Left = []
        self.AGV_Index = []
        # 画图的参数
        self.col_num = 5
        self.row_num = 5
        self.T = 0
        self.R = 0
        self.N = 25
        self.A = np.array((self.N,self.N))
        self.p = [0] * self.N
        self.s = [0] * self.N
        self.h = [0] * self.N
        self.e = [0] * self.N
        self.g = [0] * self.N

class MCRMFS(object):
    # 初始化：目标货架List，AGV个数List[0,0,0]，拣选站List,空位置List
    def __init__(self,target_Shelves,AGV_number,picking_Stations,empty_Position,W,L,m,n,A,s):
        self.target_Shelves = target_Shelves
        self.AGV_number = AGV_number # AGV数量
        self.AGV_num = [0] * self.AGV_number # AGV的工作状态列表
        self.picking_Stations = picking_Stations
        self.empty_Position = empty_Position
        self.W = W
        self.L = L
        self.m = m
        self.n = n
        self.coordinates_And_Indexes = self.init_Coordinates_And_Indexes(self.W,self.L,self.m,self.n)
        self.A = A
        self.s = s
        # self.dis_Matrix = self.distance_Matrix(self.A,self.s,self.coordinates_And_Indexes)

    # 模块-1————弗洛伊德算法
    """
    def Floyd(self,d):
        n = d.shape[0]
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    d[i][j] = min(d[i][j], d[i][k] + d[k][j])
        return d   # 输出距离矩阵
    """
    # 模块0————距离矩阵的计算
    # 输出坐标和索引矩阵，仓库的长，仓库的宽，输出任意两点的距离矩阵
    """
    def distance_Matrix(self,A,s,CIA):
        # 计算通道到通道的距离
        M = 10000
        m, n = A.shape
        for i in range(m):
            for j in range(n):
                if (A[i, j] == 0 and i != j) or s[i] == 0 or s[j] == 0:
                    A[i, j] = M
        dis_M = self.Floyd(A)
        return dis_M

    """
    """
    def distance_Matrix(self,CAI,W,L):
        dis_M = np.zeros((W*L,W*L))
        for i in range(W*L):
            for j in range(W*L):
                index1 = np.argwhere( CAI == i )
                x1 = index1[0][0]
                y1 = index1[0][1]
                index2 = np.argwhere( CAI == j )
                x2 = index2[0][0]
                y2 = index2[0][1]
                if x1 - x2 >=0:
                    dis1 = x1-x2
                else:
                    dis1 = x2- x1
                if y1- y2 >=0:
                    dis2 = y1- y2
                else:
                    dis2 = y2 - y1
                dis = dis1 + dis2
                dis_M[i,j] = dis
        return dis_M
    """

    # 初始化坐标和索引
    def init_Coordinates_And_Indexes(self,W,L,m,n):
        # 总个数
        N = W*L
        # 坐标与索引矩阵
        coordinates_And_Indexes = np.zeros((W,L))
        for i in range(W):
            for j in range(L):
                coordinates_And_Indexes[i,j] = i * L + j
        return coordinates_And_Indexes

    # 模块4————系统环境的更新
    def update_System_Environment(self,a,b): # 某种货架从a到b空位置
        pass


    # 模块5————计算某个目标货架的阻碍货架以及目标货架和拣选台返回货架
    def init_Obstruction_Of_Shelves(self,a,present_Empty_Position): # 输入目标货架a，输出需要移动的阻碍货架bList
        b1 = []
        b2 = []
        b3 = []
        b4 = []
        # 找到目标货架的坐标
        index = np.argwhere(self.coordinates_And_Indexes == a)
        # 纵坐标
        y = index[0,0]
        # 横坐标
        x = index[0,1]
        # 判断向上的阻碍货架个数
        for i in range(self.m):
            if (y-(i+1))%(self.m+1) == 0:
                break
            else:
                if self.coordinates_And_Indexes[y-(i+1),x] not in present_Empty_Position:
                    b1.append(int(self.coordinates_And_Indexes[y-(i+1),x]))
        # 调整顺序
        b11 = []
        for i in range(len(b1)):
            b11.append(b1[len(b1)-1-i])


        # 判断向下的阻碍货架个数
        for i in range(self.m):
            if (y+(i+1))%(self.m+1) == 0:
                break
            else:
                if self.coordinates_And_Indexes[y+(i+1),x] not in present_Empty_Position:
                    b2.append(int(self.coordinates_And_Indexes[y+(i+1),x]))
        # 调整顺序
        b22 = []
        for i in range(len(b2)):
            b22.append(b2[len(b2) - 1 - i])


        # 判断向左边的阻碍货架个数
        for i in range(self.n):
            if (x - (i + 1)) % (self.n + 1) == 0:
                break
            else:
                if self.coordinates_And_Indexes[y,x - (i + 1)] not in present_Empty_Position:
                    b3.append(int(self.coordinates_And_Indexes[y,x - (i + 1)]))
        b33 = []
        for i in range(len(b3)):
            b33.append(b3[len(b3) - 1 - i])


        # 判断向右边的阻碍货架个数
        for i in range(self.n):
            if (x + (i + 1)) % (self.n + 1) == 0 :
                break
            else:
                if self.coordinates_And_Indexes[y ,x + (i + 1) ] not in present_Empty_Position:
                    b4.append(int(self.coordinates_And_Indexes[y ,x + (i + 1) ]))
        b44 = []
        for i in range(len(b4)):
            b44.append(b4[len(b4) - 1 - i])


        # 判断阻碍货架个数，选择最少的进行搬运
        b = []
        num = []
        num1 = []
        num.append(b11)
        num.append(b22)
        num.append(b33)
        num.append(b44)
        num1.append(len(b11))
        num1.append(len(b22))
        num1.append(len(b33))
        num1.append(len(b44))
        minnum = min(num1)
        for i in range(4):
            if num1[i] == minnum:
                b = num[i]
        # 阻碍货架计算完毕，准备添加目标货架和目标货架至一个空位置
        # 添加目标货架
        b.append(a)
        b.append(1)
        """
        # 添加返回的拣选台位置：目标货架到这个拣选台的距离最短
        dis = []
        for i in range(len(self.picking_Stations)):
            dis.append(self.dis_Matrix[a, self.picking_Stations[i]])
        mindis = min(dis)
        for i in range(len(self.picking_Stations)):
            if mindis == self.dis_Matrix[a, self.picking_Stations[i]]:
                b.append(self.picking_Stations[i])
        """
        return b
    # —————————————————————————————————————————————————————————————————————————————————————————————————————————————分割线
    # --------------------------------系统的初始化------------------------------
    # 初始化系统的基本参数：包括目标货架、拣选台、空位置、AGV的工作状态————模块1
    def init_The_System(self,target_Shelves,picking_Stations,empty_Position,AGV_num):
        init_Target_Shelves = copy.deepcopy(target_Shelves) # 初始目标货架
        init_Picking_Stations = copy.deepcopy(picking_Stations) # 初始拣选台
        init_Empty_Position = copy.deepcopy(empty_Position) # 初始空位置
        init_AGV_num = copy.deepcopy(AGV_num)  # 初始机器人的工作状态[0,0,...]
        return init_Target_Shelves,init_Picking_Stations,init_Empty_Position,init_AGV_num

    # --------------------------------AGV信息参数的初始化------------------------------
    # 初始化AGV的基本参数：包括当前需要移动的货架、移动此货架的剩余时间、移动此货架的目的地————模块2
    def init_The_AGV(self):
        # 初始化AGV当前任务需要搬运的货架————二维List
        AGV_Shelves = []
        AGV_Shelves_Destinations = []
        AGV_coordinates = []
        AGV_Move_List = []
        present_T = [0] * self.AGV_number
        AGV_Index = [20,24]
        AGV_Have_Target_Shelves = [0] * self.AGV_number
        for i in range(len(self.AGV_num)):
            AGV_Shelves.append([])
            AGV_Shelves_Destinations.append([])
            AGV_Move_List.append([])
        # 初始化AGV当前任务需要搬运的货架的剩余完成时间————一维列表
        AGV_Time_Left = [0]*len(self.AGV_num)

        present_AGV_Mission = [0] * len(self.AGV_num)  # 记录当前每个机器人完成的任务
        next_Shelves = [0] * len(self.AGV_num)  # 用于判断是否可以执行下一个货架
        next_Mission = [0] * len(self.AGV_num)  # 用于判断是否可以执行下一个任务
        finished_Shelves = [0] * len((self.AGV_num))  # 用于记录当前完成的货架
        finished_Shelves_Destinations = [0] * len((self.AGV_num))  # 用于记录当前完成货架的目的地位置
        return AGV_Shelves,AGV_Time_Left,AGV_Shelves_Destinations,present_AGV_Mission,next_Shelves,next_Mission,finished_Shelves,finished_Shelves_Destinations,AGV_coordinates,AGV_Move_List, present_T,AGV_Index,AGV_Have_Target_Shelves

    # -----------------------------------t=0时的初始化-----------------------------------
    # 初始化T=0————模块3
    def init_T_Zero(self,init_Target_Shelves, init_Picking_Stations, init_Empty_Position, init_AGV_num):
        # 这几项为当前系统状态的定义项
        present_Target_Shelves = copy.deepcopy(init_Target_Shelves)
        present_Picking_Stations = copy.deepcopy(init_Picking_Stations)
        present_Empty_Position = copy.deepcopy(init_Empty_Position)
        present_AGV_num = copy.deepcopy(init_AGV_num)
        # 下面为一些中间参数的定义项
        return present_Target_Shelves,present_Picking_Stations,present_Empty_Position,present_AGV_num

    # -----------------------------------机器人的状态更新-----------------------------------
    # 若机器人完成了某个任务，那么它将可以接受下一个任务：next_Shelves[i]=1,并且记录被完成的任务以及将其更新为空位置————模块4
    def robot_Updata(self,AGV_Time_Left,t,AGV_Shelves,next_Shelves,finished_Shelves,present_Empty_Position):
        for i in range(len(AGV_Time_Left)):
            if AGV_Time_Left[i] == 0 and t !=0:
                next_Shelves[i] = 1
                for j in range(len(AGV_Shelves[i])):
                    if AGV_Shelves[i][j] !=0:
                        finished_Shelves[i] = AGV_Shelves[i][j]
                        if finished_Shelves[i] not in self.picking_Stations:  # 这里不能将拣选站加入空位置中
                            present_Empty_Position.append(finished_Shelves[i])
                            #print("AGV_",i,"完成了任务",finished_Shelves[i])
                        AGV_Shelves[i][j] = 0
                        break
            if AGV_Time_Left[i] != 0:
                next_Shelves[i] = 0
                #print("AGV_", i, "正在完成任务中")
        return next_Shelves, finished_Shelves, present_Empty_Position, AGV_Shelves

    # -----------------------------------任务的状态更新-----------------------------------
    # 若机器人完成了某个任务，那么这个任务需要从present_Target_Shelves中移除，从而更新任务的状态————模块5
    def mission_Updata(self):
        pass

    # ---------------------------------任务的分配--------------------------------------
    # 若子任务队列为空了，就需要分解目标任务为多个子任务————模块6
    def mission_Allocation(self,present_Target_Shelves,present_Sub_Target_Shelves,present_Empty_Position):
        # 判断应该分解那个任务为子任务
        for i in range(len(present_Target_Shelves)):
            if present_Target_Shelves[i] !=0:
                present_Sub_Target_Shelves = self.init_Obstruction_Of_Shelves(present_Target_Shelves[i], present_Empty_Position)
                present_Target_Shelves[i] = 0
                break
        return present_Sub_Target_Shelves

    # ---------------------------------子任务的分配--------------------------------------
    # 若机器人完成了某个任务，那么要为机器人分配一个新的子任务————模块7
    def sub_Mission_Allocation(self,present_AGV_num,present_Sub_Target_Shelves,present_Target_Shelves,present_Empty_Position,mission_AGV,AGV_Shelves,AGV_Index):
        print("--" * 20, "子任务分配过程", "--" * 20)
        posi = 0
        for i in range(len(present_AGV_num)):
            if present_AGV_num[i] == 0:
                # 此处应该判断子任务是否已经都被完成，若都完成则应该进行任务分配，调用模块————6
                if sum(present_Sub_Target_Shelves) == 0:
                    present_Sub_Target_Shelves = self.mission_Allocation(present_Target_Shelves,present_Sub_Target_Shelves,present_Empty_Position)
                    # 将子任务分配给此AGV
                for j in range(len(present_Sub_Target_Shelves)):
                    if present_Sub_Target_Shelves[j] != 0 :
                        if j == 0:
                            AGV_Shelves[i].append(AGV_Index[i])
                            mission_AGV[i] = present_Sub_Target_Shelves[j]
                            AGV_Shelves[i].append(present_Sub_Target_Shelves[j])
                            posi = 1
                        else:
                            mission_AGV[i] = present_Sub_Target_Shelves[j]
                            AGV_Shelves[i] = [present_Sub_Target_Shelves[j]]
                        if present_Sub_Target_Shelves[j+1] in self.picking_Stations:
                            AGV_Shelves[i].append(present_Sub_Target_Shelves[j+1])
                            present_Sub_Target_Shelves[j+1] = 0
                        present_AGV_num[i] = 1
                        print("子任务", mission_AGV[i], "被分配给了AGV-", i, "----需要移动的货架有：", AGV_Shelves[i])
                        present_Empty_Position.append(mission_AGV[i])
                        present_Sub_Target_Shelves[j] = 0
                        break
        return present_Sub_Target_Shelves,mission_AGV,AGV_Shelves,present_AGV_num,present_Target_Shelves,present_Empty_Position,posi

    # ---------------------------------机器人的调度--------------------------------------
    # 输入当前需要移动的货架，输出该货架要到的货位编号finished_Position,以及需要花费的时间finished_Time————模块8
    def AGV_Scheduling(self,present_Move,present_Empty_Position,A,AGV_Shelves,agv_i,posi):
        FTime = []
        RList = []
        # 暂存点前往目标货架
        if posi == 1:
            finished_Position = AGV_Shelves[int(agv_i)][1]
            L, finished_Time = self.Path_Planning(A, present_Empty_Position, present_Move, finished_Position)
        else:
            # 目标货架与阻碍货架的去向不同
            if present_Move in self.target_Shelves:
                finished_Position = self.picking_Stations[0]
                L,finished_Time = self.Path_Planning(A, present_Empty_Position, present_Move, finished_Position)
            else:
                for i in range(len(present_Empty_Position)):
                    # 一个一个空货架的计算位置。
                    a, b = self.Path_Planning(A, present_Empty_Position, present_Move, present_Empty_Position[i])
                    RList.append(a)
                    FTime.append(b)
                finished_Time = min(FTime)
                index = FTime.index(finished_Time)
                finished_Position = present_Empty_Position[index]
                L = RList[index]
        return finished_Position, finished_Time,L,posi

    # ---------------------------------路径规划--------------------------------------
    # 路径规划，输入:可达矩阵、当前的空位置，出发点和到达点，输出路线、时间————模块9
    def Path_Planning(self,A,present_Empty_Position,present_Move,finished_Position):
        route_List = []
        tabu_List = []
        route_List.append(present_Move)
        tabu_List.append(present_Move)
        while(present_Move != finished_Position):
            pob_Move = []
            Index = np.argwhere(self.coordinates_And_Indexes == finished_Position)
            Index1 = np.argwhere(self.coordinates_And_Indexes == present_Move)
            X = Index[0, 1]
            Y = Index[0, 0]
            X1 = Index1[0, 1]
            Y1 = Index1[0, 0]
            if X1!=0 and present_Move-1 not in tabu_List:
                pob_Move.append(present_Move-1)
            if X1!=self.L-1 and (present_Move+1 not in tabu_List or present_Move+1 in self.picking_Stations):
                pob_Move.append(present_Move + 1)
            if Y1 != self.W-1 and present_Move + self.L not in tabu_List:
                pob_Move.append(present_Move + self.L)
            if Y1 != 0 and present_Move - self.L not in tabu_List:
                pob_Move.append(present_Move - self.L)
            if len(pob_Move) == 0:
                route_List.pop()
                present_Move = route_List[-1]
                continue
            dis = []
            out = []
            for i in range(len(pob_Move)):
                # 找到下一步货架的编号
                index = np.argwhere(self.coordinates_And_Indexes == pob_Move[i] )
                # 纵坐标
                y = index[0, 0]
                # 横坐标
                x = index[0, 1]
                if A[present_Move,pob_Move[i]] != 1:
                    out.append(pob_Move[i])
                if pob_Move[i] not in present_Empty_Position and y%(self.m+1) != 0 and x%(self.n+1) != 0:
                    out.append(pob_Move[i])
                if X - x >= 0:
                    dis1 = X - x
                else:
                    dis1 = x-X
                if Y-y >= 0:
                    dis2 = Y-y
                else:
                    dis2 = y- Y
                dis12 = dis1 + dis2
                dis.append(dis12)
            # 选择距离最小的（如果有多个候选点）
            j = 0
            while j < len(pob_Move):
                if pob_Move[j] in out :
                    pob_Move.pop(j)
                    dis.pop(j)
                else:
                    j += 1
            choice = min(dis)
            present_Move = copy.deepcopy(pob_Move[dis.index(choice)])
            route_List.append(present_Move)
            tabu_List.append(present_Move)
        finished_Time = len(route_List)-1
        return route_List,finished_Time

    # ---------------------------------子任务分配、机器人调度、路径规划的集成--------------------------------------
    # 若机器人完成了某个任务，那么要为机器人分配一个新的子任务————模块10
    def sub_Mission_Allocation_Integrate(self,t ,present_T, AGV_Time_Left, AGV_Shelves, finished_Shelves,finished_Shelves_Destinations,present_Empty_Position,AGV_Move_List,AGV_Index,posi,AGV_Have_Target_Shelves,present_Target_Shelves):
        AGV_coordinates = []
        AGV_Shelves_Coordinates = []
        for i in range(len(AGV_Time_Left)):
            if AGV_Time_Left[i] == 0:
                present_T[i] = 0
        print("--" * 20, "AGV调度过程与路径规划过程", "--" * 20)
        for i in range(len(AGV_Time_Left)):
            if AGV_Time_Left[i] == 0:
                for j in range(len(AGV_Shelves[i])):
                    if AGV_Shelves[i][j] != 0:  # 开始规划下一个货架的路径
                        present_Move = AGV_Shelves[i][j]
                        finished_Shelves[i] = present_Move
                        # 规路规划当前货架应该前往那个空位置
                        finished_Shelves_Destinations[i], AGV_Time_Left[i], move_List ,posi = self.AGV_Scheduling(
                            present_Move, present_Empty_Position, self.A,AGV_Shelves,int(i),posi)
                        # 路径规划后就占用了一个空位置
                        print("AGV调度————","当前货架", present_Move, "移动去了", finished_Shelves_Destinations[i])
                        print("路径规划————","移动的过程为：",move_List)
                        # 目标货架位置更新
                        if present_Move in self.target_Shelves:
                            AGV_Have_Target_Shelves[i] = 1
                        else:
                            AGV_Have_Target_Shelves[i] = 0
                        AGV_Move_List[i] = move_List
                        if finished_Shelves_Destinations[i] in present_Empty_Position:
                            present_Empty_Position.remove(finished_Shelves_Destinations[i])
                        break
        # 记录当前AGV的位置
        for point in range(25):
            if point == AGV_Move_List[0][present_T[0]] or point == AGV_Move_List[1][present_T[1]]:
                AGV_coordinates.append(1)
                if point == AGV_Move_List[0][present_T[0]]:
                    AGV_Index[0] = point
                if point == AGV_Move_List[1][present_T[1]]:
                    AGV_Index[1] = point
            else:
                AGV_coordinates.append(0)
        # 记录目标货架的位置
        for point in range(25):
            for i in range(len(AGV_Have_Target_Shelves)):
                if AGV_Have_Target_Shelves == 1 or (point in present_Target_Shelves):
                    AGV_Shelves_Coordinates.append(1)
                else:
                    AGV_Shelves_Coordinates.append(0)
        present_T[0] = present_T[0] + 1
        present_T[1] = present_T[1] + 1
        #print("分配任务和路径规划“后”空位置的集合：", present_Empty_Position)
        return finished_Shelves,finished_Shelves_Destinations,AGV_Time_Left,present_Empty_Position,present_T ,AGV_Move_List, AGV_coordinates,AGV_Shelves_Coordinates,AGV_Index

    # 主程序run
    def run(self):
        # 1. 初始化时间
        t = 0
        # 2. 初始化系统的基本参数：包括目标货架、拣选台、空位置、AGV的工作状态————模块1
        init_Target_Shelves, init_Picking_Stations, init_Empty_Position, init_AGV_num = self.init_The_System(
            self.target_Shelves, self.picking_Stations, self.empty_Position, self.AGV_num)
        # 3. 初始化AGV的基本参数：包括当前需要移动的货架、移动此货架的剩余时间、移动此货架的目的地————模块2
        AGV_Shelves, AGV_Time_Left, AGV_Shelves_Destinations, present_AGV_Mission, next_Shelves, next_Mission, finished_Shelves, finished_Shelves_Destinations, AGV_coordinates, AGV_Move_List , present_T ,AGV_Index,AGV_Have_Target_Shelves= self.init_The_AGV()
        # 4. 初始化T=0————模块3
        if t == 0:
            present_Target_Shelves, present_Picking_Stations, present_Empty_Position, present_AGV_num = self.init_T_Zero(
                init_Target_Shelves, init_Picking_Stations, init_Empty_Position, init_AGV_num)
            # 以下为一些中间参数
            mission_AGV = [0] * len(self.AGV_num)  # 用于记录AGV当前任务的编号
            present_Sub_Target_Shelves = []  # 用于记录子任务
            system_Environment_List = []
        # 5. 开始处理
        while(sum(present_Target_Shelves) != 0 or sum(present_AGV_num) != 0): # 所有目标货架都应该为空，并且机器人都完成了任务
            # 6. 若机器人完成了某个任务，那么它将可以接受下一个任务：next_Shelves[i]=1,并且记录被完成的任务以及将其更新为空位置————模块4
            next_Shelves, finished_Shelves, present_Empty_Position, AGV_Shelves = self.robot_Updata(AGV_Time_Left,t,AGV_Shelves,next_Shelves,finished_Shelves,present_Empty_Position)
            #print("分配任务和路径规划“前”空位置的集合：", present_Empty_Position)

            # 3.1.2 判断每个AGV的当前任务是否完成
            for i in range(len(AGV_Shelves)): # 判断
                if sum(AGV_Shelves[i]) == 0 :  # 意味着这个AGV的当前任务已经完成
                    present_AGV_num[i] = 0
                    print("AGV-", i, "处于空闲")
                    AGV_Shelves = []
                    for i in range(len(self.AGV_num)):
                        AGV_Shelves.append([])
                if sum(AGV_Shelves[i]) != 0:
                    print("AGV-", i, "正在工作")

            # 8. 子任务分配————模块7
            present_Sub_Target_Shelves, mission_AGV, AGV_Shelves, present_AGV_num, present_Target_Shelves, present_Empty_Position,posi= self.sub_Mission_Allocation(
                present_AGV_num, present_Sub_Target_Shelves, present_Target_Shelves, present_Empty_Position,
                mission_AGV, AGV_Shelves,AGV_Index)

            # 9. 子任务分配、机器人调度、路径规划的集成————模块7~9，7为子任务分配，8为机器人调度，9为路径规划————模块10
            finished_Shelves,finished_Shelves_Destinations,AGV_Time_Left,present_Empty_Position ,present_T,AGV_Move_List,AGV_coordinates,AGV_Shelves_Coordinates,AGV_Index= self.sub_Mission_Allocation_Integrate(t,present_T ,AGV_Time_Left, AGV_Shelves, finished_Shelves,finished_Shelves_Destinations,present_Empty_Position,AGV_Move_List,AGV_Index,posi,AGV_Have_Target_Shelves,present_Target_Shelves)
            # 10. 更新系统环境类
            # 10.1 更新基本数据
            system_Environment = system_Env()
            system_Environment.picking_Stations = present_Picking_Stations.copy()
            system_Environment.target_Shelves = present_Target_Shelves.copy()
            system_Environment.empty_Position = present_Empty_Position.copy()
            system_Environment.sub_Target_Shelves = present_Sub_Target_Shelves.copy()
            system_Environment.AGV_num = present_AGV_num.copy()
            system_Environment.AGV_Present_Mission = finished_Shelves.copy()
            system_Environment.AGV_Present_Mission_Destination = finished_Shelves_Destinations.copy()
            system_Environment.AGV_Time_Left = AGV_Time_Left.copy()
            # 10.2 更新画图数据
            system_Environment.T = t
            system_Environment.R = len(AGV_Time_Left)
            # 拣选站
            for i1 in range(25):
                if i1 in present_Picking_Stations:
                    system_Environment.p[i1] = 1
                else:
                    system_Environment.p[i1] = 0
            # 目标货架位置
            system_Environment.g = AGV_Shelves_Coordinates
            for i2 in range(25):
                if i2 in present_Target_Shelves and i2 !=0:
                    system_Environment.g[i2] = 1
            # 通道
            system_Environment.s = Instance_mini.s
            # 机器人位置
            system_Environment.h = AGV_coordinates
            # 空位置
            for i2 in range(25):
                if i2  in present_Empty_Position or Instance_mini.s[i2] == 1:
                    system_Environment.e[i2] = 0
                else:
                    system_Environment.e[i2] = 1
            # 把类存储进列表中
            system_Environment_List.append(system_Environment)
            # 更新系统时间
            t = t + 1
            print("----------"*10)
            print("当前的时间:",t)
            # 更新每个机器人的剩余完成时间
            for i in range(len(self.AGV_num)):
                if AGV_Time_Left[i] != 0:
                    AGV_Time_Left[i] = AGV_Time_Left[i] -1
        return system_Environment_List

# 输入基本参数
W = 10  # 仓库的宽
L = 10  # 仓库的长
m = 3  # 块布局的宽
n = 5  # 块布局的长
# 可达矩阵的信息
A = Instance_mini.A
s = Instance_mini.s
# 目标货架12，机器人2个，拣选站1，空位置，16,19,20,A为可达矩阵
problem1 = MCRMFS([6,12,13],10,[1],[7,8],W,L,m,n,A,s)
# run
XX = problem1.run()
# 画图
Draw(XX)