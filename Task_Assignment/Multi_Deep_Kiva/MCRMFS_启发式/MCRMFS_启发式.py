# MCRMFS启发式算法的主程序
# 2022.10.16
import numpy as np
import copy
from MCRMFS_Instance import Instance_mini
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
    # 模块1————基本数据初始化
    # 初始化
    def init_Status(self):
        # 初始化AGV当前任务需要搬运的货架————二维List
        AGV_Shelves = []
        AGV_Shelves_Destinations = []
        for i in range(len(self.AGV_num)):
            AGV_Shelves.append([])
            AGV_Shelves_Destinations.append([])
        # 初始化AGV当前任务需要搬运的货架的剩余完成时间————一维列表
        AGV_Time_Left = [0]*len(self.AGV_num)
        return AGV_Shelves,AGV_Time_Left,AGV_Shelves_Destinations

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

    # 模块2————任务分配
    # 输入当前任务订单List1，和机器人状态List2，记录任务分配的List3，机器人的货架List4
    def mission_Assignment(self,List1,List2,List3,List4,present_Empty_Position):
        for i in range(len(List2)):
            if i == 0:
                for j in range(len(List1)):
                    if j !=0:
                        List3[j] = List1[j]
                        List2[i] = 1
                        # 计算这个任务的所有需移动的货架!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                        List4[i] = [5,6,8]
                        # List4[i] = self.initial_Obstruction_Of_Shelves(List1[j],present_Empty_Position)
                        List1[j] = 0
                        # 任务分配完成后，给机器人分配货架
        return List1,List2,List3,List4


    # 路径规划的子程序，输入:可达矩阵、当前的空位置，出发点和到达点，输出路线、时间
    def sub_Route_Planning(self,A,present_Empty_Position,present_Move,finished_Position):
        route_List = []
        route_List.append(present_Move)
        while(present_Move != finished_Position):
            pob_Move = []
            Index = np.argwhere(self.coordinates_And_Indexes == finished_Position)
            Index1 = np.argwhere(self.coordinates_And_Indexes == present_Move)
            X = Index[0, 1]
            Y = Index[0, 0]
            X1 = Index1[0, 1]
            Y1 = Index1[0, 0]
            if X1!=0 and present_Move-1 not in route_List:
                pob_Move.append(present_Move-1)
            if X1!=self.L-1 and present_Move+1 not in route_List:
                pob_Move.append(present_Move + 1)
            if Y1 != self.W-1 and present_Move + self.L not in route_List:
                pob_Move.append(present_Move + self.L)
            if Y1 != 0 and present_Move - self.L not in route_List:
                pob_Move.append(present_Move - self.L)
            dis = []
            out = []
            for i in range(len(pob_Move)):
                # 找到下一步货架的编号
                index = np.argwhere(self.coordinates_And_Indexes == pob_Move[i] )
                # 纵坐标
                try:
                    y = index[0, 0]
                except:
                    print(pob_Move[i])
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
            try:
                choice = min(dis)
            except:
                print(dis)
            present_Move = copy.deepcopy(pob_Move[dis.index(choice)])
            route_List.append(present_Move)
        finished_Time = len(route_List)-1
        return route_List,finished_Time

    # 模块3————路径规划
    # 输入当前需要移动的货架，输出该货架要到的货位编号finished_Position,以及需要花费的时间finished_Time
    def route_planning(self,present_Move,present_Empty_Position,A):
        FTime = []
        RList = []
        # 目标货架与阻碍货架的去向不同
        if present_Move in self.target_Shelves:
            finished_Position = self.picking_Stations[0]
            L,finished_Time = self.sub_Route_Planning(A, present_Empty_Position, present_Move, finished_Position)
            pass
        else:
            for i in range(len(present_Empty_Position)):
                # 一个一个空货架的计算位置。
                a, b = self.sub_Route_Planning(A, present_Empty_Position, present_Move, present_Empty_Position[i])
                RList.append(a)
                FTime.append(b)
            finished_Time = min(FTime)
            index = FTime.index(finished_Time)
            finished_Position = present_Empty_Position[index]
            L = RList[index]
        return finished_Position, finished_Time,L
    """"
    def route_planning(self,present_Move,present_Empty_Position):
        dis = []
        # 分两种情况，当前移动的到底是目标货架还是阻碍货架还是返回货架
        if present_Move not in self.target_Shelves:
            for i in range(len(present_Empty_Position)):
                dis.append(self.dis_Matrix[present_Move, present_Empty_Position[i]])
            mindis = min(dis)
            for i in range(len(present_Empty_Position)):
                if mindis == self.dis_Matrix[present_Move, present_Empty_Position[i]]:
                    finished_Position = present_Empty_Position[i]
                    finished_Time = copy.deepcopy(mindis)
        else:
            for i in range(len(self.picking_Stations)):
                dis.append(self.dis_Matrix[present_Move, self.picking_Stations[i]])
            mindis = min(dis)
            for i in range(len(self.picking_Stations)):
                if mindis == self.dis_Matrix[present_Move, self.picking_Stations[i]]:
                    finished_Position = self.picking_Stations[i]
                    finished_Time = copy.deepcopy(mindis)

        return finished_Position,finished_Time  # 返回输入货架的对应目的地货架编号，以及要花费的时间
    """
    # 模块4————系统环境的更新
    def update_System_Environment(self,a,b): # 某种货架从a到b空位置
        pass

    # 模块5————计算某个目标货架的阻碍货架以及目标货架和拣选台返回货架
    def initial_Obstruction_Of_Shelves(self,a,present_Empty_Position): # 输入目标货架a，输出需要移动的阻碍货架bList
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




    # --------------------------------系统的初始化------------------------------
    # 初始化基本参数：包括目标货架、拣选台、空位置、AGV的工作状态————模块1
    def init_The_System(self,target_Shelves,picking_Stations,empty_Position,AGV_num):
        init_Target_Shelves = copy.deepcopy(target_Shelves) # 初始目标货架
        init_Picking_Stations = copy.deepcopy(picking_Stations) # 初始拣选台
        init_Empty_Position = copy.deepcopy(empty_Position) # 初始空位置
        init_AGV_num = copy.deepcopy(AGV_num)  # 初始机器人的工作状态[0,0,...]
        return init_Target_Shelves,init_Picking_Stations,init_Empty_Position,init_AGV_num
    # 主程序run
    def run(self):
        # 1. 初始化时间
        t = 0
        # 2. 初始化基本参数：包括目标货架、拣选台、空位置、AGV的工作状态————模块1
        init_Target_Shelves, init_Picking_Stations, init_Empty_Position, init_AGV_num = self.init_The_System(
            self.target_Shelves, self.picking_Stations, self.empty_Position, self.AGV_num)
        #init_Target_Shelves = copy.deepcopy(self.target_Shelves)
        #init_AGV_num = copy.deepcopy(self.AGV_num) # 用于记录AGV是否处于工作状态
        #init_Picking_Stations = copy.deepcopy(self.picking_Stations)
        #init_Empty_Position = copy.deepcopy(self.empty_Position)
        # 进入循坏主体(只要还有任务没完成就不停止)
        # 首先将所有目标货架赋值给记录任务是否完成的mission_List，若完成则为1，若未完成则为0
        #mission_List = [0]*len(init_Target_Shelves)  # 用于记录每个任务是否被完成
        #mission_Assignment_List = [] # 用于记录每个任务都是由那个机器人完成的
        # 初始化AGV
        AGV_Shelves, AGV_Time_Left, AGV_Shelves_Destinations = self.init_Status()

        # 特殊处理t=0的更新状态
        if t == 0:
            # 这几项为当前的系统状态定义项
            present_Target_Shelves = copy.deepcopy(init_Target_Shelves)
            present_AGV_num = copy.deepcopy(init_AGV_num)  # 用于记录AGV是否处于工作状态
            present_Picking_Stations = copy.deepcopy(init_Picking_Stations)
            present_Empty_Position = copy.deepcopy(init_Empty_Position)
            # 以下为一些中间参数
            present_AGV_Mission = [0] * len(self.AGV_num)  # 记录当前每个机器人完成的任务
            next_Shelves = [0] * len(self.AGV_num)  # 用于判断是否可以执行下一个操作
            next_Mission = [0] * len(self.AGV_num)
            mission_AGV = [0] * len(self.AGV_num)  # 用于记录AGV当前任务的编号
            finished_Shelves = [0] * len((self.AGV_num))     # 用于记录当前完成的货架
            finished_Shelves_Destinations = [0] * len((self.AGV_num))   # 用于记录当前完成货架的目的地位置
        while(sum(present_Target_Shelves) != 0 or sum(present_AGV_num) != 0): # 所有目标货架都应该为空0
            # 3. 更新状态————模块2
            # 3.1 更新机器人状态(机器人的任务List、待完成货架List、待完成货架对应的剩余时间List，如果有机器人完成了一次某个货架的搬运，则修改对应List)
            # 3.1.1 判断每个AGV的当前货架的是否完成
            for i in range(len(AGV_Time_Left)):
                if AGV_Time_Left[i] == 0 and t !=0: # 意味着这个AGV的当前货架已经完成
                    next_Shelves[i] = 1   # 可以执行当前任务的下一个货架的移动了
                    for j in range(len(AGV_Shelves[i])):
                        if AGV_Shelves[i][j] !=0: # 第i个机器人的第j个货架被完成了
                            # 记录被完成的货架以及它对应的目的地货架位置________________________这些东西在路径规划的时候就已经确定了
                            finished_Shelves[i] = AGV_Shelves[i][j]
                            #finished_Shelves_Destinations[i] = 1  # !!!! 路径规划中，某个货架对应的目的地。
                            #present_Empty_Position.append(finished_Shelves[i])
                            if finished_Shelves[i] not in self.picking_Stations:  # 这里不能将拣选站加入空位置中
                                present_Empty_Position.append(finished_Shelves[i])
                            AGV_Shelves[i][j] = 0
                            break
                if AGV_Time_Left[i] != 0:
                    next_Shelves[i] = 0
            print("分配任务和路径规划“前”空位置的集合：", present_Empty_Position)
            # 3.1.2 判断每个AGV的当前任务是否完成
            for i in range(len(AGV_Shelves)): # 判断
                if sum(AGV_Shelves[i]) == 0 :  # 意味着这个AGV的当前任务已经完成
                    next_Mission[i] = 1
                    present_AGV_num[i] = 0
                    print("AGV-", i, "处于空闲")
                if sum(AGV_Shelves[i]) != 0:
                    next_Mission[i] = 0
                    print("AGV-", i, "正在工作")

            # 3.2 更新任务状态(系统的任务List，如果有机器人完成了一个任务，则修改List)
            for i in range(len((AGV_Shelves))):
                if present_AGV_num[i] == 0 and t !=0:
                    for j in range(len(present_Target_Shelves)):
                        if present_Target_Shelves[j] == present_AGV_Mission[i]:
                            #mission_AGV[i] = self.target_Shelves.index(present_Target_Shelves[j]) # 记录完成这个任务的机器人编号
                            present_Target_Shelves[j] = 0  # 将完成的任务变为0



            # 4. 任务分配（若有空闲机器人，则将未完成的任务分配给此机器人）————模块2
            # 输入当前的任务List，以及空闲的机器人。
            # present_Target_Shelves, present_AGV_num, mission_AGV, AGV_Shelves = self.mission_Assignment
            # (present_Target_Shelves,present_AGV_num,mission_AGV,AGV_Shelves ,present_Empty_Position)

            for i in range(len(present_AGV_num)):
                if present_AGV_num[i] == 0:
                    for j in range(len(present_Target_Shelves)):
                        if present_Target_Shelves[j] != 0:
                            mission_AGV[i] = present_Target_Shelves[j]
                            present_AGV_num[i] = 1
                            # 计算这个任务的所有需移动的货架
                            # 任务分配完成后，给机器人分配货架
                            AGV_Shelves[i] = self.initial_Obstruction_Of_Shelves(present_Target_Shelves[j],present_Empty_Position)
                            print("目标货架：",present_Target_Shelves[j],"被分配给了AGV-",i,"----需要移动的货架有：",AGV_Shelves[i])
                            # List4[i] = self.initial_Obstruction_Of_Shelves(List1[j],present_Empty_Position)
                            present_Target_Shelves[j] = 0
                            break



            # 5. 路径规划（对每个机器人进行路径规划）————模块3
            # 5.1 阻碍货架路径规划————模块4.1
            # 5.2 目标货架路径规划————模块4.2
            # 5.3 返回货架路径规划————模块4.3
            for i in range(len(AGV_Time_Left)):
                if AGV_Time_Left[i] == 0:
                    for j in range(len(AGV_Shelves[i])):
                        if AGV_Shelves[i][j] !=0: # 开始规划下一个货架的路径
                            present_Move = AGV_Shelves[i][j]
                            finished_Shelves[i] = present_Move
                            # 规路规划当前货架应该前往那个空位置
                            finished_Shelves_Destinations[i], AGV_Time_Left[i], move_List = self.route_planning(present_Move,present_Empty_Position,self.A)
                            # 路径规划后就占用了一个空位置
                            print("当前货架：",present_Move,"移动去了：",finished_Shelves_Destinations[i],"移动的move过程为：",move_List)
                            #present_Empty_Position.append(finished_Shelves[i])
                            if finished_Shelves_Destinations[i] in present_Empty_Position:
                                present_Empty_Position.remove(finished_Shelves_Destinations[i])
                            break
            print("分配任务和路径规划“后”空位置的集合：", present_Empty_Position)
            # 6. 任务分配后要更新系统状态，任务分配有2种，1种是给一个机器人分配另一个货架，1种是给一个机器人分配另一个任务————模块4
            # 更新系统环境状态(系统中各种货架、位置的状态改变)
            # 如果AGV完成了一个货架，需要更新地图(每个T每个AGV最多完成一个货架)
            """
            for i in range(len(next_Shelves)):
                if next_Shelves[i] == 1:
                    if finished_Shelves_Destinations[i] in present_Empty_Position:
                        present_Empty_Position.remove(finished_Shelves_Destinations[i])
                    present_Empty_Position.append(mission_AGV[i])
                    break
            # 如果AGV完成了一个任务，需要更新地图(每个T每个AGV最多完成一个任务)
            for i in range(len((self.AGV_num))):
                pass
            """
            # 时间-1
            # 更新时间
            t = t+1
            print("----------"*10)
            print("当前的时间:",t)
            # 更新系统的时间
            for i in range(len(self.AGV_num)):
                if AGV_Time_Left[i] != 0:
                    AGV_Time_Left[i] = AGV_Time_Left[i] -1

# 输入基本参数
W = 5  # 仓库的宽
L = 5  # 仓库的长
m = 3  # 块布局的宽
n = 3  # 块布局的长
# 可达矩阵的信息
A = Instance_mini.A
s = Instance_mini.s
# 目标货架12，机器人2个，拣选站1，空位置，16,19,20,A为可达矩阵
problem1 = MCRMFS([16,12],2,[1],[8,11,7],W,L,m,n,A,s)
# run
problem1.run()

