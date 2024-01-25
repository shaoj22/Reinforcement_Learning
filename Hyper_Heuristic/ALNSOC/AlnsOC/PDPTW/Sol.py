import numpy as np
class Sol:
    def __init__(self , data):
        self.data = data
        self.obj = 0

        ## 二维数组，第一维是第几辆车，第二维是车辆路径中的第几个点
        self.routes = [] #point的二维数组，初始赋值不要写成“[[]]”，以免造成路径与后续属性长度不匹配
        self.time_record = [] #二维数组，各辆车服务完各个point时的时间
        self.cap_record = [] #二维数组，各辆车服务完各个point时的载重

        ## 按照车辆划分的属性
        self.objs = [] # 各辆车的目标值
        self.cost = [] # 各辆车的费用
        self.overload = [] # 各辆车的累积超载量
        self.delays = [] # 各辆车的延误
        self.finish_time = [] # 各辆车的结束时间
        self.has_transfer = [] # 各辆车是否有转运

        ## 按照订单划分的属性
        self.transfer_time = [np.inf for i in range(len(self.data.orders))]
        self.T1_dis = [[-1, -1] for i in range(len(data.orders))]  # 各个订单的T1在不同车辆中的分布（二元数组第一个值代表在那辆车，第二个值代表在车上第几个）
        self.T2_dis = [[-1, -1] for i in range(len(data.orders))]  # 各个订单的T2在不同车辆中的分布（二元数组第一个值代表在那辆车，第二个值代表在车上第几个）

        ## 按照points划分的属性
        self.times = [-1 for i in range(len(self.data.points))] # 服务完point之后的时间
        self.caps = [-1 for i in range(len(self.data.points))]  # 服务完point之后的载重
        self.points_dis = [[-1, -1] for i in range(len(self.data.points))]  # 各个point在node上的分布（二元数组第一个值代表在那辆车，第二个值代表在车上第几个）

        # 以上所有属性都会在 Fitness_transfer 计算适应度的时候被更新

    def add_route(self, car:list):
        self.routes.append(car)
        self.time_record.append([])
        self.cap_record.append([])
        self.finish_time.append(0)
        self.has_transfer.append(False)
        self.objs.append(0)
        self.cost.append(0)
        self.overload.append(0)
        self.delays.append(0)

    def pop_route(self, route_index:int):
        self.objs.pop(route_index)
        self.time_record.pop(route_index)
        self.cap_record.pop(route_index)
        self.finish_time.pop(route_index)
        self.has_transfer.pop(route_index)
        self.cost.pop(route_index)
        self.overload.pop(route_index)
        self.delays.pop(route_index)
        return self.routes.pop(route_index)