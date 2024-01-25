'''
File: ModelBuilder.py
Project: code
File Created: Wednesday, 24th May 2023 7:34:50 pm
Author: Charles Lee (lmz22@mails.tsinghua.edu.cn)
'''

import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp
from Data import Data
from utils import DrawTools
from Data_small import Data_small


class ModelBuilder_small:
    def __init__(self, data):
        self.data = data
        self.vehicle_num = 1
        # build nodesets
        self.R = list(range(data.order_num))
        self.K = list(range(self.vehicle_num))
        self.o = 0 # virtual depot
        self.P = data.P_set
        self.D = data.D_set
        self.T = data.T_set
        self.N = [self.o] + self.P + self.D + self.T
        self.A = [(i, j) for i in self.N for j in self.N]
    
    def build_PDPTW_T_model(self, MODEL): 
        bigM = 1e6
        """ add variables """
        x_list = [(i, j, k) for (i, j) in self.A for k in self.K]
        x = MODEL.addVars(x_list, vtype=gp.GRB.BINARY, name='x')
        y_list = [(i, j, k, r) for (i, j) in self.A for k in self.K for r in self.R]
        y = MODEL.addVars(y_list, vtype=gp.GRB.BINARY, name='y')
        s_list = [(t, r, k1, k2) for t in self.T for r in self.R for k1 in self.K for k2 in self.K]
        s = MODEL.addVars(s_list, vtype=gp.GRB.BINARY, name='s')
        a_list = [(i, k) for i in self.N for k in self.K]
        a = MODEL.addVars(a_list, vtype=gp.GRB.CONTINUOUS, name='a')
        b = MODEL.addVars(a_list, vtype=gp.GRB.CONTINUOUS, name='b')
        """ set objective """
        ## 车辆固定成本
        fixed_cost = self.data.vehicle_fixed_cost * gp.quicksum(x[self.o, j, k] for j in self.N for k in self.K)
        ## 车辆时间成本
        travel_cost = self.data.vehicle_unit_travel_cost * gp.quicksum(x[i, j, k] * self.data.time_matrix[i][j] for (i, j) in self.A for k in self.K)
        ## 转运成本
        transfer_cost = self.data.parcel_transfer_unit_cost * gp.quicksum(self.data.orders[r].quantity * s[t, r, k1, k2] for t in self.T for r in self.R for k1 in self.K for k2 in self.K if k1 != k2)
        MODEL.setObjective(fixed_cost + travel_cost + transfer_cost, gp.GRB.MINIMIZE)
        """ add constraints """
        ## 任务完成约束
        MODEL.addConstrs(gp.quicksum(y[i, j, k, r] for (i, j) in self.A if i == self.P[r] for k in self.K) == 1 for r in self.R) # 任务起点出度为1
        MODEL.addConstrs(gp.quicksum(y[i, j, k, r] for (i, j) in self.A if j == self.D[r] for k in self.K) == 1 for r in self.R) # 任务终点入度为1
        # MODEL.addConstrs(gp.quicksum(y[i, j, k, r] for (i, j) in self.A if j == self.P[r] for k in self.K) <= 0 for r in self.R) # 任务起点入度为0 #!
        # MODEL.addConstrs(gp.quicksum(y[i, j, k, r] for (i, j) in self.A if i == self.D[r] for k in self.K) <= 0 for r in self.R) # 任务终点出度为0 #!
        ## y任务流不经过virtual depot #? 是否需要这一约束
        MODEL.addConstrs(gp.quicksum(y[self.o, j, k, r] for j in self.N for k in self.K) <= 0 for r in self.R)
        ## y总体流平衡约束(对于T点)
        MODEL.addConstrs(gp.quicksum(y[i, j, k, r] for j in self.N for k in self.K) == gp.quicksum(y[j, i, k, r] for j in self.N for k in self.K) for r in self.R for i in self.T)
        ## y分车流平衡约束(对于其他点,不包括任务起终点和T点) #!
        MODEL.addConstrs(gp.quicksum(y[i, j, k, r] for j in self.N) == gp.quicksum(y[j, i, k, r] for j in self.N) for r in self.R for k in self.K for i in self.N if i not in self.T + [self.P[r], self.D[r]])
        ## 转运约束
        MODEL.addConstrs(gp.quicksum(y[j, t, k1, r] for j in self.N) + gp.quicksum(y[t, j, k2, r] for j in self.N) <= s[t, r, k1, k2] + 1 for r in self.R for t in self.T for k1 in self.K for k2 in self.K if k1 != k2)
        ## 转运同步约束
        MODEL.addConstrs(a[t, k1] + self.data.parcel_transfer_time <= b[t, k2] + bigM * (1 - s[t, r, k1, k2]) for t in self.T for r in self.R for k1 in self.K for k2 in self.K if k1 != k2)
        ## 只转运一次约束
        MODEL.addConstrs(gp.quicksum(s[t, r, k1, k2] for k1 in self.K for k2 in self.K if k1 != k2 for t in self.T) <= 1 for r in self.R)
        ## y与x连接约束
        MODEL.addConstrs(y[i, j, k, r] <= x[i, j, k] for (i, j) in self.A for k in self.K for r in self.R)
        ## 车辆容量约束
        MODEL.addConstrs(gp.quicksum(self.data.orders[r].quantity * y[i, j, k, r] for r in self.R) <= self.data.vehicle_capacity * x[i, j, k] for k in self.K for (i, j) in self.A)
        ## 车辆depot出度最多为1
        MODEL.addConstrs(gp.quicksum(x[self.o, j, k] for j in self.N) <= 1 for k in self.K)
        ## x流平衡约束(对于所有点所有车) #!
        MODEL.addConstrs(gp.quicksum(x[i, j, k] for j in self.N) == gp.quicksum(x[j, i, k] for j in self.N) for i in self.N for k in self.K)
        # MODEL.addConstrs(gp.quicksum(x[i, j, k] for j in self.N for k in self.K) == gp.quicksum(x[j, i, k] for j in self.N for k in self.K) for i in self.N)
        ## PD点出入度为1
        # MODEL.addConstrs(gp.quicksum(x[i, j, k] for j in self.N for k in self.K) == 1 for i in self.P + self.D)
        ## 运输时间约束(破子环)
        MODEL.addConstrs(b[i, k] + self.data.time_matrix[i][j] <= a[j, k] + bigM * (1 - x[i, j, k]) for k in self.K for (i, j) in self.A if j != self.o)
        ## 时间窗约束(约束到达时间)
        MODEL.addConstrs(a[i, k] >= self.data.points[i].time_window[0] for i in self.N for k in self.K) 
        MODEL.addConstrs(a[i, k] <= self.data.points[i].time_window[1] for i in self.N for k in self.K)
        ## 到达离开时间约束
        MODEL.addConstrs(a[i, k] + self.data.vehicle_service_time <= b[i, k] for i in self.N for k in self.K)
        
        MODEL.update()
        return MODEL
    
    def get_solution(self, MODEL):
        """ get routes, transfer points from model """
        routes = []
        transfer_points = []
        for k in self.K:
            route = [] # not include virtual depot
            cur_p = self.o
            while True:
                for j in self.N:
                    if MODEL.getVarByName('x[%d,%d,%d]' % (cur_p, j, k)).x == 1:
                        cur_p = j
                        break
                if cur_p == self.o:
                    break
                else:
                    route.append(j)
            routes.append(route)
        for k1 in self.K:
            for k2 in self.K:
                if k1 != k2:
                    for t in self.T:
                        for r in self.R:
                            if MODEL.getVarByName('s[%d,%d,%d,%d]' % (t, r, k1, k2)).x == 1:
                                transfer_points.append(t)
        transfer_points = list(set(transfer_points))
        return routes, transfer_points
    
    def run(self):
        MODEL = gp.Model('PDPTW-T')
        self.build_PDPTW_T_model(MODEL)
        MODEL.optimize()
        routes, transfer_points = self.get_solution(MODEL)
        result_info = {
            "model" : MODEL, 
            "routes" : routes,
            "transfer_points" : transfer_points
        }
        return result_info

if __name__ == '__main__':
    data = Data_small(node_num=3)
    model_builder = ModelBuilder_small(data)
    result_info = model_builder.run()
    draw_tool = DrawTools(data)
    draw_tool.show_routes(result_info['routes'], result_info['transfer_points'])