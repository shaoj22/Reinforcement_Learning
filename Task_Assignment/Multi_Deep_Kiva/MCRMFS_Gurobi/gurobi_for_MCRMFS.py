# applying gurobipy to solve kiva
import gurobipy as gp
from gurobipy import GRB
import time
from 多深紧致化RMFS中机器人的任务调度.MCRMFS_Gurobi.MCRMFS_Instance_A import *
from 多深紧致化RMFS中机器人的任务调度.MCRMFS_Gurobi.MCRMFS_tools import Draw_Opt

class gurobi_MCRMFS():
    def __init__(self, problem, time_limit=None):
        self.T = problem.T # max time
        self.R = problem.R # robot num
        self.N = problem.N # place num
        self.A = problem.A # available matrix
        self.p = problem.p # picking station
        self.s = problem.s # aisle
        self.h0 = problem.h0 # initial positions of robots
        self.e0 = problem.e0 # none-empty space
        self.g0 = problem.g0 # target shelf
        self.time_limit = time_limit

    def get_adjacent_list(self, A):
        adjacent = []
        inv_adjacent = []
        for i in range(len(A)):
            adjacent.append([])
            inv_adjacent.append([])
        for i in range(len(A)):
            for j in range(len(A[i])):
                if A[i][j] == 1:
                    adjacent[i].append(j)
                if A[j][i] == 1:
                    inv_adjacent[i].append(j)
        return adjacent, inv_adjacent

    def run(self):
        start_Time = time.time()
        # building model
        MODEL = gp.Model('MCRMFS')
        # MODEL.setParam('OutputFlag', 0)

        points = list(range(self.N))
        arc = [(i, j) for i in points for j in points if self.A[i][j] == 1]
        adjacent, inv_adjacent = self.get_adjacent_list(self.A)
        ## add variates
        x_list = [(t, r, i, j) for t in range(self.T) for r in range(self.R) for i,j in arc]
        x = MODEL.addVars(x_list, vtype=GRB.BINARY, name="x")
        m = MODEL.addVars(x_list, vtype=GRB.BINARY, name="m")
        b = MODEL.addVars(x_list, vtype=GRB.BINARY, name="b")
        e_list = [(t, i) for t in range(self.T) for i in points]
        h = MODEL.addVars(e_list, vtype=GRB.BINARY, name="h")
        e = MODEL.addVars(e_list, vtype=GRB.BINARY, name="e")
        g = MODEL.addVars(e_list, vtype=GRB.BINARY, name="g")
        t_list = [t for t in range(self.T)]
        f = MODEL.addVars(t_list, vtype=GRB.BINARY, name="f")
        ## set objective
        MODEL.modelSense = GRB.MINIMIZE
        # MODEL.setObjective(gp.quicksum(x[t, r, i, j] for t in range(T) for r in range(R) for i in points for j in points))
        MODEL.setObjective(gp.quicksum(f[t] for t in range(self.T)))
        ## set constraints
        ### 1. independence
        MODEL.addConstrs(gp.quicksum(m[t, r, i, j] for i, j in arc ) <= 1 for r in range(self.R) for t in range(self.T)) 
        MODEL.addConstrs(gp.quicksum(m[t, r, i, j] for r in range(self.R) for i in inv_adjacent[j]) <= 1 for j in points for t in range(self.T)) 
        MODEL.addConstrs(gp.quicksum(m[t, r, i, j] for r in range(self.R) for j in adjacent[i]) <= 1 for i in points for t in range(self.T)) 
        ### 2. availability
        MODEL.addConstrs(m[t, r, i, j] <= self.A[i, j] for t in range(self.T) for r in range(self.R) for i, j in arc)
        MODEL.addConstrs(m[t, r, i, j] <= h[t, i] for t in range(self.T) for r in range(self.R) for i, j in arc)
        MODEL.addConstrs(m[t, r, i, j] <= 1-h[t, j] for t in range(self.T) for r in range(self.R) for i, j in arc)
        MODEL.addConstrs(x[t, r, i, j] <= e[t, i] for t in range(self.T) for r in range(self.R) for i, j in arc)
        MODEL.addConstrs(x[t, r, i, j] <= 1-e[t, j] for t in range(self.T) for r in range(self.R) for i, j in arc)
        MODEL.addConstrs(h[0, i] == self.h0[i] for i in points)
        MODEL.addConstrs(e[0, i] == self.e0[i] for i in points)
        MODEL.addConstrs(g[0, i] == self.g0[i] for i in points)
        ### 3. rack situation
        MODEL.addConstrs(h[t, i] + gp.quicksum(m[t-1, r, i, j] for r in range(self.R) for j in adjacent[i]) - \
            gp.quicksum(m[t-1, r, j, i] for r in range(self.R) for j in inv_adjacent[i]) == h[t-1, i] \
            for i in points for t in range(1, self.T))
        MODEL.addConstrs(x[t, r, i, j] <= m[t, r, i, j] for t in range(self.T) for r in range(self.R) for i, j in arc)
        MODEL.addConstrs(e[t, i] + gp.quicksum(x[t-1, r, i, j] for r in range(self.R) for j in adjacent[i]) - \
            gp.quicksum(x[t-1, r, j, i] for r in range(self.R) for j in inv_adjacent[i]) == e[t-1, i] \
            for i in points for t in range(1, self.T))
        MODEL.addConstrs(b[t, r, i, j] <= x[t, r, i, j] for t in range(self.T) for r in range(self.R) for i, j in arc)
        MODEL.addConstrs(b[t, r, i, j] <= g[t, i] for t in range(self.T) for r in range(self.R) for i, j in arc)
        MODEL.addConstrs(g[t, i] + gp.quicksum(b[t-1, r, i, j] for r in range(self.R) for j in adjacent[i]) - \
            gp.quicksum(b[t-1, r, j, i] for r in range(self.R) for j in inv_adjacent[i]) == g[t-1, i] \
            for i in points if self.p[i] == 0 for t in range(1, self.T))
        MODEL.addConstrs(g[t, i] == 0 for i in points if self.p[i] == 1 for t in range(1, self.T))
        ### 4. remove all targets 
        MODEL.addConstrs(f[t] >= g[t, i] for i in points for t in range(self.T))
        MODEL.addConstr(f[self.T-1] == 0)
        MODEL.addConstrs(x[t, r, i, j] <= f[t] for t in range(self.T) for r in range(self.R) for i, j in arc)
        ## 5. not stay in sideway
        MODEL.addConstrs(gp.quicksum(x[t, r, j, i] for i in adjacent[j]) >= gp.quicksum(x[t-1, r, i, j] for i in inv_adjacent[j]) \
            for j in points if self.s[j] == 1 for r in range(self.R) for t in range(1, self.T))
        
        # solve the model
        if self.time_limit is not None:
            MODEL.setParam("TimeLimit", self.time_limit)
        MODEL.optimize()
        if MODEL.status == 2:
            Obj = MODEL.ObjVal
        else:
            Obj = 0
        objBound = MODEL.objBound
        end_Time = time.time()
        Time =  end_Time - start_Time
        return MODEL, Obj, Time, objBound

if __name__ == "__main__":
    problem = Instance_all()
    # time1 = time()
    time1 = 1
    alg = gurobi_MCRMFS(problem=problem, time_limit=3600)
    model, Obj, Time, objBound= alg.run()
    # time2 = time()
    time2 = 2
    print(objBound)

    print("optimal obj: {}\ntime consumption: {}".format(model.ObjVal, time2-time1))
    draw_opt = Draw_Opt(problem)
    draw_opt.show_solution(model)
    draw_opt.show_animation(model)

