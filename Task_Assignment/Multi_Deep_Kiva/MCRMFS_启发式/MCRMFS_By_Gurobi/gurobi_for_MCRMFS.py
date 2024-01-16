# applying gurobipy to solve kiva
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt
from time import time
import math

class MCRMFS_problem():
    T = 30 # max time
    R = 2 # robot num
    col_num = 5
    N = col_num**2 # place num
    p = np.zeros(N) # 拣选台
    p[0] = 1 # [0, 0]
    p[4] = 1  # [0, 0]
    s = np.zeros(N) # 通道
    for i in range(col_num):
        s[i] = 1 # [0, :] 
        s[20+i] = 1 # [4, :]
        #s[72+i] = 1 # [8, :]
        s[i*5] = 1 # [:, 0]
        s[4+i*5] = 1 # [:, 4]
        #s[8+i*9] = 1 # [:, 8]
    e0 = np.zeros(N) # 非空位
    for i in range(N):
        if s[i] == 0:
            e0[i] = 1 
        if i in [18]:
            e0[i] = 0
    g0 = np.zeros(N) # 目标货架
    for i in [6,12,13]:
        g0[i] = 1
    A = np.zeros((N, N)) # 可达矩阵
    for i in range(col_num):
        for j in range(col_num):
            if i < col_num-1:
                A[i*5+j, (i+1)*5+j] = 1
                A[(i+1)*5+j, i*5+j] = 1
            if i in [0, 4] and j < col_num-1:
                A[i*5+j, i*5+j+1] = 1
                A[i*5+j+1, i*5+j] = 1
            if i in [1,2,3] and j in [1,2,3]:
                A[i*5+j,i*5+j-1]=1
                A[i * 5 + j - 1,i*5+j ] = 1
                A[i*5+j,i*5+j+1]=1
                A[i * 5 + j+1, i * 5 + j ] = 1
                A[i * 5 + j, i * 5 + j - 5] = 1
                A[i * 5 + j-5, i * 5 + j ] = 1
                A[i * 5 + j, i * 5 + j + 5] = 1
                A[i * 5 + j + 5, i * 5 + j] = 1
    A[5,0]=0
    A[10,5]=0
    A[15,10]=0
    A[20,15]=0
    A[4,9]=0
    A[9,14]=0
    A[14,19]=0
    A[19,24]=0
    A[3,4]=0
    A[2,3]=0
    A[1,2]=0
    A[0,1]=0
    A[24,23]=0
    A[23,22]=0
    A[21,20]=0
    A[5,6]=1
def gurobi_MCRMFS(problem):
    T = problem.T # max time
    R = problem.R # robot num
    N = problem.N # place num
    A = problem.A # 可达矩阵
    p = problem.p # 拣选台
    s = problem.s # 通道
    e0 = problem.e0 # 非空位
    g0 = problem.g0 # 目标货架
    bigM = R+1

    # building model
    MODEL = gp.Model('MCRMFS_By_Gurobi')

    points = list(range(N))

    ## add variates
    x_list = [(t, r, i, j) for t in range(T) for r in range(R) for i in points for j in points]
    x = MODEL.addVars(x_list, vtype=GRB.BINARY, name="x")
    e_list = [(t, i) for t in range(T) for i in points]
    e = MODEL.addVars(e_list, vtype=GRB.BINARY, name="e")
    g = MODEL.addVars(e_list, vtype=GRB.BINARY, name="g")
    t_list = [t for t in range(T)]
    f = MODEL.addVars(t_list, vtype=GRB.BINARY, name="f")
    ## set objective
    MODEL.modelSense = GRB.MINIMIZE
    # MODEL.setObjective(gp.quicksum(x[t, r, i, j] for t in range(T) for r in range(R) for i in points for j in points))
    MODEL.setObjective(gp.quicksum(f[t] for t in range(T)))
    ## set constraints
    ### 1. independence
    MODEL.addConstrs(gp.quicksum(x[t, r, i, j] for i in points for j in points) <= 1 for r in range(R) for t in range(T)) 
    MODEL.addConstrs(gp.quicksum(x[t, r, i, j] for r in range(R) for i in points) <= 1 for j in points for t in range(T)) 
    MODEL.addConstrs(gp.quicksum(x[t, r, i, j] for r in range(R) for j in points) <= 1 for i in points for t in range(T)) 
    ### 2. availabel
    MODEL.addConstrs(x[t, r, i, j] <= A[i, j] for t in range(T) for r in range(R) for i in points for j in points)
    MODEL.addConstrs(x[t, r, i, j] <= 1-e[t, j] for t in range(T) for r in range(R) for i in points for j in points)
    MODEL.addConstrs(x[t, r, i, j] <= e[t, i] for t in range(T) for r in range(R) for i in points for j in points)
    MODEL.addConstrs(e[0, i] == e0[i] for i in points)
    MODEL.addConstrs(g[0, i] == g0[i] for i in points)
    ### 3. rack situation
    # MODEL.addConstrs(e[t, j] + bigM * (1 - gp.quicksum(x[t-1, r, i, j] for r in range(R) for i in points)) >= 1\
    #     for t in range(1, T) for j in points)
    # MODEL.addConstrs(e[t, i] - bigM * (1 - gp.quicksum(x[t-1, r, i, j] for r in range(R) for j in points))  <= 0\
    #     for t in range(1, T) for i in points)
    MODEL.addConstrs(e[t, i] + gp.quicksum(x[t-1, r, i, j] - x[t-1, r, j, i] for r in range(R) for j in points) == e[t-1, i] \
          for i in points for t in range(1, T))
    # MODEL.addConstrs(e[t, i] + bigM * gp.quicksum(x[t-1, r, i, j] for r in range(R) for j in points) >= e[t-1, i]\
    #     for t in range(1, T) for i in points)
    # MODEL.addConstrs(gp.quicksum(e[t, i] for i in points) == gp.quicksum(e[0, i] for i in points ) \
    #     for t in range(1,T) )
    MODEL.addConstrs(g[t, j] + bigM * (1 - gp.quicksum(x[t-1, r, i, j] for r in range(R)))\
        + bigM * (1 - g[t-1, i]) + bigM * p[i] >= 1\
        for t in range(1, T) for i in points for j in points)
    MODEL.addConstrs(g[t, i] - bigM * (1 - gp.quicksum(x[t-1, r, i, j] for r in range(R))) <= 0\
        for t in range(1, T) for i in points for j in points)
    MODEL.addConstrs(g[t, i] + bigM * gp.quicksum(x[t-1, r, i, j] for r in range(R) for j in points) >= g[t-1, i]\
        for t in range(1, T) for i in points)
    MODEL.addConstrs(g[t, i] <= e[t, i] for t in range(T) for i in points)
    ### 4. remove all targets  e
    MODEL.addConstrs(gp.quicksum(g[t, j] for j in points) <= bigM * f[t] for t in range(T))
    MODEL.addConstrs(gp.quicksum(x[t, r, i, j] for r in range(R) for i in points for j in points) <= bigM * f[t] for t in range(T))
    ## 5. not stay in sideway
    MODEL.addConstrs(gp.quicksum(x[t, r, j, i] for i in points) + bigM * (1 - s[j]) \
        >= gp.quicksum(x[t-1, r, i, j] for i in points) for j in points for r in range(R) for t in range(1, T))
    
    # solve the model
    MODEL.setParam("TimeLimit", 300)
    MODEL.optimize()

    draw_opt = Draw_Opt(problem, MODEL)
    draw_opt.draw()

    return MODEL.ObjVal

class Draw_Opt():
    def __init__(self, problem, model=None):
        # draw param
        self.psize = 1 # 方格长宽

        # read problem data
        self.T = problem.T  # max time
        self.R = problem.R  # robot num
        self.N = problem.N  # place num
        self.A = problem.A  # 可达矩阵
        self.p = problem.p  # 拣选台
        self.s = problem.s  # 通道
        self.e0 = problem.e0  # 非空位
        self.g0 = problem.g0  # 目标货架

        # read model data
        self.model = model
        if model is None:
            return
        self.e = np.zeros((self.T, self.N))
        self.g = np.zeros((self.T, self.N))
        self.f = np.zeros(self.T)
        for t in range(self.T):
            var_name = "f[{}]".format(t)
            self.f[t] = model.getVarByName(var_name).X
            for i in range(self.N):
                var_name = "e[{},{}]".format(t, i)
                self.e[t, i] = model.getVarByName(var_name).X
                var_name = "g[{},{}]".format(t, i)
                self.g[t, i] = model.getVarByName(var_name).X

    def draw_t(self, ax, p, s, e, g):
        for i in range(self.N):
            row = i // 5
            col = i % 5
            if g[i] == 1: # 目标货架
                self.draw_pixel(ax, row, col, 'r')
            elif e[i] == 1: # 非空货架
                self.draw_pixel(ax, row, col, 'grey')
            elif p[i] == 1: # 拣选台
                self.draw_pixel(ax, row, col, 'b')
            elif s[i] == 1: # 通道
                self.draw_pixel(ax, row, col, 'antiquewhite')
            else: # 空货架
                self.draw_pixel(ax, row, col, 'w')
        plt.xlim(0, np.sqrt(self.N)*self.psize)
        plt.ylim(0, np.sqrt(self.N) * self.psize)
        plt.axis('off')

    def draw_pixel(self, ax, row, col, color):
        rect = plt.Rectangle(
            (row*self.psize, col*self.psize),
            self.psize,
            self.psize,
            facecolor=color,
            edgecolor='black',
            alpha = 1
        )
        ax.add_patch(rect)

    def draw(self):
        if self.model is None:
            ax = plt.subplot(111)
            self.draw_t(ax, self.p, self.s, self.e0, self.g0)
        else:
            fig = plt.figure()
            for t in range(self.T):
                col_num = math.ceil(np.sqrt(sum(self.f)+1))
                ax = fig.add_subplot(col_num, col_num, t + 1)
                self.draw_t(ax, self.p, self.s, self.e[t], self.g[t])
                if self.f[t] == 0:
                    break
        plt.show()

if __name__ == "__main__":
    problem = MCRMFS_problem()
    # opt = Draw_Opt(problem)
    # opt.draw()
    time1 = time()
    obj = gurobi_MCRMFS(problem)
    time2 = time()
    print("optimal obj: {}\ntime consumption: {}".format(obj, time2-time1))

