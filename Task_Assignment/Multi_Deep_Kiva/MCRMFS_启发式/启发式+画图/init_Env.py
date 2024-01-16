# Instances for MCRMFS
# 产生不同的环境&生成随机的案例
import random
import numpy as np
from MCRMFS_tools import Draw_Opt
class Instance_All(object):
    def __init__(self,W,L,m,n,R,p_num,percent_g,percent_e):
        self.T = 0
        self.p_num = p_num
        self.percent_g = percent_g
        self.percent_e = percent_e
        self.col_num = W
        self.row_num = L
        self.m = m
        self.n = n
        self.R = R
        self.N = self.col_num * self.row_num
        # 初始化函数处理
        self.s = self.inin_s()
        self.p = self.init_p()
        self.g0 = self.init_g0()
        self.e0 = self.init_e0()
        self.h0 = [0] * self.N
        self.A = self.init_A()
        self.Coordinates_And_Indexes  = self.init_Coordinates_And_Indexes(self.col_num,self.row_num,self.m,self.n)
    def init_Coordinates_And_Indexes(self,W,L,m,n):
        # 总个数
        N = W*L
        # 坐标与索引矩阵
        coordinates_And_Indexes = np.zeros((W,L))
        for i in range(W):
            for j in range(L):
                coordinates_And_Indexes[i,j] = i * L + j
        return coordinates_And_Indexes
    def inin_s(self):
        s = np.zeros(self.N)
        for i in range(self.N):
            if (i//self.row_num)%(self.m+1) == 0 or (i%self.row_num)%(self.n+1) == 0:
                s[i] = 1
        return s
    def init_p(self):
        # 拣选站的随机个数
        p_num = self.p_num
        p_num = 0
        p = np.zeros(self.N)
        for i in range(self.N):
            if i%(self.row_num) == 0:
                yes_or_no = random.randint(0,5)# 是否加入拣选站
                if yes_or_no == 1 and p_num != 0:
                    p[i] = 1
                    p_num = p_num - 1
        return p
    def init_g0(self):
        g0 = np.zeros(self.N)
        g0_List = []
        for i in range(int(self.N*self.percent_g)):
            num = random.randint(0, self.N - 1)
            if self.s[num] == 0:
                g0[num] = 1
        return g0
    def init_e0(self):
        e0 = np.zeros(self.N)
        e0_List = []
        # 把非通道的置为非空
        for i in range(self.N):
            if self.s[i] == 0:
                e0[i] == 1
        # 随机产生空位置
        for i in range(int(self.N*self.percent_e)):
            num = random.randint(0,self.N-1)
            if self.s[num] == 0 and self.g0[num] == 0:
                e0_List.append(num)
        e0 = np.zeros(self.N)
        # 先把货架去除去目标货架的地方变成货架
        for i in range(self.N):
            if self.s[i] == 0 and self.p[i] ==0:
                e0[i] = 1
        for i in range(len(e0_List)):
            e0[e0_List[i]] = 0
        return e0
    def init_A(self):
        A = np.zeros((self.N, self.N))
        for i in range(self.N-1):
            if (i//self.row_num) == 0:
                A[i,i+self.row_num] = 1
                A[i,i+1] = 1
                A[i,i-1] = 1
            elif (i // self.row_num) == self.col_num - 1:
                A[i, i - self.row_num] = 1
                A[i, i + 1] = 1
                A[i, i - 1] = 1
            elif (i%self.row_num) == 0:
                A[i, i + self.row_num] = 1
                A[i, i - self.row_num] = 1
                A[i, i + 1] = 1
            elif (i%self.row_num) == self.col_num - 1:
                A[i, i + self.row_num] = 1
                A[i, i - self.row_num] = 1
                A[i, i - 1] = 1
            else:
                A[i, i + self.row_num] = 1
                A[i, i - self.row_num] = 1
                A[i, i - 1] = 1
                A[i, i + 1] = 1
        return A
if __name__ == "__main__":
    W = 31
    L = 31
    m = 4
    n = 4
    R = 2
    p_num = 4        # 拣选站个数
    percent_g = 0.02 # 目标货架比例
    percent_e = 0.05 # 空位置比例
    problem = Instance_All(W,L,m,n,R,p_num,percent_g,percent_e)
    draw_opt = Draw_Opt(problem)
    draw_opt.show_map()
