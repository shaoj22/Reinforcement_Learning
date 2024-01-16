# Instances for MCRMFS

import numpy as np
import random
from 多深紧致化RMFS中机器人的任务调度.数值实验.MCRMFS_tools import Draw_Opt

class Instance_all():
    def __init__(self,w_num=5,l_num=10,m=10,n=3,R=2,percent_g=0.05,percent_e=0.25,seed=None):
        """instance of kiva

        Args:
            w_num (int, optional): 纵向块数. Defaults to 3.
            l_num (int, optional): 横向块数. Defaults to 4.
            m (int, optional): 块纵向长度. Defaults to 3.
            n (int, optional): 块横向长度. Defaults to 4.
            R (int, optional): 机器人个数. Defaults to 8.
            percent_g (float, optional): 目标货架占比. Defaults to 0.05.
            percent_e (float, optional): 空位置占比. Defaults to 0.2.
            seed (_type_, optional): 随机种子. Defaults to None.
        """
        self.w_num = w_num
        self.l_num = l_num
        W = self.w_num * m + self.w_num + 1
        L = self.l_num * n + self.l_num + 1
        self.T = 0
        self.percent_g = percent_g
        self.percent_e = percent_e
        self.col_num = W
        self.row_num = L
        self.m = m
        self.n = n
        self.R = R
        self.N = self.col_num * self.row_num

        self.generate(seed)

    def generate(self, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        else:
            random.seed()
            np.random.seed()
        # 初始化函数处理
        self.s = self.init_s()
        self.p = self.init_p()
        self.h0 = self.init_h0()
        self.Coordinates_And_Indexes  = self.init_Coordinates_And_Indexes(self.col_num,self.row_num,self.m,self.n)
        self.A = self.init_A()
        self.g0 = self.init_g0()
        self.e0 = self.init_e0()

    def init_Coordinates_And_Indexes(self,W,L,m,n):
        # 总个数
        N = W*L
        # 坐标与索引矩阵
        coordinates_And_Indexes = np.zeros((W,L))
        for i in range(W):
            for j in range(L):
                coordinates_And_Indexes[i,j] = i * L + j
        return coordinates_And_Indexes

    def init_s(self):
        s = np.zeros(self.N)
        for i in range(self.N):
            if (i//self.row_num)%(self.m+1) == 0 or (i%self.row_num)%(self.n+1) == 0:
                s[i] = 1
        return s

    def init_p(self):
        p = np.zeros(self.N)
        for i in range(self.row_num):
            if i%(self.n+1) == 0:
                # 每个通道尽头一个拣选站
                p[i] = 1
        return p

    def init_g0(self):
        g0 = np.zeros(self.N)
        candidates = [i for i in range(self.N) if self.s[i] == 0]
        g0_num = int(len(candidates) * self.percent_g)
        np.random.shuffle(candidates)
        for i in range(g0_num):
            g0[candidates[i]] = 1
        return g0
    
    def init_e0(self):
        e0 = np.zeros(self.N)
        e0_List = []
        # 把非通道的置为非空
        for i in range(self.N):
            if self.s[i] == 0:
                e0[i] = 1
        # 随机产生空位置
        candidates = [i for i in range(self.N) if self.s[i] == 0 and self.g0[i] == 0] # 是货架且非目标货架
        e0_num = int(len(candidates) * self.percent_e)
        np.random.shuffle(candidates)
        for i in range(e0_num):
            e0[candidates[i]] = 0
        return e0

    def init_h0(self):
        h0 = [0]*self.N
        for i in range(1, self.R+1):
            h0[-i] = 1
        return h0

    def init_A(self):
        A = np.zeros((self.N, self.N))
        for i in range(self.N):
            if i // self.row_num < self.col_num-1: # 非最后一行
                A[i, i + self.row_num] = 1
            if i // self.row_num > 0: #非第一行
                A[i, i - self.row_num] = 1
            if i % self.row_num > 0: # 非第一列
                A[i, i - 1] = 1
            if i % self.row_num < self.row_num-1: # 非最后一列
                A[i, i + 1] = 1
        return A

if __name__ == "__main__":
    # problem = Instance_all(w_num=1, l_num=1, m=3, n=3, R=2, percent_g=0.2, percent_e=0.2, seed=2)
    problem = Instance_all()

    draw_opt = Draw_Opt(problem)
    draw_opt.show_map()



