# Instances for MCRMFS
import numpy as np
import random
from Drawing_Tools import Draw_Opt


class Instance_all():
    def __init__(self, w_num=1, l_num=1, m=4, n=4, R=1, percent_g=0.05, percent_e=0.2, seed=None, A_size=0.2,
                 B_size=0.3):
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
        self.A_size = A_size
        self.B_size = B_size
        self.C_size = self.A_size - self.B_size

        self.generate(seed)

    def generate(self, seed=None):
        if seed is not None:
            random.seed(seed)
        else:
            random.seed()
        # 初始化函数处理
        self.s = self.init_s()
        self.p = self.init_p()
        self.h0 = self.init_h0()
        self.Coordinates_And_Indexes = self.init_Coordinates_And_Indexes(self.col_num, self.row_num, self.m, self.n)
        self.A = self.init_A()
        self.st, self.A_Index, self.B_Index, self.C_Index = self.init_st()
        self.g0 = self.init_g0()
        self.e0 = self.init_e0()

    def init_Coordinates_And_Indexes(self, W, L, m, n):
        # 总个数
        N = W * L
        # 坐标与索引矩阵
        coordinates_And_Indexes = np.zeros((W, L))
        for i in range(W):
            for j in range(L):
                coordinates_And_Indexes[i, j] = i * L + j
        return coordinates_And_Indexes

    def init_st(self):
        st = [0] * self.N
        # ABC_size的货架总数
        N = self.w_num * self.l_num * self.m * self.n
        A_num = int(N * self.A_size)
        B_num = int(N * self.B_size)
        C_num = N - A_num - B_num
        number1 = 0
        for i in range(self.N):
            if self.s[i] != 1:
                number1 += 1
            if number1 == A_num:
                A_Index = i
                break
        number2 = 0
        for i in range(A_Index + 1, self.N):
            if self.s[i] != 1:
                number2 += 1
            if number2 == B_num:
                B_Index = i
                break
        for i in range(self.N):
            if i <= A_Index and self.s[i] == 0:
                st[i] = 1
            if i > A_Index and i < B_Index and self.s[i] == 0:
                st[i] = 2
            if i > B_Index and self.s[i] == 0:
                st[i] = 3
        return st, A_Index, B_Index, (self.N - 1)

    def init_s(self):
        s = np.zeros(self.N)
        for i in range(self.N):
            if (i // self.row_num) % (self.m + 1) == 0 or (i % self.row_num) % (self.n + 1) == 0:
                s[i] = 1
        return s

    def init_p(self):
        # 拣选站的随机个数
        p = np.zeros(self.N)
        for i in range(self.N):
            if i % (self.m + 1) == 0 and i <= self.row_num:
                # 每个通道尽头一个拣选站
                p[i] = 1
        return p

    def init_ABC_demand(self):
        s = 0.067
        fa = ((1 + s) * self.A_size) / (s + self.A_size)
        fb = ((1 + s) * (self.B_size + self.A_size) / (s + self.B_size + self.A_size)) - fa
        fc = 1 - fa - fb
        return fa, fb, fc

    def init_g0(self):
        g0 = np.zeros(self.N)
        fa, fb, fc = self.init_ABC_demand()
        g0_num = int(self.N * self.percent_g)
        g0_A_num = int(g0_num * fa)
        g0_B_num = int(g0_num * fb) + 1
        g0_C_num = int(g0_num * fc) + 1
        # ABC_size的货架总数
        N = self.w_num * self.l_num * self.m * self.n
        A_num = int(N * self.A_size)
        B_num = int(N * self.B_size)
        C_num = N - A_num - B_num
        # ABC的最大索引
        A_Index = self.A_Index
        B_Index = self.B_Index
        C_Index = self.N - 1
        # 分别产生ABC目标货架
        a = 0
        b = 0
        c = 0
        while (1):
            num_1 = random.randint(0, A_Index)
            if self.s[num_1] == 0 and g0[num_1] != 1:
                g0[num_1] = 1
                a = a + 1
            if a == g0_A_num:
                break
        while (1):
            num_2 = random.randint(A_Index, B_Index)
            if self.s[num_2] == 0 and g0[num_2] != 1:
                g0[num_2] = 1
                b = b + 1
            if b == g0_B_num:
                break
        while (1):
            num_3 = random.randint(B_Index, C_Index)
            if self.s[num_3] == 0 and g0[num_3] != 1:
                g0[num_3] = 1
                c = c + 1
            if c == g0_C_num:
                break
        return g0

    def init_e0(self):
        e0 = np.zeros(self.N)
        e0_List = []
        # 把非通道的置为非空
        for i in range(self.N):
            if self.s[i] == 0:
                e0[i] == 1
        # 随机产生空位置
        for i in range(int(self.N * self.percent_e)):
            num = random.randint(0, self.N - 1)
            if self.s[num] == 0 and self.g0[num] == 0:
                e0_List.append(num)
        e0 = np.zeros(self.N)
        # 先把货架去除去目标货架的地方变成货架
        for i in range(self.N):
            if self.s[i] == 0 and self.p[i] == 0:
                e0[i] = 1
        for i in range(len(e0_List)):
            e0[e0_List[i]] = 0
        return e0

    def init_h0(self):
        robots = random.choices(range(self.N), k=self.R)
        h0 = [0] * self.N
        for ri in robots:
            h0[ri] = 1
        return h0

    def init_A(self):
        A = np.zeros((self.N, self.N))
        for i in range(self.N):
            if i // self.col_num < self.row_num - 1:  # 非最后一行
                A[i, i + self.col_num] = 1
            if i // self.col_num > 0:  # 非第一行
                A[i, i - self.col_num] = 1
            if i % self.col_num > 0:  # 非第一列
                A[i, i - 1] = 1
            if i % self.col_num < self.row_num - 1 and i != self.N - 1:  # 非最后一列
                A[i, i + 1] = 1
        return A


if __name__ == "__main__":
    # problem = Instance_total()
    # problem = Instance_mini()
    # problem = Instance_median()
    # problem = Instance_all(w_num=1, l_num=1, m=3, n=3, R=2, percent_g=0.2, percent_e=0.2, seed=2)
    problem = Instance_all()
    draw_opt = Draw_Opt(problem)
    draw_opt.show_map()
