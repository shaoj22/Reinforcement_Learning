# Instances for MCRMFS
import numpy as np
import random
from MCRMFS_tools import Draw_Opt


class Instance_all():
    def __init__(self, w_num=15, l_num=6, m=3, n=3, R=5, num_g=10, percent_e=0.2, seed=0,num_p = 100):
        """instance of kiva

        Args:
            w_num (int, optional): 纵向块数. Defaults to 3.
            l_num (int, optional): 横向块数. Defaults to 4.
            m (int, optional): 块纵向长度. Defaults to 3.
            n (int, optional): 块横向长度. Defaults to 4.
            R (int, optional): 机器人个数. Defaults to 8.
            num_g (int, optional): 目标货架占比. Defaults to 1.
            percent_e (float, optional): 空位置占比. Defaults to 0.2.
            seed (_type_, optional): 随机种子. Defaults to None.
        """
        # 求解最大时间参数
        self.T = 100000
        self.w_num = w_num
        self.l_num = l_num
        # 总行数
        self.W = self.w_num * m + self.w_num + 3
        # 总列数
        self.L = self.l_num * n + self.l_num + 3
        self.num_g = num_g
        self.percent_e = percent_e
        self.col_num = self.W
        self.row_num = self.L
        self.m = m
        self.n = n
        self.R = R
        # 总的方块数量
        self.N = self.col_num * self.row_num
        # 拣选站的个数
        self.num_p = num_p
        # 随机种子
        self.generate(seed)

    # 二维列表转换成一维列表的工具
    def tf(self, double_list):
        single_list = []
        for i in range(len(double_list)):
            for j in range(len(double_list[i])):
                single_list.append(double_list[i][j])

        return single_list

    # 产生随机环境
    def generate(self, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        else:
            random.seed()
            np.random.seed()
        # 初始化函数处理
        self.s, self.S = self.init_s()
        self.p = self.init_p()
        self.h0 = self.init_h0()
        self.Coordinates_And_Indexes = self.init_Coordinates_And_Indexes(self.col_num, self.row_num, self.m, self.n)
        # 初始化邻接矩阵
        self.A = self.init_A()
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

    # 产生通道s：若是通道则为1，否则为0
    def init_s(self):
        # 初始化二维列表
        S = [list([0] * self.L) for i in range(self.W)]
        # 从第1列 or 第1行开始：第m+1行 or n+1列为通道
        for i in range(self.W):
            for j in range(self.L):
                if i % (self.m + 1) == 1 or j % (
                        self.n + 1) == 1 or i == 0 or j == 0 or i == self.col_num - 1 or j == self.row_num - 1:
                    S[i][j] = 1
        # 转换
        s = self.tf(S)

        return s, S

    # 产生拣选站
    def init_p(self):
        # 初始化二维列表
        P = [list([0] * self.L) for i in range(self.W)]
        num = 0
        for i in range(self.W):
            if i % (self.m + 1) == 1:
                # 每个通道尽头一个拣选站
                P[i][0] = 1
                num += 1
            if num >= self.num_p:
                break
        for i in range(self.L):
            if i % (self.n + 1) == 1:
                P[0][i] = 1
        p = self.tf(P)

        return p

    # 产生目标货架
    def init_g0(self):
        # # 初始化二维列表
        # G = [list([0] * self.L) for i in range(self.W)]
        # g0 = np.zeros(self.N)
        # candidates = [i for i in range(self.W) if self.S[i][self.n // 2 + 2] == 0]
        # g0_num = self.num_g
        # np.random.shuffle(candidates)
        # for i in range(g0_num):
        #     G[candidates[i]][self.n // 2 + 2] = 1
        # g0 = self.tf(G)
        g0 = np.zeros(self.N)
        candidates = [i for i in range(self.N) if self.s[i] == 0]
        g0_num = self.num_g
        np.random.shuffle(candidates)
        for i in range(g0_num):
            g0[candidates[i]] = 1

        return g0

    # 产生空位置
    def init_e0(self):
        e0 = np.zeros(self.N)
        e0_List = []
        # 把非通道的置为非空
        for i in range(self.N):
            if self.s[i] == 0:
                e0[i] = 1
        # 随机产生空位置
        candidates = [i for i in range(self.N) if self.s[i] == 0 and self.g0[i] == 0]  # 是货架且非目标货架
        e0_num = int(len(candidates) * self.percent_e)
        np.random.shuffle(candidates)
        for i in range(e0_num):
            e0[candidates[i]] = 0

        return e0

    # 机器人的初始位置
    def init_h0(self):
        # 初始化二维列表
        H0 = [list([0] * self.L) for i in range(self.W)]
        for i in range(self.R):
            H0[self.W - 1][i+1*i] = 1
        # 转换
        h0 = self.tf(H0)

        return h0

    # 初始化邻接矩阵
    def init_A(self):
        A = np.zeros((self.N, self.N))
        # 初始化坐标与编号
        coordinate = np.zeros((self.W, self.L))
        for i in range(self.W):
            for j in range(self.L):
                coordinate[i, j] = i * self.L + j
        # 准备工作：统计奇数、偶数的行、列
        odd_row = []
        even_row = []
        odd_col = []
        even_col = []
        i = 0
        j = 0
        # 初始化行
        while i <= self.W:
            # 既能够整除且为奇数行
            if (i - 1) % (self.m + 1) == 0 and ((i - 1) / (self.m + 1)) % 2 == 0:
                odd_row.append(i)
            # 既能够整除但为偶数行
            if (i - 1) % (self.m + 1) == 0 and ((i - 1) / (self.m + 1)) % 2 != 0:
                even_row.append(i)
            i += 1
        # 初始化列
        while j <= self.L:
            # 既能够整除且为奇数列
            if (j - 1) % (self.n + 1) == 0 and ((j - 1) / (self.n + 1)) % 2 == 0:
                odd_col.append(j)
            # 既能够整除但为偶数行
            if (j - 1) % (self.n + 1) == 0 and ((j - 1) / (self.n + 1)) % 2 != 0:
                even_col.append(j)
            j += 1
        i = 0
        j = 0
        # 开始构建邻接矩阵，一共分为了9种情况：奇数行、偶数行、奇数列、偶数列、0行、尾行、0列、尾列、货架区域
        # case1：奇数行→只能右边至左边
        for i in range(len(odd_row)):
            for j in range(self.L - 1):
                # 第odd_row[i]行，第j列，编号为coordinate[odd_row[i], j]，后面一个点可以至前面一个点
                A[int(coordinate[odd_row[i], j + 1]), int(coordinate[odd_row[i], j])] = 1
        # case2：偶数行→只能左边至右边
        for i in range(len(even_row)):
            for j in range(self.L - 1):
                # 第even_row[i]行，第j列，编号为coordinate[even_row[i], j]，前面一个点可以至后面一个点
                A[int(coordinate[even_row[i], j]), int(coordinate[even_row[i], j + 1])] = 1
        # case3：奇数列→只能上面至下面
        for i in range(len(odd_col)):
            for j in range(self.W - 1):
                A[int(coordinate[j, odd_col[i]]), int(coordinate[j + 1, odd_col[i]])] = 1
        # case4: 偶数列→只能下面至上面
        for i in range(len(even_col)):
            for j in range(self.W - 1):
                A[int(coordinate[j + 1, even_col[i]]), int(coordinate[j, even_col[i]])] = 1
        # case5: 0行
        for i in range(self.L - 1):
            A[int(coordinate[0, i]), int(coordinate[0, i + 1])] = 1
        # case6：尾行
        # 判断尾行的方向，如果它的上一行是奇数行，那么尾行向右
        if (self.W - 2) in odd_row:
            for i in range(self.L - 1):
                A[int(coordinate[self.W - 1, i]), int(coordinate[self.W - 1, i + 1])] = 1
        else:
            for i in range(self.L - 1):
                A[int(coordinate[self.W - 1, i + 1]), int(coordinate[self.W - 1, i])] = 1
        # case7: 0列
        for i in range(self.W - 1):
            A[int(coordinate[i + 1, 0]), int(coordinate[i, 0])] = 1
        # case8：尾列
        # 判断尾列的方向，如果它的左侧一列是奇数列，那么尾列向上
        if (self.L - 2) in odd_col:
            for i in range(self.W - 1):
                A[int(coordinate[i + 1, 0]), int(coordinate[i, 0])] = 1
        else:
            for i in range(self.W - 1):
                A[int(coordinate[i, 0]), int(coordinate[i + 1, 0])] = 1
        # case9：货架区域
        # 不是通道的区域为货架区域，即既不是奇数行、偶数行，也不是奇数列、偶数列，还有0行、0列和尾行和尾列
        for i in range(self.W):
            for j in range(self.L):
                # 选出货架区域
                if i not in odd_row and i not in even_row and j not in odd_col and j not in even_col and i != 0 and i != (
                        self.W - 1) and j != 0 and j != (self.L - 1):
                    # 到右边
                    A[int(coordinate[i, j]), int(coordinate[i, j + 1])] = 1
                    A[int(coordinate[i, j + 1]), int(coordinate[i, j])] = 1
                    # 到左边
                    A[int(coordinate[i, j]), int(coordinate[i, j - 1])] = 1
                    A[int(coordinate[i, j - 1]), int(coordinate[i, j])] = 1
                    # 到上面
                    A[int(coordinate[i, j]), int(coordinate[i - 1, j])] = 1
                    A[int(coordinate[i - 1, j]), int(coordinate[i, j])] = 1
                    # 到下面
                    A[int(coordinate[i, j]), int(coordinate[i + 1, j])] = 1
                    A[int(coordinate[i + 1, j]), int(coordinate[i, j])] = 1

        # case10：上下左右的通道
        # 上
        i = 0
        j = 0
        for i in range(2, self.L - 2):
            if i % 2 == 0:
                A[int(coordinate[1, i]), int(coordinate[0, i])] = 1
                A[int(coordinate[self.W - 1, i]), int(coordinate[self.W - 2, i])] = 1
            else:
                A[int(coordinate[0, i]), int(coordinate[1, i])] = 1
                A[int(coordinate[self.W - 2, i]), int(coordinate[self.W - 1, i])] = 1
        for i in range(2, self.W - 2):
            if i % 2 == 0:
                A[int(coordinate[i, 0]), int(coordinate[i, 1])] = 1
                A[int(coordinate[i, self.L - 2]), int(coordinate[i, self.L - 1])] = 1
            else:
                A[int(coordinate[i, 1]), int(coordinate[i, 0])] = 1
                A[int(coordinate[i, self.L - 1]), int(coordinate[i, self.L - 2])] = 1
        return A


if __name__ == "__main__":
    # problem = Instance_all(w_num=1, l_num=1, m=3, n=3, R=2, percent_g=0.2, percent_e=0.2, seed=2)
    problem = Instance_all()

    draw_opt = Draw_Opt(problem)
    draw_opt.show_map()
