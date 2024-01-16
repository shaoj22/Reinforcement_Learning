import numpy as np
# 初始化一些基本参数
class MCRMFS_problem():
    def init_Parameters(layout_row,layout_column,m_row,n_column):
        layout_row = layout_row
        layout_column = layout_column
        m_row = m_row
        n_column = n_column
        # 时间上限
        T = 30 # max time
        # 机器人数量
        R = 2 # robot num
        # 货架总数
        N = layout_row*layout_column # place num
        # 设置拣选台
        p = np.zeros(N) # 拣选台
        p[0] = 1
        p[4] = 1
        # 设置通道
        s = np.zeros(N) # 通道
        # 设置行通道
        for i in range(layout_row):
            if i%(m_row+1) == 0:
                for j in range(layout_column):
                    s[int((i/(m_row+1))*(m_row+1)+j*layout_row)]=1
        # 设置列通道
        for i in range(layout_column):
            if i%(n_column+1) == 0:
                for j in range(layout_row):
                    s[int((i/(n_column+1))*layout_row*(n_column+1)+j)]=1
        e0 = np.zeros(N) # 非空位
        for i in range(N):
            if s[i] == 0:
                e0[i] = 1
            if i in [18]: # 设置空位置
                e0[i] = 0
        g0 = np.zeros(N) # 目标货架
        for i in [6,12]: # 设置目标货架
            g0[i] = 1

        # 计算可达矩阵
        A = np.zeros((N, N)) # 可达矩阵
        for i in range(layout_row):
            for j in range(layout_column):
                if i > 0 and i <4:
                    if j >0 and j < 4:
                        A[i+j*5,i+j*5-1]=1
                        A[i + j * 5, i + j * 5 + 1] = 1
                        A[i + j * 5,i + j * 5-5]=1
                        A[i + j * 5, i + j * 5 + 5] = 1
                        A[ i + j * 5 - 1,i + j * 5] = 1
                        A[ i + j * 5 + 1,i + j * 5] = 1
                        A[ i + j * 5 - 5,i + j * 5] = 1
                        A[ i + j * 5 + 5,i + j * 5] = 1
                    if j>4 and j <8:
                        A[i + j * 5, i + j * 5 - 1] = 1
                        A[i + j * 5, i + j * 5 + 1] = 1
                        A[i + j * 5, i + j * 5 - 5] = 1
                        A[i + j * 5, i + j * 5 + 5] = 1
                        A[i + j * 5 - 1, i + j * 5] = 1
                        A[i + j * 5 + 1, i + j * 5] = 1
                        A[i + j * 5 - 5, i + j * 5] = 1
                        A[i + j * 5 + 5, i + j * 5] = 1
                if i == 0 and j !=8:
                    A[5*j,5*(j+1)] =1
                if i == 4 and j !=8:
                    A[4+5*(j+1),4+5*j] = 1
                if j == 0 and i != 4:
                    A[(i+1)+j*5,i+j*5]=1
                if j == 4 and i !=4:
                    A[i+j*5,(i+1)+j*5] =1
                if j == 8 and i !=4:
                    A[(i+1)+j*5,i+j*5]=1
        return(A)
problem = MCRMFS_problem.init_Parameters(5,9,3,3)
