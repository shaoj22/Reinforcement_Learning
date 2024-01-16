# Instances for MCRMFS

import numpy as np
from MCRMFS_tools import Draw_Opt
class Instance_total():
    T = 20 # max time
    R = 3 # robot num
    col_num = row_num = 9
    N = col_num**2 # place num
    p = np.zeros(N) # 拣选台
    p[0] = 1 # [0, 0]
    p[72] = 1 # [0, 72]
    s = np.zeros(N) # 通道
    for i in range(col_num):
        s[i] = 1 # [0, :] 
        s[36+i] = 1 # [4, :]
        s[72+i] = 1 # [8, :]
        s[i*9] = 1 # [:, 0]
        s[4+i*9] = 1 # [:, 4]
        s[8+i*9] = 1 # [:, 8]
    e0 = np.zeros(N) # 非空位
    for i in range(N):
        if s[i] == 0:
            e0[i] = 1 
        if i in [15, 29, 34, 55]:
            e0[i] = 0
    g0 = np.zeros(N) # 目标货架
    for i in [20, 56, 60]:# [20, 56, 60]:
        g0[i] = 1
    A = np.zeros((N, N)) # 可达矩阵
    for i in range(col_num):
        for j in range(row_num):
            if i < col_num-1:
                if j not in [4, 8]: # non-down list
                    A[i*9+j, (i+1)*9+j] = 1 # me -> down
                if j not in [0]: # non-up list
                    A[(i+1)*9+j, i*9+j] = 1 # me <- down
            if j < col_num-1:
                if i not in [0, 4]: # non-right list
                    A[i*9+j, i*9+j+1] = 1 # me -> right
                if i not in [col_num-1]: # non-left list
                    A[i*9+j+1, i*9+j] = 1 # me <- right

# unfinished
class Instance_median():
    T = 100 # max time
    R = 2 # robot num
    col_num = row_num = 9
    N = col_num**2 # place num
    p = np.zeros(N) # 拣选台
    p[0] = 1 # [0, 0]
    s = np.zeros(N) # 通道
    for i in range(col_num):
        s[i] = 1 # [0, :] 
        s[36+i] = 1 # [4, :]
        s[72+i] = 1 # [8, :]
        s[i*9] = 1 # [:, 0]
        s[4+i*9] = 1 # [:, 4]
        s[8+i*9] = 1 # [:, 8]
    e0 = np.zeros(N) # 非空位
    for i in range(N):
        if s[i] == 0:
            e0[i] = 1 
        if i in [15, 29, 34]:
            e0[i] = 0
    g0 = np.zeros(N) # 目标货架
    for i in [20, 56, 60]:
        g0[i] = 1
    A = np.zeros((N, N)) # 可达矩阵
    for i in range(col_num):
        for j in range(row_num):
            if i < col_num-1:
                if j not in [4, 8]: # non-down list
                    A[i*9+j, (i+1)*9+j] = 1 # me -> down
                if j not in [0]: # non-up list
                    A[(i+1)*9+j, i*9+j] = 1 # me <- down
            if j < col_num-1:
                if i not in [0, 4]: # non-right list
                    A[i*9+j, i*9+j+1] = 1 # me -> right
                if i not in [col_num-1]: # non-left list
                    A[i*9+j+1, i*9+j] = 1 # me <- right

class Instance_mini():
    h0 = [0] * 25
    h0[3] = 1
    T = 10 # max time
    R = 2 # robot num
    col_num = row_num = 5
    N = col_num**2 # place num
    p = np.zeros(N) # 拣选台
    p[0] = 1 # [0, 0]
    p[4] = 1  # [0, 0]
    s = np.zeros(N) # 通道
    for i in range(col_num):
        s[i] = 1 # [0, :] 
        s[20+i] = 1 # [4, :]
        s[i*5] = 1 # [:, 0]
        s[4+i*5] = 1 # [:, 4]
    e0 = np.zeros(N) # 非空位
    for i in range(N):
        if s[i] == 0:
            e0[i] = 1 
        if i in [18]:
            e0[i] = 0
    g0 = np.zeros(N) # 目标货架
    for i in [6,12]:
        g0[i] = 1
    A = np.zeros((N, N)) # 可达矩阵
    for i in range(col_num):
        for j in range(col_num):
            if i < col_num-1:
                if j not in [0]: # non-down list
                    A[i*col_num+j, (i+1)*col_num+j] = 1 # me -> down
                if j not in [row_num-1]: # non-up list
                    A[(i+1)*col_num+j, i*col_num+j] = 1 # me <- down
            if j < row_num-1:
                if i not in [col_num-1]: # non-right list
                    A[i*col_num+j, i*col_num+j+1] = 1 # me -> right
                if i not in [0]: # non-left list
                    A[i*col_num+j+1, i*col_num+j] = 1 # me <- right

if __name__ == "__main__":
    # problem = Instance_total()
    problem = Instance_mini()
    draw_opt = Draw_Opt(problem)
    draw_opt.show_map()



