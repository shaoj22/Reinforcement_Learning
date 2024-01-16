# tools for MCRMFS
import numpy as np
import math
import matplotlib.pyplot as plt

class Draw_Opt():
    def __init__(self, problem):
        # draw param
        """
        :param problem:
        """


        self.psize = 1 # 方格长宽
        # read problem data
        self.col_num = problem.col_num # column number
        self.row_num = problem.row_num # row number
        self.T = problem.T  # max time
        self.R = problem.R  # robot num
        self.N = problem.N  # place num
        self.A = problem.A  # 可达矩阵
        self.p = problem.p  # 拣选台
        self.s = problem.s  # 通道
        self.h0 = problem.h0 # 机器人位置
        self.e0 = problem.e0  # 非空位
        self.g0 = problem.g0  # 目标货架

    def _draw_t(self, ax, p, s, h, e, g):
        for i in range(self.N):
            row = i % self.row_num
            col = self.col_num - 1 - i // self.row_num
            fill = 0
            if g[i] > 0: # 目标货架
                self._draw_pixel(ax, row, col, facecolor='r')
            elif e[i] > 0: # 非空货架
                self._draw_pixel(ax, row, col, facecolor='grey')
            else: # draw body of robot
                fill = 1
                if p[i] == 1: # 拣选台
                    self._draw_pixel(ax, row, col, facecolor='b')
                elif s[i] == 1: # 通道
                    self._draw_pixel(ax, row, col, facecolor='antiquewhite')
                else: # 空货架
                    self._draw_pixel(ax, row, col, facecolor='w')
            if h[i] > 0: # 机器人
                self._draw_robot(ax, row, col, fill)
        plt.xlim(0, self.row_num * self.psize)
        plt.ylim(0, self.col_num * self.psize)
        plt.axis('off')

    def _draw_pixel(self, ax, row, col, facecolor, edgecolor='black'):
        rect = plt.Rectangle(
            (row*self.psize, col*self.psize),
            self.psize,
            self.psize,
            facecolor=facecolor,
            edgecolor=edgecolor,
            alpha = 1
        )
        ax.add_patch(rect)

    def _draw_robot(self, ax, row, col, fill, facecolor='y', edgecolor='black'):
        circle = plt.Circle(
            ((row+0.5)*self.psize, (col+0.5)*self.psize),
            self.psize/2.5,
            fill=fill,
            facecolor=facecolor,
            edgecolor=edgecolor,
            alpha = 1
        )
        ax.add_patch(circle)

    def _set_model(self, model):
        # read model data
        self.model = model
        if model is None:
            return
        self.h = np.zeros((self.T, self.N))
        self.e = np.zeros((self.T, self.N))
        self.g = np.zeros((self.T, self.N))
        self.f = np.zeros(self.T)
        for t in range(self.T):
            var_name = "f[{}]".format(t)
            self.f[t] = model.getVarByName(var_name).X
            for i in range(self.N):
                var_name = "h[{},{}]".format(t, i)
                self.h[t, i] = model.getVarByName(var_name).X
                var_name = "e[{},{}]".format(t, i)
                self.e[t, i] = model.getVarByName(var_name).X
                var_name = "g[{},{}]".format(t, i)
                self.g[t, i] = model.getVarByName(var_name).X

    def show_map(self):
        ax = plt.subplot(111)
        self._draw_t(ax, self.p, self.s, self.h0, self.e0, self.g0)
        plt.show()

    def show_solution(self, model):
        self.model = model
        self._set_model(model)
        fig = plt.figure()
        for t in range(self.T):
            fig_col_num = math.ceil(np.sqrt(sum(self.f)+1))
            ax = fig.add_subplot(fig_col_num, fig_col_num, t + 1)
            self._draw_t(ax, self.p, self.s, self.h[t], self.e[t], self.g[t])
            if self.f[t] == 0: # stop after mission finished
                break
        plt.show()

    def show_animation(self, model):
        self.model = model
        self._set_model(model)
        fig = plt.figure()
        for t in range(self.T):
            plt.clf()
            ax = plt.subplot(111)
            self._draw_t(ax, self.p, self.s, self.h[t], self.e[t], self.g[t])
            plt.draw()
            plt.pause(0.5)
            if self.f[t] == 0: # stop after mission finished
                break
        plt.clf()

        
        
