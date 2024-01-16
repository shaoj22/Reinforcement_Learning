'''
File: utils.py
Project: code
File Created: Thursday, 25th May 2023 4:03:48 pm
Author: Charles Lee (lmz22@mails.tsinghua.edu.cn)
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import copy
import time


class DrawTools:
    def __init__(self, data):
        self.data = data
    
    def draw_map(self, ax):
        for order in self.data.orders:
            origin = self.data.name2node[order.origin]
            destination = self.data.name2node[order.destination]
            ax.scatter(origin.x, origin.y, c='g', s=100)
            ax.scatter(destination.x, destination.y, c='r', s=100)
    
    def draw_routes(self, ax, routes):
        for route in routes:
            for i in range(len(route) - 1):
                node1 = self.data.points[route[i]].node
                node2 = self.data.points[route[i + 1]].node
                # ax.plot([node1.x, node2.x], [node1.y, node2.y], c='b')
                # 画箭头
                dx = node2.x - node1.x
                dy = node2.y - node1.y
                ax.arrow(node1.x, node1.y, dx, dy, length_includes_head=True, head_width=0.2, head_length=0.3, fc='b', ec='b')
    
    def draw_transfer(self, ax, transfer_points):
        for p in transfer_points:
            node = self.data.points[p].node
            ax.scatter(node.x, node.y, c='y', s=150, marker='*') 
    
    def show_map(self):
        fig, ax = plt.subplots()
        self.draw_map(ax)
        plt.show()
    
    def show_routes(self, routes, transfer_points=[]): 
        fig, ax = plt.subplots()
        self.draw_map(ax)
        self.draw_routes(ax, routes)
        self.draw_transfer(ax, transfer_points)
        plt.show()
        



