'''
File: Data.py
Project: SF-X code
File Created: Wednesday, 24th May 2023 7:09:17 pm
Author: Charles Lee (lmz22@mails.tsinghua.edu.cn)
'''

import numpy as np
import pandas as pd

# main data class

class Data:
    def __init__(self, 
                 demand_file_path='城市配送系统优化资料包\\输入数据\\demand_50.csv',
                 node_file_path='城市配送系统优化资料包\\输入数据\\node_50.csv',
                 time_matrix_file_path='城市配送系统优化资料包\\输入数据\\time_matrix_50.csv',
                 parameter_file_path='城市配送系统优化资料包\\输入数据\\parameter.csv', 
                 limited_order_num=np.inf
                 ):
        """ 
        read data from files 
        """
        # read with pandas
        demand_df = pd.read_csv(demand_file_path)
        node_df = pd.read_csv(node_file_path)
        time_matrix_df = pd.read_csv(time_matrix_file_path)
        parameter_df = pd.read_csv(parameter_file_path)
        # tranform demand data to orders
        self.orders = []
        self.id2order = {}
        order_num = min(len(demand_df), limited_order_num)
        for i in range(order_num):
            order_id = demand_df['id'][i]
            origin = demand_df['origin'][i]
            destination = demand_df['destination'][i]
            start_time = demand_df['start_time'][i]
            end_time = demand_df['end_time'][i]
            quantity = demand_df['quantity'][i]
            order = Order(order_id, origin, destination, start_time, end_time, quantity)
            self.orders.append(order)
            self.id2order[order_id] = order
        # transform node data to nodes
        self.nodes = []
        self.name2node = {}
        for i in range(len(node_df)):
            node_id = i
            name = node_df['name'][i]
            x = node_df['x'][i]
            y = node_df['y'][i]
            transfer = node_df['transfer'][i]
            node = Node(node_id, name, x, y, transfer)
            self.nodes.append(node)
            self.name2node[name] = node
        # transform time matrix data to time matrix
        self.node_time_matrix = np.ones((len(self.nodes), len(self.nodes))) * np.inf
        for i in range(len(time_matrix_df)):
            name_x = time_matrix_df['name_x'][i]
            name_y = time_matrix_df['name_y'][i]
            time = time_matrix_df['time'][i]
            id_x = self.name2node[name_x].node_id
            id_y = self.name2node[name_y].node_id
            self.node_time_matrix[id_x][id_y] = time
        # get parameters
        self.vehicle_capacity = parameter_df['value'][0]
        self.vehicle_service_time = parameter_df['value'][1]
        self.parcel_transfer_time = parameter_df['value'][2]
        self.vehicle_fixed_cost = parameter_df['value'][3]
        self.vehicle_unit_travel_cost = parameter_df['value'][4]
        self.parcel_transfer_unit_cost = parameter_df['value'][5]

        """ 
        build PDP data 
        """
        self.points = []
        self.P_set, self.D_set, self.T_set = [], [], []
        # virtual depot
        self.virtual_depot = PDT_Point(0, type="virtual depot")
        self.points.append(self.virtual_depot)
        for order in self.orders:
            node = self.get_node_by_name(order.origin)
            point_id = len(self.points)
            p_point = PDT_Point(point_id, type="P", node=node, order=order)
            self.points.append(p_point)
            self.P_set.append(point_id)
        for order in self.orders:
            node = self.get_node_by_name(order.destination)
            point_id = len(self.points)
            d_point = PDT_Point(point_id, type="D", node=node, order=order)
            self.points.append(d_point)
            self.D_set.append(point_id)
        for node in self.nodes:
            if node.transfer: 
                point_id = len(self.points)
                t_point = PDT_Point(point_id, type="T", node=node)
                self.points.append(t_point)
                self.T_set.append(point_id)
        self.time_matrix = np.ones((len(self.points), len(self.points))) * np.inf
        for i in range(len(self.points)):
            for j in range(i, len(self.points)):
                if i == 0 or j == 0:
                    dist = 0
                else:
                    dist = self.node_time_matrix[self.points[i].node.node_id, self.points[j].node.node_id]
                self.time_matrix[i, j] = dist
                self.time_matrix[j, i] = dist
    
    # properties
    @property
    def order_num(self) -> int:
        return len(self.orders)
    
    @property
    def node_num(self) -> int:
        return len(self.nodes)
    
    @property
    def point_num(self) -> int:
        return len(self.points)
    
    @property
    def transfer_num(self) -> int:
        return sum([node.transfer for node in self.nodes])
        
    # methods
    def get_order_by_id(self, id: int):
        return self.id2order[id]
    
    def get_node_by_name(self, name):
        return self.name2node[name]
    
    def get_time_with_names(self, node_name1: str, node_name2: str) -> float:
        return self.node_time_matrix[self.get_node_by_name(node_name1).node_id][self.get_node_by_name(node_name2).node_id]


# basic object classes

class Order:
    def __init__(self, order_id, origin, destination, start_time, end_time, quantity):
        self.order_id = order_id
        self.origin = origin
        self.destination = destination
        self.start_time = start_time
        self.end_time = end_time
        self.quantity = quantity
    
    def __str__(self):
        return f"<Order {self.order_id}: from {self.origin} to {self.destination} with {self.quantity} parcels>"

class Node:
    def __init__(self, node_id, name, x, y, transfer):
        self.node_id = node_id
        self.name = name
        self.x = x
        self.y = y
        self.transfer = transfer
    
    def __str__(self):
        return f"<Node {self.node_id}: {self.name}, ({self.x}, {self.y}), transfer: {self.transfer}>"

class PDT_Point:
    def __init__(self, point_id: int, type: str, node: Node=None, order: Order=None):
        self.point_id = point_id
        self.type = type
        self.node = node
        self.order = order
    
    @property
    def demand(self) -> int:
        if self.type == "P":
            return self.order.quantity
        elif self.type == "D":
            return -self.order.quantity
        else:
            return 0
    
    @property
    def time_window(self) -> tuple:
        if self.type == "P":
            return (self.order.start_time, np.inf)
        elif self.type == "D":
            return (0, self.order.end_time)
        else:
            return (0, np.inf) 
    
    def __str__(self):
        info = f"<Point {self.point_id}: {self.type}, {self.node.name}"
        if self.type != "T":
            info += f", {self.order.order_id}>"
        return info

if __name__ == '__main__':
    # 设定文件路径
    demand_file_path = "城市配送系统优化资料包\\输入数据\\demand_50.csv"
    node_file_path = "城市配送系统优化资料包\\输入数据\\node_50.csv"
    time_matrix_file_path = "城市配送系统优化资料包\\输入数据\\time_matrix_50.csv"
    parameter_file_path = "城市配送系统优化资料包\\输入数据\\parameter.csv"
    # 读取数据
    # data = Data(demand_file_path, node_file_path, time_matrix_file_path, parameter_file_path)
    data = Data() # use default file path
    print(len(data.orders))
    print(len(data.nodes))
    print(data.node_time_matrix)
    print(data.get_order_by_id(1))
    print(data.get_node_by_name('node_0'))
