import copy
import queue

import numpy as np

from Fitness_transfer import Fitness_transfer
from Sol import Sol
from Data import Node
from Data2 import Data2

class Solomon_3:

    def __init__(self):
        return

    def init_solotion(self, data:Data2) -> Sol:
        order_num = data.order_num
        end_time = 2500

        Ps = [[-1 for i in range(len(data.nodes))] for i in range(len(data.nodes))]
        Ds = [[-1 for i in range(len(data.nodes))] for i in range(len(data.nodes))]
        P_num_in_nodes = [0 for i in range(len(data.nodes))]
        D_num_in_nodes = [0 for i in range(len(data.nodes))]
        for P in range(1, 1+order_num):
            order = data.points[P].order
            origin = int(order.origin[5:])
            destination = int(order.destination[5:])
            Ps[int(origin)][int(destination)] = P
            Ds[int(origin)][int(destination)] = P + order_num
            P_num_in_nodes[origin] += 1
            D_num_in_nodes[destination] += 1

        near_nodes = [[] for i in range(len(data.nodes))]
        for node_id in range(len(data.nodes)):
            node_que = queue.PriorityQueue()
            for other_node_id in range(len(data.nodes)):
                if other_node_id != node_id:
                    travel_time = data.node_time_matrix[node_id][other_node_id]
                    node_que.put((travel_time, other_node_id))
            for other_node_id in range(len(data.nodes)-1):
                near_node_id = node_que.get()[1]
                if near_node_id != node_id:
                    near_nodes[node_id].append(near_node_id)

        solution = Sol(data)
        while True:
            flag_find_task = False
            for node_id in range(len(data.nodes)):
                if P_num_in_nodes[node_id] > 0:
                    flag_find_task = True

                    max_step = 20
                    nodes = [-1 for i in range(max_step)]
                    P_points = [[] for i in range(max_step)]
                    D_points = [[] for i in range(max_step)]
                    fast = 0
                    cap = 0
                    time = data.vehicle_service_time
                    nodes[0] = node_id
                    this_node = node_id

                    for slow in range(max_step):
                        if nodes[slow] < 0:
                            break

                        # 慢指针点的D点处理
                        for D in D_points[slow]:
                            cap -= data.points[D].order.quantity

                        # 快指针往后搜索
                        from_node = nodes[slow]
                        cap_fast = cap
                        while fast < max_step - 1:
                            flag_has_new_node = False
                            for next_node in near_nodes[this_node]:
                                P = Ps[from_node][next_node]
                                if P < 0:
                                    continue
                                new_time = data.vehicle_service_time
                                new_time += data.node_time_matrix[this_node][next_node]
                                if time + new_time > end_time:
                                    break
                                new_cap = data.points[P].order.quantity
                                if cap_fast + new_cap > data.vehicle_capacity:
                                    continue

                                # 接受next_node作为fast指针的下一个点
                                fast += 1
                                nodes[fast] = next_node
                                time += new_time
                                cap_fast += new_cap
                                P_points[slow].append(P)
                                D_points[fast].append(P+order_num)
                                Ps[from_node][next_node] = -1
                                Ds[from_node][next_node] = -1
                                P_num_in_nodes[from_node] -= 1
                                D_num_in_nodes[next_node] -= 1

                                this_node = next_node
                                flag_has_new_node = True
                                break
                            if not flag_has_new_node:
                                break

                        # 慢指针点的P点处理
                        for P in P_points[slow]:
                            cap += data.points[P].order.quantity

                    car = []
                    for step in range(max_step):
                        for D in D_points[step]:
                            car.append(D)
                        for P in P_points[step]:
                            car.append(P)

                    solution.add_route(car)

            if flag_find_task:
                continue
            else:
                break

        fit = Fitness_transfer(data)
        fit.check_PD_sol(solution)
        obj = fit.cal_objective(solution=solution, change_cars=[])
        # print(obj)
        return solution




