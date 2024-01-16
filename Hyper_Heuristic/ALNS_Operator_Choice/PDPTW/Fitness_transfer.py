import numpy as np

from Sol import Sol
from Data import Node
from Data2 import Data2

class Fitness_transfer:

    def __init__(self, data: Data2):
        self.data = data
        self.order_num = self.data.order_num

    def cal_objective(self, solution: Sol, change_cars: list):

        if change_cars == None or len(change_cars) == 0:
            change_cars = [i for i in range(len(solution.routes))]
            solution.T1_dis = [[-1, -1] for i in range(len(self.data.orders))]
            solution.T2_dis = [[-1, -1] for i in range(len(self.data.orders))]
            solution.points_dis = [[-1, -1] for i in range(len(self.data.points))]

        self.car_nodesss = [[] for i in range(len(solution.routes))]
        self.arrival_times = [[] for i in range(len(solution.routes))]
        self.departure_times = [[] for i in range(len(solution.routes))]

        start_time = 0
        end_time = 2500

        # 统计各个订单的T1和T2在不同车辆中的分布
        for i in change_cars:
            for j in range(len(solution.routes[i])):
                point_id = solution.routes[i][j]
                order_id = self.data.points[point_id].order.order_id
                solution.points_dis[point_id] = [i, j]
                if self.data.points[point_id].type == "T1":
                    solution.T1_dis[order_id] = [i, j]
                    solution.has_transfer[i] = True
                elif self.data.points[point_id].type == "T2":
                    solution.T2_dis[order_id] = [i, j]
                    solution.has_transfer[i] = True

        # # 求解各个转运点的最佳node
        # for car_id in change_cars:
        #     car = solution.route[car_id]
        #     for i in range(len(car)):
        #         point_id = car[i]
        #         if self.data.points[point_id].type == "T1" or self.data.points[point_id].type == "T2":
        #             order_id = self.data.points[point_id].order.order_id
        #             if solution.T1_dis[order_id][0] != solution.T2_dis[order_id][0]:  # 不同车才转运
        #                 self.cal_T_node_for_fitness(solution, car, i, point_id)

        # 初始化路径指标
        times = [np.inf for i in range(len(solution.routes))]  # 各辆车下一个事件发生的时间
        nodes = [-1 for i in range(len(solution.routes))]  # 各辆车下一个节点编号
        steps = [0 for i in range(len(solution.routes))]  # 各辆车已经走过的points数量
        caps = [0 for i in range(len(solution.routes))]  # 各辆车的载重
        point_num = 0
        for car_id in change_cars:
            solution.cost[car_id] = 0  # 重置部分车辆的费用
            solution.overload[car_id] = 0  # 重置部分车辆的累积超载量
            solution.delays[car_id] = 0  # 重置部分车辆的延误
            solution.time_record[car_id] = []
            solution.cap_record[car_id] = []
            if len(solution.routes[car_id]) > 0:  # 非空车辆有固定费用，且需要设定开始时间为0
                solution.cost[car_id] += self.data.vehicle_fixed_cost
                times[car_id] = start_time
                point_num += len(solution.routes[car_id])
                for point_id in solution.routes[car_id]:
                    if self.data.points[point_id].type == "T1":
                        order_id = self.data.points[point_id].order.order_id
                        if solution.T1_dis[order_id][0] != solution.T2_dis[order_id][0]:  # 不同车才转运
                            solution.transfer_time[order_id] = end_time * 10  # 重置部分订单的转运时间

        # 开始求解目标值
        while True:

            # 找出最早的下一个事件的车辆，及其进行到哪一个订单
            early_car_id = -1
            min_time = end_time * 1000
            for car_id in change_cars:
                if len(solution.routes[car_id]) > 0 and times[car_id] < min_time:

                    # this_point跳过不需要转运的T1 T2
                    while steps[car_id] < len(solution.routes[car_id]):
                        this_point_id = solution.routes[car_id][steps[car_id]]
                        order_id = self.data.points[this_point_id].order.order_id
                        type = self.data.points[this_point_id].type
                        if type == "P" or type == "D":
                            break
                        elif solution.T1_dis[order_id][0] != solution.T2_dis[order_id][0]:
                            break
                        else:
                            # 跳过一个point，需要记录一下时间和载重
                            solution.time_record[car_id].append(times[car_id])
                            solution.cap_record[car_id].append(caps[car_id])
                            solution.times[this_point_id] = times[car_id]
                            solution.caps[this_point_id] = caps[car_id]
                            steps[car_id] += 1

                    if steps[car_id] == len(solution.routes[car_id]):
                        solution.finish_time[car_id] = times[car_id]
                        self.departure_times[car_id].append(times[car_id])
                        solution.delays[car_id] += max(0, times[car_id] - end_time)
                        times[car_id] = np.inf
                        continue

                    # 该点旅行时间和费用，服务时间
                    node = self.data.points[this_point_id].node
                    if node.node_id != nodes[car_id]:
                        travel_time = self.dist2(nodes[car_id], node.node_id)
                        solution.cost[car_id] += travel_time * self.data.vehicle_unit_travel_cost
                        if nodes[car_id] != -1:
                            self.departure_times[car_id].append(times[car_id])
                        times[car_id] += travel_time
                        self.car_nodesss[car_id].append(node.node_id)
                        self.arrival_times[car_id].append(times[car_id])
                        times[car_id] += self.data.vehicle_service_time
                        nodes[car_id] = node.node_id
                    solution.times[this_point_id] = times[car_id]

                    if self.data.points[this_point_id].type == "T2":
                        this_time = max(solution.transfer_time[order_id], times[car_id])
                    else:
                        this_time = times[car_id]

                    if this_time < min_time:
                        min_time = this_time
                        early_car_id = car_id
                # print(f"car {car_id} time {this_time} point {this_point_id} min time {min_time}")

            if early_car_id < 0:
                for car_id in change_cars:
                    if steps[car_id] < len(solution.routes[car_id]):
                        print("警告！车辆", car_id, "未能完成所有订单！")
                break

            # print()
            # print(f"car = {early_car_id} time before = {min_time}")

            times[early_car_id] = min_time

            car = solution.routes[early_car_id]
            this_point_id = car[steps[early_car_id]]
            order_id = self.data.points[this_point_id].order.order_id
            type = self.data.points[this_point_id].type

            # 计算载重变化
            if type == "P" or type == "T2":
                caps[early_car_id] += self.data.points[this_point_id].order.quantity
            elif type == "D" or type == "T1":
                caps[early_car_id] -= self.data.points[this_point_id].order.quantity
            if caps[early_car_id] > self.data.vehicle_capacity:
                solution.overload[early_car_id] += caps[early_car_id] - self.data.vehicle_capacity
            solution.caps[this_point_id] = caps[early_car_id]

            # 计算转运费用，更新转运时间
            if type == "T1":
                solution.cost[early_car_id] += self.data.points[
                                                   this_point_id].order.quantity * self.data.parcel_transfer_unit_cost
                solution.transfer_time[order_id] = times[early_car_id] + self.data.parcel_transfer_time

            # 服务完成一个point，需要记录一下时间和载重
            solution.time_record[early_car_id].append(times[early_car_id])
            solution.cap_record[early_car_id].append(caps[early_car_id])
            steps[early_car_id] += 1

        for car_id in change_cars:
            solution.objs[car_id] = solution.cost[car_id] + 100 * solution.delays[car_id] + 100 * solution.overload[
                car_id]
        solution.obj = sum(solution.objs)
        return solution.obj

    def statistic(self, solution:Sol):
        """
        统计solution中的各种信息，方便算子计算
        """

        # 初始化统计变量
        self.node_on_cars = [[] for i in range(len(self.data.nodes))]  # 各个node被哪些car经过
        self.node_has_points = [[] for i in range(len(self.data.nodes))]  # 各个node包含哪些点
        self.car_node_point = [[] for i in range(len(solution.routes))] # car-node-point三维列表
        self.car_node = [[] for i in range(len(solution.routes))] # car-node二维列表
        self.point_on_car = [-1 for i in range(len(self.data.points))]

        # 统计信息
        for car_index in range(len(solution.routes)):
            car = solution.routes[car_index]
            node_points = []
            this_node_id = -1
            for point_index in range(len(car)):
                point_id = car[point_index]
                self.point_on_car[point_id] = car_index
                node = self.data.points[point_id].node
                if node == None:
                    print(f"警告！VNS.by_the_way_operator()中出现没有计算node的point {point_id}")
                    continue

                # 统计car-node-point三维列表和其他信息
                self.node_has_points[node.node_id].append(point_id)
                if this_node_id < 0:
                    this_node_id = node.node_id
                    node_points.append(point_id)
                elif this_node_id == node.node_id:
                    node_points.append(point_id)
                else:
                    self.car_node_point[car_index].append(node_points)
                    self.car_node[car_index].append(this_node_id)
                    self.node_on_cars[this_node_id].append(car_index)
                    this_node_id = node.node_id
                    node_points = [point_id]

            self.car_node_point[car_index].append(node_points)
            self.car_node[car_index].append(this_node_id)
            self.node_on_cars[this_node_id].append(car_index)

    def check_PD_sol(self, solution: Sol) -> bool:
        points_nums = [0 for i in range(len(self.data.points))]
        P_dis = [[] for i in range(len(self.data.orders))]
        D_dis = [[] for i in range(len(self.data.orders))]
        for i in range(len(solution.routes)):
            car = solution.routes[i]
            for j in range(len(car)):
                point = car[j]
                points_nums[point] += 1
                order_id = self.data.points[point].order.order_id
                type = self.data.points[point].type
                if type == "P":
                    P_dis[order_id] = [i, j]
                elif type == "D":
                    D_dis[order_id] = [i, j]
                else:
                    print("警告！！解中存在PD之外的点！！")
                    return False

        for i in range(1, 1 + self.order_num + self.order_num):
            point_num = points_nums[i]
            if point_num != 1:
                print(f"point {i} finish {point_num} times!!!!")
                return False

        for i in range(len(self.data.orders)):
            if P_dis[i][0] != D_dis[i][0]:
                print(f"order {i} P at car {P_dis[i][0]} D at car {D_dis[i][0]} !!!!")
                return False
            if P_dis[i][1] >= D_dis[i][1]:
                print(f"order {i} P at index {P_dis[i][1]} D at index {D_dis[i][1]} !!!!")
                return False

        # print("perfect solution")
        return True

    def check_PDT_sol(self, solution: Sol) -> bool:
        points_nums = [0 for i in range(len(self.data.points))]
        P_dis = [[] for i in range(len(self.data.orders))]
        D_dis = [[] for i in range(len(self.data.orders))]
        T1_dis = [[] for i in range(len(self.data.orders))]
        T2_dis = [[] for i in range(len(self.data.orders))]
        for i in range(len(solution.routes)):
            car = solution.routes[i]
            for j in range(len(car)):
                point = car[j]
                points_nums[point] += 1
                order_id = self.data.points[point].order.order_id
                type = self.data.points[point].type
                if type == "P":
                    P_dis[order_id] = [i, j]
                elif type == "D":
                    D_dis[order_id] = [i, j]
                elif type == "T1":
                    T1_dis[order_id] = [i, j]
                elif type == "T2":
                    T2_dis[order_id] = [i, j]

        for P in range(1, 1 + self.order_num):
            D = P + self.order_num
            T1 = D + self.order_num
            T2 = T1 + self.order_num
            if points_nums[P] != 1:
                print(f"P point {P} finish {points_nums[P]} times!!!!")
                return False
            if points_nums[D] != 1:
                print(f"D point {D} finish {points_nums[D]} times!!!!")
                return False
            if points_nums[T1] > 1 or points_nums[T2] > 1 or points_nums[T1] != points_nums[T2]:
                print(f"T1 point {T1} finish {points_nums[T1]} times!!!!")
                print(f"T2 point {T2} finish {points_nums[T2]} times!!!!")
                return False

        for i in range(len(self.data.orders)):
            if P_dis[i][0] != D_dis[i][0]:
                if len(T1_dis[i]) < 2 or len(T2_dis[i]) < 2:
                    print(f"order {i} P at car {P_dis[i][0]} D at car {D_dis[i][0]} !!!!")
                    return False
                if P_dis[i][0] != T1_dis[i][0]:
                    print(f"order {i} P at car {P_dis[i][0]} T1 at car {T1_dis[i][0]} !!!!")
                    return False
                if D_dis[i][0] != T2_dis[i][0]:
                    print(f"order {i} D at car {D_dis[i][0]} T2 at car {T2_dis[i][0]} !!!!")
                    return False
                if P_dis[i][1] >= T1_dis[i][1]:
                    print(f"order {i} P at index {P_dis[i][1]} T1 at index {T1_dis[i][1]} !!!!")
                    return False
                if T2_dis[i][1] >= D_dis[i][1]:
                    print(f"order {i} T2 at index {T2_dis[i][1]} D at index {D_dis[i][1]} !!!!")
                    return False
            elif P_dis[i][1] >= D_dis[i][1]:
                print(f"order {i} P at index {P_dis[i][1]} D at index {D_dis[i][1]} !!!!")
                return False

        # print("perfect solution")
        return True

    def cal_T_node_for_fitness(self, solution: Sol, point_list_a: list, Ta_index:int, Ta_point_id:int):

        # 找出孪生的Tb
        order_id = self.data.points[Ta_point_id].order.order_id
        if self.data.points[Ta_point_id].type == "T1":
            car_id = solution.T2_dis[order_id][0]
            Tb_index = solution.T2_dis[order_id][1]
            Tb_point_id = Ta_point_id + self.order_num
        elif self.data.points[Ta_point_id].type == "T2":
            car_id = solution.T1_dis[order_id][0]
            Tb_index = solution.T1_dis[order_id][1]
            Tb_point_id = Ta_point_id - self.order_num
        else:
            return
        point_list_b = solution.routes[car_id]

        # 找到对应的左右端点
        Ta_left_node = self.left_point(solution, point_list_a, Ta_index).node
        Ta_right_node = self.right_point(solution, point_list_a, Ta_index+1).node
        Tb_left_node = self.left_point(solution, point_list_b, Tb_index).node
        Tb_right_node = self.right_point(solution, point_list_b, Tb_index+1).node

        min_time = np.inf
        best_node = None
        for node in self.data.nodes:
            if node.transfer:
                time = self.dist(Ta_left_node, node)
                time += self.dist(node, Ta_right_node)
                time += self.dist(Tb_left_node, node)
                time += self.dist(node, Tb_right_node)
                if time < min_time:
                    min_time = time
                    best_node = node

        self.data.points[Ta_point_id].node = best_node
        self.data.points[Tb_point_id].node = best_node

    def left_point(self, solution, point_list, T_index):

        # 初始左端点
        T_left_point_index = T_index - 1

        # 跳过T右边没有转运的T点
        while True:
            if T_left_point_index >= 0:
                T_left_point_id = point_list[T_left_point_index]
                T_left_point_type = self.data.points[T_left_point_id].type
                order_id = self.data.points[T_left_point_id].order.order_id
                if T_left_point_type == "T1" or T_left_point_type == "T2":
                    if solution.T1_dis[order_id][0] == solution.T2_dis[order_id][0]:  # 没有转运的T点跳过
                        T_left_point_index -= 1
                    elif self.data.points[T_left_point_id].node == None:
                        T_left_point_index -= 1
                    else:
                        return self.data.points[T_left_point_id]
                else:
                    return self.data.points[T_left_point_id]
            else:  # 索引已经小于0，返回虚拟point 0
                return self.data.points[0]

    def right_point(self, solution: Sol, point_list: list, T_index: int):

        # 初始右端点
        T_right_point_index = T_index

        # 跳过T右边没有转运的T点
        while True:
            if T_right_point_index < len(point_list):
                T_right_point_id = point_list[T_right_point_index]
                T_right_point_type = self.data.points[T_right_point_id].type
                order_id = self.data.points[T_right_point_id].order.order_id
                if T_right_point_type == "T1" or T_right_point_type == "T2":
                    if solution.T1_dis[order_id][0] == solution.T2_dis[order_id][0]:  # 没有转运的T点跳过
                        T_right_point_index += 1
                    elif self.data.points[T_right_point_index].node == None:
                        T_right_point_index += 1
                    else:
                        return self.data.points[T_right_point_id]
                else:
                    return self.data.points[T_right_point_id]
            else:  # 索引已经=len(point_list)，返回虚拟point 0
                return self.data.points[0]

    def dist(self, node1:Node, node2:Node) -> float:
        if node1 == None or node2 == None:
            return 0
        else:
            return self.data.node_time_matrix[node1.node_id][node2.node_id]


    def dist2(self, node1_id:int, node2_id:int) -> float:
        if node1_id < 0 or node2_id < 0:
            return 0
        else:
            return self.data.node_time_matrix[node1_id][node2_id]