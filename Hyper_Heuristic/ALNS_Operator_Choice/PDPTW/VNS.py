import copy
import numpy as np

import Save_Solution
from Fitness_transfer import Fitness_transfer
from Sol import Sol
from Data2 import Data2
from Solomon_3 import Solomon_3


class VNS:

    def __init__(self, data:Data2):
        self.data = data
        self.order_num = self.data.order_num
        self.fitness = Fitness_transfer(data)
        self.end_time = 2500

    def solve(self, solution:Sol):
        neighbor_solutions = []
        self.best_solution = solution
        self.statistic(solution)
        for i in range(30):
            self.car_has_T = [False for i in range(len(solution.routes))] # car是否包含转运（这影响到算子操作）
            new_solution = copy.deepcopy(solution)
            self.by_the_way_operator(new_solution)
            neighbor_solutions.append(new_solution)
            print(f"VNS 顺路算子 第{i}代 obj = {new_solution.obj}")
            if new_solution.obj < self.best_solution.obj:
                self.best_solution = copy.deepcopy(new_solution)
        print(f"\nVNS 顺路算子 最优解 obj = {self.best_solution.obj}")

        return

    def statistic(self, solution:Sol):
        """
        统计solution中的各种信息，方便算子计算
        """

        # 初始化统计变量
        self.node_on_cars = [[] for i in range(len(self.data.nodes))]  # 各个node被哪些car经过
        self.node_has_points = [[] for i in range(len(self.data.nodes))]  # 各个node包含哪些点
        self.car_node_point = [[] for i in range(len(solution.routes))] # car-node-point三维列表
        self.car_node = [[] for i in range(len(solution.routes))] # car-node二维列表

        # 统计信息
        for car_index in range(len(solution.routes)):
            car = solution.routes[car_index]
            node_points = []
            this_node_id = -1
            for point_index in range(len(car)):
                point_id = car[point_index]
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

    def by_the_way_operator(self, solution: Sol) -> bool:
        """
        邻域算子：路过顺便帮你转运
        """

        has_visit_car = [False for i in range(len(solution.routes))]
        change_cars = []
        find_new_solution = False

        # 搜索邻域
        rand_car_i_list = [-1 for i in range(len(self.car_node_point))]
        all_car_i_list = [i for i in range(len(self.car_node_point))]
        for i in range(len(self.car_node_point)):
            index = np.random.randint(0, len(all_car_i_list))
            rand_car_i_list[i] = all_car_i_list.pop(index)

        for car_i in rand_car_i_list:
            # if find_new_solution: #
            #     break
            if has_visit_car[car_i]:
                continue
            flag_find_other_car = False
            car = self.car_node_point[car_i]

            rand_node_i_list = [-1 for i in range(len(car))]
            all_node_i_list = [i for i in range(len(car))]
            for i in range(len(car)):
                index = np.random.randint(0, len(all_node_i_list))
                rand_node_i_list[i] = all_node_i_list.pop(index)

            for node_i_car in rand_node_i_list:
                if has_visit_car[car_i] or flag_find_other_car:
                    break
                node_id = self.car_node[car_i][node_i_car]
                if not self.data.nodes[node_id].transfer:
                    continue
                car_node_points = self.car_node_point[car_i][node_i_car]

                # 判断是否全是Deliver点，顺便计算重量
                flag_is_all_D = True
                weight = 0
                max_P_index = 0
                for point in car_node_points:
                    type = self.data.points[point].type
                    P_point = point - self.order_num
                    order_id = self.data.points[point].order.order_id
                    if type != "D":
                        flag_is_all_D = False
                        break
                    elif not (P_point in solution.routes[car_i]):
                        flag_is_all_D = False
                        break
                    else:
                        P_index = solution.routes[car_i].index(P_point)
                        max_P_index = max(max_P_index, P_index)
                        weight += self.data.points[point].order.quantity

                if not flag_is_all_D:
                    continue

                # 寻找顺路车辆
                node_cars = self.node_on_cars[node_id]
                car_nodes = self.car_node[car_i]

                rand_car2_i_list = [-1 for i in range(len(node_cars))]
                all_car2_i_list = copy.copy(node_cars)
                for i in range(len(node_cars)):
                    index = np.random.randint(0, len(all_car2_i_list))
                    rand_car2_i_list[i] = all_car2_i_list.pop(index)

                for car2_i in rand_car2_i_list:
                    if flag_find_other_car:
                        break
                    if has_visit_car[car2_i] or (car2_i == car_i) or self.car_has_T[car2_i]:
                        continue

                    rand_node2_i_list = [-1 for i in range(node_i_car)]
                    all_node2_i_list = [i for i in range(node_i_car)]
                    for i in range(node_i_car):
                        index = np.random.randint(0, len(all_node2_i_list))
                        rand_node2_i_list[i] = all_node2_i_list.pop(index)

                    car2_nodes = self.car_node[car2_i]
                    for node2_i_car in rand_node2_i_list:
                        if flag_find_other_car:
                            break
                        node2_id = car_nodes[node2_i_car]
                        if (node2_id == node_id) or (node2_id not in car2_nodes) or (not self.data.nodes[node2_id].transfer):
                            continue

                        node_i_car2 = car2_nodes.index(node_id)
                        node2_i_car2 = car2_nodes.index(node2_id)
                        if node2_i_car2 > node_i_car2:
                            continue

                        # print(car_index, other_node_id, self.car_node_point[car_index])
                        car_node2_points = self.car_node_point[car_i][node2_i_car]
                        car_node2_last_point = car_node2_points[-1]
                        car_node2_first_point = car_node2_points[0]
                        car_node2_last_point_i = solution.routes[car_i].index(car_node2_last_point)
                        car_node2_first_point_i = solution.routes[car_i].index(car_node2_first_point)
                        if car_node2_first_point_i <= max_P_index:
                            continue

                        car_time = solution.time_record[car_i][car_node2_last_point_i]
                        car2_node2_points = self.car_node_point[car2_i][node2_i_car2]
                        car2_node2_last_point = car2_node2_points[-1]
                        car2_node2_last_point_i = solution.routes[car2_i].index(car2_node2_last_point)
                        car2_time = solution.time_record[car2_i][car2_node2_last_point_i]

                        max_wait_time = max(0, self.end_time - solution.finish_time[car2_i])
                        if car_time + self.data.parcel_transfer_time > car2_time + max_wait_time:
                            continue

                        car2_node_points = self.car_node_point[car2_i][node_i_car2]
                        car2_node_first_point = car2_node_points[0]
                        car2_node_first_point_i = solution.routes[car2_i].index(car2_node_first_point)
                        flag_overlode = False
                        for car2_pass_point_i in range(car2_node2_last_point_i, car2_node_first_point_i):
                            car2_pass_point_cap = solution.cap_record[car2_i][car2_pass_point_i]
                            if car2_pass_point_cap + weight > self.data.vehicle_capacity:
                                flag_overlode = True
                                break
                        if flag_overlode:
                            continue

                        car_last_node = -1
                        car_next_node = -1
                        if node_i_car > 0:
                            car_last_node = self.car_node[car_i][node_i_car-1]
                        if node_i_car+1 < len(self.car_node[car_i]):
                            car_next_node = self.car_node[car_i][node_i_car+1]
                        old_travel_time = self.travel_time(car_last_node, node_id)
                        old_travel_time += self.travel_time(node_id, car_next_node)
                        new_travel_time = self.travel_time(car_last_node, car_next_node)
                        save_cost = (old_travel_time - new_travel_time) * self.data.vehicle_unit_travel_cost
                        if save_cost < self.data.parcel_transfer_unit_cost * weight:
                            continue

                        for remove_point in car_node_points:
                            solution.routes[car_i].remove(remove_point)
                            add_D_point = remove_point
                            add_T1_point = add_D_point + self.order_num
                            add_T2_point = add_T1_point + self.order_num
                            self.data.points[add_T1_point].node = self.data.nodes[node2_id]
                            self.data.points[add_T2_point].node = self.data.nodes[node2_id]
                            T1_insert_index = car_node2_first_point_i
                            T2_insert_index = car2_node2_last_point_i + 1
                            D_insert_index = car2_node_first_point_i
                            solution.routes[car_i].insert(T1_insert_index, add_T1_point)
                            solution.routes[car2_i].insert(D_insert_index, add_D_point)
                            solution.routes[car2_i].insert(T2_insert_index, add_T2_point)
                            self.car_has_T[car_i] = True
                            self.car_has_T[car2_i] = True

                        has_visit_car[car_i] = True
                        has_visit_car[car2_i] = True
                        change_cars.append(car_i)
                        change_cars.append(car2_i)
                        find_new_solution = True
                        flag_find_other_car = True
                        break

        self.fitness.cal_objective(solution, change_cars)
        return find_new_solution

    def travel_time(self, node_id1, node_id2):
        if node_id1 < 0 or node_id2 < 0:
            return 0
        else:
            return self.data.node_time_matrix[node_id1][node_id2]


if __name__ == "__main__":
    data = Data2()
    solution = Solomon_3().init_solotion(data)
    vns = VNS(data)
    vns.solve(solution)
    Save_Solution.save(data, vns.best_solution)