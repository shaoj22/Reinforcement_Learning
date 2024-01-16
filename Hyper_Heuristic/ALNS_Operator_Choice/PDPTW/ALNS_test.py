import numpy as np
import matplotlib.pyplot as plt
import math
import tqdm
import copy
import time 
from Operators import *
from Solomon_insert import solomon_insert_algorithm as SI 
from Data import Node
from Data2 import Data2

class ALNS_base:
    """
    Base class of ALNS algorithm
        ALNS算法的通用框架，实际使用时，需要继承该类，并实现以下方法：
            1. get_operators_list: 返回一个operator的list，每个operator都是一个类，实现了get方法 
            2. solution_init: 返回一个初始解
            3. cal_objective: 计算解的目标函数值
    """
    def __init__(self, iter_num):
        self.iter_num = iter_num
        
        # set params
        ## 1. ALNS params
        self.adaptive_period = 1000
        self.sigma1 = 2
        self.sigma2 = 1
        self.sigma3 = 0.1
        ## 2. SA params
        self.max_temp = 0.01
        self.min_temp = 1e-10
        self.cooling_rate = 0.97
        self.cooling_period = 30
    
    # to be implemented in subclass
    def set_operators_list(self):
        self.break_operators_list = []
        self.repair_operators_list = []
        raise NotImplementedError

    # to be implemented in subclass
    def solution_init(self):
        raise NotImplementedError
    
    # to be implemented in subclass
    def cal_objective(self):
        raise NotImplementedError 

    def reset(self):
        self.reset_operators_scores()
        self.obj_iter_process = []
    
    def reset_operators_scores(self):
        self.break_operators_scores = np.ones(len(self.break_operators_list))
        self.repair_operators_scores = np.ones(len(self.repair_operators_list))
        self.break_operators_steps = np.ones(len(self.break_operators_list))
        self.repair_operators_steps = np.ones(len(self.repair_operators_list))
    
    def SA_accept(self, detaC, temperature):
        return math.exp(-detaC / temperature)

    def temperature_update(self, temperature, step):
        if step % self.cooling_period == 0: # update temperature by static steps
            temperature *= self.cooling_rate
        temperature = max(self.min_temp, temperature)
        return temperature

    def choose_operator(self):
        break_weights = self.break_operators_scores / self.break_operators_steps
        repair_weights = self.repair_operators_scores / self.repair_operators_steps
        break_prob = break_weights / sum(break_weights)
        repair_prob = repair_weights / sum(repair_weights)
        break_opt_i = np.random.choice(range(len(self.break_operators_list)), p=break_prob)
        repair_opt_i = np.random.choice(range(len(self.repair_operators_list)), p=repair_prob)
        return break_opt_i, repair_opt_i
    
    def get_neighbour(self, solution, break_opt_i, repair_opt_i):
        solution = copy.deepcopy(solution)
        solution , removed_pair_list , remove_car_list = self.break_operators_list[break_opt_i].set(solution)
        self.repair_operators_list[repair_opt_i].set(solution , removed_pair_list , remove_car_list)
        return solution

    def show_process(self):
        y = self.obj_iter_process
        x = np.arange(len(y))
        plt.plot(x, y)
        plt.title("Iteration Process of ALNS")
        plt.xlabel("Iteration")
        plt.ylabel("Objective")
        plt.show()

    def run(self):
        self.reset()
        cur_solution = self.solution_init() 
        cur_obj = self.cal_objective(cur_solution)
        self.best_solution = cur_solution
        self.best_obj = cur_obj
        temperature = self.max_temp
        pbar = tqdm.tqdm(range(self.iter_num), desc="ALNS Iteration")
        time_ = 0
        for step in pbar:
            break_opt_i, repair_opt_i = self.choose_operator()
            new_solution = self.get_neighbour(cur_solution, break_opt_i, repair_opt_i) 
            
            time_s = time.time()
            new_obj = self.cal_objective(new_solution)
            time_e = time.time()
            time_ += time_e - time_s
            
            # obj: minimize the total distance 
            if new_obj < self.best_obj:
                self.best_solution = new_solution
                self.best_obj = new_obj
                cur_solution = new_solution
                cur_obj = new_obj
                self.break_operators_scores[break_opt_i] += self.sigma1
                self.break_operators_steps[break_opt_i] += 1
                self.repair_operators_scores[repair_opt_i] += self.sigma1
                self.repair_operators_steps[repair_opt_i] += 1
            elif new_obj < cur_obj: 
                cur_solution = new_solution
                cur_obj = new_obj
                self.break_operators_scores[break_opt_i] += self.sigma2
                self.break_operators_steps[break_opt_i] += 1
                self.repair_operators_scores[repair_opt_i] += self.sigma2
                self.repair_operators_steps[repair_opt_i] += 1
            elif np.random.random() < self.SA_accept((new_obj-cur_obj)/(cur_obj+1e-10), temperature): # percentage detaC
                cur_solution = new_solution
                cur_obj = new_obj
                self.break_operators_scores[break_opt_i] += self.sigma3
                self.break_operators_steps[break_opt_i] += 1
                self.repair_operators_scores[repair_opt_i] += self.sigma3
                self.repair_operators_steps[repair_opt_i] += 1
            # reset operators weights
            if step % self.adaptive_period == 0: 
                self.reset_operators_scores()
            # update SA temperature
            temperature = self.temperature_update(temperature, step)
            # record
            self.obj_iter_process.append(cur_obj)
            pbar.set_postfix({
                "best_obj" : self.best_obj, 
                "cur_obj" : cur_obj, 
                "temperature" : temperature
            })
        print("time: ", time_)
        return self.best_solution, self.best_obj
    
class Sol:
    def __init__(self):
        self.route = [[]]
        self.obj = 0
        self.objs = []
        self.cost = []
        self.caps = []
        self.overload = []
        self.delays = []
    
class ALNS(ALNS_base):
    
    def __init__(self, data , iter_num):
        super().__init__(iter_num)
        self.data = data
        self.set_operators_list(data)
    
    def set_operators_list(self, data):
        self.break_operators_list = [
            RandomBreak(data)
            ]
        self.repair_operators_list = [RandomRepair(data)]
        
    def solution_init(self):
        si = SI(data)
        solution = Sol()
        solution.route = si.solomon_insert()
        
        
        return solution
    
    def _get_dist_from_point_idx(self, pi, pj):
        return self.dist(self.data.points[pi].node, self.data.points[pj].node)
    
    def _backupon(self, solution, reward, pi2arrive_time, pi2leave_time, pi2car_idx, T2toT1, tabu, point):
        """ 从头遍历point所在route，知道求出point的时间 """
        if pi2arrive_time[point] != -1:
            # 若已经知道该点时间，则不再计算该点时间
            return 
        if point in tabu:
            # 如果出现交叉等待
            reward += 10000
            pi2arrive_time[point] = pi2arrive_time[point-1] + self._get_dist_from_point_idx(point-1, point) + 10000
            pi2leave_time[point] = pi2arrive_time[point] + self.data.vehicle_service_time
            return
        car_idx = pi2car_idx[point]
        car_route = solution.route[car_idx]
        for i in range(len(car_route)):
            pi = car_route[i]
            if pi2arrive_time[pi] != -1:
                # 若已经知道该点时间，则不再计算该点时间
                continue
            if i == 0:
                pi2arrive_time[pi] = 0
            else:
                pi2arrive_time[pi] = pi2leave_time[car_route[i-1]] + self._get_dist_from_point_idx(car_route[i-1], pi)
            pi2arrive_time[pi] = max(pi2arrive_time[pi], self.data.points[pi].time_window[0])
            if pi2arrive_time[pi] > self.data.points[pi].time_window[1]:
                # 超时惩罚
                reward += 10000
            leave_time = pi2arrive_time[pi] + self.data.vehicle_service_time
            if self.data.points[pi].type != "T2":
                # 若不是T2，正常更新时间
                pi2leave_time[pi] = leave_time
            else:
                # 如果是T2，需要先计算T1的时间
                tabu[pi] = True # 防止交叉等待
                self._backupon(solution, reward, pi2arrive_time, pi2leave_time, pi2car_idx, T2toT1, tabu, T2toT1[pi])
                transfer_time = pi2arrive_time[T2toT1[pi]] + self.data.parcel_transfer_time
                pi2leave_time[pi] = max(transfer_time, leave_time)
            if pi == point:
                break
    
    def cal_objective(self, solution):
        """ 求出solution的目标函数值 """
        solution = copy.deepcopy(solution)
        # 1. 预处理路径，使路径中的T1,T2删除, 计算pi2arrive_time, pi2leave_time, pi2car_idx, T2toT1
        pi2arrive_time = [-1 for i in range(len(self.data.points))]
        pi2leave_time = [-1 for i in range(len(self.data.points))]
        pi2car_idx = [-1 for i in range(len(self.data.points))]
        T2toT1 = {pi : pi-1 for pi in range(len(self.data.points)) if self.data.points[pi].type == "T2"}
        T1toT2 = {pi : pi+1 for pi in range(len(self.data.points)) if self.data.points[pi].type == "T1"}
        T1_isvalid = {pi : True for pi in range(len(self.data.points)) if self.data.points[pi].type == "T1"}
        for car_idx, route in enumerate(solution.route):
            for pi in route:
                pi2car_idx[pi] = car_idx
        for t1 in T1toT2.keys():
            t2 = T1toT2[t1]
            if pi2car_idx[t1] == pi2car_idx[t2]:
                # 若T1和T2在同一车辆中，则删除T1和T2
                solution.route[pi2car_idx[t1]].remove(t1)
                solution.route[pi2car_idx[t2]].remove(t2)
                T1_isvalid[t1] = False
        pi2arrive_time[0] = 0
        pi2leave_time[0] = 0
        # 2. 递归计算未求出的时间
        reward = 0
        tabu = {}
        for car_idx, route in enumerate(solution.route):
            for pi in route:
                if pi2arrive_time[pi] != -1:
                    continue
                self._backupon(solution, reward, pi2arrive_time, pi2leave_time, pi2car_idx, T2toT1, tabu, pi)
        # 3. 计算reward
        ## 1. 车辆固定成本
        reward += self.data.vehicle_fixed_cost * len(solution.route)
        ## 2. 车辆时间成本
        for car_idx, route in enumerate(solution.route):
            for i in range(1, len(route)):
                pi = route[i]
                reward += self.data.vehicle_unit_travel_cost * self._get_dist_from_point_idx(route[i-1], pi)
        ## 3. 转运成本
        for t1 in T1toT2.keys():
            t2 = T1toT2[t1]
            if T1_isvalid[t1]:
                reward += self.data.parcel_transfer_unit_cost * self.data.points[t1].order.quantity
        return reward   

    def dist(self, node1:Node, node2:Node) -> float:
        if node1 == None or node2 == None:
            return 0
        else:
            return self.data.node_time_matrix[node1.node_id][node2.node_id]
    
    def choose_operator(self):
        # choose break operator
        break_weights = self.break_operators_scores / self.break_operators_steps
        break_prob = break_weights / sum(break_weights)
        break_opt_i = np.random.choice(range(len(self.break_operators_list)), p=break_prob)
        # filter repair operators with the same type
        
        # choose repair operator
        repair_weights = self.repair_operators_scores / self.repair_operators_steps
        repair_prob = repair_weights / sum(repair_weights)
        repair_opt_i = np.random.choice(range(len(self.repair_operators_list)), p=repair_prob)
        return break_opt_i, repair_opt_i
    
    def test_run(self):
        cur_solution = self.solution_init() 
        cur_obj = self.cal_objective(cur_solution)
        return cur_solution, cur_obj
    
if __name__ == "__main__":
    data = Data2()
    iter_num = 300
    alg = ALNS(data, iter_num)
    alg.run()
    alg.show_process()