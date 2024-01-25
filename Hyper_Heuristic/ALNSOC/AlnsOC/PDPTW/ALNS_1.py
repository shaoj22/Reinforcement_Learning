import numpy as np
import matplotlib.pyplot as plt
import math
import tqdm
import copy
import time 
import json
from Fitness import *
from Operators import *
from Solomon_insert import solomon_insert_algorithm as SI 
from Solomon_3 import Solomon_3 as SI_3
from Sol import Sol
from Data import Node
from Data2 import Data2
from TSP import TSP_model
# import heartrate

# heartrate.trace(browser=True)

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
        self.cooling_rate = 0.95
        self.cooling_period = 30
    
    # to be implemented in subclass
    def set_operators_list(self):
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
        solution , break_info = self.break_operators_list[break_opt_i].set(solution)
        
        return self.repair_operators_list[repair_opt_i].set(solution , break_info)

    def show_process(self):
        y = self.obj_iter_process
        x = np.arange(len(y))
        plt.plot(x, y)
        plt.title("Iteration Process of ALNS")
        plt.xlabel("Iteration")
        plt.ylabel("Objective")
        # plt.show()
        plt.savefig('{}/iters.png'.format(self.scale))
        plt.close()

    def run(self):
        self.reset()
        cur_solution = self.solution_init() 
        
        
        cur_obj = self.cal_objective(cur_solution , None)
        self.best_solution = cur_solution
        self.best_obj = cur_obj
        temperature = self.max_temp
        pbar = tqdm.tqdm(range(self.iter_num), desc="ALNS Iteration")      
        
        for step in pbar:
            if ( step % 10 == 0):
                self.del_car(cur_solution)
            
            break_opt_i, repair_opt_i = self.choose_operator()
            new_solution , changed_car_list = self.get_neighbour(cur_solution, break_opt_i, repair_opt_i) 
            
            
            new_obj = self.cal_objective(new_solution , changed_car_list)
            
            # obj: minimize the total distance 
            if new_obj < self.best_obj:
                self.best_solution = copy.deepcopy(new_solution)
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
            
        # self.plot_routes(self.best_solution)
        
        return self.best_solution, self.best_obj
    

    
class ALNS(ALNS_base):
    
    def __init__(self, data , iter_num , scale ):
        super().__init__(iter_num)
        self.data = data
        self.set_operators_list(data)
        self.tsp_model = TSP_model(data)
        self.scale = scale
    
    def set_operators_list(self, data):
        self.break_operators_list = [
            RandomBreak(data),
            RandomFarestBreak(data),
            CarBreak(data),
            GreedyBreak(data),
            ShawBreak(data)      
        ]
        self.repair_operators_list = [
            RandomRepair(data),
            RandomNearestRepair(data),
            GreedyRepair(data), 
            # RegretKRepair(data),  // too slow , 全局贪
            RandomRegretKRepair(data)  # 局部贪
        ]
        
    def solution_init(self):
        # si = SI(data)
        # solution = Sol(self.data)
        # si.solve(solution)
        
        solution = SI_3().init_solotion(self.data)
        
        
        return solution
    
    def del_car(self, solution):
        empty_car_list = []
        for i in range(len(solution.routes)):
            if len(solution.routes[i]) == 0:
                empty_car_list.append(i)
        while(len(empty_car_list) > 5):
            solution.pop_route(empty_car_list.pop())
    

    def cal_objective(self , solution , change_cars):
        if(change_cars == None):
            return cal_fitness(self.data, solution)
        else:
            return cal_fitness_changed(self.data, solution, change_cars)
    
    
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
    
    
    def plot_routes(self  ):
        solution = self.best_solution
        data = self.data
        plt.figure(figsize=(10,10))
        for i in range(len(data.nodes)):
            plt.scatter(data.nodes[i].x, data.nodes[i].y, c='b', s=10)
        
        
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k'] # 颜色列表
        for j in range(len(solution.routes)):
            
            route = solution.routes[j]
            if(len(route) == 0):
                continue
            color = colors[j % len(colors)] # 为该路线指定一个颜色
            for i in range(len(route)-1):
                plt.plot([data.name2node[data.points[route[i]].node.name].x , data.name2node[data.points[route[i+1]].node.name].x],
                        [data.name2node[data.points[route[i]].node.name].y , data.name2node[data.points[route[i+1]].node.name].y], color + '-')
                
            if(j % 50 == 0 and j != 0):  
                # plt.show()
                plt.savefig('{}/routes_{}.png'.format(self.scale , j))
                plt.close()
                
                plt.figure(figsize=(10,10))
                
                for i in range(len(data.nodes)):
                    plt.scatter(data.nodes[i].x, data.nodes[i].y, c='b', s=10)
        plt.close()
    
    def save_data(self):
        with open('{}/result_{}.json'.format(self.scale,self.scale), 'w') as f:
            json.dump(self.best_solution.routes, f)
        with open('{}/fitness_{}.txt'.format(self.scale,self.scale), 'w') as f:
            f.write(str(self.best_obj))
    
    def run(self):
        self.reset()
        cur_solution = self.solution_init() 
        
        
        cur_obj = self.cal_objective(cur_solution , None)
        self.best_solution = cur_solution
        self.best_obj = cur_obj
        temperature = self.max_temp
        pbar = tqdm.tqdm(range(self.iter_num), desc="ALNS Iteration")      
        
        for step in pbar:
            if ( step % 10 == 0):
                self.del_car(cur_solution)
                self.TSP_process(cur_solution)
                
            break_opt_i, repair_opt_i = self.choose_operator()
            new_solution , changed_car_list = self.get_neighbour(cur_solution, break_opt_i, repair_opt_i) 
            
            # new_obj = self.cal_objective(new_solution , None)
            new_obj = self.cal_objective(new_solution , changed_car_list)
            
            # obj: minimize the total distance 
            if new_obj < self.best_obj:
                self.best_solution = copy.deepcopy(new_solution)
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
        self.TSP_bestSol_process(solution = self.best_solution)
        
        
        return self.best_solution, self.best_obj
    
    def TSP_bestSol_process(self , solution):
        car_num = len(solution.routes)
        for i in range(car_num):
            if(len(solution.routes[i]) <= 2):
                continue
            solution.routes[i] = self.TSP_route(solution.routes[i])
        self.cal_objective(solution , None)
        
    
    def TSP_process(self , cur_solution):
        # print("old " ,cur_solution.obj)
        car_num = len(cur_solution.routes)
        changed_car_list = []
        changed_car = np.random.randint(low=0, high=car_num)
        if len(cur_solution.routes[changed_car]) <= 2:
            return 
        # print(cur_solution.routes[changed_car])
        changed_car_list.append(changed_car)
        cur_solution.routes[changed_car] = self.TSP_route(cur_solution.routes[changed_car])
        self.cal_objective(cur_solution, changed_car_list)
        # print("new " ,cur_solution.obj)
        
    def TSP_route(self , route):
        return self.tsp_model.solve_route(route)
        pass
    
    def plot_all_route(self ):
        with open('result.json', 'r') as f:
            routes = json.load(f)
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k'] # 颜色列表
        for j in range(len(routes)):
            plt.figure(figsize=(10,10))
            for i in range(len(self.data.nodes)):
                plt.scatter(self.data.nodes[i].x, self.data.nodes[i].y, c='b', s=10)
                
            route = routes[j]
            if(len(route) == 0):
                continue
            color = colors[j % len(colors)] # 为该路线指定一个颜色
            for i in range(len(route)-1):
                plt.plot([self.data.name2node[self.data.points[route[i]].node.name].x , self.data.name2node[self.data.points[route[i+1]].node.name].x],
                        [self.data.name2node[self.data.points[route[i]].node.name].y , self.data.name2node[self.data.points[route[i+1]].node.name].y], color + '-')
            # plt.show()
            plt.savefig('{}\\picture\\routes_{}.png'.format(self.scale , j))
            plt.close()
    
if __name__ == "__main__":
    scale = 50
    iter_num = 1000
    location = "城市配送系统优化资料包\\输入数据"
    data = Data2(location ,scale , limited_order_num= 100 )
    # data = Data2(location , scale)
    alg = ALNS(data, iter_num ,scale)
    alg.run()
    alg.plot_routes()
    alg.show_process()
    alg.save_data()
    alg.plot_all_route()