import numpy as np
import random
import queue
from ALNS_1 import * 
from Fitness import *

# import heartrate
# heartrate.trace(browser=True)

class Operator:
    def __init__(self , data):
        
        self.data = data
        self.order_num = data.order_num
        self.rand_d_max = 8
        self.rand_d_min = 1
        self.regretK = 3
        self.limit_car_order_max = 10
        self.select_car_max_regret = 30
    
    def search_pair(self , solution , car_index , index):
        solution.routes[car_index].remove(index)
        if(self.data.points[index].type == 'D'):
            solution.routes[car_index].remove(index-self.data.order_num)
            return [index-self.data.order_num,index]
        else:
            solution.routes[car_index].remove(index+self.data.order_num)
            return [index,index+self.data.order_num]
    
    def search_pair_in_route(self , route , index):
        route.remove(index)
        if(self.data.points[index].type == 'D'):
            route.remove(index-self.data.order_num)
        else:
            route.remove(index+self.data.order_num)
    
    def insert_order_to_route(self , solution , pair , car_index , pos):
        if(pos >= len(solution.routes[car_index])):
            solution.routes[car_index].append(pair)
        else:
            solution.routes[car_index].insert(pos,pair)
         
    # 耗时严重  
    # @profile 
    def findBestInsertPos(self , solution , pair , car_index):
        best_i = -1
        best_j = -1
        
        obj_old = cal_car_fitness(self.data, solution.routes[car_index])
        
        obj_best = np.inf
        route = solution.routes[car_index].copy()
        for i in range(len(route)+1):
            if i == len(route):
                route.append(pair[0])
            else:
                route.insert(i,pair[0])
            obj_new = cal_car_fitness(self.data, route)
            if(obj_new < obj_best):
                obj_best = obj_new
                best_i = i
            route.pop(i)
        route.insert(best_i,pair[0])
        obj_best = np.inf
        for j in range(best_i+1 , len(route)+1):
            if j == len(route):
                route.append(pair[1]) 
            else:
                route.insert(j,pair[1])
            obj_new = cal_car_fitness(self.data, route)
            if(obj_new < obj_best):
                obj_best = obj_new
                best_j = j
            route.pop(j)
        return best_i , best_j , obj_best - obj_old
    
    # def findBestInsertPos(self , solution , pair , car_index):
    #     obj_best = np.inf
    #     best_i = -1
    #     best_j = -1
        
    #     obj_old = cal_car_fitness(self.data, solution.routes[car_index])
        
    #     for i in range(len(solution.routes[car_index])):
    #         for j in range(i+1 , len(solution.routes[car_index])+1):
    #             route = solution.routes[car_index].copy()
                
    #             route.insert(i,pair[0])
    #             route.insert(j,pair[1])
    #             obj_new = cal_car_fitness(self.data, route)
                
    #             if(obj_new < obj_best):
    #                 obj_best = obj_new
    #                 best_i = i
    #                 best_j = j
    #     return best_i , best_j , obj_best - obj_old

        
    def set(self , solution):
        # 获取近邻解
        raise NotImplementedError
        
        
class RandomBreak(Operator):
    def __init__(self, data ):
        super().__init__(data)
        
    # @profile
    def set(self , solution):
        # 随机破坏若干个point
        car_num = len(solution.routes)
        d = np.random.uniform(self.rand_d_min,self.rand_d_max / 2)
        removed_car_list = random.sample(range(car_num),int(d))
        removed_pair_list = []
        for car_index in removed_car_list:
            for i in range(int(d)):
                if( len(solution.routes[car_index]) == 0):
                    continue
                index = solution.routes[car_index][np.random.randint(0,len(solution.routes[car_index]))]
                removed_pair_list.append(self.search_pair(solution , car_index , index))
                
        break_info = {
            'removed_pair_list':removed_pair_list,
            'remove_car_list':removed_car_list
        }
        return  solution , break_info
    

class RandomFarestBreak(Operator):
    """随机参数适应度值最大的车辆，破坏若干个点

    Args:
        Operator (_type_): _description_
    """
    def __init__(self, data):
        super().__init__(data)
    
    # @profile   
    def set(self, solution):
        removed_pair_list = []
        break_car_obj = max(solution.objs)
        break_car_index = solution.objs.index(break_car_obj)
        d = np.random.uniform(self.rand_d_min,self.rand_d_max)
        for i in range(int(d)):
            if( len(solution.routes[break_car_index]) == 0):
                    continue
            index = solution.routes[break_car_index][np.random.randint(0,len(solution.routes[break_car_index]))]
            removed_pair_list.append(self.search_pair(solution , break_car_index , index))
        break_info = {
            'removed_pair_list':removed_pair_list,
            'remove_car_list':[break_car_index]
        }
        return solution , break_info

    
class CarBreak(Operator):
    def __init__(self, data:Data2):
        super().__init__(data)

    # @profile
    def set(self, solution:Sol):

        break_num_min = 1
        break_num_max = 2

        car_num = len(solution.routes)
        remove_car_num = np.random.randint(break_num_min, break_num_max)
        removed_pair_list = []
        changed_car_list = []

        que = queue.PriorityQueue()
        for car_index in range(car_num):
            que.put((len(solution.routes[car_index]), car_index))

        for i in range(remove_car_num):
            car_index = que.get()[1]
            while(len(solution.routes[car_index])==0):
                car_index = que.get()[1]
            if(len(solution.routes[car_index])  > self.limit_car_order_max ):
                break
            changed_car_list.append(car_index)
            while (len(solution.routes[car_index]) > 0):
                index = solution.routes[car_index][-1]
                removed_pair_list.append(self.search_pair(solution, car_index, index))

        break_info = {
            'removed_pair_list':removed_pair_list,
            'remove_car_list':changed_car_list
        }
        return solution, break_info 
     
class GreedyBreak(Operator):
    def __init__(self, data):
        super().__init__(data)
    # @profile  
    def set(self , solution):
        removed_pair_list = []
        changed_car_list = []

        que = queue.PriorityQueue()
        
        d = int(np.random.uniform(self.rand_d_min,self.rand_d_max))
        
        
        for car_index in range(len(solution.routes)):
            for order_index in range(len(solution.routes[car_index])):
                obj_old = solution.objs[car_index]
                route = solution.routes[car_index].copy()
                order = route[order_index]
                if(self.data.points[order].type == 'P'):
                    self.search_pair_in_route(route , order)
                    obj_new = cal_car_fitness(self.data, route)
                    diff = obj_new - obj_old  
                    # diff 越小说明移除该点对车辆的影响越大
                    que.put((diff , [car_index , order]))
                else:
                    continue
        for i in range(d):
            # order_index 是在route中的索引
            # index是是在points中的索引
            car_index , order = que.get()[1]
            changed_car_list.append(car_index)
            index = order
            removed_pair_list.append(self.search_pair(solution, car_index, index))
        changed_car_list = list(np.unique(changed_car_list))
        
        break_info = {
            'removed_pair_list':removed_pair_list,
            'remove_car_list':changed_car_list
        }
        return solution, break_info   
        
class ShawBreak(Operator):
    def __init__(self, data):
        super().__init__(data)
    # @profile    
    def set(self , solution):
        removed_pair_list = []
        changed_car_list = []
        
        que = queue.PriorityQueue()
        
        d = int(np.random.uniform(self.rand_d_min,self.rand_d_max) )
        
        break_index = np.random.choice(self.data.P_set)
        node1_s = self.data.points[break_index].node.node_id
        node1_e = self.data.points[break_index+self.data.order_num].node.node_id
        d1 = self.data.node_time_matrix[node1_s][node1_e]
        for car_index in range(len(solution.routes)):
            for order_index in range(len(solution.routes[car_index])):
                if(self.data.points[solution.routes[car_index][order_index]].type == 'P'):
                    node2_s = self.data.points[solution.routes[car_index][order_index]].node.node_id
                    node2_e = self.data.points[solution.routes[car_index][order_index]+self.data.order_num].node.node_id
                    d2 = self.data.node_time_matrix[node2_s][node2_e]
                    r12 = self.findNearestDist(node1_s , node2_s , node1_e , node2_e)
                    # o12 越大两个点越近
                    o12 = -(d1 + d2 )/r12
                    que.put((o12 , [car_index , solution.routes[car_index][order_index]]))
                    
        for i in range(d):
            # order_index 是在route中的索引
            # index是是在points中的索引
            car_index , order = que.get()[1]
            changed_car_list.append(car_index)
            index = order
            removed_pair_list.append(self.search_pair(solution, car_index, index))
        changed_car_list = list(np.unique(changed_car_list))
        
        break_info = {
            'removed_pair_list':removed_pair_list,
            'remove_car_list':changed_car_list
        }
        return solution, break_info   
                    
    def findNearestDist(self,node1_s , node2_s , node1_e , node2_e):
        r12 = np.inf 
        
        d12 = self.data.node_time_matrix[node1_s][node1_e]
        d13 = self.data.node_time_matrix[node1_s][node2_s]
        d14 = self.data.node_time_matrix[node1_s][node2_e]
        d23 = self.data.node_time_matrix[node1_e][node2_s]
        d24 = self.data.node_time_matrix[node1_e][node2_e]
        d34 = self.data.node_time_matrix[node2_s][node2_e]
        d32 = self.data.node_time_matrix[node2_s][node1_e]
        d41 = self.data.node_time_matrix[node2_e][node1_s]
        d31 = self.data.node_time_matrix[node2_e][node1_e]
        d42 = self.data.node_time_matrix[node2_e][node2_s]

        r12 = min(r12 , d12 + d23 + d34)
        r12 = min(r12 , d34 + d41 + d12)
        r12 = min(r12 , d13 + d32 + d24)
        r12 = min(r12 , d13 + d34 + d42)
        r12 = min(r12 , d31 + d12 + d24)
        r12 = min(r12 , d31 + d14 + d42)
        
        return r12
        
class RandomRepair(Operator):
    def __init__(self , data):
        super().__init__(data)
        
    # @profile
    def set(self , solution , break_info):
        removed_pair_list = break_info['removed_pair_list']
        remove_car_list = break_info['remove_car_list']
        
        # 随机修复
        car_num = len(solution.routes)
        car_index = np.random.randint(low=0, high=car_num, size=len(removed_pair_list))
        changed_car_list = remove_car_list + car_index.tolist()
        changed_car_list = list(np.unique(changed_car_list))
        for i in car_index:
            order_pair = removed_pair_list.pop()
            order_count = len(solution.routes[i])
            if(order_count > 1):
                a, b = sorted(random.sample(range(order_count), 2))
            else:
                a, b = 0, 0
            self.insert_order_to_route(solution , order_pair[0] , i , a)
            self.insert_order_to_route(solution , order_pair[1] , i , b+1)
        return solution , changed_car_list
        

                        
class RandomNearestRepair(Operator):
    def __init__(self, data):
        super().__init__(data)
        
    # @profile
    def set(self , solution , break_info):
        removed_pair_list = break_info['removed_pair_list']
        remove_car_list = break_info['remove_car_list']
        
        # 随机修复
        car_num = len(solution.routes)
        car_index = np.random.randint(low=0, high=car_num, size=len(removed_pair_list))
        changed_car_list = remove_car_list + car_index.tolist()
        changed_car_list = list(np.unique(changed_car_list))
        
        for i in car_index:
            order_pair = removed_pair_list.pop()
            order_count = len(solution.routes[i])
            if(order_count > 1):
                a , b , diff = self.findBestInsertPos(solution, order_pair, i)
                pass
            else:
                a , b = 0 , 0
            self.insert_order_to_route(solution , order_pair[0] , i , a)
            self.insert_order_to_route(solution , order_pair[1] , i , b)
        return solution , changed_car_list
    
class GreedyRepair(Operator):
    def __init__(self, data):
        super().__init__(data)
    # @profile
    def set(self , solution , break_info):
        removed_pair_list = break_info['removed_pair_list']
        removed_car_list = break_info['remove_car_list']
        
        for pair in removed_pair_list:    
            min_diff = np.inf 
            best_car_index = -1
            best_a_b = [-1 , -1]    
            for car_index in range(len(solution.routes)):
                a , b , diff = self.findBestInsertPos(solution, pair, car_index)
                if(diff < min_diff):
                    min_diff = diff
                    best_car_index = car_index
                    best_a_b = [a , b]
            self.insert_order_to_route(solution, pair[0], best_car_index , best_a_b[0])
            self.insert_order_to_route(solution, pair[1], best_car_index , best_a_b[1])
            removed_car_list.append(best_car_index)
        changed_car_list = list(np.unique(removed_car_list))
        
        return solution , changed_car_list
    
        
class RegretKRepair(Operator):
    def __init__(self, data):
        super().__init__(data)
    def set(self , solution , break_info):
        removed_pair_list = break_info['removed_pair_list']
        removed_car_list = break_info['remove_car_list']

        while(len(removed_pair_list) > 0 ):
            pair , best_car_index , a_b = self.findRegretInsert(solution , removed_pair_list)
            self.insert_order_to_route(solution, pair[0], best_car_index , a_b[0])
            self.insert_order_to_route(solution, pair[1], best_car_index , a_b[1])
            removed_pair_list.remove(pair)
            removed_car_list.append(best_car_index)
        changed_car_list = list(np.unique(removed_car_list))
        
        return solution , changed_car_list
    
    def findRegretInsert(self, solution , removed_pair_list):
        
        biggest_regret = np.inf
        best_pair = None
        best_car = None
        best_a_b = [-1 , -1]
        
        for pair in removed_pair_list:
            que = queue.PriorityQueue()
            regret = 0
            for car_index in range(len(solution.routes)):
                a , b , diff = self.findBestInsertPos(solution, pair, car_index)
                que.put((diff , pair , car_index , a , b))

            diff , pair , car_index , a , b = que.get()
            for i in range(self.regretK):
                regret += diff - que.get()[0]
            if(regret < biggest_regret):
                biggest_regret = regret
                best_pair = pair
                best_car = car_index
                best_a_b = [a , b]
                
        return best_pair , best_car , best_a_b
    
class RandomRegretKRepair(Operator):
    def __init__(self, data):
        super().__init__(data)
    # @profile    
    def set(self , solution , break_info):
        removed_pair_list = break_info['removed_pair_list']
        removed_car_list = break_info['remove_car_list']
        
        self.select_car_max_regret = int(len(solution.routes) * 0.3)
        
        while(len(removed_pair_list) > 0 ):
            pair , best_car_index , a_b = self.findRegretInsert(solution , removed_pair_list)
            self.insert_order_to_route(solution, pair[0], best_car_index , a_b[0])
            self.insert_order_to_route(solution, pair[1], best_car_index , a_b[1])
            removed_pair_list.remove(pair)
            removed_car_list.append(best_car_index)
        changed_car_list = list(np.unique(removed_car_list))
        
        return solution , changed_car_list
        
    # @profile
    def findRegretInsert(self, solution , removed_pair_list):

        biggest_regret = np.inf
        best_pair = None
        best_car = None
        best_a_b = [-1 , -1]
        
        for pair in removed_pair_list:
            que = queue.PriorityQueue()
            regret = 0
            
            car_list = np.random.randint(low=0, high=len(solution.routes), size=min (self.select_car_max_regret , len(solution.routes)))

            for car_index in car_list:
                a , b , diff = self.findBestInsertPos(solution, pair, car_index)
                que.put((diff , pair , car_index , a , b))
            diff , pair , car_index , a , b = que.get()
            
            iters = min(len(que.queue) , self.regretK)
            for i in range(iters):
                regret += diff - que.get()[0] # regret是一个负值，越小说明后悔越多

            if(regret < biggest_regret):
                biggest_regret = regret
                best_pair = pair
                best_car = car_index
                best_a_b = [a , b]

        if best_pair == None:
            for pair in removed_pair_list:
                que = queue.PriorityQueue()
                regret = 0
                
                for car_index in  range(len(solution.routes)):
                    a , b , diff = self.findBestInsertPos(solution, pair, car_index)
                    que.put((diff , pair , car_index , a , b))

                diff , pair , car_index , a , b = que.get()
                for i in range(self.regretK):
                    regret += diff - que.get()[0] # regret是一个负值，越小说明后悔越多
                if(regret < biggest_regret):
                    biggest_regret = regret
                    best_pair = pair
                    best_car = car_index
                    best_a_b = [a , b]
                
        return best_pair , best_car , best_a_b
        
        