'''
File: Env.py
Project: DRLH
File Created: Saturday, 27th May 2023 12:06:33 pm
Author: Charles Lee (lmz22@mails.tsinghua.edu.cn)
'''

import sys
sys.path.append('D:\\File\\Seafile_files\\3.竞赛文件\\顺丰挑战赛SF-X\\PDPTW')
import numpy as np
import gymnasium as gym
from ALNS_1 import ALNS
from Data2 import Data2
import torch

class ALNS_Env(gym.Env):
    def __init__(self):
        self.data = Data2(limited_order_num=100)
        self.alg = ALNS_Alg(self.data, iter_num=1000)
        self.action_space = gym.spaces.Discrete(self.alg.heruistics_num)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(10+self.alg.heruistics_num,), dtype=np.float32)
    
    def get_state_info(self, alg):
        state, info = alg.get_state_info()
        return state, info 
    
    def reset(self):
        self.alg.init_process()
        state, info = self.get_state_info(self.alg)
        return state, info
        
    def step(self, action):
        reward, done = self.alg.process_after_selection(action)
        state, info = self.get_state_info(self.alg)
        terminated = truncated = done
        return state, reward, truncated, terminated, info
    
class ALNS_Alg(ALNS):
    def __init__(self, data, iter_num=1000):
        super().__init__(data, iter_num)
        self.heuristics_list = [(i,j) for i in range(len(self.break_operators_list)) for j in range(len(self.repair_operators_list))]
        self.heruistics_num = len(self.heuristics_list)

    def init_process(self):
        self.reset()
        self.cur_solution = self.solution_init() 
        self.cur_obj = self.cal_objective(self.cur_solution, None)
        self.init_obj = self.cur_obj
        self.best_solution = self.cur_solution
        self.best_obj = self.cur_obj
        self.temperature = self.max_temp
        self.step = 0
        # prepare for state
        self.last_obj = self.best_obj
        self.no_improvement = 0 # steps since last improvement
        self.was_changed = 0
        self.last_action = [0] * len(self.heuristics_list)
    
    def alns_select_operator(self):
        break_i, repair_i = self.choose_operator()
        return self.heuristics_list.index((break_i, repair_i))
    
    def process_after_selection(self, action):
        # get action
        break_opt_i, repair_opt_i = self.heuristics_list[action]
        self.last_action = [0] * len(self.heuristics_list)
        self.last_action[action] = 1
        # record step
        self.step += 1
        self.no_improvement += 1
        # record last obj
        self.last_obj = self.cur_obj
        # reward preparation
        reward = 0
        # delete car if necessary
        if ( self.step % 10 == 0):
            self.del_car(self.cur_solution)
        # get new solution
        new_solution, changed_car_list = self.get_neighbour(self.cur_solution, break_opt_i, repair_opt_i) 
        new_obj = self.cal_objective(new_solution, changed_car_list)
        # obj: minimize the total distance 
        if new_obj < self.best_obj:
            self.best_solution = new_solution
            self.best_obj = new_obj
            self.cur_solution = new_solution
            self.cur_obj = new_obj
            self.break_operators_scores[break_opt_i] += self.sigma1
            self.break_operators_steps[break_opt_i] += 1
            self.repair_operators_scores[repair_opt_i] += self.sigma1
            self.repair_operators_steps[repair_opt_i] += 1
            reward += 5
            self.no_improvement = 0
            self.was_changed = 1
        elif new_obj < self.cur_obj: 
            self.cur_solution = new_solution
            self.cur_obj = new_obj
            self.break_operators_scores[break_opt_i] += self.sigma2
            self.break_operators_steps[break_opt_i] += 1
            self.repair_operators_scores[repair_opt_i] += self.sigma2
            self.repair_operators_steps[repair_opt_i] += 1
            reward += 3
            self.no_improvement = 0
            self.was_changed = 1
        elif np.random.random() < self.SA_accept((new_obj-self.cur_obj)/(self.cur_obj+1e-10), self.temperature): # percentage detaC
            self.cur_solution = new_solution
            self.cur_obj = new_obj
            self.break_operators_scores[break_opt_i] += self.sigma3
            self.break_operators_steps[break_opt_i] += 1
            self.repair_operators_scores[repair_opt_i] += self.sigma3
            self.repair_operators_steps[repair_opt_i] += 1
            reward += 1
            self.was_changed = 1
        # reset operators weights
        if self.step % self.adaptive_period == 0: 
            self.reset_operators_scores()
        # update SA temperature
        self.temperature = self.temperature_update(self.temperature, self.step)
        # record
        self.obj_iter_process.append(self.cur_obj)
        # return done information and reward
        if self.step >= self.iter_num:
            done = 1
        else:
            done = 0
        return reward, done
    
    def get_state_info(self):
        # state includes: reduced_cost, cost_from_min, cost, min_cost, temp, cs, no_improvement, 
        #                 index_step, was_changed, last_action_sign, last_action (12,)
        state = []
        reduced_cost = (self.cur_obj - self.last_obj) / self.init_obj
        cost_from_min = (self.cur_obj - self.best_obj) / self.init_obj
        cost = self.cur_obj / self.init_obj
        min_cost = self.best_obj / self.init_obj
        temp = self.temperature
        cs = self.cooling_rate
        no_improvement = self.no_improvement
        index_step = self.step
        was_changed = self.was_changed
        last_action_sign = self.cur_obj == self.best_obj
        last_action = self.last_action
        state = [reduced_cost, cost_from_min, cost, min_cost, temp, cs, no_improvement,
                    index_step, was_changed, last_action_sign] + last_action
        info = {}
        return state, info
    