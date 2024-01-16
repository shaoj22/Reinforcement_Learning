# environment class
# author: Charles Lee
# date: 2022.10.26

import numpy as np
import matplotlib.pyplot as plt
import math
from collections import deque
import heapq
from copy import deepcopy
import torch
import tensorboard
from MCRMFS_tools import Draw_Opt

class Env():
    def __init__(self, instance, args):
        # data inside instance: T, R, col_num, row_num, N, p, s, h0, e0, g0, A
        self.instance = instance
        self.render_flag = args['render']
        self.max_steps = instance.T
        self.node_dim = 8
        self.draw_opt = Draw_Opt(self.instance) # draw operator for render()
        #self.fig = plt.figure() # fig objecti for render()
        self.task_list = [] # record the task assigned

        # generate pos for each idx
        self.pos_dict = {}
        for idx in range(self.instance.N):
            raw = idx // self.instance.row_num
            col = idx % self.instance.row_num
            self.pos_dict[idx] = [raw, col]
        
    def _transfer_state(self, p, s, h, e, g, assigned):
        state = np.zeros((self.instance.N, self.node_dim))
        for i in range(self.instance.N):
            state[i, 0] = i // self.instance.row_num
            state[i, 1] = i % self.instance.row_num
            state[i, 2] = p[i]
            state[i, 3] = s[i]
            state[i, 4] = h[i]
            state[i, 5] = e[i]
            state[i, 6] = g[i]
            state[i, 7] = assigned[i]
        return state

    def _get_info(self):
        rack_mask, place_mask = self._get_mask()
        valid_flag = self._check_valid(rack_mask, place_mask)
        info = {"rack_mask": rack_mask, "place_mask": place_mask, 
                "valid_flag": valid_flag, "step_dict": self.step_dict, "move_dict" : self.move_dict, 
                "env" : self}
        return info

    def _get_mask(self):
        """
        get masks for rack selection and place selection
        """
        # search for available rack and place
        # (take assigned place as block)
        available_rack = []
        available_place = []
        visited = np.zeros(self.instance.N, dtype=bool)
        queue = [0]
        while queue:
            cur = queue.pop()
            if visited[cur]:
                continue
            visited[cur] = True
            next_list = []
            # if index valid, add to list
            if cur - self.instance.row_num >= 0: 
                next_list.append(cur - self.instance.row_num)
            if cur + self.instance.row_num < self.instance.N: 
                next_list.append(cur + self.instance.row_num)
            if cur - 1 >= 0:
                next_list.append(cur - 1)
            if cur + 1 < self.instance.N:
                next_list.append(cur + 1)
            # check each next index
            for nex in next_list:
                if self.assigned[nex] == 1: # if assigned, skip
                    continue
                if self.e[nex] == 0: # if empty
                    queue.append(nex)
                    if self.s[nex] == 0: # if empty place
                        available_place.append(nex)
                else: # if has rack
                    if self.h[nex] == 0: # if not assigned
                        available_rack.append(nex)
        available_rack = list(set(available_rack))
        available_place = list(set(available_place))

        # create masks
        rack_mask = np.ones(self.instance.N, dtype=bool)
        place_mask = np.ones(self.instance.N, dtype=bool)
        for ri in available_rack:
            rack_mask[ri] = False
        for pi in available_place:
            place_mask[pi] = False
        # mask place in other robots' routes 
        for robot in self.robots:
            for pi in robot.route:
                place_mask[pi] = True

        return rack_mask, place_mask
                    
    def _check_valid(self, rack_mask, place_mask):
        # 条件1：有可选货架与空位
        valid_flag1 = sum(rack_mask)<len(rack_mask) and sum(place_mask)<len(place_mask) # if no rack / place to choose, invalid
        # 条件2：有需要拣选的目标货架
        valid_flag2 = False
        for pi in range(self.instance.N):
            if self.g[pi] == True and self.assigned[pi] == False and self.h[pi] == False:
                valid_flag2 = True
        return valid_flag1 and valid_flag2

    def _get_block(self):
        """
        get block for routing algorithm
        """
        block = np.zeros(self.instance.N, dtype=bool)
        for i in range(self.instance.N):
            if self.e[i] == 1 or self.assigned == 1:
                block[i] = True
        for ri in range(len(self.robots)): # free rack been packed by robot
            if self.robots[ri].is_loaded:
                block[self.robots[ri].pos] = False
        return block

    def reset(self):
        """
        reset the environment and return initial state

        Return:
            state (ndarray): all information about depot and robot
            info (list[:, 2]): other informations, in this case is idxs of pickers
        """
        # reset steps
        self.steps = 0
        # reset map
        self.p = self.instance.p.copy()
        self.s = self.instance.s.copy()
        self.h = self.instance.h0.copy()
        self.e = self.instance.e0.copy()
        self.g = self.instance.g0.copy()
        self.assigned = [0] * self.instance.N
        self.sleep_robot = np.zeros(self.instance.R, dtype=bool) # whether robot is sleeping temperarily
        # reset robots
        self.robots = []
        self.robot_put_down_block = []
        self.step_dict = {} # record place i in time t is occupied, to avoid collision
        self.move_dict = {} # record move from i to j in time t is occupied, to avoid collision
        for i in range(self.instance.N):
            if self.h[i] > 0:
                robot = Robot(i)
                self.robots.append(robot)
                self.step_dict[i, 0] = 1
                self.robot_put_down_block.append([None, None])
        self.main_robot_id = 0 # the robot to assign task, initialize as 0
        # return initial state, mask
        state = self._transfer_state(self.p, self.s, self.h, self.e, self.g, self.assigned)
        info = self._get_info()
        return state, info

    def _subroute_planning(self, start_pos, end_pos, start_time, block=None):
        """plan sub route to rack or target or place (Dijkstra)

        Args:
            start_pos (int): current position idx
            end_pos (int): end position idx
            block (ndarray): whether idx is block. Defaults to None, means no loaded.
        Return:
            route (List): the subroute result, if failed, return 1 if blocked rack, 0 if blocked place
        """
        dist_list = np.ones(self.instance.N) * np.inf
        dist_list[start_pos] = 0
        finish_list = np.zeros(self.instance.N)
        route_list = [[] for _ in range(self.instance.N)]
        route_list[start_pos].append(start_pos)
        heap = [(0, start_pos)]
        while heap:
            dist, cur_pos = heapq.heappop(heap)
            # if dist > dist_list[cur_pos]: # 若该点已被更新过，则跳过
            #     continue
            route = route_list[cur_pos]
            stop_by_robot_flag = False
            next_time = start_time + len(route)
            for next_pos in range(self.instance.N):
                ## available, not repeat, aisle or (empty, not assigned end_pos),
                if (self.instance.A[cur_pos][next_pos] == 1 \
                    and finish_list[next_pos] == False\
                    and (block is None or block[next_pos] == False)
                    and self.move_dict.get((next_pos, cur_pos, next_time), False) == False # avoid exchange
                    ):
                    # check been stopped by put down rack
                    stop_by_put_down_flag = False
                    for ri in range(len(self.robots)):
                        rp_block, rp_time = self.robot_put_down_block[ri]
                        if next_pos == rp_block and next_time >= rp_time:
                            stop_by_put_down_flag = True
                            break
                    if stop_by_put_down_flag:
                        continue
                    # check been stopped by robot
                    if self.step_dict.get((next_pos, next_time), False): 
                        stop_by_robot_flag = True # if stopped by robot you can wait to avoid it 
                        continue
                    # find valid next_pos
                    stop_by_robot_flag = False # if has choice, not been stopped
                    _dist = dist_list[cur_pos] + 1
                    if _dist < dist_list[next_pos]:
                        dist_list[next_pos] = _dist
                        _route = route.copy()
                        _route.append(next_pos)
                        route_list[next_pos] = _route
                        heapq.heappush(heap, (_dist, next_pos))
                        if next_pos == end_pos:
                            return _route
            # if no valid next_pos found
            if stop_by_robot_flag: # wait if been stopped by robot
                route_list[cur_pos].append(cur_pos)
                heapq.heappush(heap, (dist_list[cur_pos], cur_pos))
                continue
        return None

    def _subroute_planning_A_star(self, cur_pos, end_pos, start_time, block=None):
        """plan sub route to rack or target or place

        Args:
            cur_pos (int): current position idx
            end_pos (int): end position idx
            block (ndarray): whether idx is block. Defaults to None, means no loaded.
        Return:
            route (List): the subroute result, if failed, return 1 if blocked rack, 0 if blocked place
        """
        tabu = {}
        route = [cur_pos]
        while cur_pos != end_pos:
            stop_flag = "stop by block"
            min_d = np.inf
            best_next_pos = None
            next_time = start_time + len(route)
            for next_pos in range(self.instance.N):
                ## available, not repeat, aisle or (empty, not assigned end_pos),
                if (self.instance.A[cur_pos][next_pos] == 1 \
                    and next_pos not in tabu\
                    and (block is None or block[next_pos] == False)
                    ):
                    # check been stopped by put down rack
                    stop_by_put_down_flag = False
                    for ri in range(len(self.robots)):
                        rp_block, rp_time = self.robot_put_down_block[ri]
                        if next_pos == rp_block and next_time >= rp_time:
                            stop_by_put_down_flag = True
                            break
                    if stop_by_put_down_flag:
                        continue
                    # check been stopped by robot
                    if self.step_dict.get((next_pos, next_time), False): 
                        stop_flag = "stop by robot" # if stopped by robot you can wait to avoid it 
                        continue
                    # dist = np.sqrt((self.pos_dict[end_pos][0]-self.pos_dict[next_pos][0])**2 + \
                    #     (self.pos_dict[end_pos][1]-self.pos_dict[next_pos][1])**2)
                    dist = abs(self.pos_dict[end_pos][0]-self.pos_dict[next_pos][0]) + \
                        abs(self.pos_dict[end_pos][1]-self.pos_dict[next_pos][1]) # manhattan distance to end_pos
                    if dist < min_d: # choose the nearest pos to end_pos
                        min_d = dist
                        best_next_pos = next_pos
            # if no valid next_pos found
            if best_next_pos is None:
                if stop_flag == "stop by robot": # wait if been stopped by robot
                    route.append(cur_pos)
                else:
                    ni = route.pop() # pop route while tabu add
                    # assert len(route) > 0, "fail route planning" # if 0, fail planning route #!
                    cur_pos = route[-1]
                continue
            # if found valid next_pos
            route.append(best_next_pos)
            tabu[best_next_pos] = 1
            cur_pos = best_next_pos
        return route

    def _robot_path_planning(self, robot, action, block):
        """nearest neighbour path planning for robot

        Args:
            robot (Robot): Robot class, containing information about robot
            action (List[2]): containing rack and place as subtask
            block (ndarray[N]): whether idx is block, for routing algorithm

        Updates:
            route (List): route generated for main robot (reverse for pop)
            task (List): rack and place
        """
        # assert len(robot.route)==0, "planning path for robot with task"
        # assert robot.is_loaded==0, "planning path for robot with task"
        # assert robot.is_loaded_target==0, "planning path for robot with task"
        targets = [i for i in range(self.instance.N) if self.g[i] == 1]
        pickers = [i for i in range(self.instance.N) if self.p[i] == 1]
        rack, place = action 
        cur_pos = robot.pos
        cur_time = self.steps
        # route to rack
        route1 = self._subroute_planning(cur_pos, rack, cur_time) 
        cur_pos = route1[-1]
        cur_time += len(route1) - 1 
        # route to target (if rack is target)
        if rack in targets:
            # choose the nearest picker first
            min_d = np.inf
            picker = None
            for pi in pickers:
                dist = abs(self.pos_dict[pi][0]-self.pos_dict[cur_pos][0]) + \
                    abs(self.pos_dict[pi][1]-self.pos_dict[cur_pos][1]) # manhattan distance to pi
                if dist < min_d:
                    min_d = dist
                    picker = pi
            # route to picker
            route2 = self._subroute_planning(cur_pos, picker, cur_time, block) # remove repeated cur_pos
            if route2 is None:
                return None
            route2 = route2[1:]
            cur_pos = route2[-1]
            cur_time += len(route2) 
        else:
            route2 = []
        # route to place
        route3 = self._subroute_planning(cur_pos, place, cur_time, block) # remove repeated cur_pos
        if route3 is None:
            return None
        route3 = route3[1:]
        
        # update route and task
        route = route1 + route2 + route3
        return route

    def _check_robots_task(self, robots):
        """
        check whether all robots has task: 
            return -1 if all has
            return the main robot idx if not
        """
        ti = -1
        info = None
        for robot_idx in range(len(robots)):
            if self.sleep_robot[robot_idx] == 1: # skip sleep robot
                continue
            if len(robots[robot_idx].route) <= 1:
                ti = robot_idx
                break
        if ti != -1:
            info = self._get_info()
            if info["valid_flag"] == 0: # if invalid, make all robots with no task sleep
                for robot_idx in range(len(robots)):
                    if (len(robots[robot_idx].route) <= 1):
                        self.sleep_robot[robot_idx] = 1
                ti, info = -1, None
            else: # if valid, make sleep robots awake
                if sum(self.sleep_robot) > 0: 
                    self.sleep_robot[:] = 0
        return ti, info

    def _return_state(self, reward, done, info):
        state = self._transfer_state(self.p, self.s, self.h, self.e, self.g, self.assigned)
        # assert done or info["valid_flag"], "invalid return, has no rack or place to choose"
        return state, reward, done, info

    def step(self, action):
        """
        step function, apply action to change env and reply

        Args:
            action (list[2]): constains rack idx and empty space idx
        Return: (implement with _return_state())
            state (ndarray): all information about depot and robot
            reward (double): reward gain in this step
            done (int): whether the episode is done
            info (list): other informations, in this case is idxs of pickers
        """
        reward = 0
        if action is None:
            # 已无任务可分配，使所有无工作机器人休眠
            for i in range(len(self.robots)):
                if len(self.robots[i].route) <= 1:
                    self.sleep_robot[i] = 1
        else:
            # path planning according to action
            main_robot = self.robots[self.main_robot_id]
            block = self._get_block()
            route = self._robot_path_planning(main_robot, action, block)
            if route is not None:
                main_robot.route = route[::-1] # reverse the route for pop()
                main_robot.task = action
                main_robot.is_loaded_target = self.g[action[0]]
                self.assigned[action[0]] = 1
                self.assigned[action[1]] = 1
                self.task_list.append(action+[self.main_robot_id])
                # update step_dict
                cur_time = self.steps + 1
                for i in range(1, len(route)):
                    # assert (route[i], cur_time) not in self.step_dict, "collision happens"
                    self.step_dict[route[i], cur_time] = True
                    self.move_dict[route[i-1], route[i], cur_time] = True
                    cur_time += 1
                self.robot_put_down_block[self.main_robot_id][0] = route[-1]
                self.robot_put_down_block[self.main_robot_id][1] = cur_time
                # assert self.e[action[0]] == 1, "assigned empty rack"
                
                # print(f"assigned robot {self.main_robot_id} from {action[0]} to {action[1]}")
            else:
                # 无法规划任务，使所有无工作机器人休眠
                for i in range(len(self.robots)):
                    if len(self.robots[i].route) <= 1:
                        self.sleep_robot[i] = 1
        # robots step on and change env
        # keep forging ahead
        while 1:
            # for ri in range(len(self.robots)):
            #     assert self.h[self.robots[ri].pos] <= 1, "robot overlap"
            if self.steps > self.max_steps:
                done = 1
                info = self._get_info()
                return self._return_state(reward, done, info)
            if self.render_flag:
                self.render()
            # check robots have task, break out if any robot has empty route
            ti, info = self._check_robots_task(self.robots)
            if ti != -1: # not -1 means not all has task
                self.main_robot_id = ti
                # assert self.robots[ti].is_loaded == 0, "assigned for loaded"
                done = 0
                break
            # check if robots all sleep, if so, game is done
            if all(self.sleep_robot == 1):
                done = 1
                break
            # robots forward
            for ri in range(len(self.robots)):
                if self.sleep_robot[ri] == 1: # skip sleep robot
                    continue
                robot = self.robots[ri]
                cur_idx = robot.route.pop() # robot.pos == route.pop()
                # assert self.step_dict[cur_idx, self.steps] == 1, "real step not conform to step_dict"
                next_idx = robot.route[-1]
                robot.pos = next_idx

                # update map according to move
                ## move robot
                # assert self.h[cur_idx] > 0, "empty robot move"
                self.h[cur_idx] -= 1
                self.h[next_idx] += 1
                ## move rack/target rack
                if robot.is_loaded:
                    # assert self.e[cur_idx]>0, "load empty rack"
                    # assert self.e[next_idx]==0, "rack collision"
                    self.e[cur_idx] -= 1
                    self.e[next_idx] += 1
                    if robot.is_loaded_target:
                        # assert self.g[cur_idx]>0, "load empty target"
                        self.g[cur_idx] -= 1
                        self.g[next_idx] +=1
                ## finish target rack in picker
                if self.p[next_idx] > 0 and self.g[next_idx] > 0:
                    self.g[next_idx] -= 1
                    robot.is_loaded_target = 0
                    reward += 1000
                # break out if all targets finished
                # if sum(self.g)==0:
                #     done = 1
                #     # print("finished, steps:{}".format(self.steps))
                #     reward -= 0.01 * (1 + len(robot.route))
                #     return self._return_state(reward, done, info)

                # update robot.is_load according to next_idx
                if next_idx == robot.task[0]: # if achieve rack
                    # assert robot.is_loaded == 0, "loaded when achieve rack"
                    robot.is_loaded = 1
                    # robot.is_loaded_target = self.g[next_idx]
                    self.assigned[next_idx] = 0
                elif next_idx == robot.task[1] and robot.is_loaded and self.g[next_idx] == 0: # if achieve place
                    robot.is_loaded = 0
                    robot.is_loaded_target = 0
                    self.assigned[next_idx] = 0
                    self.robot_put_down_block[ri] = [None, None]
                
            # record time step number
            self.steps += 1
            reward -= 0.01 # step cost
            
        return self._return_state(reward, done, info)

    def render(self):
        """
        render function to visualize solution
        """
        plt.clf()
        ax = plt.subplot(111)
        self.draw_opt._draw_t(ax, self.p, self.s, self.h, self.e, self.g)
        plt.draw()
        plt.pause(0.2)

class Robot():
    def __init__(self, pos):
        self.pos = pos
        self.route = []
        self.task = [None, None]
        self.is_loaded = 0
        self.is_loaded_target = 0

class Env_info():
    def __init__(self, env):
        # copy information of env
        self.p = env.p.copy()
        self.s = env.s.copy()
        self.h = env.h.copy()
        self.e = env.e.copy()
        self.g = env.g.copy()
        self.assigned = env.assigned.copy()
        self.robots = deepcopy(env.robots)
        self.main_robot_id = env.main_robot_id