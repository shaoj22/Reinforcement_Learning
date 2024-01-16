# policy class
# author: Charles Lee
# date: 2022.10.26

import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import torch.nn as nn

class PGPolicy(nn.Module):
    def __init__(self, net, optim, args):
        super(PGPolicy, self).__init__()
        self.net = net
        self.optim = optim
        self.random_seed = args['random_seed']
        if self.random_seed is not None:
            torch.manual_seed(self.random_seed)

        self.action_space = args["instance"].N
        self.max_steps = args['instance'].T
        self.col_num = args['instance'].row_num

    def get_rack_candidates(self, env, rack_mask):
        # 寻找阻碍货架作为货架候选
        rack_candidates = []
        place_tabu = []
        targets = [pi for pi in range(self.action_space) if env.g[pi] == 1 and env.assigned[pi] == 0 and env.h[pi] == 0]
        directions = [-self.col_num, self.col_num, -1, 1] # up, down, left, right
        for i in range(len(targets)):
            target = targets[i]
            min_block_num = np.inf
            rack = -1 # rack to move
            key_tabu = None # block place
            block_place_list = []
            for dire in directions:
                block_num = 0
                cur_pos = target
                cur_rack = target
                cur_tabu = []
                while env.s[cur_pos] == False:  # 假设仓库外围一圈一定是通道，通过判断通道避免了越界
                    block_place_list.append(cur_pos)
                    if env.e[cur_pos] == True:
                        block_num += 1
                        cur_rack = cur_pos
                    else:
                        cur_tabu.append(cur_pos) # 禁止放在核心通道上，防止目标货架锁死
                    cur_pos += dire
                if block_num < min_block_num:
                    min_block_num = block_num
                    rack = cur_rack
                    key_tabu = cur_tabu
            if rack_mask[rack] == False:
                rack_candidates.append(rack)
                place_tabu += key_tabu
        return rack_candidates, place_tabu
                  
    def get_place_candidates(self, env, rack):
        # 选择rack作为货架，能够到达的空位
        place_candidates = []
        visited = np.zeros(env.instance.N, dtype=bool)
        stack = [rack]
        while stack:
            cur = stack.pop()
            if visited[cur]:
                continue
            visited[cur] = True
            next_list = []
            # if index valid, add to list
            if cur - env.instance.row_num >= 0: 
                next_list.append(cur - env.instance.row_num)
            if cur + env.instance.row_num < env.instance.N: 
                next_list.append(cur + env.instance.row_num)
            if cur - 1 >= 0:
                next_list.append(cur - 1)
            if cur + 1 < env.instance.N:
                next_list.append(cur + 1)
            # check each next index
            for nex in next_list:
                if env.assigned[nex] == 1: # if assigned, skip
                    continue
                if env.e[nex] == 0: # if empty
                    stack.append(nex)
                    if env.s[nex] == 0: # if empty place
                        place_candidates.append(nex)
        return place_candidates
                
    def get_picker(self, env, rack):
        # get nearest picker for target (if not target, picker = rack)
        targets = [pi for pi in range(self.action_space) if env.g[pi] == 1 and env.assigned[pi] == 0 and env.h[pi] == 0]
        pickers = [pi for pi in range(self.action_space) if env.p[pi] == 1]
        if rack in targets:
            min_d = np.inf
            picker = None # picker to chooose
            for pi in pickers:
                rack_corr = [rack // self.col_num, rack % self.col_num]
                pi_corr = [pi // self.col_num, pi % self.col_num]
                dist = abs(rack_corr[0]-pi_corr[0]) + \
                    abs(rack_corr[1]-pi_corr[1]) # manhattan distance to rack
                if dist < min_d:
                    min_d = dist
                    picker = pi
        else:
            picker = rack
        return picker

    def forward(self, state, info, exploit=0):
        if self.random_seed is not None:
            self.random_seed+=1
            np.random.seed(self.random_seed)
        # read information
        rack_mask = info["rack_mask"]
        place_mask = info["place_mask"]
        env = info["env"]
        # update rack_mask only consider block rack
        rack_candidates, place_tabu = self.get_rack_candidates(env, rack_mask)
        if len(rack_candidates) == 0: # 无可选货架，返回None
            return None
        rack_mask[:] = True
        for rack in rack_candidates:
            rack_mask[rack] = False
        info["rack_mask"] = rack_mask
        # network calculation
        state_v = torch.FloatTensor(state)
        q = env.robots[env.main_robot_id].pos
        info["q1"] = q
        logits1 = self.net(state=state_v, q=q)
        # choose rack
        logits1[rack_mask] = -np.inf 
        prob1 = torch.softmax(logits1, dim=0).detach().numpy()
        # assert prob1[0] is not None, "invalid logits"
        if exploit == 0:
            rack = np.random.choice(self.action_space, p=prob1)
        else:
            rack = np.argmax(prob1)
        # assert rack_mask[rack] == False, "choose masked rack"
        # update place_mask with place_tabu
        place_candidates = self.get_place_candidates(env, rack)
        place_mask[:] = True
        for place in place_candidates:
            place_mask[place] = False
        place_mask[place_tabu] = True
        info["place_mask"] = place_mask
        # choose target
        picker = self.get_picker(env, rack)
        # choose the nearest empty place to put the rack
        min_d = np.inf
        place = None # place to put
        for pi in place_candidates:
            picker_corr = [picker // self.col_num, picker % self.col_num]
            pi_corr = [pi // self.col_num, pi % self.col_num]
            dist = abs(picker_corr[0]-pi_corr[0]) + \
                abs(picker_corr[1]-pi_corr[1]) # manhattan distance to rack
            if dist < min_d:
                min_d = dist
                place = pi
        action = [rack, place]
        return action

    def update(self, batch):
        self.optim.zero_grad()
        batch_states, batch_actions, batch_rewards, batch_dones, \
            batch_infos = batch
        # calculate baseline of each step
        max_step_num = 0
        for actions in batch_actions:
            max_step_num = max(max_step_num, len(actions))
        step_rewards = [[] for _ in range(max_step_num)]
        for episode_reward in batch_rewards:
            for step_i, step_reward in enumerate(episode_reward):
                step_rewards[step_i].append(step_reward)
        step_baseline = []
        for step_i in range(len(step_rewards)):
            if len(step_rewards[step_i]) > 0:
                step_baseline.append(np.mean(step_rewards[step_i]))
        
        total_loss = 0
        for episode_i in range(len(batch_states)):
            for step_i in range(len(batch_states[episode_i])):
                # read batch data
                state = batch_states[episode_i][step_i]
                action = batch_actions[episode_i][step_i]
                reward = batch_rewards[episode_i][step_i]
                info = batch_infos[episode_i][step_i]

                # get masks
                rack_mask = info["rack_mask"]
                # place_mask = info["place_mask"]

                # calculate reward
                reward -= step_baseline[step_i]
                reward = torch.tensor(reward)

                # calculate log_prob
                state_v = torch.FloatTensor(state)
                logits1 = self.net(state=state_v, q=info["q1"])
                # logits2 = self.net(state=state_v, q=info["q2"])
                logits1[rack_mask] = -np.inf
                # logits2[place_mask] = -np.inf
                # read information
                prob1 = torch.softmax(logits1, dim=0) # check
                # prob2 = torch.softmax(logits2, dim=0)
                log_prob1 = torch.log_softmax(logits1, dim=0)
                # log_prob2 = torch.log_softmax(logits2, dim=0)

                # calculate loss
                loss = -log_prob1[action[0]] * reward #- log_prob2[action[1]] * reward #!check
                loss.backward()
                total_loss += loss.detach().numpy()

        # update paraments of net
        self.optim.step()

        # return mean loss for display
        mean_loss = total_loss / len(batch)
        return mean_loss
                
class HeuristicPolicy(nn.Module):
    def __init__(self, args):
        super(HeuristicPolicy, self).__init__()
        self.col_num = args['instance'].row_num
        self.action_space = args["instance"].N

    def get_rack_candidates(self, env, rack_mask):
        # 寻找阻碍货架作为货架候选
        rack_candidates = []
        place_tabu = []
        targets = [pi for pi in range(self.action_space) if env.g[pi] == 1 and env.assigned[pi] == 0 and env.h[pi] == 0]
        directions = [-self.col_num, self.col_num, -1, 1] # up, down, left, right
        for i in range(len(targets)):
            target = targets[i]
            min_block_num = np.inf
            rack = -1 # rack to move
            key_tabu = None # block place
            block_place_list = []
            for dire in directions:
                block_num = 0
                cur_pos = target
                cur_rack = target
                cur_tabu = []
                while env.s[cur_pos] == False:  # 假设仓库外围一圈一定是通道，通过判断通道避免了越界
                    block_place_list.append(cur_pos)
                    if env.e[cur_pos] == True:
                        block_num += 1
                        cur_rack = cur_pos
                    else:
                        cur_tabu.append(cur_pos) # 禁止放在核心通道上，防止目标货架锁死
                    cur_pos += dire
                if block_num < min_block_num:
                    min_block_num = block_num
                    rack = cur_rack
                    key_tabu = cur_tabu
            if rack_mask[rack] == False:
                rack_candidates.append(rack)
                place_tabu += key_tabu
        return rack_candidates, place_tabu
                
    def get_picker(self, env, rack):
        # get nearest picker for target (if not target, picker = rack)
        targets = [pi for pi in range(self.action_space) if env.g[pi] == 1 and env.assigned[pi] == 0 and env.h[pi] == 0]
        pickers = [pi for pi in range(self.action_space) if env.p[pi] == 1]
        if rack in targets:
            min_d = np.inf
            picker = None # picker to chooose
            for pi in pickers:
                rack_corr = [rack // self.col_num, rack % self.col_num]
                pi_corr = [pi // self.col_num, pi % self.col_num]
                dist = abs(rack_corr[0]-pi_corr[0]) + \
                    abs(rack_corr[1]-pi_corr[1]) # manhattan distance to rack
                if dist < min_d:
                    min_d = dist
                    picker = pi
        else:
            picker = rack
        return picker

    def get_place_candidates(self, env, rack):
        # 选择rack作为货架，能够到达的空位
        place_candidates = []
        visited = np.zeros(env.instance.N, dtype=bool)
        stack = [rack]
        while stack:
            cur = stack.pop()
            if visited[cur]:
                continue
            visited[cur] = True
            next_list = []
            # if index valid, add to list
            if cur - env.instance.row_num >= 0: 
                next_list.append(cur - env.instance.row_num)
            if cur + env.instance.row_num < env.instance.N: 
                next_list.append(cur + env.instance.row_num)
            if cur - 1 >= 0:
                next_list.append(cur - 1)
            if cur + 1 < env.instance.N:
                next_list.append(cur + 1)
            # check each next index
            for nex in next_list:
                if env.assigned[nex] == 1: # if assigned, skip
                    continue
                if env.e[nex] == 0: # if empty
                    stack.append(nex)
                    if env.s[nex] == 0: # if empty place
                        place_candidates.append(nex)
        return place_candidates

    def forward(self, state, info, exploit=0):
        # read information
        rack_mask = info["rack_mask"]
        place_mask = info["place_mask"]
        env = info["env"]
        rack_candidates, place_tabu = self.get_rack_candidates(env, rack_mask)
        if len(rack_candidates) == 0: # 无可选，返回None
            return None
        # 启发式选择第一个阻碍货架
        rack = rack_candidates[0] 
        assert rack_mask[rack] == False, "chose wrong rack"
        # update place_candidates
        place_candidates = self.get_place_candidates(env, rack)
        for pi in place_tabu: # update candidates with place_tabu
            if pi in place_candidates:
                place_candidates.remove(pi)
        if len(place_candidates) == 0: # 无可选，返回None
            return None
        # get picker
        picker = self.get_picker(env, rack)         
        # picker = rack
        # choose the nearest empty place to put the rack
        min_d = np.inf
        place = None # place to put
        for pi in place_candidates:
            picker_corr = [picker // self.col_num, picker % self.col_num]
            pi_corr = [pi // self.col_num, pi % self.col_num]
            dist = abs(picker_corr[0]-pi_corr[0]) + \
                abs(picker_corr[1]-pi_corr[1]) # manhattan distance to rack
            if dist < min_d:
                min_d = dist
                place = pi
        # assert place_mask[place] == False, "chose wrong place"
        # return action
        action = [rack, place]
        return action
        
        
        
                

                
            
            









