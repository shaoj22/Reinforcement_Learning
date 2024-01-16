# collector class for RL
# author: Charles Lee
# date: 2022.10.26

import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import trange
import torch
import tensorboard
import collections
from Env import Env

class Collector():
    def __init__(self, policy, buffer, args):
        self.policy = policy
        self.buffer = buffer
        self.gamma = args["gamma"]
        # translate scale to instance
        self.env = Env(args["instance"], args)
        self.test_env = Env(args["instance"], args)
        self.regenerate_episode = args["regenerate_episode"]
        self.test_size = args["test_size"]
        self.map_seed = args["random_seed"]

    def wrap_exp(self, experience):
        episode_rewards = experience.episode_rewards
        ## discount sum
        sum_reward = 0
        for ri_ in range(len(episode_rewards)):
            ri = len(episode_rewards)-1-ri_ # reverse rank
            sum_reward *= self.gamma
            sum_reward += episode_rewards[ri]
            episode_rewards[ri] = sum_reward
        ## sum all
        # sum_reward = 0
        # for ri in range(len(episode_rewards)):
        #     episode_rewards[ri] = sum_reward
        self.buffer.append(experience)

    @torch.no_grad()
    def collect(self, n_episode):
        rewards = 0
        for episode_i in range(n_episode):
            # if episode_i % self.regenerate_episode == 0 and self.map_seed is None:
            #     self.env.instance.generate()
            self.env.instance.generate(episode_i % self.test_size)
            state, info = self.env.reset()
            episode_states, episode_actions, episode_rewards, episode_dones, \
                episode_infos = [], [], [], [], []
            while 1:
                # action according to state and info
                action = self.policy(state, info)
                # record state, info, action before env update
                if action is not None:
                    episode_states.append(state)
                    episode_infos.append(info) 
                    episode_actions.append(action)
                # environment update
                state, reward, done, info = self.env.step(action)
                # record reward, done after env update
                if action is not None:
                    episode_rewards.append(reward)
                    episode_dones.append(done)
                # accumulate reward
                rewards.append(reward)
                # break and save experience if done
                if done:
                    exp = Experience(episode_states, episode_actions, episode_rewards, \
                        episode_dones, episode_infos)
                    self.wrap_exp(exp)
                    break
        return rewards
        
    @torch.no_grad()
    def test(self, n_episode):
        """ test from map seed 0 to seed n_episode """
        rewards = []
        for map_seed in range(n_episode):
            epi_reward = 0
            self.test_env.instance.generate(map_seed)
            state, info = self.test_env.reset()
            while 1:
                # action according to state and info
                action = self.policy(state, info, exploit=1)
                # environment update
                state, reward, done, info = self.test_env.step(action)
                # accumulate reward
                epi_reward += reward
                # break and save experience if done
                if done:
                    break
            rewards.append(epi_reward)
        return rewards

Experience = collections.namedtuple("episode_Experience", \
    field_names=["episode_states", "episode_actions", "episode_rewards", \
        "episode_dones", "episode_infos"])

class ExperienceBuffer():
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=int(capacity))

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def clear(self):
        self.buffer.clear()

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        episode_states, episode_actions, episode_rewards, episode_dones, \
                episode_infos = zip(*[self.buffer[idx] for idx in indices])
        # 每个episode的steps不一定一样，用list形式输出
        return episode_states, episode_actions, episode_rewards, episode_dones, \
            episode_infos
        
    
    

