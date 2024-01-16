'''
File: Train.py
Project: DRLH
File Created: Wednesday, 31st May 2023 5:02:42 pm
Author: Charles Lee (lmz22@mails.tsinghua.edu.cn)
'''

import gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from Hyper_Heuristic.ALNS_Operator_Choice.Deep_Reinforcement_Learning.Env import ALNS_Env
from Hyper_Heuristic.ALNS_Operator_Choice.Deep_Reinforcement_Learning.PPO import PPO, train_on_policy_agent
from Hyper_Heuristic.ALNS_Operator_Choice.Deep_Reinforcement_Learning.SAC import SAC, train_off_policy_agent
import tqdm
import datetime
from torch.utils.tensorboard import SummaryWriter
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class Args:
    def __init__(self):
        # env params
        self.env = ALNS_Env()
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        # common alg params
        self.alg_name = 'sac'
        self.actor_lr = 1e-4
        self.critic_lr = 1e-5
        self.num_episodes = 1000
        self.hidden_dim = 256
        self.gamma = 0.5
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # ppo params
        self.lmbda = 0.95
        self.epochs = 10
        self.eps = 0.2
        # sac params
        self.alpha_lr = 1e-3
        self.target_entropy = -10 * self.action_dim
        self.tau = 0.005
        self.buffer_size = 1000000
        self.minimal_size = 10000
        self.batch_size = 256
        self.update_step = 1
        # file params
        self.cur_dir = os.path.dirname(os.path.abspath(__file__))
        self.cur_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        self.log_dir = os.path.join(self.cur_dir, 'output', self.cur_time)
        self.model_dir = os.path.join(self.log_dir, self.alg_name+'_model')
        self.save_interval = 100


def train(args):
    env = args.env
    torch.manual_seed(0)
    np.random.seed(0)
    writer = SummaryWriter(args.log_dir)
    writer.add_text("args", str(args.__dict__))
    if args.alg_name == 'ppo':
        agent = PPO(args)
        train_on_policy_agent(args, agent, env, writer)
    elif args.alg_name == 'sac':
        agent = SAC(args)
        train_off_policy_agent(args, agent, env, writer)

if __name__ == "__main__":
    args = Args()
    train(args)