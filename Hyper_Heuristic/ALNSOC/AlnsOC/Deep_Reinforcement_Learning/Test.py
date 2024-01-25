'''
File: Test.py
Project: DRLH
File Created: Sunday, 28th May 2023 10:19:05 am
Author: Charles Lee (lmz22@mails.tsinghua.edu.cn)
'''

import torch, numpy as np
from Hyper_Heuristic.ALNS_Operator_Choice.Deep_Reinforcement_Learning.Env import ALNS_Env
from Hyper_Heuristic.ALNS_Operator_Choice.Deep_Reinforcement_Learning.SAC import SAC, PolicyNet
from Hyper_Heuristic.ALNS_Operator_Choice.Deep_Reinforcement_Learning.Train import Args
device = 'cuda' if torch.cuda.is_available() else 'cpu'

args = Args()
env = args.env
agent = SAC(args)
agent.load("DRLH\\output\\20230601-164832\\sac_model")

np.random.seed(0)
total_reward = 0
state, info = env.reset()
while True:
    # action = agent.take_action(state)
    action = env.alg.alns_select_operator()
    state, reward, truncated, terminated, _ = env.step(action)
    done = truncated or terminated
    print("iter {}: best obj: {}".format(env.alg.step, env.alg.best_obj))
    total_reward += reward
    if done:
        break
print("total reward: {}".format(total_reward))

