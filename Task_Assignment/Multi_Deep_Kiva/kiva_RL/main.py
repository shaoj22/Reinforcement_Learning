# main function for RL
# author: Charles Lee
# date: 2022.10.26

import numpy as np
import matplotlib.pyplot as plt
import math
import time
import tqdm
from copy import deepcopy
import torch
from torch.utils.tensorboard import SummaryWriter
# from MCRMFS_Instance import Instance_all
from MCRMFS_Instance_A import Instance_all
from Env import Env
from Model import Attention_Net, AttentionQuery_Net
from Policy import PGPolicy, HeuristicPolicy
from Collector import Collector, Experience, ExperienceBuffer
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

args = {
    # running option
    "render" : 0, # whether render the game
    "record_data" : 0, # whether record data in tensorboard
    "load_model" : "", # whether import attributed of network, 0 for not
    "save_model" : 0, # whether save the learned network model, 1 for save, 0 for not
    "file_name_suffix" : "", # when save data or model, add the suffix after file name

    # args for model
    "test" : 1, # if test, no learning
    "method" : "heuristic", # net, heuristic, random
    "network" : "AM", # AM, FF
    "learning_rate" : 1e-4, # learning rate of optim
    "batch_num" : 1, # number of batches to train or test
    "batch_size" : 1, # batch size
    "regenerate_episode" : 2**4, # number of episodes to regenerate map
    "test_size" : 100, # number of test episode from mapseed 0 to test_size each batch
    "gamma" : 0.95, # the discount rate
    "random_seed" : None, # random seed, None for not set

    # args for environment
    "instance" : Instance_all(), # example instance for information checking (for Policy)
}

def train(args):
    # create the net, optim and setup policy
    if args["method"] == "net":
        if args["network"] == "AM":
            # net = Attention_Net(node_dim=8) # x, y, s, p, e, g, h, assigned
            net = AttentionQuery_Net(node_dim=8) # x, y, s, p, e, g, h, assigned
        else:
            assert False, "network name not exists"
        optim = torch.optim.AdamW(net.parameters(), lr=args["learning_rate"])
        policy = PGPolicy(net, optim, args)
        if args["load_model"] and args["method"] == "net":
            policy.load_state_dict(
                torch.load('saved_models/{}.pth'.format(args["load_model"]))
            )
    elif args["method"] == "heuristic":
        policy = HeuristicPolicy(args)
    # setup collector, env is created inside
    replay_buffer = ExperienceBuffer(args["batch_size"])
    collector = Collector(policy, replay_buffer, args)
    # set writer for tensorboard
    file_name = f"MCRMFS-lr{args['learning_rate']}_"+\
        f"{args['method']}_{args['file_name_suffix']}_{str(time.time())[-4:]}"
    if args['record_data']:
        writer = SummaryWriter('runs/'+file_name)
    
    # run batch_size times and collect data in buffer
    test_start = time.time()
    test_rewards = collector.test(args["test_size"])
    test_timecost = (time.time() - test_start) / args["test_size"]
    return test_rewards, test_timecost


if __name__ == "__main__":
    train(args) 
