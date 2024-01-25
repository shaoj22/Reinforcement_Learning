'''
File: PPO.py
Project: DRLH
File Created: Thursday, 1st June 2023 12:36:27 am
Author: Charles Lee (lmz22@mails.tsinghua.edu.cn)
'''


import gym
import torch
import torch.nn.functional as F
import numpy as np
import Hyper_Heuristic.ALNS_Operator_Choice.Deep_Reinforcement_Learning.rl_utils as rl_utils
import os


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.preprocess = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
        )
        self.output_layer = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.preprocess(x))
        return F.softmax(self.output_layer(x), dim=1)


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.preprocess = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
        )
        self.output_layer = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.preprocess(x))
        return self.output_layer(x)


class PPO:
    ''' PPO算法,采用截断方式 '''
    def __init__(self, args):
                #  state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                #  lmbda, epochs, eps, gamma, device):
        # read params
        state_dim = args.state_dim
        hidden_dim = args.hidden_dim
        action_dim = args.action_dim
        actor_lr = args.actor_lr
        critic_lr = args.critic_lr
        lmbda = args.lmbda
        epochs = args.epochs
        eps = args.eps
        gamma = args.gamma
        device = args.device

        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state) + 1e-5
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        td_target = rewards + self.gamma * self.critic(next_states) * (1 -
                                                                       dones)
        td_delta = td_target - self.critic(states)
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda,
                                               td_delta.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor(states).gather(1,
                                                            actions)).detach()

        actor_loss_sum = 0
        critic_loss_sum = 0
        total_loss_sum = 0
        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps,
                                1 + self.eps) * advantage  # 截断
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
            actor_loss_sum += actor_loss.item()
            critic_loss_sum += critic_loss.item()
            total_loss_sum += actor_loss.item() + critic_loss.item()
        avg_actor_loss = actor_loss_sum / self.epochs
        avg_critic_loss = critic_loss_sum / self.epochs
        avg_total_loss = total_loss_sum / self.epochs
        loss_info = {
            'actor_loss': avg_actor_loss,
            'critic_loss': avg_critic_loss,
            'total_loss': avg_total_loss, 
        }
        return loss_info

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.actor.state_dict(), path + '\\actor.pth')
        torch.save(self.critic.state_dict(), path + '\\critic.pth')
    
    def load(self, path):
        self.actor.load_state_dict(torch.load(path + 'actor.pth'))
        self.critic.load_state_dict(torch.load(path + 'critic.pth'))


def train_on_policy_agent(args, agent, env, writer):
    with tqdm.tqdm(total=args.num_episodes) as pbar:
        for i_episode in range(args.num_episodes):
            episode_reward = 0
            transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
            state, _ = env.reset()
            done = False
            while not done:
                action = agent.take_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                transition_dict['states'].append(state)
                transition_dict['actions'].append(action)
                transition_dict['next_states'].append(next_state)
                transition_dict['rewards'].append(reward)
                transition_dict['dones'].append(done)
                state = next_state
                episode_reward += reward
            loss_info = agent.update(transition_dict)
            writer.add_scalar('reward/episode_reward', episode_reward, i_episode)
            for key, value in loss_info.items():
                writer.add_scalar('loss/{}'.format(key), value, i_episode)
            pbar.set_postfix({'episode': '%d' % (i_episode), 'reward': '%.3f' % episode_reward})
            pbar.update(1)
            if i_episode % args.save_interval == 0:
                agent.save(args.model_dir)
