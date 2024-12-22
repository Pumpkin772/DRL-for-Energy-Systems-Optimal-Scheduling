import os
import pickle
import gym
import time
import torch
import torch.nn as nn
import numpy as np
import numpy.random as rd
from torch.nn.modules import loss
from random_generator_battery import ESSEnv
import pandas as pd

from tools import Arguments, get_episode_return, test_one_episode, ReplayBuffer, optimization_base_result, test_ten_episodes, test_ten_episodes_MIP
from agent import AgentDDPG, AgentPPO, AgentSAC, AgentTD3
from random_generator_battery import ESSEnv
from net import Actor_MIP, CriticQ

env = ESSEnv()
state = env.reset()

agent = AgentDDPG()
agent.init(64, env.state_space.shape[0], env.action_space.shape[0], 1e-4)
agent.act.load_state_dict(torch.load('D:\桌面\待实现\代码\DRL-for-Energy-Systems-Optimal-Scheduling\AgentDDPG\\actor.pth'))
record = test_ten_episodes(state, env, agent.act, agent.device)
print(record['unbalance'])

agent = AgentPPO()
agent.init(64, env.state_space.shape[0], env.action_space.shape[0], 1e-4)
agent.act.load_state_dict(torch.load('D:\桌面\待实现\代码\DRL-for-Energy-Systems-Optimal-Scheduling\AgentPPO\\actor.pth'))
record = test_ten_episodes(state, env, agent.act, agent.device)
print(record['unbalance'])

agent = AgentSAC()
agent.init(64, env.state_space.shape[0], env.action_space.shape[0], 1e-4)
agent.act.load_state_dict(torch.load('D:\桌面\待实现\代码\DRL-for-Energy-Systems-Optimal-Scheduling\AgentSAC\\actor.pth'))
record = test_ten_episodes(state, env, agent.act, agent.device)
print(record['unbalance'])

agent = AgentTD3()
agent.init(64, env.state_space.shape[0], env.action_space.shape[0], 1e-4)
agent.act.load_state_dict(torch.load('D:\桌面\待实现\代码\DRL-for-Energy-Systems-Optimal-Scheduling\AgentTD3\\actor.pth'))
record = test_ten_episodes(state, env, agent.act, agent.device)
print(record['unbalance'])

cri_save_path = 'D:\桌面\\test\MIP-DQN/critic.pth'
net_dim = 64
net = CriticQ(net_dim, env.state_space.shape[0], env.action_space.shape[0])
net.load_state_dict(torch.load(cri_save_path))
scaled_parameters = np.ones(8)
scaled_parameters[0] = env.battery.max_charge
scaled_parameters[1] = env.dg1.ramping_up
scaled_parameters[5] = env.dg1.power_output_max
scaled_parameters[2] = env.dg2.ramping_up
scaled_parameters[6] = env.dg2.power_output_max
scaled_parameters[3] = env.dg3.ramping_up
scaled_parameters[7] = env.dg3.power_output_max
scaled_parameters[4] = env.Netload_max
batch_size = 256
actor = Actor_MIP(scaled_parameters, batch_size, net, env.state_space.shape[0], env.action_space.shape[0], env)
record = test_ten_episodes_MIP(state,env,actor,agent.device)
print(record['unbalance'])


