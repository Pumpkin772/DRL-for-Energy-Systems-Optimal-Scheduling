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
import matplotlib.pyplot as plt
import seaborn as sns

from tools import Arguments, get_episode_return, test_one_episode, ReplayBuffer, optimization_base_result, test_ten_episodes, test_ten_episodes_MIP
from agent import AgentDDPG, AgentPPO, AgentSAC, AgentTD3
from random_generator_battery import ESSEnv
from net import Actor_MIP, CriticQ

env = ESSEnv()
state = env.reset()

agent = AgentDDPG()
agent.init(64, env.state_space.shape[0], env.action_space.shape[0], 1e-4)
agent.act.load_state_dict(torch.load('D:\桌面\待实现\代码\DRL-for-Energy-Systems-Optimal-Scheduling\AgentDDPG\\actor.pth'))
record1 = test_ten_episodes(state, env, agent.act, agent.device)

agent = AgentPPO()
agent.init(64, env.state_space.shape[0], env.action_space.shape[0], 1e-4)
agent.act.load_state_dict(torch.load('D:\桌面\待实现\代码\DRL-for-Energy-Systems-Optimal-Scheduling\AgentPPO\\actor.pth'))
record2 = test_ten_episodes(state, env, agent.act, agent.device)


agent = AgentSAC()
agent.init(64, env.state_space.shape[0], env.action_space.shape[0], 1e-4)
agent.act.load_state_dict(torch.load('D:\桌面\待实现\代码\DRL-for-Energy-Systems-Optimal-Scheduling\AgentSAC\\actor.pth'))
record3 = test_ten_episodes(state, env, agent.act, agent.device)


agent = AgentTD3()
agent.init(64, env.state_space.shape[0], env.action_space.shape[0], 1e-4)
agent.act.load_state_dict(torch.load('D:\桌面\待实现\代码\DRL-for-Energy-Systems-Optimal-Scheduling\AgentTD3\\actor.pth'))
record4 = test_ten_episodes(state, env, agent.act, agent.device)

#
# cri_save_path = 'D:\桌面\\test\MIP-DQN/critic.pth'
# net_dim = 64
# net = CriticQ(net_dim, env.state_space.shape[0], env.action_space.shape[0])
# net.load_state_dict(torch.load(cri_save_path))
# scaled_parameters = np.ones(8)
# scaled_parameters[0] = env.battery.max_charge
# scaled_parameters[1] = env.dg1.ramping_up
# scaled_parameters[5] = env.dg1.power_output_max
# scaled_parameters[2] = env.dg2.ramping_up
# scaled_parameters[6] = env.dg2.power_output_max
# scaled_parameters[3] = env.dg3.ramping_up
# scaled_parameters[7] = env.dg3.power_output_max
# scaled_parameters[4] = env.Netload_max
# batch_size = 256
# actor = Actor_MIP(scaled_parameters, batch_size, net, env.state_space.shape[0], env.action_space.shape[0], env)
# record5 = test_ten_episodes_MIP(state,env,actor,agent.device)
#print(record5['unbalance'])

days = []
record6 = []
for i in range(1,11):
    days.append(i)
    record6.append(0)
print(record1['unbalance'])
plt.plot(days, record1['unbalance'], label='DDPG', color='blue')
print(record2['unbalance'])
plt.plot(days, record2['unbalance'], label='PPO', color='yellow')
print(record3['unbalance'])
plt.plot(days, record3['unbalance'], label='SAC', color='green')
print(record4['unbalance'])
plt.plot(days, record4['unbalance'], label='TD3', color='pink')

#plt.plot(days, record6, label='MIP-DQN', color='cyan')

plt.scatter(days, record1['unbalance'], color='blue')
plt.scatter(days, record2['unbalance'], color='yellow')
plt.scatter(days, record3['unbalance'], color='green')
plt.scatter(days, record4['unbalance'], color='pink')
#plt.scatter(days, record6, color='cyan')

plt.legend()
# 添加标题和轴标签
#plt.title('Training rewards over Episodes with 95% Confidence Interval')
plt.xlabel('Days')
plt.ylabel('Cumulative unbalance')
# 添加虚线网格
plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
plt.show()
plt.close()


