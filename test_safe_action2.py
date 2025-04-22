# 测试 safe_action无解的原因

import os
import pickle
import gym
import torch
import torch.nn as nn
import numpy as np
import numpy.random as rd
from torch.nn.modules import loss
from plotDRL import plot_optimization_result
from random_generator_battery import ESSEnv
import pandas as pd
import matplotlib.pyplot as plt

from tools import Arguments,get_episode_return,test_one_episode,ReplayBuffer,optimization_base_result,test_ten_episodes, test_ten_episodes_safe
from agent import AgentTD3_with_safe_action
from random_generator_battery import ESSEnv

act_save_path = "D:\桌面\待实现\代码\DRL-for-Energy-Systems-Optimal-Scheduling\AgentTD3_with_safe_action\\actor.pth"
args=Arguments()
agent = AgentTD3_with_safe_action()
env = ESSEnv()
agent.state = env.reset()
agent.init(args.net_dim, env.state_space.shape[0], env.action_space.shape[0], args.learning_rate,
           args.if_per_or_gae)
agent.act.load_state_dict(torch.load(act_save_path))
target_step = 100
soc_list = []
flag_list = []
for i in range(target_step):
    state = agent.state
    action = agent.select_action(state)
    safe_action = env.get_safe_action(action)
    #flag_list.append(flag)
    state, next_state, reward, done, = env.step(safe_action)
    soc_list.append(env.battery.SOC())
    print(f"unbalance:{env.real_unbalance}")
    print(f"load:{env.electricity_demand}")
    print(f"pv:{env.pv_generation}")
    agent.state = next_state

print(soc_list)
plt.plot(soc_list)
#plt.plot(flag_list)
plt.show()