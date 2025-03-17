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

from tools import Arguments,get_episode_return,test_one_episode,ReplayBuffer,optimization_base_result,test_ten_episodes, test_ten_episodes_safe
from agent import AgentTD3
from random_generator_battery import ESSEnv


args=Arguments()
agent = AgentTD3()
env = ESSEnv()
agent.state = env.reset()
agent.init(args.net_dim, env.state_space.shape[0], env.action_space.shape[0], args.learning_rate,
           args.if_per_or_gae)
target_step = 10
for i in range(target_step):
    state = agent.state
    action = agent.select_action(state)
    safe_action = env.get_safe_action(action)
    state, next_state, reward, done, = env.step(safe_action)
    print(env.real_unbalance)
    agent.state = next_state