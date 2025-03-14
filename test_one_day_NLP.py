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
from plotDRL import plot_optimization_result

env = ESSEnv()
state = env.reset()
output_record = optimization_base_result(env, env.month, env.day, env.battery.current_capacity)
del output_record['min_cost']
output_record_df = pd.DataFrame.from_dict(output_record)
plot_optimization_result(output_record_df, 'D:\桌面\待实现\代码\DRL-for-Energy-Systems-Optimal-Scheduling\AgentDDPG\DRL__plots')