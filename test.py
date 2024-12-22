import pickle
import torch
import pandas as pd
from plotDRL import plot_training_rewardinfo, plot_evaluation_information, plot_cost_rewardinfo
import matplotlib.pyplot as plt
from DDPG import AgentDDPG
from tools import test_one_episode, test_ten_episodes_cost_NLP
from random_generator_battery import ESSEnv
# datasource1 = 'D:\桌面\待实现\代码\DRL-for-Energy-Systems-Optimal-Scheduling\AgentPPO\\all_seeds_reward_record.pkl'
# datasource2 = 'D:\桌面\待实现\代码\DRL-for-Energy-Systems-Optimal-Scheduling\AgentDDPG\\all_seeds_reward_record.pkl'
# with open(datasource2, 'rb') as tf:
#     train_data = pickle.load(tf)
# print(train_data[1234])
# agent = AgentDDPG()
# env = ESSEnv()
# agent.init(256, env.state_space.shape[0], env.action_space.shape[0], 1e-4)
# agent.act.load_state_dict(torch.load('D:\桌面\待实现\代码\DRL-for-Energy-Systems-Optimal-Scheduling\AgentDDPG\\actor.pth'))
# print('parameters have been reload and test')
# record = test_one_episode(env, agent.act, agent.device)
# test_data_save_path = 'D:\桌面\待实现\代码\DRL-for-Energy-Systems-Optimal-Scheduling\AgentDDPG\\test_data2.pkl'
# with open(test_data_save_path, 'wb') as tf:
#     pickle.dump(record, tf)
# #plot_evaluation_information('D:\桌面\待实现\代码\DRL-for-Energy-Systems-Optimal-Scheduling\AgentDDPG\\test_data2.pkl','D:\桌面\待实现\代码\DRL-for-Energy-Systems-Optimal-Scheduling\AgentDDPG\DRL__plots')
# with open(test_data_save_path, 'rb') as tf:
#     test2 = pickle.load(tf)
# print(test2['unbalance'])
env = ESSEnv()
state = env.reset()
record = test_ten_episodes_cost_NLP(env)
print(record)