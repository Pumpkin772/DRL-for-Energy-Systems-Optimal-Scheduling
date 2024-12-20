import pickle
import torch
import pandas as pd
from plotDRL import plot_training_rewardinfo, plot_evaluation_information, plot_cost_rewardinfo
import matplotlib.pyplot as plt
datasource1 = 'D:\桌面\待实现\代码\DRL-for-Energy-Systems-Optimal-Scheduling\AgentPPO\\all_seeds_reward_record.pkl'
datasource2 = 'D:\桌面\待实现\代码\DRL-for-Energy-Systems-Optimal-Scheduling\AgentDDPG\\all_seeds_reward_record.pkl'
with open(datasource2, 'rb') as tf:
    train_data = pickle.load(tf)
print(train_data[1234])
plot_evaluation_information('D:\桌面\待实现\代码\DRL-for-Energy-Systems-Optimal-Scheduling\AgentDDPG\\test_data.pkl','D:\桌面\待实现\代码\DRL-for-Energy-Systems-Optimal-Scheduling\AgentDDPG\DRL__plots')