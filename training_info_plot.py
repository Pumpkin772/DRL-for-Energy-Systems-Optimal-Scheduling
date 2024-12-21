import pickle
import torch
import pandas as pd
from plotDRL import plot_training_rewardinfo, plot_evaluation_information, plot_cost_rewardinfo, plot_cost_rewardinfo_MIP, plot_training_unbalanceinfo
import matplotlib.pyplot as plt
datasource1 = 'D:\桌面\待实现\代码\DRL-for-Energy-Systems-Optimal-Scheduling\AgentPPO\\all_seeds_reward_record.pkl'
name1 = 'PPO'
color1 = 'blue'
datasource2 = 'D:\桌面\待实现\代码\DRL-for-Energy-Systems-Optimal-Scheduling\AgentDDPG\\all_seeds_reward_record.pkl'
name2 = 'DDPG'
color2 = 'yellow'
datasource3 = 'D:\桌面\待实现\代码\DRL-for-Energy-Systems-Optimal-Scheduling\AgentTD3\\all_seeds_reward_record.pkl'
name3 = 'TD3'
color3 = 'pink'
datasource4 = 'D:\桌面\待实现\代码\DRL-for-Energy-Systems-Optimal-Scheduling\AgentSAC\\all_seeds_reward_record.pkl'
name4 = 'SAC'
color4 = 'green'
datasource5 = 'D:\桌面\\test\MIP-DQN\AgentMIPDQN\\all_seeds_reward_record.pkl'
name5 = 'MIP-DQN'
color5 = 'cyan'

plot_training_rewardinfo(name1,datasource1,color1)
plot_training_rewardinfo(name2,datasource2,color2)
plot_training_rewardinfo(name3,datasource3,color3)
plot_training_rewardinfo(name4,datasource4,color4)
plot_training_rewardinfo(name5,datasource5,color5)
plt.legend()
# 添加标题和轴标签
plt.title('Training rewards over Episodes with 95% Confidence Interval')
plt.xlabel('Episode')
plt.ylabel('Cost')
# 添加虚线网格
plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
plt.show()

plot_cost_rewardinfo(name1,datasource1,color1)
plot_cost_rewardinfo(name2,datasource2,color2)
plot_cost_rewardinfo(name3,datasource3,color3)
plot_cost_rewardinfo(name4,datasource4,color4)
plot_cost_rewardinfo_MIP(name5,datasource5,color5)
plt.legend()
# 添加标题和轴标签
plt.title('Training Cost over Episodes with 95% Confidence Interval')
plt.xlabel('Episode')
plt.ylabel('Cost')
# 添加虚线网格
plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
plt.show()

plot_training_unbalanceinfo(name1,datasource1,color1)
plot_training_unbalanceinfo(name2,datasource2,color2)
plot_training_unbalanceinfo(name3,datasource3,color3)
plot_training_unbalanceinfo(name4,datasource4,color4)
plot_training_unbalanceinfo(name5,datasource5,color5)
plt.legend()
# 添加标题和轴标签
plt.title('Training unbalance over Episodes with 95% Confidence Interval')
plt.xlabel('Episode')
plt.ylabel('Cost')
# 添加虚线网格
plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
plt.show()

# with open(datasource, 'rb') as tf:
#     train_data = pickle.load(tf)
# print(len(train_data[1234]['mean_episode_reward']))
#plot_evaluation_information('D:\桌面\待实现\代码\DRL-for-Energy-Systems-Optimal-Scheduling\AgentDDPG\\test_data.pkl','D:\桌面\待实现\代码\DRL-for-Energy-Systems-Optimal-Scheduling\AgentDDPG\DRL__plots')
# print(torch.cuda.is_available())
# print(torch.version.cuda)
# print(torch.cuda.device_count())
