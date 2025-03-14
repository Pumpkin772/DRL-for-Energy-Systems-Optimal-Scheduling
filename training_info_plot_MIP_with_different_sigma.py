import pickle
import torch
import pandas as pd
from plotDRL import plot_training_rewardinfo, plot_evaluation_information, plot_cost_rewardinfo, plot_cost_rewardinfo_MIP, plot_training_unbalanceinfo
import matplotlib.pyplot as plt
datasource1 = 'D:\桌面\\test\MIP-DQN\AgentMIPDQN\\all_seeds_reward_record_sigma20.pkl'
name1 = 'simga20'
color1 = 'blue'
datasource2 = 'D:\桌面\\test\MIP-DQN\AgentMIPDQN\\all_seeds_reward_record_sigma50.pkl'
name2 = 'sigma50'
color2 = 'green'
datasource3 = 'D:\桌面\\test\MIP-DQN\AgentMIPDQN\\all_seeds_reward_record_sigma100.pkl'
name3 = 'sigma100'
color3 = 'pink'

plot_training_rewardinfo(name1,datasource1,color1)
plot_training_rewardinfo(name2,datasource2,color2)
plot_training_rewardinfo(name3,datasource3,color3)
plt.legend()
# 添加标题和轴标签
plt.title('Training rewards over Episodes with 95% Confidence Interval')
plt.xlabel('Episode')
plt.ylabel('Reward')
# 添加虚线网格
plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
plt.show()

plot_cost_rewardinfo(name1,datasource1,color1)
plot_cost_rewardinfo(name2,datasource2,color2)
plot_cost_rewardinfo(name3,datasource3,color3)
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
plt.legend()
# 添加标题和轴标签
plt.title('Training unbalance over Episodes with 95% Confidence Interval')
plt.xlabel('Episode')
plt.ylabel('Unbalance')
# 添加虚线网格
plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
plt.show()