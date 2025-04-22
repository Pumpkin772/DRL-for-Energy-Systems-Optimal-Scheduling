import pickle
import torch
import pandas as pd
from plotDRL import plot_training_rewardinfo, plot_evaluation_information, plot_cost_rewardinfo, plot_cost_rewardinfo_MIP, plot_training_unbalanceinfo
import matplotlib.pyplot as plt
import numpy as np

datasource1 = 'D:\桌面\待实现\代码\DRL-for-Energy-Systems-Optimal-Scheduling\AgentTD3_with_safe_action\\all_seeds_reward_record.pkl'
name1 = 'safe_action'
color1 = 'blue'
plot_training_unbalanceinfo(name1,datasource1,color1)
# 添加标题和轴标签
plt.title('Training unbalance over Episodes with 95% Confidence Interval')
plt.xlabel('Episode')
plt.ylabel('Unbalance')
# 添加虚线网格
plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
plt.show()
