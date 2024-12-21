import pickle
import torch
import pandas as pd
from plotDRL import plot_training_rewardinfo, plot_evaluation_information, plot_cost_rewardinfo, plot_cost_rewardinfo_MIP, plot_training_unbalanceinfo
import matplotlib.pyplot as plt

datasource = 'D:\桌面\\test\MIP-DQN\AgentMIPDQN\\test_data1.pkl'
with open(datasource, 'rb') as tf:
    test_data = pickle.load(tf)
print(test_data['unbalance'])