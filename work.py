import pickle
import os
test_data_path = 'D:\桌面\待实现\代码\DRL-for-Energy-Systems-Optimal-Scheduling\AgentDDPG\\test_data.pkl'
size = os.path.getsize(test_data_path)  # 获取文件大小
print(size)
with open(test_data_path, 'rb') as tf:
    test_data = pickle.load(tf)
print(test_data)