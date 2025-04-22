import numpy as np
import matplotlib.pyplot as plt

# 假设这是你从训练过程中收集的数据，这里我们使用随机数据作为示例
np.random.seed(0)  # 为了可重复性
episodes = np.arange(1, 101)  # 假设有100个训练周期
rewards = np.random.normal(loc=0.5, scale=0.1, size=(100, 5))  # 假设的奖励数据，5个随机种子

# 计算每个周期的均值和标准误差
mean_rewards = np.mean(rewards, axis=1)
sem_rewards = np.std(rewards, axis=1) / np.sqrt(rewards.shape[1])
print(rewards.shape)
print(mean_rewards.shape)
print(episodes.shape)
# 计算95%置信区间的误差范围
critical_value = 1.96  # 对应于95%置信水平
margin_error = critical_value * sem_rewards

# 绘制均值曲线
plt.plot(episodes, mean_rewards, label='Mean Reward', color='blue')

# 绘制95%置信区间
plt.fill_between(episodes, mean_rewards - margin_error, mean_rewards + margin_error, color='lightblue', alpha=0.3, label='95% Confidence Interval')

# 添加图例
plt.legend()

# 添加标题和轴标签
plt.title('Training Rewards over Episodes with 95% Confidence Interval')
plt.xlabel('Episode')
plt.ylabel('Reward')

# 添加虚线网格
plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.7)

# 显示图形
plt.show()