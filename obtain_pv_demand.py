import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
pv_df=pd.read_csv('data/PV.csv',sep=';')
#hourly price data for a year
price_df=pd.read_csv('data/Prices.csv',sep=';')
# mins electricity consumption data for a year
electricity_df=pd.read_csv('data/H4.csv',sep=';')

pv_data=pv_df['P_PV_'].apply(lambda x: x.replace(',','.')).to_numpy(dtype=float)
price=price_df['Price'].apply(lambda x:x.replace(',','.')).to_numpy(dtype=float)
electricity=electricity_df['Power'].apply(lambda x:x.replace(',','.')).to_numpy(dtype=float)

pv_df['Date'] = pd.to_datetime(pv_df.iloc[:, 0], format='%d.%m.%y %H:%M')

# 筛选12月1日至次年1月1日之间的数据
# 定义日期范围
start_date_str = '01.11.16 00:00'
end_date_str = '31.12.16 23:00'
start_date = pd.to_datetime(start_date_str, format='%d.%m.%y %H:%M')
end_date = pd.to_datetime(end_date_str, format='%d.%m.%y %H:%M') + pd.Timedelta(days=1)  # 加1天，包含1月1日当天

# 筛选数据
filtered_df = pv_df[(pv_df['Date'] >= start_date) & (pv_df['Date'] <= end_date)]

# 查看筛选后的数据
print(filtered_df)

df = filtered_df
df.set_index('Date', inplace=True)

# 清洗P_PV_列，替换逗号并转换为浮点数
df['P_PV_'] = df['P_PV_'].replace(',', '.', regex=True).astype(float)
df['P_PV_'] *= 1000

df['Hour'] = df.index.hour

# 根据小时信息分组，并计算每组的均值和方差
grouped_mean = df.groupby('Hour')['P_PV_'].mean()
grouped_var = df.groupby('Hour')['P_PV_'].var(ddof=1)  # 设置ddof=1以计算样本方差

# 打印结果
print("Hourly Mean:")
print(grouped_mean)
print("\nHourly Variance:")
print(grouped_var)
print("\n")
fig, ax = plt.subplots()

# 绘制均值的条形图
hours = np.arange(1, 25)  # 小时从1到24
mean_demand = grouped_mean.values

# 绘制方差作为均值的填充效果
# 计算每个小时的标准差（方差的平方根）
std_dev_demand = grouped_var.apply(np.sqrt)
ax.plot(hours,mean_demand,drawstyle='steps-mid',label='Price',color='pink')
ax.fill_between(hours, mean_demand - std_dev_demand, mean_demand + std_dev_demand, color='lightpink', alpha=0.3, step='mid', label='Variance')
#ax.fill_between(hours, mean_demand - std_dev_demand, mean_demand + std_dev_demand, color='lightblue', alpha=0.3, label='Variance Demand')

# 设置图例
ax.legend()

# 设置坐标轴标签
ax.set_xlabel('Hour of the Day')
ax.set_ylabel('Mean Value and Variance')

# 设置x轴的刻度标签
ax.set_xticks(hours)
ax.set_xticklabels(hours)

# 设置标题
ax.set_title('Hourly Mean and Variance')

# 显示图形
plt.show()