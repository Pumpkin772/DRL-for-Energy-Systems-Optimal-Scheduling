import random
import numpy as np
import pandas as pd
import gym
from gym import spaces
import pyomo.environ as pyo
import os

from Parameters import battery_parameters,dg_parameters
os.environ['GUROBI_HOME'] = 'F:\\gurobi\\win64'
class Constant:
	MONTHS_LEN = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
	MAX_STEP_HOURS = 24 * 30
class DataManager():
    def __init__(self) -> None:
        self.PV_Generation=[]
        self.Prices=[]
        self.Electricity_Consumption=[]
    def add_pv_element(self,element):self.PV_Generation.append(element)
    def add_price_element(self,element):self.Prices.append(element)
    def add_electricity_element(self,element):self.Electricity_Consumption.append(element)

    # get current time data based on given month day, and day_time
    def get_pv_data(self,month,day,day_time):return self.PV_Generation[(sum(Constant.MONTHS_LEN[:month-1])+day-1)*24+day_time]
    def get_price_data(self,month,day,day_time):return self.Prices[(sum(Constant.MONTHS_LEN[:month-1])+day-1)*24+day_time]
    def get_electricity_cons_data(self,month,day,day_time):return self.Electricity_Consumption[(sum(Constant.MONTHS_LEN[:month-1])+day-1)*24+day_time]
    # get series data for one episode 得到一天的数据
    def get_series_pv_data(self,month,day): return self.PV_Generation[(sum(Constant.MONTHS_LEN[:month-1])+day-1)*24:(sum(Constant.MONTHS_LEN[:month-1])+day-1)*24+24]
    def get_series_price_data(self,month,day):return self.Prices[(sum(Constant.MONTHS_LEN[:month-1])+day-1)*24:(sum(Constant.MONTHS_LEN[:month-1])+day-1)*24+24]
    def get_series_electricity_cons_data(self,month,day):return self.Electricity_Consumption[(sum(Constant.MONTHS_LEN[:month-1])+day-1)*24:(sum(Constant.MONTHS_LEN[:month-1])+day-1)*24+24]

data_manager = DataManager()
pv_df = pd.read_csv('data/PV.csv', sep=';')
# hourly price data for a year
price_df = pd.read_csv('data/Prices.csv', sep=';')
# mins electricity consumption data for a year
electricity_df = pd.read_csv('data/H4.csv', sep=';')
pv_data = pv_df['P_PV_'].apply(lambda x: x.replace(',', '.')).to_numpy(dtype=float)
price = price_df['Price'].apply(lambda x: x.replace(',', '.')).to_numpy(dtype=float)
electricity = electricity_df['Power'].apply(lambda x: x.replace(',', '.')).to_numpy(dtype=float)
print(electricity_df)
# netload=electricity-pv_data
'''we carefully redesign the magnitude for price and amount of generation as well as demand'''
for element in pv_data:
    data_manager.add_pv_element(element * 100)
for element in price:
    element /= 10
    if element <= 0.5:
        element = 0.5
    data_manager.add_price_element(element)
for i in range(0, electricity.shape[0], 60):
    element = electricity[i:i + 60]
    data_manager.add_electricity_element(sum(element) * 300)

month = np.random.randint(1, 13)
day = np.random.randint(1, 21)
current_time = 0
electricity_demand = data_manager.get_electricity_cons_data(month, day, current_time)
print(electricity_demand)