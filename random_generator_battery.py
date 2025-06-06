

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

class DG():
    '''simulate a simple diesel generator here'''
    def __init__(self,parameters):
        self.name=parameters.keys()
        self.a_factor=parameters['a']
        self.b_factor=parameters['b']
        self.c_factor=parameters['c']
        self.power_output_max=parameters['power_output_max']
        self.power_output_min=parameters['power_output_min']
        self.ramping_up=parameters['ramping_up']
        self.ramping_down=parameters['ramping_down']
        self.last_step_output=None
    def step(self,action_gen):
        output_change=action_gen*self.ramping_up# constrain the output_change with ramping up boundary
        output=self.current_output+output_change
        if output>0:
            output=max(self.power_output_min,min(self.power_output_max,output))# meet the constrain
        else:
            output=0
        self.current_output=output
    def _get_cost(self,output):
        if output<=0:
            cost=0
        else:
            cost=(self.a_factor*pow(output,2)+self.b_factor*output+self.c_factor)
        return cost
    def reset(self):
        self.current_output=0

class Battery():
    '''simulate a simple battery here'''
    def __init__(self,parameters):
        self.capacity=parameters['capacity']
        self.max_soc=parameters['max_soc']
        self.initial_capacity=parameters['initial_capacity']
        self.min_soc=parameters['min_soc']# 0.2
        self.degradation=parameters['degradation']# degradation cost 1.2
        self.max_charge=parameters['max_charge']# nax charge ability
        self.max_discharge=parameters['max_discharge']
        self.efficiency=parameters['efficiency']
    def step(self,action_battery):
        energy=action_battery*self.max_charge
        updated_capacity=max(self.min_soc,min(self.max_soc,(self.current_capacity*self.capacity+energy)/self.capacity))
        self.energy_change=(updated_capacity-self.current_capacity)*self.capacity# if charge, positive, if discharge, negative
        self.current_capacity=updated_capacity# update capacity to current codition
    def _get_cost(self,energy):# calculate the cost depends on the energy change
        cost=energy**2*self.degradation
        return cost
    def SOC(self):
        return self.current_capacity
    def reset(self):
        self.current_capacity=np.random.uniform(0.2,0.8)
class Grid():
    def __init__(self):

        self.on=True
        if self.on:
            self.exchange_ability=100
        else:
            self.exchange_ability=0
    def _get_cost(self,current_price,energy_exchange):
        return current_price*energy_exchange
    def retrive_past_price(self):
        result=[]
        if self.day<1:
            past_price=self.past_price#
        else:
            past_price=self.price[24*(self.day-1):24*self.day]
            # print(past_price)
        for item in past_price[(self.time-24)::]:
            result.append(item)
        for item in self.price[24*self.day:(24*self.day+self.time)]:
            result.append(item)
        return result

class ESSEnv(gym.Env):
    '''ENV descirption:
    the agent learn to charge with low price and then discharge at high price, in this way, it could get benefits'''

    def __init__(self, **kwargs):
        super(ESSEnv, self).__init__()
        # parameters
        self.data_manager = DataManager()
        self._load_year_data()
        self.episode_length = kwargs.get('episode_length', 24)  # 如果键存在，返回对应的值；如果键不存在，则返回方法的第二个参数作为默认值。
        self.month = None
        self.day = None
        self.TRAIN = True
        self.current_time = None
        self.battery_parameters = kwargs.get('battery_parameters', battery_parameters)
        self.dg_parameters = kwargs.get('dg_parameters', dg_parameters)
        self.penalty_coefficient = 20  # control soft penalty constrain
        self.sell_coefficient = 0.5  # control sell benefits
        # instant the components of the environment
        self.grid = Grid()
        self.battery = Battery(self.battery_parameters)
        self.dg1 = DG(self.dg_parameters['gen_1'])
        self.dg2 = DG(self.dg_parameters['gen_2'])
        self.dg3 = DG(self.dg_parameters['gen_3'])

        # define normalized action space
        # action space here is [output of gen1,outputof gen2, output of gen3, charge/discharge of battery]
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)  # seems here doesn't used
        # state is [time_step,netload,dg_output_last_step]# this time no prive
        self.state_space = spaces.Box(low=0, high=1, shape=(7,), dtype=np.float32)
        # set state related normalization reference
        self.Length_max = 24
        self.Price_max = max(self.data_manager.Prices)
        # self.Netload_max=max(self.data_manager.Electricity_Consumption)-max(self.data_manager.PV_Generation)
        self.Netload_max = max(self.data_manager.Electricity_Consumption)
        self.SOC_max = self.battery.max_soc
        self.DG1_max = self.dg1.power_output_max
        self.DG2_max = self.dg2.power_output_max
        self.DG3_max = self.dg3.power_output_max

    def reset(self):
        '''reset is used for initialize the environment, decide the day of month.'''
        self.month = np.random.randint(1, 13)  # here we choose 12 month

        if self.TRAIN:
            self.day = np.random.randint(1, 21)
        else:
            self.day = np.random.randint(21, Constant.MONTHS_LEN[self.month - 1])
        self.current_time = 0
        self.battery.reset()
        self.dg1.reset()
        self.dg2.reset()
        self.dg3.reset()
        return self._build_state()

    def get_safe_action(self,action):
        electricity_demand = self.data_manager.get_electricity_cons_data(self.month, self.day, self.current_time)
        pv_generation = self.data_manager.get_pv_data(self.month, self.day, self.current_time)
        price = self.data_manager.get_price_data(self.month, self.day, self.current_time) / self.Price_max
        self.electricity_demand = electricity_demand
        self.pv_generation = pv_generation
        self.price = price
        dg1_output = float(self.dg1.current_output)  # 转换为Python float
        dg2_output = float(self.dg2.current_output)
        dg3_output = float(self.dg3.current_output)
        action = action.flatten()  # 转换为一维数组
        N = action.shape[0]
        ramping_up1 = 100
        ramping_up2 = 100
        ramping_up3 = 200
        power_output_max1 = 150
        power_output_max2 = 375
        power_output_max3 = 500
        power_output_min1 = 10
        power_output_min2 = 10
        power_output_min3 = 10
        exchange_ability = 100
        max_charge = 100
        current_capacity = float(self.battery.SOC())
        capacity = 500

        m = pyo.ConcreteModel()
        m.N = pyo.Set(initialize=range(N), ordered=False)
        # 使用Python原生类型初始化参数
        m.dg1 = pyo.Param(default=dg1_output, mutable=False)
        m.dg2 = pyo.Param(default=dg2_output, mutable=False)
        m.dg3 = pyo.Param(default=dg3_output, mutable=False)
        m.ramp1 = pyo.Param(default=ramping_up1, mutable=False)
        m.ramp2 = pyo.Param(default=ramping_up2, mutable=False)
        m.ramp3 = pyo.Param(default=ramping_up3, mutable=False)
        m.power_output_max1 = pyo.Param(default=power_output_max1, mutable=False)
        m.power_output_max2 = pyo.Param(default=power_output_max2, mutable=False)
        m.power_output_max3 = pyo.Param(default=power_output_max3, mutable=False)
        m.power_output_min1 = pyo.Param(default=power_output_min1, mutable=False)
        m.power_output_min2 = pyo.Param(default=power_output_min2, mutable=False)
        m.power_output_min3 = pyo.Param(default=power_output_min3, mutable=False)
        m.exchange_ability = pyo.Param(default=exchange_ability, mutable=False)
        m.max_charge = pyo.Param(default=max_charge, mutable=False)
        m.current_capacity = pyo.Param(default=current_capacity, mutable=False)
        m.capacity = pyo.Param(default=capacity, mutable=False)

        # 正确初始化动作变量
        initial_action = {i: float(action[i]) for i in range(N)}  # 确保标量初始化
        m.action = pyo.Var(m.N, initialize=initial_action, bounds=(-1, 1))
        # m.exchange = pyo.Var(initialize=0.0, bounds=(-1, 1))  # 使用Python float



        m.con11 = pyo.Constraint(expr=(m.action[1]*m.ramp1 + m.dg1) >= m.power_output_min1)
        m.con12 = pyo.Constraint(expr=(m.action[1] * m.ramp1 + m.dg1) <= m.power_output_max1)
        m.con21 = pyo.Constraint(expr=(m.action[2]*m.ramp2 + m.dg2) >= m.power_output_min2)
        m.con22 = pyo.Constraint(expr=(m.action[2] * m.ramp2 + m.dg2) <= m.power_output_max2)
        m.con31 = pyo.Constraint(expr=(m.action[3]*m.ramp3 + m.dg3) >= m.power_output_min3)
        m.con32 = pyo.Constraint(expr=(m.action[3] * m.ramp3 + m.dg3) <= m.power_output_max3)
        m.con_battery1 = pyo.Constraint(expr=(m.max_charge*m.action[0] + m.current_capacity*m.capacity)/m.capacity >= 0.2)
        m.con_battery2 = pyo.Constraint(expr=(m.max_charge * m.action[0] + m.current_capacity*m.capacity) / m.capacity <= 0.8)
        m.con_balance1 = pyo.Constraint(expr=m.action[1]*m.ramp1 + m.dg1 + m.action[2]*m.ramp2 + m.dg2 + m.action[3]*m.ramp3 + m.dg3
                                                + pv_generation - electricity_demand - (m.max_charge*m.action[0])<= m.exchange_ability)
        m.con_balance2 = pyo.Constraint(
            expr=m.action[1] * m.ramp1 + m.dg1 + m.action[2] * m.ramp2 + m.dg2 + m.action[3] * m.ramp3 + m.dg3
                     + pv_generation - electricity_demand - (
                                 m.max_charge * m.action[0]) >= -m.exchange_ability)
        def obj_rule(m):
            return sum((m.action[i]-action[i])**2 for i in m.N)
        m.obj = pyo.Objective(expr=obj_rule, sense=pyo.minimize)
        results = pyo.SolverFactory('gurobi').solve(m, tee=False) # Gurobi求解器在不可行时仍保存最近可行解

        '''here we need to check the printed model for each constrain, are they satisfied for the real part'''
        unbalance = (
                pyo.value(m.action[1]) * m.ramp1 + m.dg1
                + pyo.value(m.action[2]) * m.ramp2 + m.dg2
                + pyo.value(m.action[3]) * m.ramp3 + m.dg3
                + pv_generation
                - electricity_demand
                - (m.max_charge * pyo.value(m.action[0]))
        )


        # print(f"safe当前时刻参数：\n"
        #       f"DG1当前输出：{pyo.value(m.action[1]) * m.ramp1 + m.dg1} | \n"
        #       f"DG2当前输出：{pyo.value(m.action[2]) * m.ramp2 + m.dg2} | \n"
        #       f"DG3当前输出：{pyo.value(m.action[3]) * m.ramp3 + m.dg3} | \n"
        #       f"电池当前SOC：{(pyo.value(m.max_charge)*pyo.value(m.action[0]) + pyo.value(m.current_capacity)*pyo.value(m.capacity))/pyo.value(m.capacity)} | \n"
        #       f"光伏发电：{pv_generation} | 电力需求：{electricity_demand}\n"
        #       f"电池输出：{m.max_charge * pyo.value(m.action[0])}\n"
        #       f"初始动作值：{safe_action}")
        # print(f"safe不平衡度{unbalance}")
        # print(f"obj:{pyo.value(m.obj)}")
        # print(f"unbalance:{unbalance}")
        flag = 0
        if results.solver.termination_condition == pyo.TerminationCondition.infeasible or results.solver.termination_condition == pyo.TerminationCondition.unbounded:
            flag = 1
            print("求解失败")
            if pyo.value(m.action[1]*m.ramp1 + m.dg1 + m.action[2]*m.ramp2 + m.dg2 + m.action[3]*m.ramp3 + m.dg3
                                                + pv_generation - electricity_demand - (m.max_charge*m.action[0])) >= m.exchange_ability:
                # 强制各机组降到最低出力
                m.action[1].value = max((m.power_output_min1 - m.dg1)/m.ramp1, -1.0)
                m.action[2].value = max((m.power_output_min2 - m.dg2)/m.ramp2, -1.0)
                m.action[3].value = max((m.power_output_min3 - m.dg3)/m.ramp3, -1.0)
            elif pyo.value(m.action[1]*m.ramp1 + m.dg1 + m.action[2]*m.ramp2 + m.dg2 + m.action[3]*m.ramp3 + m.dg3
                                                + pv_generation - electricity_demand - (m.max_charge*m.action[0])) <= -m.exchange_ability:
                # 强制各机组升到最高出力
                m.action[1].value = min((m.power_output_max1 - m.dg1) / m.ramp1, 1.0)
                m.action[2].value = min((m.power_output_max2 - m.dg2) / m.ramp2, 1.0)
                m.action[3].value = min((m.power_output_max3 - m.dg3) / m.ramp3, 1.0)
            print(f"dg1_origin:{pyo.value(m.dg1)}")
            print(f"dg2_origin:{pyo.value(m.dg2)}")
            print(f"dg3_origin:{pyo.value(m.dg3)}")
            print(f"dg1:{pyo.value(m.action[1]*m.ramp1 + m.dg1)}")
            print(f"dg2:{pyo.value(m.action[2]*m.ramp2 + m.dg2)}")
            print(f"dg3:{pyo.value(m.action[3]*m.ramp3 + m.dg3)}")
            print(f"battery:{pyo.value((m.max_charge*m.action[0] + m.current_capacity*m.capacity)/m.capacity)}")
            print(f"battery_power:{pyo.value(m.max_charge*m.action[0])}")
            print(f"pv:{pv_generation}")
            print(f"demand:{electricity_demand}")
            print(f"dg_action:f{m.action[1].value},{m.action[2].value},{m.action[3].value}")

        # 提取安全动作
        safe_action = [pyo.value(m.action[i]) for i in m.N]
        return np.array(safe_action), flag


    def _build_state(self):
        # we put all original information into state and then transfer it into normalized state
        soc = self.battery.SOC() / self.SOC_max
        dg1_output = self.dg1.current_output / self.DG1_max
        dg2_output = self.dg2.current_output / self.DG2_max
        dg3_output = self.dg3.current_output / self.DG3_max
        time_step = self.current_time / (self.Length_max - 1) # current_time为当前小时数
        electricity_demand = self.data_manager.get_electricity_cons_data(self.month, self.day, self.current_time)
        pv_generation = self.data_manager.get_pv_data(self.month, self.day, self.current_time)
        price = self.data_manager.get_price_data(self.month, self.day, self.current_time) / self.Price_max
        net_load = (electricity_demand - pv_generation) / self.Netload_max
        obs = np.concatenate((np.float32(time_step), np.float32(price), np.float32(soc), np.float32(net_load),
                              np.float32(dg1_output), np.float32(dg2_output), np.float32(dg3_output)), axis=None)
        return obs

    def step(self, action):  # state transition here current_obs--take_action--get reward-- get_finish--next_obs
        ## here we want to put take action into each components
        current_obs = self._build_state()
        self.battery.step(action[0])  # here execute the state-transition part, battery.current_capacity also changed
        self.dg1.step(action[1])
        self.dg2.step(action[2])
        self.dg3.step(action[3])
        current_output = np.array((self.dg1.current_output, self.dg2.current_output, self.dg3.current_output,
                                   -self.battery.energy_change))  # truely corresonding to the result
        self.current_output = current_output
        actual_production = sum(current_output)
        # transfer to normal_state
        netload = current_obs[3] * self.Netload_max
        price = current_obs[1] * self.Price_max

        unbalance = actual_production - netload
        electricity_demand = self.data_manager.get_electricity_cons_data(self.month, self.day, self.current_time)
        pv_generation = self.data_manager.get_pv_data(self.month, self.day, self.current_time)
        # print(f"env当前时刻参数：\n"
        #       f"DG1当前输出：{self.dg1.current_output} | \n"
        #       f"DG2当前输出：{self.dg2.current_output} | \n"
        #       f"DG3当前输出：{self.dg3.current_output} | \n"
        #       f"电池当前SOC：{self.battery.current_capacity} |\n"
        #       f"光伏发电：{pv_generation} | 电力需求：{electricity_demand}\n"
        #       f"电池输出：{self.battery.energy_change}\n"
        #       f"初始动作值：{action}")
        reward = 0
        excess_penalty = 0
        deficient_penalty = 0
        sell_benefit = 0
        buy_cost = 0
        self.excess = 0
        self.shedding = 0
        # logic here is: if unbalance >0 then it is production excess, so the excessed output should sold to power grid to get benefits
        # print(f"env不平衡度_env{unbalance}")
        if unbalance >= 0:  # it is now in excess condition
            if unbalance <= self.grid.exchange_ability:
                sell_benefit = self.grid._get_cost(price,
                                                   unbalance) * self.sell_coefficient  # sell money to grid is little [0.029,0.1]
            else:
                sell_benefit = self.grid._get_cost(price, self.grid.exchange_ability) * self.sell_coefficient
                # real unbalance that even grid could not meet
                self.excess = unbalance - self.grid.exchange_ability
                excess_penalty = self.excess * self.penalty_coefficient
        else:  # unbalance <0, its load shedding model, in this case, deficient penalty is used
            if abs(unbalance) <= self.grid.exchange_ability:
                buy_cost = self.grid._get_cost(price, abs(unbalance))
            else:
                buy_cost = self.grid._get_cost(price, self.grid.exchange_ability)
                self.shedding = abs(unbalance) - self.grid.exchange_ability
                deficient_penalty = self.shedding * self.penalty_coefficient
        battery_cost = self.battery._get_cost(self.battery.energy_change)  # we set it as 0 this time
        dg1_cost = self.dg1._get_cost(self.dg1.current_output)
        dg2_cost = self.dg2._get_cost(self.dg2.current_output)
        dg3_cost = self.dg3._get_cost(self.dg3.current_output)

        reward = -(battery_cost + dg1_cost + dg2_cost + dg3_cost + excess_penalty +
                   deficient_penalty - sell_benefit + buy_cost) / 2e3

        self.operation_cost = battery_cost + dg1_cost + dg2_cost + dg3_cost + buy_cost - sell_benefit + (
                self.shedding + self.excess) * self.penalty_coefficient

        self.unbalance = unbalance
        self.real_unbalance = self.shedding + self.excess
        '''here we also need to store the final step outputs for the final steps including, soc, output of units for seeing the final states'''
        final_step_outputs = [self.dg1.current_output, self.dg2.current_output, self.dg3.current_output,
                              self.battery.current_capacity]
        self.current_time += 1
        finish = (self.current_time == self.episode_length)
        if finish:
            self.final_step_outputs = final_step_outputs
            self.current_time = 0
            next_obs = self.reset()

        else:
            next_obs = self._build_state()
        return current_obs, next_obs, float(reward), finish

    def render(self, current_obs, next_obs, reward, finish):
        print('day={},hour={:2d}, state={}, next_state={}, reward={:.4f}, terminal={}\n'.format(self.day,
                                                                                                self.current_time,
                                                                                                current_obs, next_obs,
                                                                                                reward, finish))

    def _load_year_data(self):
        '''this private function is used to load the electricity consumption, pv generation and related prices in a year as
        a one hour resolution, with the cooperation of class DataProcesser and then all these data are stored in data processor'''
        pv_df = pd.read_csv('data/PV.csv', sep=';')
        # hourly price data for a year
        price_df = pd.read_csv('data/Prices.csv', sep=';')
        # mins electricity consumption data for a year
        electricity_df = pd.read_csv('data/H4.csv', sep=';')
        pv_data = pv_df['P_PV_'].apply(lambda x: x.replace(',', '.')).to_numpy(dtype=float)
        price = price_df['Price'].apply(lambda x: x.replace(',', '.')).to_numpy(dtype=float)
        electricity = electricity_df['Power'].apply(lambda x: x.replace(',', '.')).to_numpy(dtype=float)
        # netload=electricity-pv_data
        '''we carefully redesign the magnitude for price and amount of generation as well as demand'''
        for element in pv_data:
            self.data_manager.add_pv_element(element * 100)
        for element in price:
            element /= 10
            if element <= 0.5:
                element = 0.5
            self.data_manager.add_price_element(element)
        for i in range(0, electricity.shape[0], 60):
            element = electricity[i:i + 60]
            self.data_manager.add_electricity_element(sum(element) * 300)
    ## test environment
if __name__ == '__main__':
    env=ESSEnv()
    env.TRAIN=False
    rewards=[]

    current_obs=env.reset()
    tem_action=[0.1,0.1,0.1,0.1]
    for _ in range (144):
        print(f'current month is {env.month}, current day is {env.day}, current time is {env.current_time}')
        current_obs,next_obs,reward,finish=env.step(tem_action)
        env.render(current_obs,next_obs,reward,finish)
        current_obs=next_obs
        rewards.append(reward)
