import torch
import os
import numpy as np
import numpy.random as rd
import pandas as pd
import pyomo.environ as pyo
import pyomo.kernel as pmo
from omlt import OmltBlock

from gurobipy import *
from omlt.neuralnet import NetworkDefinition, FullSpaceNNFormulation, ReluBigMFormulation
from omlt.io.onnx import write_onnx_model_with_bounds, load_onnx_neural_network_with_bounds
import tempfile
import torch.onnx
import torch.nn as nn
from copy import deepcopy
from random_generator_battery import ESSEnv


class Actor(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                 nn.Linear(mid_dim, action_dim))

    def forward(self, state):
        return self.net(state).tanh()  # action.tanh()

    def get_action(self, state, action_std):
        action = self.net(state).tanh()
        noise = (torch.randn_like(action) * action_std).clamp(-0.5, 0.5)
        return (action + noise).clamp(-1.0, 1.0)


class ActorSAC(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net_state = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                       nn.Linear(mid_dim, mid_dim), nn.ReLU(), )
        self.net_a_avg = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                       nn.Linear(mid_dim, action_dim))  # the average of action
        self.net_a_std = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                       nn.Linear(mid_dim, action_dim))  # the log_std of action
        self.log_sqrt_2pi = np.log(np.sqrt(2 * np.pi))

    def forward(self, state):
        tmp = self.net_state(state)
        return self.net_a_avg(tmp).tanh()  # action

    def get_action(self, state):
        t_tmp = self.net_state(state)
        a_avg = self.net_a_avg(t_tmp)  # NOTICE! it is a_avg without .tanh()
        a_std = self.net_a_std(t_tmp).clamp(-20, 2).exp()
        return torch.normal(a_avg, a_std).tanh()  # re-parameterize

    def get_action_logprob(self, state):
        t_tmp = self.net_state(state)
        a_avg = self.net_a_avg(t_tmp)  # NOTICE! it needs a_avg.tanh()
        a_std_log = self.net_a_std(t_tmp).clamp(-20, 2)
        a_std = a_std_log.exp()

        noise = torch.randn_like(a_avg, requires_grad=True)
        a_tan = (a_avg + a_std * noise).tanh()  # action.tanh()

        log_prob = a_std_log + self.log_sqrt_2pi + noise.pow(2).__mul__(0.5)  # noise.pow(2) * 0.5
        log_prob = log_prob + (-a_tan.pow(2) + 1.000001).log()  # fix log_prob using the derivative of action.tanh()
        return a_tan, log_prob.sum(1, keepdim=True)


class ActorPPO(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                 nn.Linear(mid_dim, action_dim), )

        # the logarithm (log) of standard deviation (std) of action, it is a trainable parameter
        self.a_std_log = nn.Parameter(torch.zeros((1, action_dim)) - 0.5, requires_grad=True)
        self.sqrt_2pi_log = np.log(np.sqrt(2 * np.pi))

    def forward(self, state):
        return self.net(state).tanh()  # action.tanh()# in this way limit the data output of action

    def get_action(self, state):
        #mean
        a_avg = self.net(state)
        #standard deviation 
        a_std = self.a_std_log.exp()

        noise = torch.randn_like(a_avg)
        action = a_avg + noise * a_std
        return action, noise

    def get_logprob_entropy(self, state, action):
        a_avg = self.net(state)
        a_std = self.a_std_log.exp()

        delta = ((a_avg - action) / a_std).pow(2) * 0.5
        logprob = -(self.a_std_log + self.sqrt_2pi_log + delta).sum(1)  # new_logprob

        dist_entropy = (logprob.exp() * logprob).mean()  # policy entropy
        return logprob, dist_entropy

    def get_old_logprob(self, _action, noise):  # noise = action - a_noise
        delta = noise.pow(2) * 0.5
        return -(self.a_std_log + self.sqrt_2pi_log + delta).sum(1)  # old_logprob


class Critic(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim + action_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                 nn.Linear(mid_dim, 1))

    def forward(self, state, action):
        return self.net(torch.cat((state, action), dim=1))  # q value


class CriticAdv(nn.Module):
    def __init__(self, mid_dim, state_dim, _action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                 nn.Linear(mid_dim, 1))

    def forward(self, state):
        return self.net(state)  # advantage value


class CriticTwin(nn.Module):  # shared parameter
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net_sa = nn.Sequential(nn.Linear(state_dim + action_dim, mid_dim), nn.ReLU(),
                                    nn.Linear(mid_dim, mid_dim), nn.ReLU())  # concat(state, action)
        self.net_q1 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                    nn.Linear(mid_dim, 1))  # q1 value
        self.net_q2 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                    nn.Linear(mid_dim, 1))  # q2 value

    def forward(self, state, action):
        tmp = self.net_sa(torch.cat((state, action), dim=1))
        return self.net_q1(tmp)  # one Q value

    def get_q1_q2(self, state, action):
        tmp = self.net_sa(torch.cat((state, action), dim=1))
        return self.net_q1(tmp), self.net_q2(tmp)  # two Q values

# after adding layer normalization, it doesn't work
class ActorPPO(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim, layer_norm=False):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                 nn.Linear(mid_dim, action_dim), )

        # the logarithm (log) of standard deviation (std) of action, it is a trainable parameter
        self.a_logstd = nn.Parameter(torch.zeros((1, action_dim)) - 0.5, requires_grad=True)
        self.sqrt_2pi_log = np.log(np.sqrt(2 * np.pi))
        if layer_norm:
            self.layer_norm(self.net)

    @staticmethod
    def layer_norm(layer, std=1.0, bias_const=0.0):
        for l in layer:
            if hasattr(l, 'weight'):
                torch.nn.init.orthogonal_(l.weight, std)
                torch.nn.init.constant_(l.bias, bias_const)

    def forward(self, state):
        return self.net(state).tanh()  # action.tanh()

    def get_action(self, state):
        a_avg = self.net(state)  # too big for the action
        a_std = self.a_logstd.exp()

        noise = torch.randn_like(a_avg)
        action = a_avg + noise * a_std
        return action, noise

    def get_logprob_entropy(self, state, action):
        a_avg = self.net(state)
        a_std = self.a_logstd.exp()

        delta = ((a_avg - action) / a_std).pow(2) * 0.5  # delta here is the diverse between the
        logprob = -(self.a_logstd + self.sqrt_2pi_log + delta).sum(1)  # new_logprob

        dist_entropy = (logprob.exp() * logprob).mean()  # policy entropy
        return logprob, dist_entropy

    def get_old_logprob(self, _action, noise):  # noise = action - a_noise
        delta = noise.pow(2) * 0.5
        return -(self.a_logstd + self.sqrt_2pi_log + delta).sum(1)  # old_logprob

class CriticAdv_ppo(nn.Module):
    def __init__(self, mid_dim, state_dim, _action_dim, layer_norm=False):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                 nn.Linear(mid_dim, 1))
        if layer_norm:
            self.layer_norm(self.net, std=1.0)

    @staticmethod
    def layer_norm(layer, std=1.0, bias_const=0.0):
        for l in layer:
            if hasattr(l, 'weight'):
                torch.nn.init.orthogonal_(l.weight, std)
                torch.nn.init.constant_(l.bias, bias_const)

    def forward(self, state):
        return self.net(state)  # Advantage value

class CriticQ(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net_head = nn.Sequential(nn.Linear(state_dim + action_dim, mid_dim), nn.ReLU(),
                                      nn.Linear(mid_dim, mid_dim), nn.ReLU())
        self.net_q1 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                    nn.Linear(mid_dim, 1))  # we get q1 value
        self.net_q2 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                    nn.Linear(mid_dim, 1))  # we get q2 value

    def forward(self, value):
        mid = self.net_head(value)
        return self.net_q1(mid)

    def get_q1_q2(self, value):
        mid = self.net_head(value)
        return self.net_q1(mid), self.net_q2(mid)

class Actor_MIP:
    '''this actor is used to get the best action and Q function, the only input should be batch tensor state, action, and network, while the output should be
    batch tensor max_action, batch tensor max_Q'''
    def __init__(self,scaled_parameters,batch_size,net,state_dim,action_dim,env,constrain_on=True):
        self.batch_size = batch_size
        self.net = net
        self.state_dim = state_dim
        self.action_dim =action_dim
        self.env = env
        self.constrain_on=constrain_on
        self.scaled_parameters=scaled_parameters

    def predict_best_action(self, state):
        state=state.detach().cpu().numpy()
        v1 = torch.zeros((1, self.state_dim+self.action_dim), dtype=torch.float32)
        '''this function is used to get the best action based on current net'''
        model = self.net.to('cpu')
        input_bounds = {}
        lb_state = state
        ub_state = state
        for i in range(self.action_dim + self.state_dim):
            if i < self.state_dim:
                input_bounds[i] = (float(lb_state[0][i]), float(ub_state[0][i])) # 状态的约束
            else:
                input_bounds[i] = (float(-1), float(1)) # 动作的约束

        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            # export neural network to ONNX
            torch.onnx.export(
                model,
                v1, # 一个虚拟输入张量，用于导出模型。这个张量的形状和类型应该与模型期望的输入相匹配
                f, # 模型被导入到文件f中
                input_names=['state_action'],
                output_names=['Q_value'],
                dynamic_axes={
                    'state_action': {0: 'batch_size'},
                    'Q_value': {0: 'batch_size'}
                }
            )
            # write ONNX model and its bounds using OMLT
        write_onnx_model_with_bounds(f.name, None, input_bounds) # 将 ONNX 模型及其输入输出边界信息写入文件的函数，结合nn和优化
        # load the network definition from the ONNX model
        network_definition = load_onnx_neural_network_with_bounds(f.name)
        network_definition.scaled_input_bounds
        # global optimality
        formulation = ReluBigMFormulation(network_definition)
        m = pyo.ConcreteModel()
        m.nn = OmltBlock() # 创建一个 Pyomo 块（Block），这个块包含了与神经网络相关的变量和约束
        m.nn.build_formulation(formulation) # 将 formulation 中定义的变量和约束添加到 OmltBlock 实例 m.nn 中
        '''# we are now building the surrogate model between action and state'''
        # constrain for battery，
        if self.constrain_on:
            m.power_balance_con1 = pyo.Constraint(expr=(
                    (-m.nn.inputs[7] * self.scaled_parameters[0])+\
                    ((m.nn.inputs[8] * self.scaled_parameters[1])+m.nn.inputs[4]*self.scaled_parameters[5]) +\
                    ((m.nn.inputs[9] * self.scaled_parameters[2])+m.nn.inputs[5]*self.scaled_parameters[6]) +\
                    ((m.nn.inputs[10] * self.scaled_parameters[3])+m.nn.inputs[6]*self.scaled_parameters[7])>=\
                    m.nn.inputs[3] *self.scaled_parameters[4]-self.env.grid.exchange_ability))
            m.power_balance_con2 = pyo.Constraint(expr=(
                    (-m.nn.inputs[7] * self.scaled_parameters[0])+\
                    (m.nn.inputs[8] * self.scaled_parameters[1]+m.nn.inputs[4]*self.scaled_parameters[5]) +\
                    (m.nn.inputs[9] * self.scaled_parameters[2]+m.nn.inputs[5]*self.scaled_parameters[6]) +\
                    (m.nn.inputs[10] * self.scaled_parameters[3]+m.nn.inputs[6]*self.scaled_parameters[7])<=\
                    m.nn.inputs[3] *self.scaled_parameters[4]+self.env.grid.exchange_ability))
            # m.state_con3 = pyo.Constraint(expr=(m.nn.inputs[0] >= state[0][0]))
            # m.state_con4 = pyo.Constraint(expr=(m.nn.inputs[0] <= state[0][0]))
            # m.state_con5 = pyo.Constraint(expr=(m.nn.inputs[1] >= state[0][1]))
            # m.state_con6 = pyo.Constraint(expr=(m.nn.inputs[1] <= state[0][1]))
            # m.state_con7 = pyo.Constraint(expr=(m.nn.inputs[2] >= state[0][2]))
            # m.state_con8 = pyo.Constraint(expr=(m.nn.inputs[2] <= state[0][2]))
            # m.state_con9 = pyo.Constraint(expr=(m.nn.inputs[3] >= state[0][3]))
            # m.state_con10 = pyo.Constraint(expr=(m.nn.inputs[3] <= state[0][3]))
            # m.state_con11 = pyo.Constraint(expr=(m.nn.inputs[4] >= state[0][4]))
            # m.state_con12 = pyo.Constraint(expr=(m.nn.inputs[4] <= state[0][4]))
            # m.state_con13 = pyo.Constraint(expr=(m.nn.inputs[5] >= state[0][5]))
            # m.state_con14 = pyo.Constraint(expr=(m.nn.inputs[5] <= state[0][5]))
            # m.state_con15 = pyo.Constraint(expr=(m.nn.inputs[6] >= state[0][6]))
            # m.state_con16 = pyo.Constraint(expr=(m.nn.inputs[6] <= state[0][6]))
        m.obj = pyo.Objective(expr=(m.nn.outputs[0]), sense=pyo.maximize) # [0]降二维至一维

        pyo.SolverFactory('gurobi').solve(m, tee=False)

        best_input = pyo.value(m.nn.inputs[:])

        best_action = (best_input[self.state_dim::])
        return best_action

