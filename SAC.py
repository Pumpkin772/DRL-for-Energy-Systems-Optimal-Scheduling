import os
import pickle
import gym
import time
import torch
import torch.nn as nn
import numpy as np
import numpy.random as rd
from torch.nn.modules import loss
from random_generator_battery import ESSEnv
import pandas as pd 

from tools import Arguments,get_episode_return,test_one_episode,ReplayBuffer,optimization_base_result,test_ten_episodes
from agent import AgentSAC
from random_generator_battery import ESSEnv

def update_buffer(_trajectory):
    ten_state = torch.as_tensor([item[0] for item in _trajectory], dtype=torch.float32)
    ary_other = torch.as_tensor([item[1] for item in _trajectory])
    ary_other[:, 0] = ary_other[:, 0]   # ten_reward
    ary_other[:, 1] = (1.0 - ary_other[:, 1]) * gamma  # ten_mask = (1.0 - ary_done) * gamma

    buffer.extend_buffer(ten_state, ary_other)

    _steps = ten_state.shape[0]
    _r_exp = ary_other[:, 0].mean()  # other = (reward, mask, action)
    return _steps, _r_exp

if __name__=='__main__':
    args=Arguments()
    reward_record={'episode':[],'steps':[],'mean_episode_reward':[],'unbalance':[],'cost':[]}
    loss_record={'episode':[],'steps':[],'critic_loss':[],'actor_loss':[],'entropy_loss':[]}
    args.visible_gpu='0'
    gpu_id = 0
    all_seeds_reward_record = {}
    for seed in args.random_seed_list:
        # 奖励函数记录
        reward_record = {'episode': [], 'steps': [], 'mean_episode_reward': [], 'unbalance': [], 'cost': []}
        # 损失函数记录
        loss_record = {'episode': [], 'steps': [], 'critic_loss': [], 'actor_loss': [], 'entropy_loss': []}
        args.random_seed = seed
        args.agent=AgentSAC()
        agent_name=f'{args.agent.__class__.__name__}'
        args.agent.cri_target=True
        args.env=ESSEnv()
        args.init_before_training(if_main=True)
        '''init agent and environment'''
        agent=args.agent
        env=args.env
        all_seeds_reward_record[seed] = {'episode': [], 'steps': [], 'mean_episode_reward': [], 'unbalance': [],
                                         'cost': []}
        agent.init(args.net_dim,env.state_space.shape[0],env.action_space.shape[0],args.learning_rate,args.if_per_or_gae,gpu_id)
        '''init replay buffer'''
        buffer = ReplayBuffer(max_len=args.max_memo, state_dim=env.state_space.shape[0],
                              action_dim= env.action_space.shape[0])
        '''start training'''
        cwd=args.cwd
        gamma=args.gamma
        batch_size=args.batch_size# how much data should be used to update net
        target_step=args.target_step#how manysteps of one episode should stop

        repeat_times=args.repeat_times# how many times should update for one batch size data

        soft_update_tau = args.soft_update_tau

        agent.state=env.reset()

        '''collect data and train and update network'''
        num_episode=args.num_episode
        '''here record real unbalance'''

        ##
        # args.train=False
        # args.save_network=False
        # args.test_network=False
        # args.save_test_data=False
        # args.compare_with_pyomo=False
        #
        if args.train:
            collect_data=True
            while collect_data:
                print(f'buffer:{buffer.now_len}')
                with torch.no_grad():
                    trajectory=agent.explore_env(env,target_step)
                    steps,r_exp=update_buffer(trajectory)
                    buffer.update_now_len()
                if buffer.now_len>=10000:
                    collect_data=False
            for i_episode in range(num_episode):
                reward_record['episode'].append(i_episode)
                loss_record['episode'].append(i_episode)
                critic_loss,actor_loss,entropy_loss=agent.update_net(buffer,batch_size,repeat_times,soft_update_tau)
                loss_record['critic_loss'].append(critic_loss)
                loss_record['actor_loss'].append(actor_loss)
                loss_record['entropy_loss'].append(entropy_loss)
                with torch.no_grad():
                    episode_reward,episode_unbalance,episode_cost=get_episode_return(env,agent.act,agent.device)
                    reward_record['mean_episode_reward'].append(episode_reward)
                    reward_record['unbalance'].append(episode_unbalance)
                    reward_record['cost'].append(episode_cost)
                print(f'curren epsiode is {i_episode}, reward:{episode_reward},unbalance:{episode_unbalance},cost:{episode_cost},buffer_length: {buffer.now_len}')
                if i_episode % 10==0:
                # target_step
                    with torch.no_grad():
                        trajectory=agent.explore_env(env,target_step)
                        steps,r_exp=update_buffer(trajectory)
        all_seeds_reward_record[seed] = reward_record

    act_save_path = f'{args.cwd}/actor.pth'
    loss_record_path=f'{args.cwd}/loss_data.pkl'
    reward_record_path=f'{args.cwd}/reward_data.pkl'
    all_seeds_reward_record_path = f'{args.cwd}/all_seeds_reward_record.pkl'
    with open (loss_record_path,'wb') as tf:
        pickle.dump(loss_record,tf)
    with open (reward_record_path,'wb') as tf:
        pickle.dump(reward_record,tf)
    with open(all_seeds_reward_record_path, 'wb') as tf:
        pickle.dump(all_seeds_reward_record, tf)


    if args.save_network:
        torch.save(agent.act.state_dict(),act_save_path)
        print('actor parameters have been saved')
    
    if args.test_network:
        args.cwd=agent_name
        agent.act.load_state_dict(torch.load(act_save_path))
        print('parameters have been reload and test')
        record=test_ten_episodes(env,agent.act,agent.device)
        print(record)
        #eval_data=pd.DataFrame(record['information'])
        #eval_data.columns=['time_step','price','netload','action','real_action','soc','battery','gen1','gen2','gen3','unbalance','operation_cost']
    if args.save_test_data:
        test_data_save_path=f'{args.cwd}/test_data.pkl'
        with open(test_data_save_path,'wb') as tf:
            pickle.dump(record,tf)

    '''compare with pyomo data and results'''
    if args.compare_with_pyomo:
        month=record['init_info'][0][0]
        day=record['init_info'][0][1]
        initial_soc=record['init_info'][0][3]   
        print(initial_soc)     
        base_result=optimization_base_result(env,month,day,initial_soc)
    if args.plot_on:
        from plotDRL import PlotArgs,make_dir,plot_evaluation_information,plot_optimization_result
        plot_args=PlotArgs()
        plot_args.feature_change='2000Episode_100exchange_50penalty'
        args.cwd=agent_name
        plot_dir=make_dir(args.cwd,plot_args.feature_change)
        plot_optimization_result(base_result,plot_dir)
        plot_evaluation_information(args.cwd+'/'+'test_data.pkl',plot_dir)
    '''compare the different cost get from pyomo and SAC'''
    #ration=sum(eval_data['operation_cost'])/sum(base_result['step_cost'])
    #print(sum(eval_data['operation_cost']))
    #print(sum(base_result['step_cost']))
    #print(ration)