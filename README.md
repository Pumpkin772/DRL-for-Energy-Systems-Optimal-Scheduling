
# Abstract 
* Taking advantage of their data-driven and model-free features, Deep Reinforcement Learning (DRL) algorithms have the potential to deal with the increasing level of uncertainty due to the introduction of renewable-based generation. To deal simultaneously with the energy systems' operational cost and technical constraints (e.g, generation-demand power balance) DRL algorithms must consider a trade-off when designing the reward function. This trade-off introduces extra hyperparameters that impact the DRL algorithms' performance and capability of providing feasible solutions. In this paper, a performance comparison of different DRL algorithms, including DDPG, TD3, SAC, and PPO, are presented. We aim to provide a fair comparison of these DRL algorithms for energy systems optimal scheduling problems. Results show DRL algorithms' capability of providing in real-time good-quality solutions, even in unseen operational scenarios, when compared with a mathematical programming model of the energy system optimal scheduling problem. Nevertheless, in the case of large peak consumption, these algorithms failed to provide feasible solutions, which can impede their practical implementation.
# Organization
* 文件夹 "Data" -- 处理过的历史数据.
* 脚本 "agent" and "net"-- 通用神经网络和智能体.
* 脚本 "DDPG","SAC","TD3" and "PPO"-- 训练、测试和绘图主函数的集成.
* 脚本 "tools"-- 主函数所需的一般功能 
* 脚本 "random_generator_battery" -- 能源系统环境
* 安装所有软件包后，运行脚本（如 DDPG.py）。请查看代码结构。
# Dependencies
This code requires installation of the following libraries: ```PYOMO```,```pandas 1.1.4```, ```numpy 1.20.1```, ```matplotlib 3.3.4```, ```pytorch 1.11.0```,  ```math```, you can find more information [at this page](https://ieeexplore.ieee.org/document/9960642).
# Recommended citation
A preprint is available, and you can check this paper for more details  [Link of the paper](https://ieeexplore.ieee.org/document/9960642).
* Paper authors: Hou Shengren, Edgar Mauricio Salazar, Pedro P. Vergara, Peter Palensky
* Accepted for publication at IEEE PES ISGT 2022
* If you use (parts of) this code, please cite the preprint or published paper
## Additional Information 
* Sorry some people reported the PPO not works on my environment. I guess I uploaded the wrong version code. I will fix it when I am free. Right now, please use DDPG and TD3 first. If you have any other questions on code, please submit an issue. 
