# 融合经验共享的分层跟踪模型项目
## 1. 代码结构
### 1.1 config开头的文件即为配置文件，记录各种参数。
- config.py是训练能力智能体所使用的参数
- config_dqn.py是训练离散型决策智能体所使用的参数（即直接输出要选择的能力编号）
- config_ppo.py是训练连续性决策智能体所使用的参数（输出各项能力对应的智能体动作，此处为两类速度，的加权参数）
### 1.2 get开头的文件为对应的模型结构文件。
- get_ability_policy.py定义了能力智能体的结构
- get_decision_policy.py定义了决策智能体的结构
- get_test_hierarchical_policy.py定义了在测试分层跟踪模型时所使用的不同模型的结构
- get_test_nav.py定义了在测试安全能力时所使用的不同模型的结构
- get_test_track.py定义了在测试追踪能力时所使用的不同模型的结构
### 1.3 test开头的文件为不同测试任务的入口文件
- test_hierarchical.py是分层模型的测试入口
- test_nav.py是安全技能的测试入口
- test_track.py是追踪技能的测试入口
### 1.4 train开头的文件为不同训练任务的入口文件
- train_KURL_ppo.py是使用PPO算法训练KURL模型
- train_KURL_ppo_join.py是使用PPO算法训练KURL模型的同时使用经验共享
- train_KURL_sac.py是使用SAC算法训练KURL模型
- train_convNet_LSTM_ppo.py是使用PPO算法训练convNet_LSTM模型
- train_convNet_LSTM_sac.py是使用SAC算法训练convNet_LSTM模型
- train_hierarchical_dqn.py是使用DQN算法训练离散型决策智能体
- train_hierarchical_ppo.py是使用PPO算法训练连续型决策智能体
- train_join_ppo.py是使用PPO算法采用经验共享进行模型训练
- tran_nav_d3qn.py是使用Dueling DQN算法在导航任务下训练安全能力
- tran_nav_d3qn_image.py是以RGB图像作为状态空间，使用Dueling DQN算法在导航任务下训练安全能力
- tran_nav_ppo.py是使用PPO算法在导航任务下训练安全能力
- tran_nav_sac.py是使用SAC算法在导航任务下训练安全能力
- train_track_d3qn.py是使用Dueling DQN算法在跟踪任务下训练追踪能力
- train_track_d3qn_image.py是以RGB图像作为状态空间，使用Dueling DQN算法在跟踪任务下训练追踪能力
- train_track_ppo.py是使用PPO算法在跟踪任务下训练追踪能力
- train_track_rgb_ppo.py是以RGB图像作为状态空间，使用PPO算法在跟踪任务下训练追踪能力
- train_track_nav_ppo.py是使用PPO算法通过直接的样本共享方式进行模型训练
- train_track_nav_ppo_image.py是以RGB图像作为状态空间，使用PPO算法通过直接的样本共享方式进行模型训练
### 1.5 辅助文件
- plot开头的文件，可以根据记录的回合数据绘制轨迹图等图像
- anlalyze结尾的文件，可以绘制对应任务的测试结果分析柱状图

### 1.6 Nets文件夹
包含不同的神经网络结构实现，RNN即循环神经网络、Star表示应用了本项目所设计的经验共享结构、attention对应了注意力机制

### 1.7 Seg文件夹
包含了Nested Unet模型的实现即状态空间压缩方法的流程（Get_super.py）

### 1.8 Bacteria_APF文件夹
一类基于人工势场法的导航算法

### 1.9 detect_module文件夹
使用一个小网络实现判断Turtlebot是否在图像中出现的功能



