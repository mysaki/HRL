#!/usr/bin/env python
# -*- coding: UTF-8 -*-

""" parameters setting """
import torch
import numpy as np
from gymnasium import spaces
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
task="Hierarchical-v1"
# resume_model = "hierarchical_track_dqn_12_6_14_19"
resume_model = "hierarchical_track_dqn_3_11_11_38"
resume = False
reward_threshold=4900000
seed=1   
buffer_size=20000
lr=1e-3
gamma=0.995
epoch=1000
step_per_epoch=20000
episode_per_collect=16
repeat_per_collect=2
step_per_collect = 3000
eps_test = 0.05
eps_train =0.3
buffer_alpha = 0.6
beta = 0.4
batch_size=512
hidden_sizes=[64, 64]
headless = True if resume == False else False
training_num=10
test_num=2
num_atoms = 51
logdir="log"
render=0.0
noisy_std = 0.1
device = "cuda" if torch.cuda.is_available() else "cpu"
save_interval = 4
update_per_step = 0.01
high_level_action = spaces.Discrete(2)
# high_level_action = spaces.Box(
#     low=np.array([0, 0]), high=np.array([1, 1]), dtype=np.float32
# )
low_action_space = spaces.Box(
                        low=np.array([-1, -1]), high=np.array([1, 1]),  dtype=np.float32
                        )
observation_space = spaces.Box(
                        low=-10, high=10, shape=(34,), dtype=np.float32
                    )
state_shape = observation_space.shape or observation_space.n
action_shape = low_action_space.shape
low_policy_params={
    "discount_factor":0.995,
    "max_grad_norm":0.5,
    "eps_clip":0.2,
    "vf_coef":0.25,
    "ent_coef":0.0,
    "reward_normalization":1,
    "advantage_normalization":1,
    "recompute_advantage":0,
    "dual_clip":None,
    "value_clip":1,
    "gae_lambda":0.95,
    'feature_dim':256,
    'hidden_size':[128],

}
# high_policy_params={
#     "discount_factor":0.995,
#     "max_grad_norm":0.5,
#     "eps_clip":0.2,
#     "vf_coef":0.25,
#     "ent_coef":0.0,
#     "reward_normalization":1,
#     "advantage_normalization":1,
#     "recompute_advantage":0,
#     "dual_clip":None,
#     "value_clip":1,
#     "gae_lambda":0.95,
#     'feature_dim':256,
#     'hidden_size':[128],

# }
high_policy_params = {
    "n_step": 3,
    "target_update_freq": 10,
    "hidden_size": [256,256,256,256,256],
    "feature_dim": 256,
}

restore = True
valid_actions = ['forward', 'backward', 'right_turn', 'left_turn', 'stop']
speed = 0.4 # rad/s (pioneer 3dx: 5.6 rad/s: ~ 0.56m/s)  # similar to human's normal speed
robot_speed=1.2
wait_response = False # True: Synchronous response(too much delay)
# 动作空间
valid_actions_dict = {valid_actions[0]: np.array([speed, 0]),
                      valid_actions[1]: np.array([-speed, 0]),
                      valid_actions[2]: np.array([speed, -robot_speed]),
                      valid_actions[3]: np.array([speed, robot_speed]),
                      valid_actions[4]: np.array([0, 0])}


# network
update_freq = 20  # How often to perform a training step.
gamma = .99 # Discount factor on the target Q-values
startE = 0.5  # Starting chance of random action
endE = 0.1  # Final chance of random action
path = "./trainedModel"   # The path to save our model to.
annealing_steps = 10000.  # How many steps of training to reduce startE to endE.
num_episodes = 100  # How many episodes of game environment to train network with.,./
pre_train_steps = 1000 # How many steps of random actions before training begins.
max_epLength = 100         # The max allowed length of our episode.
tau = 0.001               # Rate to update target network toward primary network
replay_memory = 50000
aciton = 'discrete'
max_x_vel=0.9
max_y_vel=1.46
chassis_radius=0.07
delta_v=1.5
max_count=2000
time_step =0.05
safe_distance=0.51
best_angle=0
best_distance=0.5
min_distance=0.3
max_distance=4
maxDetectDistance=4
minDetectDistance=0.1
max_angle=30
wheel_radius=0.036
random_action_thre=0.5
decision_interval = 3
ego_safe_distance= 0.4
