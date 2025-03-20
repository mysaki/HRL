import argparse
import os
import datetime
from typing import Tuple
import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from torch.distributions import Distribution, Independent, Normal
from torch.utils.tensorboard import SummaryWriter
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from tianshou.data import Collector,VectorReplayBuffer,Batch,ReplayBuffer,PrioritizedVectorReplayBuffer
from tianshou.env import ShmemVectorEnv
from tianshou.policy import PPOPolicy, RainbowPolicy,DQNPolicy
from tianshou.policy.base import BasePolicy
from tianshou.trainer import OnpolicyTrainer 
from tianshou.utils.net.discrete import NoisyLinear
from tianshou.utils import TensorboardLogger
from Nets.Critic_Net import Critic_Preprocess_Net
from Nets.Actor_Net import Actor_Preprocess_Net
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.continuous import ActorProb, Critic
from Nets.Track_net import Track_Net
import config_dqn as config
from Nets.Star_net import STAR
def get_decision_policy():
    def noisy_linear(x: int, y: int):
        return NoisyLinear(x, y, config.noisy_std)
    # Q_param = V_param = {"hidden_sizes": [64,64]}
    # net=Track_Net(
    #    config.state_shape[0],
    #    config.high_level_action.shape or config.high_level_action.n, 
    #    concat= False,
    #    device = config.device,
    #    use_dueling = True,
    #    get_feature=True,
    #    dueling_param=(Q_param, V_param),
    #    feature_dim=config.high_policy_params['feature_dim']
    # ).to(config.device)
    net = Net(
        state_shape=config.state_shape[0],
        action_shape=config.high_level_action.shape or config.high_level_action.n,
        hidden_sizes=config.high_policy_params["hidden_size"],
        device=config.device,
        softmax=True,
        num_atoms=config.num_atoms,
        dueling_param=({"linear_layer": noisy_linear}, {"linear_layer": noisy_linear}),
    ).to(config.device)
    high_optim = torch.optim.Adam(net.parameters(), lr=config.lr)
    high_level_policy = RainbowPolicy(
        model=net,
        optim=high_optim,
        discount_factor=config.gamma,
        estimation_step=config.high_policy_params["n_step"],
        target_update_freq=config.high_policy_params["target_update_freq"],
        action_space=config.high_level_action,
    )
    return high_level_policy,high_optim