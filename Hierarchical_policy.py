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
class HierarchicalPolicy(BasePolicy):
    def __init__(self, high_level_policy, low_level_policy,**kwargs):
        super().__init__(
            action_scaling="True",
            action_bound_method="clip",
            **kwargs)
        self.high_level_policy = high_level_policy
        self.low_level_policy = low_level_policy
        self.action_space = low_level_policy.action_space
        self.count_safe = 0
        self.count_track = 0
    def forward(self, batch, state=None):
        # 高层策略决定方向目标
        high_level_batch = self.high_level_policy.forward(batch)
        high_level_action = high_level_batch['act']
        # print("action:",high_level_action)
        high_level_action = self.high_level_policy.exploration_noise(high_level_action, batch)
        high_level_action = [
            [0,1] if action == 0 else
            [1,0] if action == 1 else
            [int(bit) for bit in format(action,'b')]
            for action in high_level_action]
        if high_level_action[0] == [0,1]:
            self.count_safe+=1
        elif high_level_action[0] == [1,0]:
            self.count_track+=1
        # if high_level_action == 0 :
        #     high_level_action =[0,0]
        # elif high_level_action == 1:
        #     high_level_action = [0,1]
        # elif high_level_action == 2:
        #     high_level_action = [1,0]
        # elif high_level_action == 3:
        #     high_level_action = [0,1]
        # 假设高层动作表示不同的技能
        # 将高层动作嵌入到低层策略的输入中
        origin_obs = torch.from_numpy(batch.obs).to(config.device)
        modified_obs = torch.cat([origin_obs[:,:2], torch.tensor(high_level_action).to(config.device),origin_obs[:,2:]], dim=1)
        # print(modified_obs)
        low_level_batch = batch.copy()
        low_level_batch['obs']=modified_obs
        # 低层策略根据高层的方向执行具体的动作
        low_level_batch= self.low_level_policy.forward(Batch(low_level_batch))
        low_level_batch['low_level_obs'] = modified_obs
        low_level_batch['high_level_act'] = high_level_batch['act']
        low_level_batch['obs']=batch.obs
        return Batch(low_level_batch)

    def learn(self, batch, **kwargs):
        high_level_batch = batch.copy()
        # high_level_batch['obs'] = batch['low_level_obs']
        high_level_batch['act'] = batch['high_level_act']
        high_level_loss = self.high_level_policy.learn(Batch(high_level_batch), **kwargs)
        # low_level_loss = self.low_level_policy.learn(batch, **kwargs)
        
        return high_level_loss#, low_level_loss
    
    def update(self, sample_size: int, buffer: Optional[ReplayBuffer],
               **kwargs: Any) -> Dict[str, Any]:
        high_level_loss = self.high_level_policy.update(sample_size, buffer, level='high')
        #low_level_loss = self.low_level_policy.update(sample_size, buffer, level='low')
        # loss = {}
        # for key in high_level_loss.keys():
        #     temp = []
        #     for i in range(len(high_level_loss[key])):
        #         temp.append(high_level_loss[key][i])#+low_level_loss[key][i]
        #     loss[key] = temp
        return high_level_loss

    def map_action(self, act: Union[Batch, np.ndarray]) -> Union[Batch, np.ndarray]:
        """Map raw network output to action range in gym's env.action_space.

        This function is called in :meth:`~tianshou.data.Collector.collect` and only
        affects action sending to env. Remapped action will not be stored in buffer
        and thus can be viewed as a part of env (a black box action transformation).

        Action mapping includes 2 standard procedures: bounding and scaling. Bounding
        procedure expects original action range is (-inf, inf) and maps it to [-1, 1],
        while scaling procedure expects original action range is (-1, 1) and maps it
        to [action_space.low, action_space.high]. Bounding procedure is applied first.

        :param act: a data batch or numpy.ndarray which is the action taken by
            policy.forward.

        :return: action in the same form of input "act" but remap to the target action
            space.
        """

        # print("raw_act",act)
        if isinstance(self.action_space, gym.spaces.Box) and \
                isinstance(act, np.ndarray):
            # currently this action mapping only supports np.ndarray action
            if self.action_bound_method == "clip":
                act = np.clip(act, -1.0, 1.0)
            elif self.action_bound_method == "tanh":
                act = np.tanh(act)
            if self.action_scaling:
                assert np.min(act) >= -1.0 and np.max(act) <= 1.0, \
                    "action scaling only accepts raw action range = [-1, 1]"
                low, high = self.action_space.low, self.action_space.high
                act = low + (high - low) * (act + 1.0) / 2.0  # type: ignore
        if isinstance(act, np.ndarray):
            act = torch.from_numpy(act)
        # act= torch.where(act[:, 0] < act[:, 1], torch.tensor(1), torch.tensor(0))
        # act.unsqueeze(1)
        # act = act.cpu().numpy()
        # print("final_act",act)
        return act