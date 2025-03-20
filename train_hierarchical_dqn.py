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
from get_ability_policy import *
from get_decision_policy import *
from Hierarchical_policy import HierarchicalPolicy
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def test_ppo() -> None:
    # model
    low_level_policy = get_low_policy()
    high_level_policy,high_optim = get_decision_policy()
    env = gym.make(config.task, headless=config.headless,mode=config.resume,ability_agent =low_level_policy)
    config.state_shape = env.observation_space.shape or env.observation_space.n
    config.action_shape = env.action_space.shape
    # seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)


    # 分层策略控制器
    hierarchical_policy = HierarchicalPolicy(high_level_policy, low_level_policy)
    # log
    time = datetime.datetime.now()  # 获取当前时间
    log_path = os.path.join(config.logdir, config.task,"hierarchical_track_dqn_{}_{}_{}_{}".format(
            time.month, time.day, time.hour, time.minute, time.second))
    if not config.resume:

        writer = SummaryWriter(log_path)
        logger = TensorboardLogger(writer, save_interval=config.save_interval)

    def save_best_fn(policy: BasePolicy,tag = 'reset') -> None:
        torch.save(
            {
            'high_level_policy_state_dict': high_level_policy.state_dict(),
            # 'low_level_policy_state_dict': low_level_policy.state_dict(),
            # 'low_level_optim':low_level_optim.state_dict(),
            'high_level_optim':high_optim.state_dict(),
            },
            os.path.join(log_path, "policy_{}.pth".format(tag))
            )
        print("Model Saved!")

    def stop_fn(mean_rewards: float) -> bool:
        return mean_rewards >= config.reward_threshold

    def train_fn(epoch, env_step):
        # eps annnealing, just a demo
        if env_step <= 100000:
            high_level_policy.set_eps(config.eps_train)
        elif env_step <= 500000:
            eps = config.eps_train - (env_step - 100000) / 400000 * (0.9 * config.eps_train)
            high_level_policy.set_eps(eps)
        else:
            # policy.set_eps(0.1 * args.eps_train)
            high_level_policy.set_eps(0.1)
    def test_fn(epoch, env_step):
        high_level_policy.set_eps(config.eps_test)
    def save_checkpoint_fn(epoch: int, env_step: int, gradient_step: int) -> str:
        # see also: https://pytorch.org/tutorials/beginner/saving_loading_models.html
        ckpt_path = os.path.join(log_path, "checkpoint.pth")
        # Example: saving by epoch num
        # ckpt_path = os.path.join(log_path, f"checkpoint_{epoch}.pth")
        torch.save(
            {
            'high_level_policy_state_dict': high_level_policy.state_dict(),
            # 'low_level_policy_state_dict': low_level_policy.state_dict(),
            # 'low_level_optim':low_level_optim.state_dict(),
            'high_level_optim':high_optim.state_dict(),
            },
            ckpt_path,
        )
        print("Model Saved!")
        return ckpt_path
    def create_env(env_name, headless, mode,ability_agent):
        return gym.make(env_name, headless=headless, mode=mode,ability_agent=ability_agent)
    if config.resume:
        resume_path = os.path.join('log', config.task,config.resume_model)
        # load from existing checkpoint
        print(f"Loading high level agent under {resume_path}")
        # ckpt_path = os.path.join(resume_path, "policy_reset.pth")
        ckpt_path = os.path.join(resume_path,"Track_test.pth")
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path)
            # high_level_policy.load_state_dict(checkpoint['high_level_policy_state_dict'])
            hierarchical_policy.high_level_policy.load_state_dict(checkpoint,strict = False)
            # low_level_policy.load_state_dict(checkpoint['low_level_policy_state_dict'])
            # low_level_optim.load_state_dict(checkpoint["low_level_optim"])
            # high_optim.load_state_dict(checkpoint["high_level_optim"])
            print("Successfully restore  high level policy and optim.")
        else:
            print("Fail to restore policy and optim.")
        hierarchical_policy.eval()
        collector = Collector(hierarchical_policy, 
                              env,
                              log_path=log_path,
                              label='try'
                              )
        result = collector.collect(render=0.1,n_episode=100)
        # 将字典转换为DataFrame
        df = pd.DataFrame(result)
        # 保存为Excel文件
        df.to_excel(
            "result_hierarchical_dqn_{}_{}_{}_{}_{}.xlsx".format(
                time.month, time.day, time.hour, time.minute, time.second
            ),
            index=False,
        )
        print(result)
    else:
        envs = []
        for i in range(0,config.training_num):
            envs.append(lambda i=i: create_env(config.task, config.headless, config.resume,ability_agent=low_level_policy))
            # envs.append(lambda i=i+1: create_env("Hierarchical-v1.1", config.headless, config.resume))
            # envs.append(lambda i=i+2: create_env("Hierarchical-v1.2", config.headless, config.resume))
        # test_envs = gym.make(args.task)
        train_envs = ShmemVectorEnv(envs)
        test_envs = []
        for i in range(0,config.test_num):
            test_envs.append(lambda i=i: create_env(config.task, config.headless, config.resume,ability_agent=low_level_policy))
            # test_envs.append(lambda i=i+1: create_env("Hierarchical-v1.1", config.headless, config.resume))
            # test_envs.append(lambda i=i+2: create_env("Hierarchical-v1.2", config.headless, config.resume))
        test_envs = ShmemVectorEnv(test_envs)
        train_envs.seed(config.seed)
        test_envs.seed(config.seed)
        # collector
        train_collector = Collector(
            hierarchical_policy,
            train_envs,
            PrioritizedVectorReplayBuffer(config.buffer_size, len(train_envs), alpha=config.buffer_alpha, beta=config.beta),
            exploration_noise=True,
            log_path=log_path,
            label='train'
        )
        test_collector = Collector(hierarchical_policy, 
                                   test_envs,
                                   log_path=log_path,
                                   label='test')
        # trainer
        trainer = OnpolicyTrainer(
            policy=hierarchical_policy,
            train_collector=train_collector,
            test_collector=test_collector,
            max_epoch=config.epoch,
            step_per_epoch=config.step_per_epoch,
            step_per_collect=config.step_per_collect,
            repeat_per_collect=config.repeat_per_collect,
            episode_per_test=config.test_num,
            batch_size=config.batch_size,
            # episode_per_collect=config.episode_per_collect,
            update_per_step=config.update_per_step,
            stop_fn=stop_fn,
            save_best_fn=save_best_fn,
            train_fn=train_fn,
            test_fn=test_fn,
            logger=logger,
            resume_from_log=config.resume,
            save_checkpoint_fn=save_checkpoint_fn,
        )

        for epoch_stat in trainer:
            print(f"Epoch: {epoch_stat}")
            print(epoch_stat)
            # print(info)

        assert stop_fn(epoch_stat[2]['best_reward'])


def test_ppo_resume() -> None:
    config.resume = True
    test_ppo()

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    print("可用的GPU数量：",torch.cuda.device_count())
    test_ppo()
