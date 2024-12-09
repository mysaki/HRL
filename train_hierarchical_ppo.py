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
from tianshou.data import Collector,VectorReplayBuffer,Batch,ReplayBuffer
from tianshou.env import ShmemVectorEnv
from tianshou.policy import PPOPolicy,DQNPolicy
from tianshou.policy.base import BasePolicy
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils import TensorboardLogger
from Nets.Critic_Net import Critic_Preprocess_Net
from Nets.Actor_Net import Actor_Preprocess_Net
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.continuous import ActorProb, Critic
from Nets.Track_net import Track_Net
import config_ppo as config
from Nets.Star_net_rnn_attention import STAR
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
        # print("high level action:",high_level_action)
        # high_level_action = self.high_level_policy.exploration_noise(high_level_action, batch)
        # 假设高层动作表示不同的技能
        # 将高层动作嵌入到低层策略的输入中
        origin_obs = torch.from_numpy(batch.obs).to(config.device)
        # 获得底层跟踪能力智能体对应的动作
        size = origin_obs.shape[0]
        modified_obs = torch.cat([origin_obs[:,:2], torch.tensor([1,0]).unsqueeze(0).repeat(size,1).to(config.device),origin_obs[:,2:]], dim=1)
        low_level_track_batch = batch.copy()
        low_level_track_batch['obs']=modified_obs
        low_level_track_batch = self.low_level_policy.forward(
            Batch(low_level_track_batch)
        )
        batch["low_level_track_act"] = low_level_track_batch["act"]
        batch["low_level_track_obs"] = modified_obs
        # 获得底层避障能力智能体对应的动作
        low_level_safe_batch = batch.copy()
        modified_obs = torch.cat(
            [
                origin_obs[:, :2],
                torch.tensor([0,1]).unsqueeze(0).repeat(size,1).to(config.device),
                origin_obs[:, 2:],
            ],
            dim=1,
        )
        low_level_safe_batch["obs"] = modified_obs
        low_level_safe_batch = self.low_level_policy.forward(
            Batch(low_level_safe_batch)
        )
        batch["low_level_safe_act"] = low_level_safe_batch["act"]
        batch['high_level_act'] = high_level_batch['act']
        s = torch.exp(high_level_batch["act"][0][0]) + torch.exp(
            high_level_batch["act"][0][1]
        )
        high_level_batch["act"][0][0] = torch.exp(high_level_batch["act"][0][0]) / s
        high_level_batch["act"][0][1] = torch.exp(high_level_batch["act"][0][1]) / s
        batch["act"] = (
            high_level_batch["act"][0][0] * batch["low_level_track_act"]
            + high_level_batch["act"][0][1] * batch["low_level_safe_act"]
        )
        # print("weight:",high_level_batch["act"],"track_act:",batch["low_level_track_act"],"safe_act:",batch["low_level_safe_act"],"final_act:",batch["act"])
        # print("model_act:",batch["act"])
        batch["low_level_safe_obs"] = modified_obs
        return Batch(batch)

    def learn(self, batch, **kwargs):
        high_level_batch = batch.copy()
        # high_level_batch['obs'] = batch['low_level_obs']
        high_level_batch['act'] = batch['high_level_act']
        high_level_loss = self.high_level_policy.learn(Batch(high_level_batch), **kwargs)
        # low_level_loss = self.low_level_policy.learn(batch, **kwargs)

        return high_level_loss#, low_level_loss

    def update(self, sample_size: int, buffer: Optional[ReplayBuffer],
               **kwargs: Any) -> Dict[str, Any]:
        high_level_loss = self.high_level_policy.update(sample_size, buffer, level='high',batch_size=config.batch_size,repeat=config.repeat_per_collect)
        # low_level_loss = self.low_level_policy.update(sample_size, buffer, level='low')
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


def test_ppo() -> None:
    env = gym.make(config.task, headless=config.headless,mode=config.resume)
    config.state_shape = env.observation_space.shape or env.observation_space.n
    config.action_shape = env.action_space.shape

    # you can also use tianshou.env.SubprocVectorEnv
    # train_envs = gym.make(args.task)

    # seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    low_policy_state_dim = list(config.state_shape)
    low_policy_state_dim[-1]+=2
    low_policy_state_dim = tuple(low_policy_state_dim)
    low_policy_APNet = STAR(
        input_dim=np.prod(low_policy_state_dim)+32, device=config.device, feature_dim=256, hidden_dim=[128,128])
    low_policy_CPNet=Critic_Preprocess_Net(input_dim=np.prod(low_policy_state_dim), 
                                device=config.device,
                                feature_dim=256, 
                                hidden_size=[128,128]
                                ).to(config.device)
    low_policy_actor = ActorProb(low_policy_APNet,
                    config.action_shape,
                    unbounded=True,
                    hidden_sizes=[128],
                    device=config.device
                    ).to(config.device)
    low_policy_critic = Critic(
        low_policy_CPNet,
        hidden_sizes=[128],
        device=config.device,
    ).to(config.device)
    low_policy_actor_critic = ActorCritic(low_policy_actor, low_policy_critic)
    # orthogonal initialization
    for m in low_policy_actor_critic.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight)
            torch.nn.init.zeros_(m.bias)
    for parameter in low_policy_actor_critic.parameters():
        parameter.requires_grad = False
    low_level_optim = torch.optim.Adam(low_policy_actor_critic.parameters(), lr=config.lr)

    # replace DiagGuassian with Independent(Normal) which is equivalent
    # pass *logits to be consistent with policy.forward
    def dist(loc_scale) -> Distribution:
        loc, scale = loc_scale
        return Independent(Normal(loc, scale), 1)

    low_level_policy = PPOPolicy(
        actor=low_policy_actor,
        critic=low_policy_critic,
        optim=low_level_optim,
        dist_fn=dist,
        discount_factor=config.gamma,
        max_grad_norm=config.low_policy_params["max_grad_norm"],
        eps_clip=config.low_policy_params["eps_clip"],
        vf_coef=config.low_policy_params["vf_coef"],
        ent_coef=config.low_policy_params["ent_coef"],
        reward_normalization=config.low_policy_params["reward_normalization"],
        advantage_normalization=config.low_policy_params["advantage_normalization"],
        recompute_advantage=config.low_policy_params["recompute_advantage"],
        dual_clip=config.low_policy_params["dual_clip"],
        value_clip=config.low_policy_params["value_clip"],
        gae_lambda=config.low_policy_params["gae_lambda"],
        action_space=env.action_space,
        action_bound_method='clip',
    )
    log_low_model_path = "Log/join_train_track_ppo_11_17_0_0"

    print(f"Loading low level agent under {log_low_model_path}")
    ckpt_path = os.path.join(log_low_model_path, "Track_train.pth")
    if os.path.exists(ckpt_path):
        # checkpoint = torch.load(ckpt_path, map_location=args.device)
        # policy.load_state_dict(checkpoint["model"])
        # optim.load_state_dict(checkpoint["optim"])
        low_level_policy.load_state_dict(torch.load(ckpt_path))
        low_level_policy.eval()
        print('Low level policy load!')
    else:
        print("Fail to restore low level policy.")
    # model
    high_net = Net(state_shape=config.state_shape, hidden_sizes=config.high_policy_params['hidden_size'], device=config.device)
    high_actor = ActorProb(high_net, config.high_level_action.shape,
                            # unbounded=True,
                            conditioned_sigma = True,
                            device=config.device).to(config.device)
    high_critic = Critic(
        Net(state_shape=config.state_shape, hidden_sizes=config.high_policy_params['hidden_size'], device=config.device),
        device=config.device,
    ).to(config.device)
    high_actor_critic = ActorCritic(high_actor, high_critic)
    # orthogonal initialization
    for m in high_actor_critic.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight)
            torch.nn.init.zeros_(m.bias)
    high_optim = torch.optim.Adam(high_actor_critic.parameters(), lr=config.lr)

    # replace DiagGuassian with Independent(Normal) which is equivalent
    # pass *logits to be consistent with policy.forward

    high_level_policy = PPOPolicy(
        actor=high_actor,
        critic=high_critic,
        optim=high_optim,
        dist_fn=dist,
        discount_factor=config.gamma,
        max_grad_norm=config.high_policy_params["max_grad_norm"],
        eps_clip=config.high_policy_params["eps_clip"],
        vf_coef=config.high_policy_params["vf_coef"],
        ent_coef=config.high_policy_params["ent_coef"],
        reward_normalization=config.high_policy_params["reward_normalization"],
        advantage_normalization=config.high_policy_params["advantage_normalization"],
        recompute_advantage=config.high_policy_params["recompute_advantage"],
        dual_clip=config.high_policy_params["dual_clip"],
        value_clip=config.high_policy_params["value_clip"],
        gae_lambda=config.high_policy_params["gae_lambda"],
        action_space=config.high_level_action,
    )
    # 分层策略控制器
    hierarchical_policy = HierarchicalPolicy(high_level_policy, low_level_policy)

    # log
    time = datetime.datetime.now()  # 获取当前时间
    log_path = os.path.join(config.logdir, config.task,"hierarchical_track_ppo_{}_{}_{}_{}".format(
        time.month, time.day, time.hour, time.minute, time.second))
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
    def create_env(env_name, headless, mode):
        return gym.make(env_name, headless=headless, mode=mode)
    if config.resume:
        resume_path = os.path.join('log', config.task,config.resume_model)
        # load from existing checkpoint
        print(f"Loading high level agent under {resume_path}")
        # ckpt_path = os.path.join(resume_path, "policy_reset.pth")
        ckpt_path = os.path.join(resume_path,"Track_train.pth")
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path)
            # high_level_policy.load_state_dict(checkpoint['high_level_policy_state_dict'])
            hierarchical_policy.load_state_dict(checkpoint)
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
            "result_hierarchical_{}_{}_{}_{}_{}.xlsx".format(
                time.month, time.day, time.hour, time.minute, time.second
            ),
            index=False,
        )
        print(result)
    else:
        envs = []
        for i in range(config.training_num):
            envs.append(lambda i=i: create_env(config.task, config.headless, config.resume))
        # test_envs = gym.make(args.task)
        train_envs = ShmemVectorEnv(envs)
        test_envs = []
        for i in range(config.test_num):
            test_envs.append(lambda i=i: create_env(config.task, config.headless, config.resume))
        test_envs = ShmemVectorEnv(test_envs)
        train_envs.seed(config.seed)
        test_envs.seed(config.seed)
        # collector
        train_collector = Collector(
            hierarchical_policy,
            train_envs,
            VectorReplayBuffer(config.buffer_size, len(train_envs)),
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
            # step_per_collect=config.step_per_collect,
            repeat_per_collect=config.repeat_per_collect,
            episode_per_test=config.test_num,
            batch_size=config.batch_size,
            episode_per_collect=config.episode_per_collect,
            update_per_step=config.update_per_step,
            stop_fn=stop_fn,
            save_best_fn=save_best_fn,

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
    test_ppo()
