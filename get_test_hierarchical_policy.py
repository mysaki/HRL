import argparse
import os
import pprint
import datetime
import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.distributions import Distribution, Independent, Normal
from torch.utils.tensorboard import SummaryWriter
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from gymnasium import spaces
from tianshou.data import Collector, PrioritizedVectorReplayBuffer
from tianshou.data import Collector,VectorReplayBuffer,Batch,ReplayBuffer
from tianshou.env import ShmemVectorEnv
from tianshou.utils.net.discrete import NoisyLinear
from tianshou.policy import PPOPolicy, RainbowPolicy
from tianshou.policy.base import BasePolicy
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.continuous import ActorProb, Critic
from Nets.Actor_Net import Actor_Preprocess_Net
from Nets.Critic_Net import Critic_Preprocess_Net
from Nets.Star_net_rnn_attention import STAR
from Nets.Star_net_rnn import STAR as STAR_rnn
from Nets.Star_net_attention import STAR as STAR_mha
from Nets.Star_net import STAR as STAR_only
from Nets.Track_net import Track_Net
import config_dqn as config
from get_test_nav_policy import *
from get_test_track_policy import *
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def get_policy_nav(path=None):
    def get_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        parser.add_argument('--headless', type=bool, default=True)
        parser.add_argument("--task", type=str, default="Navigation-v0")
        parser.add_argument('--test', type=bool, default=False)
        parser.add_argument("--load-model", type=bool, default=False)
        parser.add_argument("--reward-threshold", type=float, default=150000000)
        parser.add_argument("--seed", type=int, default=1)
        parser.add_argument("--buffer-size", type=int, default=20000)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--gamma", type=float, default=0.995)
        parser.add_argument("--epoch", type=int, default=100)
        parser.add_argument("--step-per-epoch", type=int, default=100000)
        parser.add_argument("--episode-per-collect", type=int, default=20)
        parser.add_argument("--repeat-per-collect", type=int, default=2)
        parser.add_argument("--batch-size", type=int, default=1024)
        parser.add_argument("--actor-hidden-sizes", type=int,
                            nargs="*", default=[128])
        parser.add_argument("--critic-hidden-sizes", type=int,
                            nargs="*", default=[128])
        parser.add_argument("--training-num", type=int, default=20)
        parser.add_argument("--test-num", type=int, default=4)
        parser.add_argument("--logdir", type=str, default="Log")
        parser.add_argument("--render", type=float, default=0.0)
        parser.add_argument(
            "--device",
            type=str,
            default="cuda" if torch.cuda.is_available() else "cpu",
        )
        # ppo special
        parser.add_argument("--vf-coef", type=float, default=0.25)
        parser.add_argument("--ent-coef", type=float, default=0.0)
        parser.add_argument("--eps-clip", type=float, default=0.2)
        parser.add_argument("--max-grad-norm", type=float, default=0.5)
        parser.add_argument("--gae-lambda", type=float, default=0.95)
        parser.add_argument("--rew-norm", type=int, default=1)
        parser.add_argument("--dual-clip", type=float, default=None)
        parser.add_argument("--value-clip", type=int, default=1)
        parser.add_argument("--norm-adv", type=int, default=1)
        parser.add_argument("--recompute-adv", type=int, default=0)
        parser.add_argument("--resume", action="store_true")
        parser.add_argument('--buffer_alpha', type=float, default=0.6)
        parser.add_argument('--beta', type=float, default=0.4)
        parser.add_argument("--save-interval", type=int, default=4)
        return parser.parse_known_args()[0]
    args =get_args()
    args.state_shape = (36,)
    args.action_space = spaces.Box(
        low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float64
    )
    args.action_shape = (2,)
    # model
    CPNet = Critic_Preprocess_Net(
        input_dim=np.prod(args.state_shape), device=args.device, feature_dim=256, hidden_size=[128,128])
    # Q_param = V_param = {"hidden_sizes": [64, 64]}
    # CPNet = Critic_Preprocess_Net(
    #     input_dim=34, action_shape=2, device=args.device, num_atoms=2, dueling_param=(Q_param, V_param), feature_dim=64, hidden_size=64)
    # APNet=STAR(input_dim=np.prod(args.state_shape)+32, feature_dim=256, device=args.device,hidden_dim=[128,128])
    APNet = Actor_Preprocess_Net(
        input_dim=np.prod(args.state_shape), device=args.device, feature_dim=256, hidden_size=[128,128])
    actor = ActorProb(APNet,
                      args.action_shape,
                      unbounded=True,
                      hidden_sizes=args.actor_hidden_sizes,
                      device=args.device
                      ).to(args.device)
    critic = Critic(
        CPNet,
        hidden_sizes=args.critic_hidden_sizes,
        device=args.device,
    ).to(args.device)
    actor_critic = ActorCritic(actor, critic)
    # orthogonal initialization
    for m in actor_critic.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight)
            torch.nn.init.zeros_(m.bias)
    optim = torch.optim.Adam(actor_critic.parameters(), lr=args.lr)

    # replace DiagGuassian with Independent(Normal) which is equivalent
    # pass *logits to be consistent with policy.forward
    def dist(loc_scale) -> Distribution:
        loc, scale = loc_scale
        return Independent(Normal(loc, scale), 1)

    policy = PPOPolicy(
        actor=actor,
        critic=critic,
        optim=optim,
        dist_fn=dist,
        discount_factor=args.gamma,
        max_grad_norm=args.max_grad_norm,
        eps_clip=args.eps_clip,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        reward_normalization=args.rew_norm,
        advantage_normalization=args.norm_adv,
        recompute_advantage=args.recompute_adv,
        dual_clip=args.dual_clip,
        value_clip=args.value_clip,
        gae_lambda=args.gae_lambda,
        action_space=args.action_space,
        action_bound_method='clip',
    )

    if path != None:
        # load from existing checkpoint
        print(f"Loading agent under {path}")
        ckpt_path = os.path.join(path, "Track_train.pth")
        if os.path.exists(ckpt_path):
            # checkpoint = torch.load(ckpt_path, map_location=args.device)
            # policy.load_state_dict(checkpoint["model"])
            # optim.load_state_dict(checkpoint["optim"])
            policy.load_state_dict(torch.load(ckpt_path))
            print("Policy load!")
        else:
            print("Fail to restore policy.")
    return policy
        


def get_dqn_policy(path=None):
    env = gym.make(config.task, headless=config.headless,mode=config.resume)
    config.state_shape = env.observation_space.shape or env.observation_space.n
    config.action_shape = env.action_space.shape
    def noisy_linear(x: int, y: int):
        return NoisyLinear(x, y, config.noisy_std)
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
    policy = RainbowPolicy(
        model=net,
        optim=high_optim,
        discount_factor=config.gamma,
        estimation_step=config.high_policy_params["n_step"],
        target_update_freq=config.high_policy_params["target_update_freq"],
        action_space=config.high_level_action,
    )
    if path:
        # load from existing checkpoint
        print(f"Loading high level agent under {path}")
        # ckpt_path = os.path.join(resume_path, "policy_reset.pth")
        ckpt_path = os.path.join(path, "Track_train.pth")
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path)
            # high_level_policy.load_state_dict(checkpoint['high_level_policy_state_dict'])
            policy.load_state_dict(checkpoint,strict = False)
            # low_level_policy.load_state_dict(checkpoint['low_level_policy_state_dict'])
            # low_level_optim.load_state_dict(checkpoint["low_level_optim"])
            # high_optim.load_state_dict(checkpoint["high_level_optim"])
            print("Successfully restore  high level policy and optim.")
        else:
            print("Fail to restore policy and optim.")
    policy.eval()
    return policy

def get_Both_STAR_Only_policy(path=None):
    def get_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        parser.add_argument('--headless', type=bool, default=True)
        parser.add_argument("--task", type=str, default="Dynamic-v0")
        parser.add_argument('--test', type=bool, default=False)
        parser.add_argument("--load-model", type=bool, default=False)
        parser.add_argument("--reward-threshold", type=float, default=150000000)
        parser.add_argument("--seed", type=int, default=1)
        parser.add_argument("--buffer-size", type=int, default=20000)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--gamma", type=float, default=0.995)
        parser.add_argument("--epoch", type=int, default=100)
        parser.add_argument("--step-per-epoch", type=int, default=100000)
        parser.add_argument("--episode-per-collect", type=int, default=20)
        parser.add_argument("--repeat-per-collect", type=int, default=2)
        parser.add_argument("--batch-size", type=int, default=1024)
        parser.add_argument("--actor-hidden-sizes", type=int,
                            nargs="*", default=[128])
        parser.add_argument("--critic-hidden-sizes", type=int,
                            nargs="*", default=[128])
        parser.add_argument("--training-num", type=int, default=20)
        parser.add_argument("--test-num", type=int, default=4)
        parser.add_argument("--logdir", type=str, default="Log")
        parser.add_argument("--render", type=float, default=0.0)
        parser.add_argument(
            "--device",
            type=str,
            default="cuda" if torch.cuda.is_available() else "cpu",
        )
        # ppo special
        parser.add_argument("--vf-coef", type=float, default=0.25)
        parser.add_argument("--ent-coef", type=float, default=0.0)
        parser.add_argument("--eps-clip", type=float, default=0.2)
        parser.add_argument("--max-grad-norm", type=float, default=0.5)
        parser.add_argument("--gae-lambda", type=float, default=0.95)
        parser.add_argument("--rew-norm", type=int, default=1)
        parser.add_argument("--dual-clip", type=float, default=None)
        parser.add_argument("--value-clip", type=int, default=1)
        parser.add_argument("--norm-adv", type=int, default=1)
        parser.add_argument("--recompute-adv", type=int, default=0)
        parser.add_argument("--resume", action="store_true")
        parser.add_argument('--buffer_alpha', type=float, default=0.6)
        parser.add_argument('--beta', type=float, default=0.4)
        parser.add_argument("--save-interval", type=int, default=4)
        return parser.parse_known_args()[0]
    args =get_args()
    args.state_shape = (36,)
    args.action_space = spaces.Box(
        low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float64
    )
    args.action_shape = (2,)
    # model
    # CPNet = Critic_Preprocess_Net(
    #     input_dim=np.prod(args.state_shape), device=args.device, feature_dim=256, hidden_size=[128,128])
    CPNet = STAR_only(input_dim=np.prod(args.state_shape), feature_dim=256, device=args.device,hidden_dim=[128,128])
    # Q_param = V_param = {"hidden_sizes": [64, 64]}
    # CPNet = Critic_Preprocess_Net(
    #     input_dim=34, action_shape=2, device=args.device, num_atoms=2, dueling_param=(Q_param, V_param), feature_dim=64, hidden_size=64)
    APNet=STAR_only(input_dim=np.prod(args.state_shape), feature_dim=256, device=args.device,hidden_dim=[128,128])
    # APNet=Actor_Preprocess_Net(
    #     input_dim=np.prod(args.state_shape), device=args.device, feature_dim=256, hidden_size=[128,128])
    actor = ActorProb(APNet,
                      args.action_shape,
                      unbounded=True,
                      hidden_sizes=args.actor_hidden_sizes,
                      device=args.device
                      ).to(args.device)
    critic = Critic(
        CPNet,
        hidden_sizes=args.critic_hidden_sizes,
        device=args.device,
    ).to(args.device)
    actor_critic = ActorCritic(actor, critic)
    # orthogonal initialization
    for m in actor_critic.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight)
            torch.nn.init.zeros_(m.bias)
    optim = torch.optim.Adam(actor_critic.parameters(), lr=args.lr)

    # replace DiagGuassian with Independent(Normal) which is equivalent
    # pass *logits to be consistent with policy.forward
    def dist(loc_scale) -> Distribution:
        loc, scale = loc_scale
        return Independent(Normal(loc, scale), 1)

    policy = PPOPolicy(
        actor=actor,
        critic=critic,
        optim=optim,
        dist_fn=dist,
        discount_factor=args.gamma,
        max_grad_norm=args.max_grad_norm,
        eps_clip=args.eps_clip,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        reward_normalization=args.rew_norm,
        advantage_normalization=args.norm_adv,
        recompute_advantage=args.recompute_adv,
        dual_clip=args.dual_clip,
        value_clip=args.value_clip,
        gae_lambda=args.gae_lambda,
        action_space=args.action_space,
        action_bound_method='clip',
    )
    if path != None:
        # load from existing checkpoint
        print(f"Loading agent under {path}")
        ckpt_path = os.path.join(path, "Track_train.pth")
        if os.path.exists(ckpt_path):
            # checkpoint = torch.load(ckpt_path, map_location=args.device)
            # policy.load_state_dict(checkpoint["model"])
            # optim.load_state_dict(checkpoint["optim"])
            policy.load_state_dict(torch.load(ckpt_path))
            print("Policy load!")
        else:
            print("Fail to restore policy.")
    return policy
def get_A_STAR_Only_policy(path=None):
    def get_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        parser.add_argument('--headless', type=bool, default=True)
        parser.add_argument("--task", type=str, default="Dynamic-v0")
        parser.add_argument('--test', type=bool, default=False)
        parser.add_argument("--load-model", type=bool, default=False)
        parser.add_argument("--reward-threshold", type=float, default=150000000)
        parser.add_argument("--seed", type=int, default=1)
        parser.add_argument("--buffer-size", type=int, default=20000)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--gamma", type=float, default=0.995)
        parser.add_argument("--epoch", type=int, default=100)
        parser.add_argument("--step-per-epoch", type=int, default=100000)
        parser.add_argument("--episode-per-collect", type=int, default=20)
        parser.add_argument("--repeat-per-collect", type=int, default=2)
        parser.add_argument("--batch-size", type=int, default=1024)
        parser.add_argument("--actor-hidden-sizes", type=int,
                            nargs="*", default=[128])
        parser.add_argument("--critic-hidden-sizes", type=int,
                            nargs="*", default=[128])
        parser.add_argument("--training-num", type=int, default=20)
        parser.add_argument("--test-num", type=int, default=4)
        parser.add_argument("--logdir", type=str, default="Log")
        parser.add_argument("--render", type=float, default=0.0)
        parser.add_argument(
            "--device",
            type=str,
            default="cuda" if torch.cuda.is_available() else "cpu",
        )
        # ppo special
        parser.add_argument("--vf-coef", type=float, default=0.25)
        parser.add_argument("--ent-coef", type=float, default=0.0)
        parser.add_argument("--eps-clip", type=float, default=0.2)
        parser.add_argument("--max-grad-norm", type=float, default=0.5)
        parser.add_argument("--gae-lambda", type=float, default=0.95)
        parser.add_argument("--rew-norm", type=int, default=1)
        parser.add_argument("--dual-clip", type=float, default=None)
        parser.add_argument("--value-clip", type=int, default=1)
        parser.add_argument("--norm-adv", type=int, default=1)
        parser.add_argument("--recompute-adv", type=int, default=0)
        parser.add_argument("--resume", action="store_true")
        parser.add_argument('--buffer_alpha', type=float, default=0.6)
        parser.add_argument('--beta', type=float, default=0.4)
        parser.add_argument("--save-interval", type=int, default=4)
        return parser.parse_known_args()[0]
    args =get_args()
    args.state_shape = (36,)
    args.action_space = spaces.Box(
        low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float64
    )
    args.action_shape = (2,)
    # model
    CPNet = Critic_Preprocess_Net(
        input_dim=np.prod(args.state_shape), device=args.device, feature_dim=256, hidden_size=[128,128])
    # CPNet = STAR_only(input_dim=np.prod(args.state_shape), feature_dim=256, device=args.device,hidden_dim=[128,128])
    # Q_param = V_param = {"hidden_sizes": [64, 64]}
    # CPNet = Critic_Preprocess_Net(
    #     input_dim=34, action_shape=2, device=args.device, num_atoms=2, dueling_param=(Q_param, V_param), feature_dim=64, hidden_size=64)
    APNet=STAR_only(input_dim=np.prod(args.state_shape), feature_dim=256, device=args.device,hidden_dim=[128,128])
    # APNet=Actor_Preprocess_Net(
    #     input_dim=np.prod(args.state_shape), device=args.device, feature_dim=256, hidden_size=[128,128])
    actor = ActorProb(APNet,
                      args.action_shape,
                      unbounded=True,
                      hidden_sizes=args.actor_hidden_sizes,
                      device=args.device
                      ).to(args.device)
    critic = Critic(
        CPNet,
        hidden_sizes=args.critic_hidden_sizes,
        device=args.device,
    ).to(args.device)
    actor_critic = ActorCritic(actor, critic)
    # orthogonal initialization
    for m in actor_critic.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight)
            torch.nn.init.zeros_(m.bias)
    optim = torch.optim.Adam(actor_critic.parameters(), lr=args.lr)

    # replace DiagGuassian with Independent(Normal) which is equivalent
    # pass *logits to be consistent with policy.forward
    def dist(loc_scale) -> Distribution:
        loc, scale = loc_scale
        return Independent(Normal(loc, scale), 1)

    policy = PPOPolicy(
        actor=actor,
        critic=critic,
        optim=optim,
        dist_fn=dist,
        discount_factor=args.gamma,
        max_grad_norm=args.max_grad_norm,
        eps_clip=args.eps_clip,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        reward_normalization=args.rew_norm,
        advantage_normalization=args.norm_adv,
        recompute_advantage=args.recompute_adv,
        dual_clip=args.dual_clip,
        value_clip=args.value_clip,
        gae_lambda=args.gae_lambda,
        action_space=args.action_space,
        action_bound_method='clip',
    )
    if path != None:
        # load from existing checkpoint
        print(f"Loading agent under {path}")
        ckpt_path = os.path.join(path, "Track_train.pth")
        if os.path.exists(ckpt_path):
            # checkpoint = torch.load(ckpt_path, map_location=args.device)
            # policy.load_state_dict(checkpoint["model"])
            # optim.load_state_dict(checkpoint["optim"])
            policy.load_state_dict(torch.load(ckpt_path))
            print("Policy load!")
        else:
            print("Fail to restore policy.")
    return policy

class HierarchicalPolicy(BasePolicy):
    def __init__(self, high_level_policy, low_level_policy, **kwargs):
        super().__init__(action_scaling="True", action_bound_method="clip", **kwargs)
        self.high_level_policy = high_level_policy
        self.low_level_policy = low_level_policy
        self.action_space = low_level_policy.action_space
        self.count_safe = 0
        self.count_track = 0

    def forward(self, batch, state=None):
        # 高层策略决定方向目标
        high_level_batch = self.high_level_policy.forward(batch)
        high_level_action = high_level_batch["act"]
        # print("action:", high_level_action)
        high_level_action = self.high_level_policy.exploration_noise(
            high_level_action, batch
        )
        high_level_action = [
            (
                [0, 1]
                if action == 0
                else (
                    [1, 0] if action == 1 else [int(bit) for bit in format(action, "b")]
                )
            )
            for action in high_level_action
        ]
        if high_level_action[0] == [0, 1]:
            self.count_safe += 1
        elif high_level_action[0] == [1, 0]:
            self.count_track += 1
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
        modified_obs = torch.cat(
            [
                origin_obs[:, :2],
                torch.tensor(high_level_action).to(config.device),
                origin_obs[:, 2:],
            ],
            dim=1,
        )
        # print(modified_obs)
        low_level_batch = batch.copy()
        low_level_batch["obs"] = modified_obs
        # 低层策略根据高层的方向执行具体的动作
        low_level_batch = self.low_level_policy.forward(Batch(low_level_batch))
        low_level_batch["low_level_obs"] = modified_obs
        low_level_batch["high_level_act"] = high_level_batch["act"]
        low_level_batch["obs"] = batch.obs
        return Batch(low_level_batch)

    def learn(self, batch, **kwargs):
        high_level_batch = batch.copy()
        # high_level_batch['obs'] = batch['low_level_obs']
        high_level_batch["act"] = batch["high_level_act"]
        high_level_loss = self.high_level_policy.learn(
            Batch(high_level_batch), **kwargs
        )
        # low_level_loss = self.low_level_policy.learn(batch, **kwargs)

        return high_level_loss  # , low_level_loss

    def update(
        self, sample_size: int, buffer: Optional[ReplayBuffer], **kwargs: Any
    ) -> Dict[str, Any]:
        high_level_loss = self.high_level_policy.update(
            sample_size, buffer, level="high"
        )
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
        if isinstance(self.action_space, gym.spaces.Box) and isinstance(
            act, np.ndarray
        ):
            # currently this action mapping only supports np.ndarray action
            if self.action_bound_method == "clip":
                act = np.clip(act, -1.0, 1.0)
            elif self.action_bound_method == "tanh":
                act = np.tanh(act)
            if self.action_scaling:
                assert (
                    np.min(act) >= -1.0 and np.max(act) <= 1.0
                ), "action scaling only accepts raw action range = [-1, 1]"
                low, high = self.action_space.low, self.action_space.high
                act = low + (high - low) * (act + 1.0) / 2.0  # type: ignore
        if isinstance(act, np.ndarray):
            act = torch.from_numpy(act)
        # act= torch.where(act[:, 0] < act[:, 1], torch.tensor(1), torch.tensor(0))
        # act.unsqueeze(1)
        # act = act.cpu().numpy()
        # print("final_act",act)
        return act

def get_ASTAR_hierarchical_policy(low_level_policy_path=None, high_level_policy_path=None):
    low_level_policy = get_A_STAR_Only_policy(low_level_policy_path)
    high_level_policy = get_dqn_policy(high_level_policy_path)
    hierarchical_policy = HierarchicalPolicy(high_level_policy, low_level_policy)
    hierarchical_policy.eval()
    return hierarchical_policy
def get_hierarchical_policy(low_level_policy_path=None, high_level_policy_path=None):
    low_level_policy = get_Both_STAR_Only_policy(low_level_policy_path)
    high_level_policy = get_dqn_policy(high_level_policy_path)
    hierarchical_policy = HierarchicalPolicy(high_level_policy, low_level_policy)
    hierarchical_policy.eval()
    return hierarchical_policy

class Rule_Base_Agent(BasePolicy):
    def __init__(self,model_path ,**kwargs):
        super().__init__(action_scaling="True", action_bound_method="clip", **kwargs)
        self.agent = get_Both_STAR_Only_policy(path=model_path)
    def learn(self):
        return
    def forward(self,batch):
        flag = batch.flag
        # print("target in sight?",flag)
        if flag == 0:
            origin_obs = torch.from_numpy(batch.obs).to(config.device)
            modified_obs = torch.cat([origin_obs[:,:2], torch.tensor([[0,1]]).to(config.device),origin_obs[:,2:]], dim=1)
        elif flag == 1:
            origin_obs = torch.from_numpy(batch.obs).to(config.device)
            modified_obs = torch.cat([origin_obs[:,:2], torch.tensor([[1,0]]).to(config.device),origin_obs[:,2:]], dim=1)
        batch['obs']=modified_obs
        # 低层策略根据高层的方向执行具体的动作
        batch= self.agent.forward(batch)
        return batch
if __name__ == '__main__':
    policy_hierarchical = get_hierarchical_policy(
        low_level_policy_path="./Log/join_train_track_ppo_12_29_11_2",
        high_level_policy_path="./log/Hierarchical-v1/hierarchical_track_dqn_1_9_9_38",
    )
