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
from gymnasium import spaces
from tianshou.data import Collector, PrioritizedVectorReplayBuffer
from tianshou.env import ShmemVectorEnv
from tianshou.policy import PPOPolicy
from tianshou.policy.base import BasePolicy
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.continuous import ActorProb, Critic
from Nets.Actor_Net import Actor_Preprocess_Net
from Nets.Critic_Net import Critic_Preprocess_Net
from Nets.Star_net_rnn_attention import STAR
from Nets.Track_net_v3 import Track_Net
from Nets.Star_net_rnn import STAR as STAR_rnn
from Nets.Star_net_attention import STAR as STAR_mha
from Nets.Star_net import STAR as STAR_only
def get_track_policy(path=None):
    def get_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        parser.add_argument('--headless', type=bool, default=True)
        parser.add_argument("--task", type=str, default="Dynamic-v0")
        parser.add_argument('--test', type=bool, default=True)
        parser.add_argument("--load-model", type=bool, default=True)
        parser.add_argument("--reward-threshold", type=float, default=150000000)
        parser.add_argument("--seed", type=int, default=1)
        parser.add_argument("--buffer-size", type=int, default=20000)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--gamma", type=float, default=0.995)
        parser.add_argument("--epoch", type=int, default=25)
        parser.add_argument("--step-per-epoch", type=int, default=100000)
        parser.add_argument("--episode-per-collect", type=int, default=20)
        parser.add_argument("--repeat-per-collect", type=int, default=2)
        parser.add_argument("--batch-size", type=int, default=1024)
        parser.add_argument("--actor-hidden-sizes", type=int,
                            nargs="*", default=[128])
        parser.add_argument("--critic-hidden-sizes", type=int,
                            nargs="*", default=[128])
        parser.add_argument("--training-num", type=int, default=20)
        parser.add_argument("--test-num", type=int, default=2)
        parser.add_argument("--logdir", type=str, default="Log")
        parser.add_argument("--render", type=float, default=0.0)
        parser.add_argument(
            "--device",
            type=str,
            default="cuda:0" if torch.cuda.is_available() else "cpu",
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
    APNet = Actor_Preprocess_Net(
        input_dim=np.prod(args.state_shape), device=args.device, feature_dim=256, hidden_size=[128,128])
    # Q_param = V_param = {"hidden_sizes": [64, 64]}
    # CPNet = Critic_Preprocess_Net(
    #     input_dim=34, action_shape=2, device=args.device, num_atoms=2, dueling_param=(Q_param, V_param), feature_dim=64, hidden_size=64)
    CPNet = Critic_Preprocess_Net(
        input_dim=np.prod(args.state_shape), device=args.device, feature_dim=256, hidden_size=[128,128])
    # CPNet=STAR(
    #     input_dim=np.prod(args.state_shape)+32, device=args.device, feature_dim=256, hidden_dim=[128,128],norm_layers=None)
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


def get_rgb_policy(path=None):
    def get_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        parser.add_argument("--headless", type=bool, default=True)
        parser.add_argument("--task", type=str, default="Track-v0")
        parser.add_argument("--test", type=bool, default=False)
        parser.add_argument("--load-model", type=bool, default=False)
        parser.add_argument("--reward-threshold", type=float, default=150000000)
        parser.add_argument("--seed", type=int, default=1)
        parser.add_argument("--buffer-size", type=int, default=10000)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--gamma", type=float, default=0.995)
        parser.add_argument("--epoch", type=int, default=100)
        parser.add_argument("--step-per-epoch", type=int, default=100000)
        parser.add_argument("--episode-per-collect", type=int, default=20)
        parser.add_argument("--repeat-per-collect", type=int, default=2)
        parser.add_argument("--batch-size", type=int, default=256)
        parser.add_argument("--actor-hidden-sizes", type=int, nargs="*", default=[64, 64])
        parser.add_argument("--critic-hidden-sizes", type=int, nargs="*", default=[64, 64])
        parser.add_argument("--training-num", type=int, default=20)
        parser.add_argument("--test-num", type=int, default=2)
        parser.add_argument("--logdir", type=str, default="Log")
        parser.add_argument("--render", type=float, default=0.0)
        parser.add_argument(
            "--device",
            type=str,
            default="cuda:1" if torch.cuda.is_available() else "cpu",
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
        parser.add_argument("--buffer_alpha", type=float, default=0.6)
        parser.add_argument("--beta", type=float, default=0.4)
        parser.add_argument("--save-interval", type=int, default=4)
        return parser.parse_known_args()[0]
    args =get_args()
    args.state_shape = (3, 250, 250)
    args.action_space = spaces.Box(
        low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float64
    )
    args.action_shape = (2,)

    time = datetime.datetime.now()  # 获取当前时间

    # Q_param = V_param = {"hidden_sizes": [64,64,64,64]}
    Q_param = V_param = {"hidden_sizes": [64, 64, 64, 64]}
    net = Track_Net(
        *args.state_shape,
        get_feature=True,
        action_shape=args.action_shape,
        feature_dim=128,
        device=args.device,
    ).to(args.device)
    # net = Net(args.state_shape, hidden_sizes=[64,64], device=args.device)
    # net=Track_Net(
    #     *args.state_shape,
    #     get_feature = True,
    #     action_shape=args.action_shape,
    #     feature_dim=128,
    #     device=args.device,
    # ).to(args.device)
    actor = ActorProb(
        net,
        args.action_shape,
        unbounded=True,
        hidden_sizes=args.actor_hidden_sizes,
        device=args.device,
    ).to(args.device)
    critic = Critic(
        net,
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
        action_bound_method="clip",
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

def get_C_STAR_policy(path=None):
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
    CPNet = STAR(input_dim=np.prod(args.state_shape)+32, feature_dim=256, device=args.device,hidden_dim=[128,128])
    # Q_param = V_param = {"hidden_sizes": [64, 64]}
    # CPNet = Critic_Preprocess_Net(
    #     input_dim=34, action_shape=2, device=args.device, num_atoms=2, dueling_param=(Q_param, V_param), feature_dim=64, hidden_size=64)
    # APNet=STAR(input_dim=np.prod(args.state_shape)+32, feature_dim=256, device=args.device,hidden_dim=[128,128])
    APNet=Actor_Preprocess_Net(
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

def get_C_STAR_rnn_policy(path=None):
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
    CPNet = STAR_rnn(input_dim=np.prod(args.state_shape)+32, feature_dim=256, device=args.device,hidden_dim=[128,128])
    # Q_param = V_param = {"hidden_sizes": [64, 64]}
    # CPNet = Critic_Preprocess_Net(
    #     input_dim=34, action_shape=2, device=args.device, num_atoms=2, dueling_param=(Q_param, V_param), feature_dim=64, hidden_size=64)
    # APNet=STAR(input_dim=np.prod(args.state_shape)+32, feature_dim=256, device=args.device,hidden_dim=[128,128])
    APNet=Actor_Preprocess_Net(
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

def get_C_STAR_mha_policy(path=None):
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
    CPNet = STAR_mha(input_dim=np.prod(args.state_shape)+32, feature_dim=256, device=args.device,hidden_dim=[128,128])
    # Q_param = V_param = {"hidden_sizes": [64, 64]}
    # CPNet = Critic_Preprocess_Net(
    #     input_dim=34, action_shape=2, device=args.device, num_atoms=2, dueling_param=(Q_param, V_param), feature_dim=64, hidden_size=64)
    # APNet=STAR(input_dim=np.prod(args.state_shape)+32, feature_dim=256, device=args.device,hidden_dim=[128,128])
    APNet=Actor_Preprocess_Net(
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

def get_C_STAR_only_policy(path=None):
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
    CPNet = STAR_only(input_dim=np.prod(args.state_shape), feature_dim=256, device=args.device,hidden_dim=[128,128])
    # Q_param = V_param = {"hidden_sizes": [64, 64]}
    # CPNet = Critic_Preprocess_Net(
    #     input_dim=34, action_shape=2, device=args.device, num_atoms=2, dueling_param=(Q_param, V_param), feature_dim=64, hidden_size=64)
    # APNet=STAR(input_dim=np.prod(args.state_shape)+32, feature_dim=256, device=args.device,hidden_dim=[128,128])
    APNet=Actor_Preprocess_Net(
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

def get_Both_STAR_policy(path=None):
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
    CPNet = STAR(input_dim=np.prod(args.state_shape)+32, feature_dim=256, device=args.device,hidden_dim=[128,128])
    # Q_param = V_param = {"hidden_sizes": [64, 64]}
    # CPNet = Critic_Preprocess_Net(
    #     input_dim=34, action_shape=2, device=args.device, num_atoms=2, dueling_param=(Q_param, V_param), feature_dim=64, hidden_size=64)
    APNet=STAR(input_dim=np.prod(args.state_shape)+32, feature_dim=256, device=args.device,hidden_dim=[128,128])
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

def get_Both_STAR_rnn_policy(path=None):
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
    CPNet = STAR_rnn(input_dim=np.prod(args.state_shape)+32, feature_dim=256, device=args.device,hidden_dim=[128,128])
    # Q_param = V_param = {"hidden_sizes": [64, 64]}
    # CPNet = Critic_Preprocess_Net(
    #     input_dim=34, action_shape=2, device=args.device, num_atoms=2, dueling_param=(Q_param, V_param), feature_dim=64, hidden_size=64)
    APNet=STAR_rnn(input_dim=np.prod(args.state_shape)+32, feature_dim=256, device=args.device,hidden_dim=[128,128])
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

def get_Both_STAR_mha_policy(path=None):
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
    CPNet = STAR_mha(input_dim=np.prod(args.state_shape)+32, feature_dim=256, device=args.device,hidden_dim=[128,128])
    # Q_param = V_param = {"hidden_sizes": [64, 64]}
    # CPNet = Critic_Preprocess_Net(
    #     input_dim=34, action_shape=2, device=args.device, num_atoms=2, dueling_param=(Q_param, V_param), feature_dim=64, hidden_size=64)
    APNet=STAR_mha(input_dim=np.prod(args.state_shape)+32, feature_dim=256, device=args.device,hidden_dim=[128,128])
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

def get_Both_STAR_only_policy(path=None):
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


def get_A_STAR_policy(path=None):
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
    # CPNet = STAR(input_dim=np.prod(args.state_shape)+32, feature_dim=256, device=args.device,hidden_dim=[128,128])
    # Q_param = V_param = {"hidden_sizes": [64, 64]}
    # CPNet = Critic_Preprocess_Net(
    #     input_dim=34, action_shape=2, device=args.device, num_atoms=2, dueling_param=(Q_param, V_param), feature_dim=64, hidden_size=64)
    APNet=STAR(input_dim=np.prod(args.state_shape)+32, feature_dim=256, device=args.device,hidden_dim=[128,128])
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

def get_A_STAR_mha_policy(path=None):
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
    # CPNet = STAR(input_dim=np.prod(args.state_shape)+32, feature_dim=256, device=args.device,hidden_dim=[128,128])
    # Q_param = V_param = {"hidden_sizes": [64, 64]}
    # CPNet = Critic_Preprocess_Net(
    #     input_dim=34, action_shape=2, device=args.device, num_atoms=2, dueling_param=(Q_param, V_param), feature_dim=64, hidden_size=64)
    APNet=STAR_mha(input_dim=np.prod(args.state_shape)+32, feature_dim=256, device=args.device,hidden_dim=[128,128])
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
def get_A_STAR_rnn_policy(path=None):
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
    # CPNet = STAR(input_dim=np.prod(args.state_shape)+32, feature_dim=256, device=args.device,hidden_dim=[128,128])
    # Q_param = V_param = {"hidden_sizes": [64, 64]}
    # CPNet = Critic_Preprocess_Net(
    #     input_dim=34, action_shape=2, device=args.device, num_atoms=2, dueling_param=(Q_param, V_param), feature_dim=64, hidden_size=64)
    APNet=STAR_rnn(input_dim=np.prod(args.state_shape)+32, feature_dim=256, device=args.device,hidden_dim=[128,128])
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
def get_A_STAR_only_policy(path=None):
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
    # CPNet = STAR(input_dim=np.prod(args.state_shape)+32, feature_dim=256, device=args.device,hidden_dim=[128,128])
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

def get_A_STAR_C_STAR_Only_policy(path=None):
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
    APNet=STAR(input_dim=np.prod(args.state_shape)+32, feature_dim=256, device=args.device,hidden_dim=[128,128])
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