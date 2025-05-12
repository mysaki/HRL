import argparse
import os
import pprint
import datetime
import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Distribution, Independent, Normal
from torch.utils.tensorboard import SummaryWriter
from gymnasium import spaces
from tianshou.data import Collector, PrioritizedVectorReplayBuffer
from tianshou.env import ShmemVectorEnv
from tianshou.policy import SACPolicy
from tianshou.policy.base import BasePolicy
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.continuous import ActorProb, Critic
from Nets.Actor_Net import Actor_Preprocess_Net
# from Nets.Critic_Net import Critic_Preprocess_Net
from Nets.Critic_Net import Critic_Preprocess_Net
from tianshou.env.worker import SubprocEnvWorker

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--headless', type=bool, default=True)
    parser.add_argument("--task", type=str, default="Navigation-v0")
    parser.add_argument('--test', type=bool, default=False)
    parser.add_argument("--load-model", type=bool, default=False)
    parser.add_argument("--reward-threshold", type=float, default=150000000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--buffer-size", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--step-per-epoch", type=int, default=24000)
    parser.add_argument("--episode-per-collect", type=int, default=10)
    parser.add_argument("--repeat-per-collect", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--actor-hidden-sizes", type=int,
                        nargs="*", default=[64, 64])
    parser.add_argument("--critic-hidden-sizes", type=int,
                        nargs="*", default=[64, 64])
    parser.add_argument("--training-num", type=int, default=40)
    parser.add_argument("--test-num", type=int, default=2)
    parser.add_argument("--logdir", type=str, default="Log")
    parser.add_argument("--render", type=float, default=0.0)
    parser.add_argument('--buffer_alpha', type=float, default=0.6)
    parser.add_argument('--beta', type=float, default=0.4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    # sac special
    parser.add_argument("--actor-lr", type=float, default=1e-3)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--auto-alpha", type=int, default=1)
    parser.add_argument("--alpha-lr", type=float, default=1e-4)
    parser.add_argument("--update-per-step", type=float, default=0.1)
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--save-interval", type=int, default=4)
    return parser.parse_known_args()[0]


def test_ppo(args: argparse.Namespace = get_args()) -> None:
    args.state_shape = (34,)
    args.action_space = spaces.Box(
        low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float64
    )
    args.action_shape = (2,)
    action_dim = 2
    time = datetime.datetime.now()  # 获取当前时间
    # log
    log_path = os.path.join(args.logdir, "sac_{}_{}_{}_{}".format(
        time.month, time.day, time.hour, time.minute, time.second))
    log_mode_path = os.path.join(args.logdir, "ppo_6_1_20_26")
    writer = SummaryWriter(log_path)
    APNet = Actor_Preprocess_Net(
        input_dim=34, device=args.device, feature_dim=64, hidden_size=[64]).to(args.device)
    # model
    actor = ActorProb(APNet, args.action_shape, device=args.device, unbounded=True).to(args.device)
    # actor = torch.nn.DataParallel(actor,[0,1]).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    net_c1 = Critic_Preprocess_Net(input_dim=34, device=args.device,hidden_size=[64]).to(args.device)
    critic1 = Critic(net_c1, device=args.device).to(args.device)
    # critic1 = torch.nn.DataParallel(critic1,[0,1]).to(args.device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    net_c2 = Critic_Preprocess_Net(input_dim=34, device=args.device,hidden_size=[64]).to(args.device)
    critic2 = Critic(net_c2, device=args.device).to(args.device)
    # critic2 = torch.nn.DataParallel(critic2,[0,1]).to(args.device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)
    if args.auto_alpha:
        target_entropy = -action_dim
        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        args.alpha = (target_entropy, log_alpha, alpha_optim)

    policy= SACPolicy(
        actor=actor,
        actor_optim=actor_optim,
        critic1=critic1,
        critic1_optim=critic1_optim,
        critic2=critic2,
        critic2_optim=critic2_optim,
        tau=args.tau,
        gamma=args.gamma,
        alpha=args.alpha,
        estimation_step=args.n_step,
        action_space=args.action_space,
    )

    if args.load_model:

        # load from existing checkpoint
        print(f"Loading agent under {log_mode_path}")
        ckpt_path = os.path.join(log_mode_path, "Track_train.pth")
        if os.path.exists(ckpt_path):
            # checkpoint = torch.load(ckpt_path, map_location=args.device)
            # policy.load_state_dict(checkpoint["model"])
            # optim.load_state_dict(checkpoint["optim"])
            policy.load_state_dict(torch.load(ckpt_path))
            print('Policy load!')
        else:
            print("Fail to restore policy.")

    def save_best_fn(policy: BasePolicy, label) -> None:
        torch.save(policy.state_dict(), os.path.join(
            log_path, "Track_{}.pth".format(label)))
        print('{} policy is saved'.format(label))

    def stop_fn(mean_rewards: float) -> bool:
        return mean_rewards >= args.reward_threshold
    if args.test == False:  # trainer
        logger = TensorboardLogger(writer, save_interval=args.save_interval)
        env = gym.make(args.task, headless=args.headless,mode=args.test)
        train_envs = ShmemVectorEnv(
            [lambda: env for _ in range(args.training_num)])
        test_envs = ShmemVectorEnv([lambda: env for _ in range(args.test_num)])
        # seed
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        train_envs.seed(args.seed)
        test_envs.seed(args.seed)

        # collector
        train_collector = Collector(
            policy,
            train_envs,
            PrioritizedVectorReplayBuffer(args.buffer_size, len(
                train_envs), alpha=args.buffer_alpha, beta=args.beta),
            log_path=log_path,
            label='train',
        )
        test_collector = Collector(
            policy, test_envs, log_path=log_path, label='test')

        # trainer
        result = OnpolicyTrainer(
            policy=policy,
            train_collector=train_collector,
            test_collector=test_collector,
            max_epoch=args.epoch,
            step_per_epoch=args.step_per_epoch,
            repeat_per_collect=args.repeat_per_collect,
            episode_per_test=args.test_num,
            batch_size=args.batch_size,
            episode_per_collect=args.episode_per_collect,
            stop_fn=stop_fn,
            save_best_fn=save_best_fn,
            logger=logger,
            resume_from_log=args.resume,
            # save_checkpoint_fn=save_checkpoint_fn,
        ).run()
        assert stop_fn(result['best_reward'])
        pprint.pprint(result)

    else:
        # Let's watch its performance!
        args.headless = False
        env = gym.make(args.task, headless=args.headless,mode=args.test)
        policy.eval()
        collector = Collector(policy, env, log_path=log_path, label='try')
        result = collector.collect(render=args.render, n_episode=100)
        print(f"Final reward: {result['rew']}, length: {result['len']}")


def test_ppo_resume(args: argparse.Namespace = get_args()) -> None:
    args.resume = True
    test_ppo(args)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    test_ppo()
