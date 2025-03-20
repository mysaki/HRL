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
# from Nets.Critic_Net import Critic_Preprocess_Net
from Nets.Critic_Net import Critic_Preprocess_Net
from Nets.Star_net_rnn_attention import STAR
from Nets.Star_net import STAR as STAR_ONLY
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
    parser.add_argument("--test-num", type=int, default=2)
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


def test_ppo(args: argparse.Namespace = get_args()) -> None:
    args.state_shape = (36,)
    args.action_space = spaces.Box(
        low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float64
    )
    args.action_shape = (2,)

    time = datetime.datetime.now()  # 获取当前时间
    label = "train" if args.test == False else 'test'
    # log
    log_path = os.path.join(args.logdir, "join_{}_track_ppo_{}_{}_{}_{}".format(
        label,time.month, time.day, time.hour, time.minute, time.second))
    log_model_name = "join_train_track_ppo_12_27_11_28"
    log_mode_path = os.path.join("Log", log_model_name)
    writer = SummaryWriter(log_path)

    # if args.load_model:
    #    policy.load_state_dict(torch.load('./log/20240402/Safe_test.pth'))
    #    print('Policy load!')                                                                                                                                                                                                                                                                                   ')

    #  
    # Q_param = V_param = {"hidden_sizes": [64,64,64,64]}
    # APNet = STAR(
    #     input_dim=np.prod(args.state_shape)+32, device=args.device, feature_dim=256, hidden_dim=[128,128],norm_layers=None)
    # CPNet = Critic_Preprocess_Net(
    #     input_dim=np.prod(args.state_shape), device=args.device, feature_dim=256, hidden_size=[128,128])
    CPNet = STAR_ONLY(input_dim=np.prod(args.state_shape), feature_dim=256, device=args.device,hidden_dim=[128,128])
    # Q_param = V_param = {"hidden_sizes": [64, 64]}
    # CPNet = Critic_Preprocess_Net(
    #     input_dim=34, action_shape=2, device=args.device, num_atoms=2, dueling_param=(Q_param, V_param), feature_dim=64, hidden_size=64)
    APNet = STAR_ONLY(input_dim=np.prod(args.state_shape), feature_dim=256, device=args.device,hidden_dim=[128,128])
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

    def save_best_fn(policy: BasePolicy, label = "save_best") -> None:
        torch.save(policy.state_dict(), os.path.join(
            log_path, "Track_{}.pth".format(label)))
        print('{} policy is saved'.format(label))

    def stop_fn(mean_rewards: float) -> bool:
        return mean_rewards >= args.reward_threshold
    def create_env(env_name, headless, mode):
        return gym.make(env_name, headless=headless, mode=mode)
    if args.test == False:  # trainer
        logger = TensorboardLogger(writer, save_interval=args.save_interval)
        # envs = [lambda: create_env(args.task, args.headless, args.test) for _ in range(args.training_num)]
        # track_envs = [lambda: create_env("Dynamic-v0", args.headless, args.test) for _ in range(args.training_num)]

        # train_envs = ShmemVectorEnv(envs + track_envs)
        envs = []
        # for i in range(args.training_num):
        #     envs.append(lambda: create_env(args.task, args.headless, args.test) if i % 2 == 0 else create_env("Dynamic-v0", args.headless, args.test))
        for i in range(args.training_num):
            if i % 2 == 0:
                envs.append(lambda i=i: create_env(args.task, args.headless, args.test))
            else:
                envs.append(lambda i=i: create_env("Dynamic-v0", args.headless, args.test))
        train_envs = ShmemVectorEnv(envs)
        # test_envs = [lambda: create_env(args.task, args.headless, args.test) for _ in range(args.test_num)]
        # test_track_envs = [lambda: create_env("Dynamic-v0", args.headless, args.test) for _ in range(args.test_num)]
        tenvs = []
        # for i in range(args.test_num):
        #     tenvs.append(lambda: create_env(args.task, args.headless, args.test) if i % 2 == 0 else create_env("Dynamic-v0", args.headless, args.test))
        for i in range(args.test_num):
            if i % 2 == 0:
                tenvs.append(lambda i=i: create_env(args.task, args.headless, args.test))
            else:
                tenvs.append(lambda i=i: create_env("Dynamic-v0", args.headless, args.test))
        test_envs = ShmemVectorEnv(tenvs)
        # test_envs = ShmemVectorEnv(test_envs + test_track_envs)
        # seed
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        train_envs.seed(args.seed)
        test_envs.seed(args.seed)
        buffer = PrioritizedVectorReplayBuffer(args.buffer_size, len(
                train_envs), alpha=args.buffer_alpha, beta=args.beta)
        # collector
        train_collector = Collector(
            policy,
            train_envs,
            buffer,
            log_path=log_path,
            label='train',
        )
        # train_track_collector = Collector(
        #     policy,
        #     train_track_envs,
        #     buffer,
        #     log_path=log_path,
        #     label='train',
        # )
        test_collector = Collector(
            policy, test_envs, log_path=log_path, label='test')

        # trainer
        result = OnpolicyTrainer(
            policy=policy,
            train_collector=train_collector,
            test_collector=test_collector,
            # side_task_train_collector=None,
            # side_task_test_collector=None,
            max_epoch=args.epoch,
            step_per_epoch=args.step_per_epoch,
            repeat_per_collect=args.repeat_per_collect,
            episode_per_test=args.test_num*2,
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
        env = gym.make("Navigation-v0", headless=args.headless,mode=args.test)
        policy.eval()
        collector = Collector(policy, env, log_path=log_path, label='try')
        result = collector.collect(render=args.render, n_episode=1000)
        # 将字典转换为DataFrame
        df = pd.DataFrame(result)
        # 保存为Excel文件
        df.to_excel(
            "result_{}_{}_{}_{}_{}_{}.xlsx".format(
                log_model_name,time.month, time.day, time.hour, time.minute, time.second
            ),
            index=False,
        )
        print(result)
        print(f"Final reward: {result['rew']}, length: {result['len']}")


def test_ppo_resume(args: argparse.Namespace = get_args()) -> None:
    args.resume = True
    test_ppo(args)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    test_ppo()
#
