import argparse
import os
import pprint
import datetime
import threading
import time
import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Distribution, Independent, Normal
from torch.utils.tensorboard import SummaryWriter
from gymnasium import spaces
from tianshou.data import Collector, PrioritizedVectorReplayBuffer
from tianshou.env import ShmemVectorEnv
from tianshou.policy import PPOPolicy,DQNPolicy
from tianshou.policy.base import BasePolicy
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.continuous import ActorProb, Critic
from Nets.Actor_Net import Actor_Preprocess_Net
# from Nets.Critic_Net import Critic_Preprocess_Net
from Nets.Critic_Net import Critic_Preprocess_Net
from Nets.Track_net import Track_Net
from tianshou.data import Batch
import config_ppo
last_dist=1.0

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--headless', type=bool, default=False)
    parser.add_argument("--task", type=str, default="Joint-v0")
    parser.add_argument('--test', type=bool, default=True)
    parser.add_argument("--load-model", type=bool, default=True)
    parser.add_argument("--reward-threshold", type=float, default=150000000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--buffer-size", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--step-per-epoch", type=int, default=150000)
    parser.add_argument("--episode-per-collect", type=int, default=16)
    parser.add_argument("--repeat-per-collect", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--actor-hidden-sizes", type=int,
                        nargs="*", default=[64, 64])
    parser.add_argument("--critic-hidden-sizes", type=int,
                        nargs="*", default=[64, 64])
    parser.add_argument("--training-num", type=int, default=30)
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


def build_agent_net(mode, args: argparse.Namespace = get_args(),algom_type = "continue",env =None) -> None:
    args.state_shape = (34,)
    args.action_shape = env.action_space.shape or env.action_space.n
    if algom_type == "continue":
        args.action_space = spaces.Box(
            low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float64
        )
        APNet = Actor_Preprocess_Net(
        input_dim=np.prod(args.state_shape), device=args.device, feature_dim=64, hidden_size=[64])
        # Q_param = V_param = {"hidden_sizes": [64, 64]}
        # CPNet = Critic_Preprocess_Net(
        #     input_dim=34, action_shape=2, device=args.device, num_atoms=2, dueling_param=(Q_param, V_param), feature_dim=64, hidden_size=64)
        CPNet=Critic_Preprocess_Net(input_dim=np.prod(args.state_shape), device=args.device,hidden_size=[64])
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
    elif algom_type == "discrete":
        args.action_space = spaces.Discrete(5)
        args.feature_dim = 512
        args.n_step = 3
        args.target_update_freq = 200
        Q_param = V_param = {"hidden_sizes": [64,64,64,64]}
        net=Track_Net(
        34,
        args.action_shape, 
        concat= False,
        device = args.device,
        use_dueling = True,
        get_feature=True,
        dueling_param=(Q_param, V_param),
        feature_dim=args.feature_dim
        ).to(args.device)
        optim = torch.optim.Adam(net.parameters(), lr=args.lr)
        policy = DQNPolicy(
            model=net,
            optim=optim,
            discount_factor=args.gamma,
            estimation_step=args.n_step,
            target_update_freq=args.target_update_freq,
            action_space=env.action_space,
        )
    
    # if args.load_model:
    #    policy.load_state_dict(torch.load('./log/20240402/Safe_test.pth'))
    #    print('Policy load!')                                                                                                                                                                                                                                                                                   ')

    # model
    # Q_param = V_param = {"hidden_sizes": [64,64,64,64]}


    if mode == "track":
        if algom_type == "continue":
            log_mode_path='Log/track_ppo_6_16_0_30'
        elif algom_type == "discrete":
            log_mode_path='Log/track_d3qn_7_3_8_7'
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
    elif mode == "obstacle":
        if algom_type == "continue":
            log_mode_path='Log/nav_ppo_6_17_11_27'
        elif algom_type == "discrete":
            log_mode_path='Log/nav_d3qn_7_7_15_28'
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
    return policy

def mode_choose(state,env):
    """
    根据当前环境信息及历史环境信息变化来决定进入哪一个模式
    """
    global last_dist
    theta = state[0]*180
    
    if -101.25<theta<-78.75:
        bin_idx=8
    elif -78.75 < theta <-56.25:
        bin_idx=7
    elif -56.25 <theta <-33.75:
        bin_idx=6
    elif -33.75 < theta < -11.25:
        bin_idx=5
    elif -11.25 <theta < 11.25:
        bin_idx = 4
    elif 11.25 <theta <33.75:
        bin_idx=3
    elif 33.75 < theta < 56.25:
        bin_idx = 2
    elif 56.25 <theta <78.75:
        bin_idx = 1
    elif 78.75 < theta < 101.25:
        bin_idx = 0
    elif 101.25 <theta < 123.75:
        bin_idx=15
    elif 123.75 < theta <146.25:
        bin_idx = 14
    elif 146.25 < theta <168.75:
        bin_idx = 13
    elif 168.75<theta<=180 or -180<=theta<-168.75:
        bin_idx = 12
    elif -168.75 < theta < -146.25:
        bin_idx = 11
    elif -146.25 < theta <-123.75:
        bin_idx = 10
    elif -123.75 < theta <= -101.25:
        bin_idx = 9
    
    now_dist = state[2+bin_idx]*config_ppo.maxDetectDistance
    # if env.task == "track":
    #     now_dist = state[1]*config.maxDetectDistance+config.best_distance
    # elif env.task == "obstacle":
    #     now_dist = state[1]*env.init_distance
    # print("relative_angle:",theta,"bin_idx:",bin_idx)
    delta_dist = now_dist - last_dist
    dist_change_ratio=now_dist/last_dist
    # print("last_dist",last_dist,"now_dist",now_dist,"now_dist/last_dist",dist_change_ratio,"delta_dist",delta_dist)
    last_dist = now_dist
    if dist_change_ratio<0.5 and delta_dist < 0 :
        print("change to mode:'obstacle'")
        return "obstacle"
    else:
        return "track"
def get_input(prompt, stop_event):
    while not stop_event.is_set():
        user_input = input(prompt)
        if user_input.lower() == "o":
            stop_event.set()
        else:
            print(f"Received input: {user_input}")

def test_agent(args: argparse.Namespace = get_args(),algom_type = "continue") -> None:

    env = gym.make("Joint-v0", headless=args.headless,mode=args.test,algom_type = algom_type)
    track_agent = build_agent_net(mode = "track",algom_type=algom_type,env=env)
    obs_agent = build_agent_net(mode = "obstacle",algom_type=algom_type,env=env)
    truncated = False
    done = False
    state,_=env.reset()
    batch= Batch(
            obs={},
            act={},
            rew={},
            terminated={},
            truncated={},
            done={},
            obs_next={},
            info={},
            policy={}
        )
    batch.update(obs=[state])
    # stop_event = threading.Event()
    # input_thread = threading.Thread(target=get_input, args=("Enter command: ", stop_event))
    
    # input_thread.start()
    mode = "track"
    step_sum = 0
    success_num = 0
    for i in range(10):
        epi_step =0
        done=False
        truncated =False
        while not done and not truncated and epi_step < config_ppo.max_count:
            # mode=mode_choose(batch.obs[0],env)
            # if stop_event.is_set():
            #     mode = "obstacle"
            if mode == "obstacle":
                print("Obstacle mode!")
                env.change_task()
                obs_done=False
                obs_truncated=False
                while mode == 'obstacle' and not obs_done and not obs_truncated and epi_step < 2000:
                    act=obs_agent(batch)
                    next_state, reward, obs_done, obs_truncated, info = env.step(act.act[0])
                    batch.update(obs=[next_state])
                    mode = info['task']
                    epi_step+=1
            env.track_task()
            act=track_agent(batch)
            next_state, reward, done, truncated, info = env.step(act.act[0])
            mode = info['task']
            # print(next_state[2:18])
            batch.update(obs=[next_state])
            epi_step+=1
        if epi_step >=config_ppo.max_count:
            success_num+=1
        step_sum+=epi_step
        print("average step:",step_sum/(i+1),"success:",success_num)
        env.reset()
    env.shutdown()
if __name__ == "__main__":
    test_agent(algom_type = "continue")
