import argparse
import os
import pprint
import datetime
import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, PrioritizedVectorReplayBuffer, VectorReplayBuffer
from tianshou.env import DummyVectorEnv,ShmemVectorEnv,SubprocVectorEnv
from tianshou.policy import DQNPolicy
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from Nets.CNN_net import CNN_Net
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Dynamic-v0")
    parser.add_argument("--test", type=bool, default=False)
    parser.add_argument("--headless", type=bool, default=True)
    parser.add_argument("--load-model", type=bool, default=False)
    parser.add_argument("--reward-threshold", type=float, default=3000000)
    parser.add_argument("--seed", type=int, default=1626)
    parser.add_argument("--eps-test", type=float, default=0.05)
    parser.add_argument("--eps-train", type=float, default=0.1)
    parser.add_argument("--buffer-size", type=int, default=20000)
    parser.add_argument("--feature-dim", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--target-update-freq", type=int, default=200)
    parser.add_argument("--epoch", type=int, default=2000)
    parser.add_argument("--step-per-epoch", type=int, default=100000)
    parser.add_argument("--step-per-collect", type=int, default=400)
    parser.add_argument("--update-per-step", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[128, 128])
    parser.add_argument("--training-num", type=int, default=20)
    parser.add_argument("--test-num", type=int, default=2)
    parser.add_argument("--logdir", type=str, default="Log")
    parser.add_argument("--render", type=float, default=0.0)
    parser.add_argument("--prioritized-replay", action="store_true", default=False)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_known_args()[0]


def test_dqn(args=get_args()):
    print("******The programe is run on :",args.device,"******")
    # if args.test==True:
    #     args.load_model=True
    env = gym.make(args.task,headless=args.headless,mode=args.test,algom_type='discrete')
    args.prioritized_replay = True
    args.gamma = 0.95
    args.seed = 1
    args.eps_train =0.2
    
    args.state_shape = (3,250,250)
    args.action_shape = env.action_space.shape or env.action_space.n
    
    # Q_param = V_param = {"hidden_sizes": [128]}
    # model
    Q_param = V_param = {"hidden_sizes": [64,64,64,64]}
    net=CNN_Net(
       *args.state_shape,
       args.action_shape,
       concat= False,
       device = args.device,
       use_dueling = True,
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
    time = datetime.datetime.now()  # 获取当前时间
    log_path = os.path.join(args.logdir, "track_d3qn_image_{}_{}_{}_{}".format(
        time.month, time.day, time.hour, time.minute, time.second))
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    args.model_name = os.path.join(args.logdir,"track_d3qn_image_7_3_16_12","Track_test.pth")
    if args.load_model:
       policy.load_state_dict(torch.load(args.model_name))
       print('Policy load!')
    torch.multiprocessing.set_start_method('spawn')
    

    if args.test==False:
        # train_envs = gym.make(args.task)
        # you can also use tianshou.env.SubprocVectorEnv
        train_envs = SubprocVectorEnv([lambda:env  for _ in range(args.training_num)])
        # test_envs = gym.make(args.task)
        test_envs = SubprocVectorEnv([lambda:env for _ in range(args.test_num)])
        # seed
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        train_envs.seed(args.seed)
        test_envs.seed(args.seed)
            # buffer
        if args.prioritized_replay:
            buf = PrioritizedVectorReplayBuffer(
                args.buffer_size,
                buffer_num=len(train_envs),
                alpha=args.alpha,
                beta=args.beta,
            )
        else:
            buf = VectorReplayBuffer(args.buffer_size, buffer_num=len(train_envs))
                # log
        
        if os.path.exists(log_path) == False:
            os.mkdir(log_path)
        # collector
        train_collector = Collector(policy, train_envs, buf, exploration_noise=True,log_path=log_path,label='train')
        test_collector = Collector(policy, test_envs, exploration_noise=True,log_path=log_path,label='test')
        # policy.set_eps(1)
        train_collector.collect(n_step=args.batch_size * args.training_num)


        writer = SummaryWriter(log_path)
        logger = TensorboardLogger(writer)

        def save_best_fn(policy,str=''):
            torch.save(policy.state_dict(), os.path.join(log_path, "Track_{}.pth".format(str)))

        def stop_fn(mean_rewards):
            return mean_rewards >= args.reward_threshold

        def train_fn(epoch, env_step):
            # eps annnealing, just a demo
            if env_step <= 100000:
                policy.set_eps(args.eps_train)
            elif env_step <= 300000:
                eps = args.eps_train - (env_step - 100000) / 200000 * (0.9 * args.eps_train)
                policy.set_eps(eps)
            else:
                # policy.set_eps(0.1 * args.eps_train)
                policy.set_eps(0.1)

        def test_fn(epoch, env_step):
            policy.set_eps(args.eps_test)

        # trainer
        result = OffpolicyTrainer(
            policy=policy,
            train_collector=train_collector,
            test_collector=test_collector,
            max_epoch=args.epoch,
            step_per_epoch=args.step_per_epoch,
            step_per_collect=args.step_per_collect,
            episode_per_test=args.test_num,
            batch_size=args.batch_size,
            update_per_step=args.update_per_step,
            train_fn=train_fn,
            test_fn=test_fn,
            stop_fn=stop_fn,
            save_best_fn=save_best_fn,
            logger=logger,
        ).run()
        assert stop_fn(result['best_reward'])
        pprint.pprint(result)

    else:
        # Let's watch its performance!
        args.headless=False
        env = gym.make(args.task,headless=args.headless,mode=args.test,algom_type='discrete')
        policy.eval()
        policy.set_eps(args.eps_test)
        collector = Collector(policy, env,log_path=log_path,label='eval')
        result = collector.collect(n_episode=100, render=args.render)
        print(result)
        print(f"Final reward: {result['rew']}, length: {result['len']}")
        #print(f"Final reward: {result['rews'].mean}, length: {result['lens'].mean}")



def test_pdqn(args=get_args()):

    test_dqn(args)


if __name__ == "__main__":

    test_dqn(get_args())