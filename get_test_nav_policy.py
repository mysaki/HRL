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
from Nets.Star_net_rnn import STAR as STAR_rnn
from Nets.Star_net_attention import STAR as STAR_mha
from Nets.Star_net import STAR as STAR_only
import time
class Bacteria_policy():
    def __init__(self,obstacles_coordinates,tracker_coordinates,target_coordinates):
        self.start_time = time.time()
        # 初始化一个字典 par 用于存储所有参数
        self.par = {}
        # 机器人的初始坐标 [x, y]
        self.par['robot_coordinates'] = tracker_coordinates
        # 障碍物的坐标，以二维数组形式存储
        self.par['obstacles_coordinates'] = np.array(obstacles_coordinates)
        # 用于记录已检测到的障碍物坐标，初始为空数组
        self.par["obstacles_coordinates_detected"] = np.empty((len(obstacles_coordinates), 3))
        # 用于记录检测到的每个障碍物的最小距离，第一行是生成的障碍物的索引，第二行是记录到该障碍物的最小距离，初始为空数组
        self.par["detected_obstacles_distances"] = np.empty((2, len(obstacles_coordinates)))
        # 目标的坐标 [x, y]
        self.par['target_coordinates'] = target_coordinates
        # print("target_coordinates",target_coordinates)
        # 传感器的探测范围
        self.par['sensor_range'] = 4
        # 细菌点的步长，为传感器范围的 0.05 倍
        self.par['step_size'] = 0.1
        # 障碍物势函数中 alpha 的取值范围
        self.par['alpha_o_range'] = np.arange(0.1, 50.1, 0.1)
        # 障碍物势函数中 mu 的取值范围
        self.par['mu_o_range'] = np.arange(1, 1001)
        # 用于存储到障碍物的平均距离，初始化为 0
        self.par['avg_r'] = np.zeros(1)
        # 用于存储到障碍物的距离总和，初始化为 0
        self.par['r_sum'] = np.zeros(1)
        # 存储最终用于机器人的障碍物势函数的 alpha 和 mu 值
        self.par['obstacle_robot'] = [1, 1]
        # 存储最终用于选定细菌点的障碍物势函数的 alpha 和 mu 值
        self.par['obstacle_bacteria'] = [1, 1]
        # 目标势函数的 [alpha, mu] 值
        self.par['target'] = [100000, 0.1]
        # 细菌点的数量
        self.par['bacteria_no'] = 60
        # 细菌点之间的角度间隔
        self.par['bacteria_degree'] = 360 / self.par['bacteria_no']
        # 细菌点的角度数组
        self.par['bacteria_angles'] = np.arange(self.par['bacteria_degree'], 360 + self.par['bacteria_degree'], self.par['bacteria_degree'])
        # 势函数计算的距离下限，低于此距离障碍物势为无穷大
        self.par['potential_lower_distance_limit'] = 0.1
        # 势函数计算的距离上限，高于此距离障碍物势为 0
        self.par['potential_upper_distance_limit'] = 4
        # 安全距离，若机器人到任何检测到的障碍物的距离低于此值，则终止运行（碰撞情况）
        self.par['safety_margin'] = 0.25
        # 找到合适细菌点时的提示信息
        self.par['confirm'] = 'Solution Found'
        # 未找到合适细菌点时的提示信息
        self.par['error'] = 'Solution Not Found'
        # 计算机器人到目标的距离
        self.par['RDTT'] = np.sqrt((self.par['robot_coordinates'][0] - self.par['target_coordinates'][0])**2 +
                            (self.par['robot_coordinates'][1] - self.par['target_coordinates'][1])**2)
        # 初始化势函数值
        self.par['J_obstRT'] = 0

        # 存储变量的初始化
        # 存储每次成功移动决策时势函数的误差
        self.par['err_J_sto_s'] = np.zeros(1)
        # 存储每次迭代中具有最小到目标距离的细菌点的势函数误差，用于分析和绘图
        self.par['err_J_sto'] = np.zeros(1)
        # 存储每次迭代中细菌点的最小势函数误差
        self.par['err_J_sto_f'] = np.zeros(1)
        # 存储每次迭代中机器人的总势函数值
        self.par['JT_sto'] = np.zeros(1)
        # 存储每次循环中具有最小到目标距离的细菌点的势函数值，用于分析和绘图
        self.par['JT_bacteria_sto'] = np.zeros(1)
        # 存储每次循环中细菌点的最小势函数误差
        self.par['JT_bacteria_sto_f'] = np.zeros(1)
        # 记录机器人移动的次数（成功迭代的次数）
        self.par['move_count'] = 0
        # 记录迭代的次数（执行循环的次数）
        self.par['loop'] = 0
        # 记录检测到的障碍物的数量
        self.par['det'] = 0

    def Bacteria_Bacteria_Random_Selection(self):
        # 随机选择细菌索引（Python 索引从 0 开始）
        i = np.random.randint(0, self.par["bacteria_no"])  # 生成 0 到 bacteria_no-1 的整数

        # 检查势能是否非无穷大
        # 更新机器人坐标并添加高斯噪声
        self.par["robot_coordinates"][0] = self.par["bx"][i] + 0.1 * np.random.randn()
        self.par["robot_coordinates"][1] = self.par["by"][i] + 0.1 * np.random.randn()
        self.par["check"] = 1  # 标记选择成功

    def Get_Bacterias(self):
        # 初始化细菌点的 x 和 y 坐标数组
        self.par["bx"] = np.zeros(self.par["bacteria_no"])
        self.par["by"] = np.zeros(self.par["bacteria_no"])
        # print("机器人坐标：",self.par["robot_coordinates"])
        # 计算每个细菌点的坐标
        for i in range(self.par["bacteria_no"]):
            # 根据机器人坐标、步长和细菌点角度计算细菌点的 x 坐标
            self.par["bx"][i] = self.par["robot_coordinates"][0] + (
                self.par["step_size"] * np.cos(np.radians(self.par["bacteria_angles"][i]))
            )
            # 根据机器人坐标、步长和细菌点角度计算细菌点的 y 坐标
            self.par["by"][i] = self.par["robot_coordinates"][1] + (
                self.par["step_size"] * np.sin(np.radians(self.par["bacteria_angles"][i]))
            )
        # print("细菌坐标：",self.par["bx"],self.par["by"])
    def Bacteria_Bacteria_Selection(self):
        # 初始化误差数组
        err_DTT = np.zeros(self.par["bacteria_no"])
        err_J = np.zeros(self.par["bacteria_no"])

        # 计算每个细菌的误差
        for i in range(self.par["bacteria_no"]):
            err_DTT[i] = self.par["BDTT"][i] - self.par["RDTT"]
            err_J[i] = self.par["J_BT"][i] - self.par["J_RT"]

        self.par["check"] = 0  # 初始标记未找到解
        # 遍历所有细菌尝试选择
        for _ in range(self.par["bacteria_no"]):
            mi = np.argmin(err_DTT)  # 找到当前最小距离误差的索引
            if err_J[mi] < 0:
                # 更新机器人坐标并添加噪声
                self.par["robot_coordinates"][0] = self.par["bx"][mi] + 0.1 * np.random.randn()
                self.par["robot_coordinates"][1] = self.par["by"][mi] + 0.1 * np.random.randn()
                self.par["check"] = 1  # 标记成功选择
                break
            else:
                err_DTT[mi] = np.inf  # 排除当前最小值，继续寻找下一个


    def Bacteria_Calculate_Distances_Obstacles(self):
        # 遍历检测到的障碍物距离矩阵的每一列
        for i in range(self.par['det']):
            # 计算机器人与当前障碍物之间的欧几里得距离
            # par['robot_coordinates'] 是机器人的坐标，par['obstacles_coordinates'] 是障碍物的坐标
            # par['detected_obstacles_distances'][0, i] 表示当前障碍物在障碍物坐标矩阵中的索引
            distance = np.sqrt((self.par['robot_coordinates'][0] - self.par['obstacles_coordinates'][int(self.par['detected_obstacles_distances'][0, i]) - 1, 0])**2 +
                            (self.par['robot_coordinates'][1] - self.par['obstacles_coordinates'][int(self.par['detected_obstacles_distances'][0, i]) - 1, 1])**2)-self.par['obstacles_coordinates'][int(self.par['detected_obstacles_distances'][0, i]) - 1, 2]
            # 如果计算得到的距离小于之前记录的该障碍物的检测距离
            if distance < self.par['detected_obstacles_distances'][1, i]:
                # 则更新该障碍物的检测距离为新计算的距离
                self.par['detected_obstacles_distances'][1, i] = distance
        
    def Bacteria_Check_Safety(self):
        # 初始化检查标志为 1，表示安全
        self.par['check'] = 1
        # 遍历检测到的障碍物坐标矩阵的每一行
        for i in range(self.par['obstacles_coordinates_detected'].shape[0]):
            # 计算机器人与当前障碍物之间的欧几里得距离
            # print("check",i,par["obstacles_coordinates_detected"][i])
            distance = np.sqrt((self.par['robot_coordinates'][0] - self.par['obstacles_coordinates_detected'][i, 0])**2 +
                            (self.par['robot_coordinates'][1] - self.par['obstacles_coordinates_detected'][i, 1])**2)
            # 如果计算得到的距离小于安全距离
            if distance < self.par["obstacles_coordinates_detected"][i, 2]:
                # 则将检查标志设为 0，表示不安全
                self.par['check'] = 0
                # 一旦发现不安全情况，跳出循环
                break
    def Bacteria_Detect_Obstacles(self):
        # 遍历所有障碍物的坐标
        for j in range(self.par['obstacles_coordinates'].shape[0]):
            # 计算机器人与当前障碍物之间的欧几里得距离
            distance = np.sqrt((self.par['robot_coordinates'][0] - self.par['obstacles_coordinates'][j, 0])**2 +
                            (self.par['robot_coordinates'][1] - self.par['obstacles_coordinates'][j, 1])**2)
            # print(self.par['robot_coordinates'],self.par['obstacles_coordinates'][j],distance)
            # 检查当前障碍物是否在传感器范围内且未被检测到过
            if distance-self.par['obstacles_coordinates'][j, 2] <= self.par['sensor_range']:
                # 如果满足条件，将该障碍物坐标添加到已检测到的障碍物坐标中
                self.par['obstacles_coordinates_detected'][self.par['det']] = self.par['obstacles_coordinates'][j]
                # print("detect", par["obstacles_coordinates_detected"])
                # 记录检测到的障碍物在障碍物坐标数组中的索引
                self.par['detected_obstacles_distances'][0, self.par['det']] = j
                # 记录检测到的障碍物与机器人的距离
                self.par['detected_obstacles_distances'][1, self.par['det']] = distance-self.par['obstacles_coordinates'][j, 2]
                # 已检测到的障碍物数量加 1
                self.par['det'] = self.par['det'] + 1
        # print("检测到的障碍物数量：",self.par["det"])
        # print("检测到的障碍物:",self.par['obstacles_coordinates_detected'][:self.par["det"]])

    def Bacteria_Robot_Potential(self):
        # 初始化障碍物对机器人的总势能为 0
        self.par["J_obstRT"] = 0
        # 遍历所有检测到的障碍物坐标
        # print(self.par["obstacles_coordinates_detected"][:self.par["det"]])
        for j in range(self.par["det"]):
            # 计算机器人与当前障碍物的距离
            dist = np.sqrt(
                (self.par["robot_coordinates"][0] - self.par["obstacles_coordinates_detected"][j, 0])
                ** 2
                + (
                    self.par["robot_coordinates"][1]
                    - self.par["obstacles_coordinates_detected"][j, 1]
                )
                ** 2
            )-self.par['obstacles_coordinates_detected'][j, 2]
            # 根据距离计算障碍物对机器人的势能
            if (
                dist >= self.par["potential_lower_distance_limit"]
                and dist <= self.par["potential_upper_distance_limit"]
            ):
                # 当距离在有效范围内，使用指数函数计算势能
                pot_val = self.par["obstacle_robot"][0] * np.exp(
                    -self.par["obstacle_robot"][1]
                    * (
                        np.power(dist,2)
                    )
                )
            elif dist < self.par["potential_lower_distance_limit"]:
                # 当距离小于下限，势能设为无穷大
                pot_val = np.inf
            else:
                # 当距离大于上限，势能设为 0
                pot_val = 0
            # 累加障碍物对机器人的势能
            self.par["J_obstRT"] += pot_val

        # 计算机器人到目标的距离
        self.par["RDTT"] = np.sqrt(
            (self.par["robot_coordinates"][0] - self.par["target_coordinates"][0]) ** 2
            + (self.par["robot_coordinates"][1] - self.par["target_coordinates"][1]) ** 2
        )
        # print("机器人到目标的距离：",self.par["RDTT"])
        # 计算目标对机器人的势能
        self.par["J_GoalRT"] = -self.par["target"][0] * np.exp(
            -self.par["target"][1]
            * (
                (self.par["robot_coordinates"][0] - self.par["target_coordinates"][0]) ** 2
                + (self.par["robot_coordinates"][1] - self.par["target_coordinates"][1]) ** 2
            )
        )

        # 计算机器人受到的总势能
        self.par["J_RT"] = self.par["J_GoalRT"] + self.par["J_obstRT"]
        # print("机器人势能：")
        # print(self.par["J_RT"],self.par["J_GoalRT"],self.par["J_obstRT"])


    def Bacteria_Bacteria_Potential(self):
        # 初始化数组（使用 NumPy）
        self.par["J_ObstBT"] = np.zeros(self.par["bacteria_no"])
        self.par["J_GoalBT"] = np.zeros(self.par["bacteria_no"])
        self.par["BDTT"] = np.zeros(self.par["bacteria_no"])
        # print("细菌势能：")
        # 遍历所有细菌
        for i in range(self.par["bacteria_no"]):  # Python 索引从 0 开始
            # 计算细菌到目标的欧氏距离
            dx = self.par["bx"][i] - self.par["target_coordinates"][0]
            dy = self.par["by"][i] - self.par["target_coordinates"][1]
            
            self.par["BDTT"][i] = np.sqrt(dx**2 + dy**2)
            # print("细菌到目标的距离：",self.par["BDTT"][i])
            # 遍历所有检测到的障碍物
            for j in range(self.par['det']):
                # 计算细菌到障碍物的欧氏距离
                obst_x = self.par["obstacles_coordinates_detected"][j, 0]
                obst_y = self.par["obstacles_coordinates_detected"][j, 1]
                dx_obst = self.par["bx"][i] - obst_x
                dy_obst = self.par["by"][i] - obst_y
                dist = np.sqrt(dx_obst**2 + dy_obst**2)-self.par["obstacles_coordinates_detected"][j, 2]
                
                # 根据距离计算势能
                if (
                    dist >= self.par["potential_lower_distance_limit"]
                    and dist <= self.par["potential_upper_distance_limit"]
                ):
                    exponent = -self.par["obstacle_bacteria"][1] * (dist**2)
                    pot_val = self.par["obstacle_bacteria"][0] * np.exp(exponent)

                elif dist < self.par["potential_lower_distance_limit"]:
                    pot_val = np.inf
                else:
                    pot_val = 0
                self.par["J_ObstBT"][i] += pot_val  # 累加障碍物势能

            # 计算目标势能
            exponent_goal = -self.par["target"][1] * (dx**2 + dy**2)
            self.par["J_GoalBT"][i] = -self.par["target"][0] * np.exp(exponent_goal)

        # 总势能 = 目标势能 + 障碍物势能
        self.par["J_BT"] = self.par["J_GoalBT"] + self.par["J_ObstBT"]  # 直接向量化操作（无需循环）
        # print(self.par["J_BT"],self.par["J_GoalBT"],self.par["J_ObstBT"])

    def step(self,robot_coordinates):
        self.par['det'] = 0
        self.par["robot_coordinates"]= robot_coordinates
        self.Get_Bacterias()
        # 记录机器人移动次数
        self.par["move_count"] = self.par["move_count"] + 1
        # # 检查机器人是否安全
        # self.Bacteria_Check_Safety()
        # 检测障碍物
        self.Bacteria_Detect_Obstacles()
        # 计算机器人与障碍物的距离
        self.Bacteria_Calculate_Distances_Obstacles()
        # 计算机器人的势能
        self.Bacteria_Robot_Potential()
        # 计算细菌点的势能
        self.Bacteria_Bacteria_Potential()
        # 选择合适的细菌点
        self.Bacteria_Bacteria_Selection()
        # 记录当前时间
        elapsed_time = time.time() - self.start_time
        self.par["et"] = elapsed_time
        # 如果不安全，进行随机选择
        if self.par["check"] == 0:
            self.Bacteria_Bacteria_Random_Selection()

        # 计算机器人与障碍物的平均距离
        if self.par["detected_obstacles_distances"].size > 0:
            self.par["average_distance"] = np.mean(self.par["detected_obstacles_distances"][1, :])
        else:
            self.par["average_distance"] = 0
        return self.par["robot_coordinates"]
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