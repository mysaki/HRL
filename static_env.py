#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)# 不显示DeprecationWarning
""" main script """
#通用库
import numpy as np
import time
import csv
import random
from sklearn.preprocessing import MinMaxScaler
from copy import copy
import os
from os.path import dirname, join, abspath

import gymnasium as gym
from gymnasium import spaces

from PIL import Image
import matplotlib.pyplot as plt
import cv2
#torch
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
# pyrep
from pyrep.objects.shape import Shape
from pyrep.robots.mobiles.tracker import Tracker
from pyrep.robots.mobiles.target import Target
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep import PyRep
#分割相关
from Segment_net import get_segment_model
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from prompts import Model
#状态提取相关
from Get_super import Transfer
from flow import get_flow
#其他自定义文件
import config_ppo
from regular import correct_vel


# 设置超参数
SCENE_FILE = join(dirname(abspath(__file__)),
                  'Safe_v3.ttt')
BATCH_SIZE = 64
CAPACITY = 10000
LEARNING_RATE = 0.001
GAMMA = 0.99

Initial_positions_target=[[-0.55,-3.2,0.065],
                          [-1,8,0.065],
                          [-2.4,-2.3,0.065],
                          [-10,9,0.065],
                          [1.8,2.5,0.065],
                          [-8,0.9,0.065],
                          [8,-6.5,0.065],
                          ]
Initial_positions_tracker=[[-0.55,-2.2,0.065],
                           [-1,9,0.065],
                           [-2.4,-1.3,0.065],
                           [-10,10,0.065],
                           [1.8,3.5,0.065],
                           [-8,1.9,0.065],
                           [8,-5.5,0.065],
                           ]



class Track(gym.Env):
    # metadata = {
    #     "name": "tictactoe_v3",
    #     "is_parallelizable": False,
    #     "render_fps": 1,
    # }
    metadata = {
        "name": "VAT_v2",
        "is_parallelizable": True,
        "headless":False
    }
    def __init__(
        self,headless:False,
    ):
        super().__init__()
        # ——————参数设置——————
        self.last_position = 0
        self.last_distance=1.0
        self.last_angle_diff=0
        self.w1=0.4
        self.w2=0.6
        self.max_distance=3.0
        self.min_distance=0.4
        self.last_driving_angle=0
        self.last_target_position=[0,0]
        self.last_tracker_position=[0,1]
        self.tracker_target_angle=0
        self.tracker_target_distance=1

        self.headless=headless
        self.time=0
        self.trained=0
        self.count=0
        self.max_count=3000
        self.save_fig_step=0
        self.epoch=0
        self.scaler=MinMaxScaler()

        self.predict_angle=0

        self.segment_model=get_segment_model(True)
        # self.prompts_model=Model()

        self.maxDetectDistance=4
        self.minDetectDistance=0.4
        self.stop_detection=0

        self.action_conduct_time=100
        self.target_action_flag=0
        self.device='cuda'
        
        self.action_space = spaces.Box(
                        low=np.array([-0.46, -1.9]), high=np.array([0.46, 1.9]),  dtype=np.float32
                    )
        self.observation_space = spaces.Box(
                        low=-4.0, high=4.0, shape=(18,), dtype=np.float32
                    )

        # self.action_space = spaces.Discrete(5)
        # self.observation_space = spaces.Box(
        #                 low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        #             )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def get_mask(self,input):
        # temp=input.copy()
        # temp=temp.astype(np.uint8)
        # # print(temp.shape)
        # temp = temp.transpose(2, 0, 1)
        # temp=np.array([temp[2],temp[1],temp[0]])
        # temp = temp.transpose(1, 2, 0)
        # # if os.path.exists('./outputs/{}'.format(self.epoch)) == False:
        # # os.mkdir('./outputs/{}'.format(self.epoch))
        # # cv2.imwrite('./outputs/{}/reslut_{}.png'.format(self.epoch,self.count),out_temp)
        img_transform = Compose([
        transforms.Normalize(),
        ])
        augmented=img_transform(image=input)
        input=augmented['image']
        input = input.astype('float32')/255
        input = input.transpose(2, 0, 1)
        input=torch.from_numpy(input)
        input=input.cuda()
        input=input.reshape([1,3,256,256])
        self.segment_model=self.segment_model.cuda()
        output=self.segment_model(input)
        output=output[0].detach().cpu().numpy()
        return output
    def seed(self, seed=None):
       np.random.seed(seed)

    # def get_laser_data(self):
    #     laser_data=[]
    #     for i in range(16):
    #         data=self.usensors[i].read()
    #         if data != -1:
    #             if data < self.minDetectDistance:
    #                 data=self.minDetectDistance
    #             # data=((data-self.minDetectDistance)/(self.maxDetectDistance-self.minDetectDistance))
    #         else:
    #             data=0
    #         laser_data.append(data) #data表示每个可行分区上的距离可行进多少，1就表示完全行，0表示完全不可通行
    #     return laser_data
    def get_laser_data(self):
        laser_data=[]
        for i in range(16):
            data=self.usensors[i].read()
            if data != -1:
                if data < self.minDetectDistance:
                    data=self.minDetectDistance
                # data=((data-self.minDetectDistance)/(self.maxDetectDistance-self.minDetectDistance))
            else:
                data=0
            laser_data.append(data) #data表示每个可行分区上的距离可行进多少，1就表示完全行，0表示完全不可通行
        return laser_data
        
    def predict(self,laser_data,angle):#根据相机的视场角判断覆盖的激光雷达探测区
        flag=0
        for i in range(16):
            if_detected,point=self.usensors[i].is_detected(self.target)
            if if_detected==True:
                # print("find target")
                dist=np.sqrt(point[0] ** 2 + point[1] ** 2 + point[2] ** 2)
                flag=1
                # dist=laser_data[i]
                break
        if flag==0:
            dist=100
        self.last_laser_data= dist
        # return dist/config.max_distance
        return dist


    def target_motion_control(self,action_conduct_time,target_action_flag):# 控制target运动
        target_action = {'forward': [5, 5], 'turn_left': [-1.2, 1.2], 'turn_right': [1.2, -1.2]}

        option = random.choices([0,1,2],weights=[0.2,0.4,0.4])
        #print("状态变化前：", action_conduct_time, target_action_flag,option)
        if option[0]== 0 and action_conduct_time <= 0:
            target_action_flag = 0
            action_conduct_time = 150
            # print("前车直行")
            # vrepInterface.move_target_wheels(target_action['forward'][0], target_action['forward'][1])
        elif option[0] == 1 and action_conduct_time <= 0:
            target_action_flag = 1
            action_conduct_time = 150
            # print("前车左转")
        # vrepInterface.move_target_wheels(target_action['turn_left'][0], target_action['turn_left'][1])
        elif option[0] == 2 and action_conduct_time <= 0:
            # print("前车右转")
            target_action_flag = 2
            action_conduct_time = 150
            # vrepInterface.move_target_wheels(target_action['turn_right'][0], target_action['turn_right'][1])

        if target_action_flag == 0:
            self.target.set_joint_target_velocities([target_action['forward'][0], target_action['forward'][1]])
            action_conduct_time -= 1  # 执行时间减1
        elif target_action_flag == 1:
            action_conduct_time -= 1  # 执行时间减1
            if action_conduct_time >= 100:
                self.target.set_joint_target_velocities([target_action['turn_left'][0], target_action['turn_left'][1]])
            else:
                self.target.set_joint_target_velocities([target_action['forward'][0], target_action['forward'][1]])
        elif target_action_flag == 2:
            action_conduct_time -= 1  # 执行时间减1
            if action_conduct_time >= 100:
                self.target.set_joint_target_velocities([target_action['turn_right'][0], target_action['turn_right'][1]])
            else:
                self.target.set_joint_target_velocities([target_action['forward'][0], target_action['forward'][1]])
        #print("状态变化后：", action_conduct_time, target_action_flag, option)
        return action_conduct_time,target_action_flag

    def get_state(self):
        self.save_fig_step+=1
        observation = self.kinect.capture_rgb()
        observation = np.rot90(observation, 3)  
        observation = np.rot90(observation, 3)  
        observation = np.rot90(observation, 3)  # 需要旋转270度才能变成标准的图像，与相机获得的图像匹配
        input=cv2.resize(observation,(256,256))
        input = np.rot90(input, 3)
        input=input*255
        # mask=self.get_mask(input)
        # self.seg_img=mask>0.5
        # angle_info,dist_info,target_center,self.temp_center=self.transfer.get_info(self.seg_img.reshape([256,256]),mask,input,self.epoch)
        # p_logist,p_mask,_=self.prompts_model.get_segments(temp)
        # p_angle_info,p_dist_info,p_target_center,p_temp_center=self.transfer.get_info(p_mask.reshape([256,256]),p_logist,now_img,self.epoch)
        # p_logist=p_logist.reshape([256,256,1])
        # obs=obs.reshape([256,256,1])
        # p_seg_img=cv2.cvtColor(p_logist,cv2.COLOR_GRAY2BGR)
        # seg_img=cv2.cvtColor(obs,cv2.COLOR_GRAY2BGR)
        # cv2.imwrite('./retune_result/p_mask_{}.jpg'.format(self.count),p_seg_img*255)
        # cv2.imwrite('./retune_result/mask_{}.jpg'.format(self.count),seg_img*255)
        self.laser_data=self.get_laser_data()
        # s=[]
        # s.append(np.degrees(p_angle_info[0]))
        # s.append(self.predict(self.laser_data,p_angle_info))
        # s.append(np.degrees(angle_info[0]))
        # s.append(self.predict(self.laser_data,angle_info[0]))
        # self.laser_data=self.get_laser_data()
        self.tracker_target_distance, _, self.tracker_target_angle,_ = self.if_in_range(self.tracker.get_2d_pose(),self.target.get_2d_pose())
        # s.extend([self.tracker_target_angle,self.tracker_target_distance])
        # s.extend([np.degrees(angle_info[0]),self.predict(self.laser_data,angle_info[0])])
        # with open('./64_64_64_LSTM_1_Current_20240418.csv','a') as f:
        #     writer=csv.writer(f)
        #     writer.writerow(s)
        # s.extend([self.tracker_target_angle,self.tracker_target_distance])
        # with open('./prompt_super_info.csv','a') as f:
        #     writer=csv.writer(f)
        #     writer.writerow(s)
        state=[self.tracker_target_angle/config_ppo.max_angle,self.tracker_target_distance]#距离不要归一化!
        # state=[-np.degrees(angle_info[0])/config.max_angle,self.predict(self.laser_data,angle_info[0])]#距离不要归一化!
        # with open('./relative_info.csv','a') as f:
        #     writer=csv.writer(f)
        #     writer.writerow(state)
        state.extend(self.laser_data)
        # state.extend(output_state)
        # state=torch.tensor(state,device=self.device)
        print("state:",state)
        return np.array([state])

    def step(self, action):
        # print("当前的进程号是：",os.getpid())
        self.count+=1
        self.x_vel=action[0]
        self.z_vel=action[1]
        wheel_dist = 0.14
        wheel_radius = 0.036
        left_vel = self.x_vel - self.z_vel * wheel_dist / 2
        right_vel = self.x_vel + self.z_vel * wheel_dist / 2
        final_left_vel = left_vel / wheel_radius
        final_right_vel = right_vel / wheel_radius
        self.tracker.set_joint_target_velocities([final_left_vel, final_right_vel])
        # self.action_conduct_time,self.target_action_flag=self.target_motion_control(self.action_conduct_time,self.target_action_flag)
        self.pr.step() # Step the physics simulation
        # 获得奖励值及其他状态标志
        reward, done, truncated= self.get_reward()
        next_state = self.get_state()
        # done=False
        # truncated=False
        return next_state, reward, done, truncated, {}
    def get_reward(self):
        done = False
        truncated=False
        now_tracker_position=self.tracker.get_2d_pose()
        now_target_position=self.target.get_2d_pose()
        # 碰撞判定
        # tracker_collision= self.tracker.check_collision(self.target) or self.tracker.check_collision(self.boder) or self.tracker.check_collision(self.obstacles)
        # target_collision= self.target.check_collision(self.tracker) or self.target.check_collision(self.boder) or self.target.check_collision(self.obstacles)
        tracker_collision= self.tracker.check_collision(self.target) or self.tracker.check_collision(self.boder)
        target_collision= self.target.check_collision(self.tracker) or self.target.check_collision(self.boder)
        for dynamic_obstacle in self.dynamic_obstacles:
            tracker_collision = tracker_collision or self.tracker.check_collision(dynamic_obstacle)
        for dynamic_obstacle in self.dynamic_obstacles:
            target_collision = target_collision or self.target.check_collision(dynamic_obstacle)
        # 前后车之间的距离，是否丢失前车，前后车相对角度
        now_distance, now_in_range_flag, now_angle,now_driving_angle = self.if_in_range(now_tracker_position,now_target_position)
        
        #print("前车与后车的距离为：",distance,"前车与后车的角度差为：",angle,"度")
        if self.count == self.max_count:
            done = True
            reward = 100
            print("圆满完成了本episode:", reward)
            return reward, done, truncated
        if target_collision:
            done = False
            truncated=True
            #print("发生了碰撞")
            reward = 0
            print("target发生碰撞")
            return reward, done, truncated

        if tracker_collision == 1:  # 如果发生了碰撞
            done = True
            #print("发生了碰撞")
            reward = -100
            print("tracker发生碰撞")
            return reward, done, truncated
            
        if now_distance < config_ppo.min_distance:
            #print("两车距离过近，发生碰撞")
            done = True
            reward = -100
            print("两车距离过近，发生碰撞：", reward)
            return reward, done, truncated
        
        if now_in_range_flag == 0:  # 如果前车超出了后车的视野范围
            #print("跟丢了")
            done = True
            reward = -100
            print("因超出视野范围得到的惩罚值是：", reward)
            return reward, done, truncated
        else:
            #在视野范围内

            reward1 = np.abs(config_ppo.max_angle-np.abs(now_angle))/(config_ppo.max_angle-config_ppo.best_angle)
            reward2 = np.abs(config_ppo.max_distance-now_distance)/(config_ppo.max_distance-config_ppo.best_distance) if now_distance >=config_ppo.best_distance else -np.abs(config_ppo.best_distance-now_distance)/(config_ppo.best_distance-config_ppo.safe_distance)
            reward=reward1+reward2
            
        # print("正常")
        
        return reward*20, done, truncated
   
    def if_in_range(self,tracker_location,target_location):
        """ judge if the people in  the range of kinect on the robot"""
        flag = 0
        relative_pos = [target_location[0] - tracker_location[0], target_location[1] - tracker_location[1]]
        distance = np.sqrt(relative_pos[0] * relative_pos[0] + relative_pos[1] * relative_pos[1])
        tracker_direction = tracker_location[2] / np.pi * 180 - 90  # 因为车辆自身坐标系与地图坐标系存在差异，所以需要减去90度修正
        target_direction = target_location[2] / np.pi * 180 - 90  # 因为车辆自身坐标系与地图坐标系存在差异，所以需要减去90度修正
        if target_direction < 0:
            target_direction += 360
        #  得到target相对于tracker的角度
        target_to_tracker_angle=np.degrees(np.arctan2(relative_pos[1],relative_pos[0]))
        angles_between_target_tracker = (target_to_tracker_angle - tracker_direction)
        if angles_between_target_tracker <-180:
            angles_between_target_tracker+=360
        elif angles_between_target_tracker >180:
            angles_between_target_tracker-=360
        # if tracker_direction>= 0 and tracker_direction<=90:
        #     if target_to_tracker_angle
        # 判断是否在视野角度内
        if np.abs(angles_between_target_tracker) <= config_ppo.max_angle and distance >= config_ppo.safe_distance and distance <= config_ppo.max_distance:
            flag = 1

        driving_angle = target_direction - tracker_direction
        return distance, flag, angles_between_target_tracker, driving_angle

    def shutdown(self):
        self.pr.stop()
        self.pr.shutdown()

    def reset(self, seed=None, options=None):
        
        self.epoch+=1
        # if self.time %20 ==0 and self.time !=0:
        #     self.shutdown()
        if self.time  == 0:
        #——————启动Pyrep——————
            self.pr = PyRep()
            self.pr.launch(SCENE_FILE, self.headless)
            self.pr.start()
            self.pr.step() # Step the physics simulation
        #——————创建环境中的实体——————
            self.usensors=[]
            for i in range(1,17):
                self.usensors.append(ProximitySensor(f'Tracker_ultrasonicSensor{i}'))
            self.target = Target()
            self.tracker=Tracker()
            self.kinect = VisionSensor('kinect_rgb')
            # self.lidar1=VisionSensor('lidar_sensor1')
            # self.obstacles=Shape("Obstacles")
            self.boder=Shape("Boder")
            self.dynamic_obstacles=[]
            for i in range(1,19):
                self.dynamic_obstacles.append(Shape("D_O_{}".format(i)))

        #——————初始化——————
            self.tracker.set_control_loop_enabled(False)
            self.tracker.set_motor_locked_at_zero_velocity(True)
            self.target.set_control_loop_enabled(False)
            self.target.set_motor_locked_at_zero_velocity(True)
            self.initial_tracker_positions = self.tracker.get_2d_pose()
            self.initial_target_positions = self.target.get_2d_pose()
            self.seed()
        self.time+=1



        index=random.randint(0, len(Initial_positions_target)-1)
        # index=6
        # self.tracker.set_2d_pose(Initial_positions_tracker[index])
        # self.target.set_2d_pose(Initial_positions_target[index])
        self.tracker.set_2d_pose([9,0,0.065])
        self.target.set_2d_pose([9,-1,0.065])
        self.last_target_position=self.target.get_2d_pose()
        self.last_tracker_position=self.tracker.get_2d_pose()
        self.target.set_motor_locked_at_zero_velocity(True)
        self.tracker.set_motor_locked_at_zero_velocity(True)
        self.transfer=Transfer()
        self.pr.step() # Step the physics simulation
        self.action_conduct_time=100
        self.target_action_flag=0
        self.count=0
        
        return self.get_state(),{}

    def render(self):
        self.pr.launch(SCENE_FILE, False)
        self.pr.start()
    
    # def get_reward(self):
    #     done = False
    #     truncated=False
    #     now_tracker_position=self.tracker.get_2d_pose()
    #     now_target_position=self.target.get_2d_pose()
    #     collision_obstacles =  self.obstacles.check_collision(None) 
    #     collision_boder =  self.boder.check_collision(None) 
    #     # collision= collision_boder
    #     collision= collision_obstacles or collision_boder
    #     # 前后车之间的距离，是否丢失前车，前后车相对角度
    #     last_distance, last_in_range_flag, last_angle,last_driving_angle = self.if_in_range(self.last_tracker_position,self.last_target_position)
    #     distance, in_range_flag, angle,driving_angle = self.if_in_range(now_tracker_position,self.last_target_position)
    #     now_distance, now_in_range_flag, now_angle,now_driving_angle = self.if_in_range(now_tracker_position,now_target_position)
    #     #print("前车与后车的距离为：",distance,"前车与后车的角度差为：",angle,"度")
    #     if self.count == self.max_count:
    #         done = True
    #         #print("发生了碰撞")
    #         reward = 100
    #         print("圆满完成了本episode:", reward)
    #         return reward, done, truncated
    #     if collision == 1:  # 如果发生了碰撞
    #         done = True
    #         #print("发生了碰撞")
    #         reward = -100
    #         print("因碰撞得到的惩罚值是:", reward)
    #         return reward, done, truncated
            
    #     if distance <config.safe_distance:
    #         #print("两车距离过近，发生碰撞")
    #         done = True
    #         reward = -100
    #         print("两车距离过近，发生碰撞：", reward)
    #         return reward, done, truncated
    #     if in_range_flag == 0:  # 如果前车超出了后车的视野范围
    #         #print("跟丢了")
    #         done = True
    #         reward = -100
    #         print("因超出视野范围得到的惩罚值是：", reward)
    #         return reward, done, truncated
    #     else:
    #         #在视野范围内
    #         # reward =(self.last_angle_diff-angle)*(self.last_distance-distance)*(50/(np.abs(angle)*np.abs(distance)+1))-self.baseline
    #         # reward = 50 / (np.abs(angle) * np.abs(distance) + 1) - self.baseline #-4.18~45
    #         # reward = 1-np.abs(distance-self.best_distance)/self.max_distance
    #         # reward = 1-self.w1*np.abs(distance-self.best_distance)/self.max_distance - self.w2*np.abs(angle-self.best_angle)/self.max_angle #0~1
            
    #         # if np.abs(angle-config.best_angle) < 1:
    #         #     angle_buff = 1
    #         # else:
    #         #     if (np.abs(last_angle)-np.abs(angle))>0:
    #         #         angle_buff = 1
    #         #     elif (np.abs(last_angle)-np.abs(angle))==0:
    #         #         angle_buff=0
    #         #     elif (np.abs(last_angle)-np.abs(angle))<0:
    #         #         angle_buff=-1
    #         # if np.abs(distance-config.best_distance) <0.1:
    #         #     dist_buff = 1
    #         # else:
    #         #     if (last_distance-distance)>0:
    #         #         dist_buff = 1
    #         #     elif (last_distance-distance)==0:
    #         #         dist_buff=0
    #         #     elif (last_distance-distance)<0:
    #         #         dist_buff=-1
    #         if (np.abs(last_angle)-np.abs(angle))>0:
    #             angle_buff = 1
    #         elif (np.abs(last_angle)-np.abs(angle))==0:
    #             angle_buff=0
    #         elif (np.abs(last_angle)-np.abs(angle))<0:
    #             angle_buff=-1
    #         if (last_distance-distance)>0:
    #             dist_buff = 1
    #         elif (last_distance-distance)==0:
    #             dist_buff=0
    #         elif (last_distance-distance)<0:
    #             dist_buff=-1

    #         # print("angle_buff",angle_buff,"dist_buff",dist_buff)
    #         # print("Current step is:",self.count)
    #         # print("Last target position:",self.last_target_position,"Current target position:",self.target.get_2d_pose())
    #         # print("Last tracker position:",self.last_tracker_position,"Current tracker position:",now_tracker_position)
    #         # print("last_distance:",last_distance,"distance:",distance)
    #         # print("last_angle:",last_angle,"angle:",angle)
    #         reward1 = angle_buff*np.abs(config.max_angle-np.abs(angle))/config.max_angle
    #         reward2 = dist_buff*np.abs(config.max_distance-distance)/(config.max_distance-config.best_distance) if distance >=config.best_distance else -np.abs(config.safe_distance/distance)
            
    #         # print("reward1:",reward1,"reward2:",reward2)
    #         #如果至少一个方面有进步，则全加
    #         if (reward1>0 and reward2>0) or (reward1>0 and reward2==0) or (reward1==0 and reward2>0):
    #             reward=reward1+reward2
    #         # 如果一个方面进步，另一个方面退步，则考虑进步那方面
    #         elif reward1 < 0 and reward2 > 0:
    #             reward=reward2
    #         # 如果一个方面进步，另一个方面退步，则考虑进步那方面
    #         elif reward1 > 0 and reward2 < 0:
    #             reward=reward1
    #         # 如果两个方面都退步，或是一个方面不变，但另一个方面退步，则惩罚
    #         elif (reward1<0 and reward2 <0) or(reward1<0 and reward2 ==0) or (reward1==0 and reward2 <0):
    #             reward = reward1+reward2
    #         #如果是在安全距离附近保持不变，则加分
    #         elif reward1==0 and reward2 ==0 and np.abs(distance-config.best_distance) <= 0.1:
    #             reward = 1
    #         #如果是在安全距离外保持不变，则惩罚
    #         elif reward1==0 and reward2 ==0 and np.abs(distance-config.best_distance) > 0.1:
    #             reward = -1


    #         # print([dist_buff*self.w1*np.abs(distance-self.best_distance)/self.max_distance,angle_buff*self.w2*np.abs(angle-self.best_angle)/self.max_angle])
    #         #reward+= self.keep_track_counter*0.1
    #         # reward += -0.4 if (self.last_distance-distance) <0 else 0.4#距离变大增加惩罚
    #         # reward += -0.3 if (self.last_angle_diff - angle) <0 else 0.3#角度偏离增加惩罚
    #         # reward += -0.3 if (self.last_driving_angle - driving_angle) < 0 else 0.3  # 角度偏离增加惩罚
    #         # if (self.last_distance-distance) <0:
    #         #     print("距离相较变大，上一时刻两机器人之间的距离为：",self.last_distance)
    #         # if (self.last_angle_diff - angle) <0:
    #         #     print("相对角度相较变大，上一时刻两机器人之间的相对角度为：",self.last_angle_diff)
    #         # if (self.last_driving_angle - driving_angle) <0:
    #         #     print("行驶方向差相较变大，上一时刻两机器人之间的行驶方向差为：",self.last_driving_angle)
   
    #     self.last_target_position=self.target.get_2d_pose()
    #     self.last_tracker_position=self.tracker.get_2d_pose()
    #     self.last_distance = distance
    #     self.last_angle_diff = angle
    #     self.last_driving_angle = driving_angle
    #     # print("正常")
        
    #     return reward*20, done, truncated
    # def if_in_range(self,tracker_position,target_position):
    #     """ judge if the people in  the range of kinect on the robot"""
    #     flag = 0
    #     tracker_location = tracker_position
    #     target_location = target_position
    #     relative_pos = [target_location[0] - tracker_location[0], target_location[1] - tracker_location[1]]
    #     distance = np.sqrt(relative_pos[0] * relative_pos[0] + relative_pos[1] * relative_pos[1])
    #     tracker_direction = tracker_location[2] / np.pi * 180 - 90  # 因为车辆自身坐标系与地图坐标系存在差异，所以需要减去90度修正
    #     if tracker_direction < 0:
    #         tracker_direction += 360
    #     target_direction = target_location[2] / np.pi * 180 - 90  # 因为车辆自身坐标系与地图坐标系存在差异，所以需要减去90度修正
    #     if target_direction < 0:
    #         target_direction += 360
    #     #print("tracker_direction:", tracker_direction)
    #     #print("target_direction:", target_direction)
    #     #  得到target相对于tracker的角度
        

    #     if relative_pos[0] > 0 and relative_pos[1] >0:
    #         target_to_tracker_angle = np.arctan(relative_pos[1] / relative_pos[0])  # 得到的是弧度值
    #         target_to_tracker_angle = target_to_tracker_angle / np.pi * 180
    #     elif relative_pos[0] < 0 and relative_pos[1] > 0:
    #         target_to_tracker_angle = np.arctan(relative_pos[1] / relative_pos[0])  # 得到的是弧度值
    #         target_to_tracker_angle = 180 + target_to_tracker_angle / np.pi * 180
    #     elif relative_pos[0] < 0 and relative_pos[1] < 0:
    #         target_to_tracker_angle = np.arctan(relative_pos[1] / relative_pos[0])  # 得到的是弧度值
    #         target_to_tracker_angle = 180 + target_to_tracker_angle / np.pi * 180
    #     elif relative_pos[0] > 0 and relative_pos[1] < 0:
    #         target_to_tracker_angle = np.arctan(relative_pos[1] / relative_pos[0])  # 得到的是弧度值
    #         target_to_tracker_angle = 360 + target_to_tracker_angle / np.pi * 180
    #     elif relative_pos[0] == 0 and relative_pos[1] >0:
    #         target_to_tracker_angle=90
    #     elif relative_pos[0] == 0 and relative_pos[1] <0:
    #         target_to_tracker_angle=-90
    #     elif relative_pos[0]>0 and relative_pos[1] == 0:
    #         target_to_tracker_angle=0
    #     elif relative_pos[0] <0 and relative_pos[1] == 0:
    #         target_to_tracker_angle=180

    #     #print("target_to_tracker_angle:", target_to_tracker_angle)
    #     # print("tracker_direction",tracker_direction)
    #     # target目标位置与tracker形成的夹角与tracker的行进方向之间的关系
    #     angles_between_target_tracker = target_to_tracker_angle - tracker_direction if np.abs(target_to_tracker_angle - tracker_direction)< np.abs(np.abs(target_to_tracker_angle - tracker_direction) - 360) else np.abs(target_to_tracker_angle - tracker_direction) - 360

    #     # 判断是否在视野角度内
    #     if np.abs(angles_between_target_tracker) <= config.max_angle and distance >= config.min_distance and distance <= config.max_distance:
    #         flag = 1

    #     driving_angle = target_direction - tracker_direction
    #     return distance, flag, -angles_between_target_tracker, driving_angle
#if __name__ == "__main__":3
    #env = VAT_env(True)
    #parallel_api_test(env, num_cycles=3)

