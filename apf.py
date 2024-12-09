import config_ppo
# pyrep
from pyrep.objects.shape import Shape
from pyrep.robots.mobiles.tracker import Tracker
from pyrep.robots.mobiles.target import Target
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep import PyRep
import random
import numpy as np
from os.path import dirname, join, abspath
# 设置超参数
SCENE_FILE = join(dirname(abspath(__file__)),
                  'icra_v2.ttt')
Initial_positions_target=[
                          [9,-1,0.065],
                          [13,2,0.065],
                          [9.8,-4,0.065],
                        #   [6.4,0.2,0.065],
                          [7.8,4,0.065],
                          ]
Initial_positions_tracker=[
                           [9,0,0.065],
                           [13,3,0.065],
                           [9.8,-3,0.065],
                        #    [6.4,1.2,0.065],
                           [7.8,5,0.065],
                           ]
class APF():
    def __init__(self,headless) -> None:
        self.time=0
        self.count=0
        self.headless=headless
    def reset(self):
        if self.time == 0:
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
            self.boder=Shape("Boder")
            self.tracker_collision=Shape("Tracker_collision")
            self.obstacles=Shape("Obstacles")
        # self.dynamic_obstacles=[]
        # for i in range(1,7):
        #     self.dynamic_obstacles.append(Shape("D_O_{}".format(i)))

            #——————初始化——————
            self.tracker.set_control_loop_enabled(False)
            self.tracker.set_motor_locked_at_zero_velocity(True)
            self.target.set_control_loop_enabled(False)
            self.target.set_motor_locked_at_zero_velocity(True)
            self.initial_tracker_positions = self.tracker.get_2d_pose()
            self.initial_target_positions = self.target.get_2d_pose()  
        self.time+=1

        index=random.randint(0, len(Initial_positions_target)-1)
        # index=6
        self.tracker.set_2d_pose(Initial_positions_tracker[index])
        self.target.set_2d_pose(Initial_positions_target[index])
        self.last_target_position=self.target.get_2d_pose()
        self.last_tracker_position=self.tracker.get_2d_pose()
        self.target.set_motor_locked_at_zero_velocity(True)
        self.tracker.set_motor_locked_at_zero_velocity(True)
        self.tracker.wheel_distance = 0.14
        self.tracker.wheel_radius = 0.036
        self.target_pos=self.last_target_position
        self.pr.step() # Step the physics simulation
        self.action_conduct_time=100
        self.target_action_flag=0
        self.count=0
    def get_bin_idx(self,angle):#根据相机的视场角判断覆盖的激光雷达探测区
        theta = angle
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
            bin_idx = 15
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
    
        return bin_idx
    def get_laser_data(self):
        laser_data=[]
        for i in range(16):
            data=self.usensors[i].read()
            if data != -1:
                if data < config_ppo.minDetectDistance:
                    data=config_ppo.minDetectDistance
                elif data>config_ppo.maxDetectDistance:
                    data=config_ppo.maxDetectDistance
                # data=((data-self.minDetectDistance)/(self.maxDetectDistance-self.minDetectDistance))
            else:
                data=config_ppo.maxDetectDistance # 扫描不到东西的时候也表明距离很远，因此直接赋值为
            laser_data.append(data/config_ppo.maxDetectDistance) #data表示tracker与光束碰到的第一个障碍物之间的距离
        return laser_data
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
        if np.abs(angles_between_target_tracker) <= config_ppo.max_angle and distance >= config_ppo.min_distance and distance <= config_ppo.max_distance:
            flag = 1

        driving_angle = target_direction - tracker_direction
        return distance, flag, angles_between_target_tracker, driving_angle

    def apf_track(self):
        braitenbergL=1*np.array([-0.1,-0.2,-0.3,-0.4,-0.5,-0.6,-0.7,-0.8, 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        braitenbergR=1*np.array([-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1, 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        v0=10
        vLeft=v0
        vRight=v0
        laser_data = self.get_laser_data()
        for i in range(16):
            dist=laser_data[i]
            vLeft=vLeft+braitenbergL[i]*(1-dist)
            vRight=vRight+braitenbergR[i]*(1-dist)
        # print("斥力-VL,VR:",vLeft,vRight)
        now_tracker_position=self.tracker.get_2d_pose()
        now_target_position=self.target.get_2d_pose()
        self.now_distance, self.now_in_range_flag, self.now_angle,self.now_driving_angle = self.if_in_range(now_tracker_position,now_target_position)
        bin_index = self.get_bin_idx(self.now_angle)
        target_dist = laser_data[bin_index]
        # print("bin_index:",bin_index)
        # print("target_dist:",target_dist)
        if target_dist*config_ppo.maxDetectDistance > 0.7:
            # print("接近")
            vLeft=vLeft-80*braitenbergL[bin_index]*(target_dist)
            vRight=vRight-80*braitenbergR[bin_index]*(target_dist)
            # print("delta-VL,VR:",50*braitenbergL[bin_index]*(target_dist),50*braitenbergR[bin_index]*(target_dist))
        else:
            # print("远离")
            vLeft=vLeft+50*braitenbergL[bin_index]*(1-target_dist)
            vRight=vRight+50*braitenbergR[bin_index]*(1-target_dist)
        # print("final-VL,VR:",vLeft,vRight)
        # if vLeft > vRight:
        #     print("right turn")
        # elif vRight > vLeft:
        #     print("left turn")
        return vLeft,vRight,self.now_in_range_flag
    
    def collision_check(self):
        now_tracker_position=self.tracker.get_2d_pose()
        now_target_position=self.target.get_2d_pose()
        self.now_distance, self.now_in_range_flag, self.now_angle,self.now_driving_angle = self.if_in_range(now_tracker_position,now_target_position)
        shape_collision=self.tracker_collision.check_collision(self.boder) or self.tracker_collision.check_collision(self.obstacles)
        if self.now_distance <config_ppo.min_distance:
            near_collision = True
        else:
            near_collision =False
        return shape_collision or near_collision
    
if __name__ == "__main__":
    num_episode = 100
    max_epi_step = 2000
    apf = APF(headless=True)
    num_collisions=0
    avg_lenth=0
    num_losts=0
    num_perfect = 0
    for episode in range(num_episode):
        print("epis:",episode)
        done = False 
        step = 0
        apf.reset()
        while done == False and step < max_epi_step:
            if apf.collision_check() == True:
                print("Collide!")
                done = True
                num_collisions+=1
                break
            v_left,v_right, if_in_range = apf.apf_track()
            if if_in_range == False:
                print("Lost!")
                num_losts+=1
                break
            apf.tracker.set_joint_target_velocities([v_left, v_right])
            apf.pr.step()
            step+=1
        avg_lenth+=step
        if step == max_epi_step:
            num_perfect+=1
            print("Perfect!")
        # apf.pr.stop()
        # apf.pr.shutdown()
    print("avg_length:",avg_lenth/100,"num_losts:",num_losts,"num_collisions:",num_collisions,"num_perfect:",num_perfect)


