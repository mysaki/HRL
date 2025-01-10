import gymnasium as gym
import datetime
from tianshou.data import Batch
import pandas as pd
import os
from get_test_track_policy import get_track_policy,get_Both_STAR_policy, get_C_STAR_policy,get_A_STAR_policy,get_rgb_policy,get_A_STAR_only_policy,get_A_STAR_mha_policy,get_A_STAR_rnn_policy,get_C_STAR_only_policy,get_C_STAR_mha_policy,get_C_STAR_rnn_policy,get_Both_STAR_only_policy,get_Both_STAR_mha_policy,get_Both_STAR_rnn_policy

policy_ASTAR = get_A_STAR_policy(path="./Log/join_train_track_ppo_12_29_11_2")
policy_ASTAR_rnn = get_A_STAR_rnn_policy(path="./Log/join_train_track_ppo_1_3_16_31")
policy_ASTAR_mha = get_A_STAR_mha_policy(path="./Log/join_train_track_ppo_1_3_16_29")
policy_ASTAR_only = get_A_STAR_only_policy(path="./Log/join_train_track_ppo_1_2_14_51")
policy_CSTAR = get_C_STAR_policy(path="./Log/join_train_track_ppo_12_27_11_21")
policy_CSTAR_rnn = get_C_STAR_rnn_policy(path="./Log/join_train_track_ppo_12_31_11_7")
policy_CSTAR_mha = get_C_STAR_mha_policy(path="./Log/join_train_track_ppo_1_4_0_5")
policy_CSTAR_only = get_C_STAR_only_policy(path="./Log/join_train_track_ppo_1_4_15_23")
policy_ACSTAR = get_Both_STAR_policy(path="./Log/join_train_track_ppo_12_27_11_28")
policy_ACSTAR_rnn = get_Both_STAR_rnn_policy(path="./Log/join_train_track_ppo_12_31_11_1")
policy_ACSTAR_mha = get_Both_STAR_mha_policy(path="./Log/join_train_track_ppo_12_31_11_22")
policy_ACSTAR_only = get_Both_STAR_only_policy(path="./Log/join_train_track_ppo_1_1_21_11")
policy_TRACK = get_track_policy(path="./Log/track_train_ppo_12_29_10_54")
policy_rgb = get_rgb_policy(path="./Log/track_ppo_image_12_26_8_44")

policy_set = [policy_TRACK,policy_rgb,policy_CSTAR,policy_ASTAR,policy_ACSTAR,policy_CSTAR_rnn,policy_ASTAR_rnn,policy_ACSTAR_rnn,policy_CSTAR_only,policy_ASTAR_only,policy_ACSTAR_only,policy_CSTAR_mha,policy_ASTAR_mha,policy_ACSTAR_mha]
for policy in policy_set:
    policy.eval()
model_name =['TRACK',"RGB",'CSTAR','ASTAR','ACSTAR','CSTAR_RNN','ASTAR_RNN','ACSTAR_RNN','CSTAR_ONLY','ASTAR_ONLY','ACSTAR_ONLY','CSTAR_MHA','ASTAR_MHA','ACSTAR_MHA']
task = "Dynamic-v0"
test_count = 0
episode_count = 0
policy_num = len(policy_set)
test_episodes = 100
env = gym.make(task, headless=True, mode=True)
time = datetime.datetime.now()
result = {
# 'reward_CSTAR':[],'reward_ASTAR':[],'reward_ACSTAR':[],'reward_TRACK':[],'reward_RGB':[],
#           'epi_len_CSTAR':[],'epi_len_ASTAR':[],'epi_len_ACSTAR':[],'epi_len_TRACK':[],'epi_len_RGB':[],
#           'angle_acc_CSTAR':[],'angle_acc_ASTAR':[],'angle_acc_ACSTAR':[],'angle_acc_TRACK':[],'angle_acc_RGB':[],
#           'distance_acc_CSTAR':[],'distance_acc_ASTAR':[],'distance_acc_ACSTAR':[],'distance_acc_TRACK':[],'distance_acc_RGB':[]
}
time = datetime.datetime.now()
folder_name = f"/media/hp/新加卷/XNW/Hiearchical_RL/test_records/track/{time.month}_{time.day}_{time.hour}_{time.minute}_{time.second}"
os.makedirs(folder_name)
for episode in range(test_episodes):
    epi_path = os.path.join(folder_name,str(episode))
    os.makedirs(epi_path)
    print("**********Current episode:",episode,"**********")
    for i in range(policy_num):
        print("**********Current policy:",model_name[i],"**********")
        reward_key = '_'.join(['reward',model_name[i]])
        if reward_key not in result:
            result[reward_key] = []
        epi_len_key ='_'.join(['epi_len',model_name[i]])
        if epi_len_key not in result:
            result[epi_len_key] = []
        angle_acc_key = '_'.join(['angle_acc',model_name[i]])
        if angle_acc_key not in result:
            result[angle_acc_key] = []
        distance_acc_key = '_'.join(['distance_acc',model_name[i]])
        if distance_acc_key not in result:
            result[distance_acc_key] = []
        if model_name[i] == 'RGB':
            state_mode = 'rgb'
        else:
            state_mode = 'task_info'
        if i == 0:
            # print("reset env")
            state, _ = env.reset(state_mode = state_mode)
            init_tracker_pos = env.tracker.get_2d_pose()
            init_target_pos = env.target.get_2d_pose()
            obstacles= env.pr.script_call(function_name_at_script_name='get_obstacle_pos@ResizableFloor_5_26', 
                script_handle_or_type=1, 
            )
            init_tracker_pos = env.tracker.get_2d_pose()
            init_target_pos = env.target.get_2d_pose()
            note = open(f"{epi_path}/map.txt",mode='a')
            note.write("walls\n")
            note.write(obstacles[2][0])
            note.write('\n')
            note.write("obstacles\n")
            note.write(obstacles[3])
            note.write('\n')
            note.close()
        else:
            # print("policy changed!")
            
            env.tracker.set_2d_pose(init_tracker_pos)
            env.target.set_2d_pose(init_target_pos)
            state = env.get_state(state_mode = state_mode)
        done = False
        Truncated = False
        epi_reward = 0
        episode_count = 0
        epi_angle_acc = 0
        epi_distance_acc = 0
        policy = policy_set[i]
        while not done:
            # if model_name[i] == 'RGB':
            #     state_mode = 'rgb'
            # else:
            #     state_mode = 'task_info'
            episode_count += 1
            batch = Batch()
            # print(state.shape)
            batch.obs = [state]
            batch.info = {}
            batch = policy(batch)
            epi_distance_acc += env.tracker_target_distance
            epi_angle_acc += env.tracker_target_angle
            batch['state_mode'] = state_mode
            next_state, reward, done, truncated, info = env.step(batch)
            epi_reward += reward
            state = next_state

        result[epi_len_key].append(episode_count)
        result[reward_key].append(epi_reward/episode_count)
        result[angle_acc_key].append(epi_angle_acc/episode_count)
        result[distance_acc_key].append(epi_distance_acc/episode_count) 
# 将字典转换为DataFrame
df = pd.DataFrame(result)
# 保存为Excel文件
df.to_excel(
    "result_track_ppo_{}_{}_{}_{}_{}.xlsx".format(
    time.month, time.day, time.hour, time.minute, time.second
    ),
    index=False,
)
