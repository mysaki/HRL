import gymnasium as gym
import datetime
from tianshou.data import Batch
import pandas as pd
import os
import csv
import config
from get_test_nav_policy import *#get_policy_nav,get_A_STAR_policy,get_C_STAR_policy,get_Both_STAR_policy,get_A_STAR_only_policy,get_A_STAR_mha_policy,get_A_STAR_rnn_policy,get_C_STAR_only_policy,get_C_STAR_mha_policy,get_C_STAR_rnn_policy,get_Both_STAR_only_policy,get_Both_STAR_mha_policy,get_Both_STAR_rnn_policy,get

policy_NAV = get_policy_nav(path="./Log/nav_train_ppo_12_29_11_27")
# policy_NAV_TRACK = get_policy_nav(path="./Log/track_train_ppo_2_18_15_50")
policy_NT_SHARE = get_policy_nav(path="./Log/track_train_ppo_2_20_9_38")
# policy_ASTAR = get_A_STAR_policy(path="./Log/join_train_track_ppo_12_29_11_2")
# policy_ASTAR_rnn = get_A_STAR_rnn_policy(path="./Log/join_train_track_ppo_1_3_16_31")
# policy_ASTAR_mha = get_A_STAR_mha_policy(path="./Log/join_train_track_ppo_1_3_16_29")
policy_ASTAR_only = get_A_STAR_only_policy(path="./Log/join_train_track_ppo_1_18_13_38")
# policy_CSTAR = get_C_STAR_policy(path="./Log/join_train_track_ppo_12_27_11_21")
# policy_CSTAR_rnn = get_C_STAR_rnn_policy(path="./Log/join_train_track_ppo_12_31_11_7")
# policy_CSTAR_mha = get_C_STAR_mha_policy(path="./Log/join_train_track_ppo_1_4_0_5")
policy_CSTAR_only = get_C_STAR_only_policy(path="./Log/join_train_track_ppo_1_4_15_23")
# policy_ACSTAR = get_Both_STAR_policy(path="./Log/join_train_track_ppo_12_27_11_28")
# policy_ASTAR_CSTAR_ONLY = get_A_STAR_C_STAR_Only_policy(path="./Log/join_train_track_ppo_1_14_20_31")
# policy_ACSTAR_rnn = get_Both_STAR_rnn_policy(path="./Log/join_train_track_ppo_12_31_11_1")
# policy_ACSTAR_mha = get_Both_STAR_mha_policy(path="./Log/join_train_track_ppo_12_31_11_22")
policy_ACSTAR_only = get_Both_STAR_only_policy(path="./Log/join_train_track_ppo_1_18_10_29")
policy_set = [policy_NAV,policy_NT_SHARE,policy_ACSTAR_only,policy_ASTAR_only,policy_CSTAR_only]
for policy in policy_set:
    policy.eval()
# model_name =['NAV','CSTAR','ASTAR','ACSTAR','CSTAR_RNN','ASTAR_RNN','ACSTAR_RNN','CSTAR_ONLY','ASTAR_ONLY','ACSTAR_ONLY','CSTAR_MHA','ASTAR_MHA','ACSTAR_MHA']
model_name = ['NAV','NAV_TRACK_SHARE','AC_SHARE','A_SHARE','C_SHARE']
task = "Navigation-v0"
test_count = 0
episode_count = 0
policy_num = len(model_name)
test_episodes = 5000
env = gym.make(task, headless=True, mode=True)
time = datetime.datetime.now()
folder_name = f"/media/hp/新加卷/XNW/Hiearchical_RL/test_records/nav/{time.month}_{time.day}_{time.hour}_{time.minute}_{time.second}"
result = {
}
os.makedirs(folder_name)
result_file_path = "{}/result_nav_{}_{}_{}_{}_{}.csv".format(
    folder_name, time.month, time.day, time.hour, time.minute, time.second
)
print("*"*20)
print("执行的任务为:",task)
print("比较的模型为：",model_name)
print("*" * 20)
for episode in range(test_episodes):
    print("**********Current episode:",episode,"**********")
    epi_path = os.path.join(folder_name,str(episode))
    os.makedirs(epi_path)
    for i in range(policy_num):
        print("**********Current policy:",model_name[i],"**********")
        reward_key = '_'.join(['reward',model_name[i]])
        if reward_key not in result:
            result[reward_key] = []
        epi_len_key ='_'.join(['epi_len',model_name[i]])
        if epi_len_key not in result:
            result[epi_len_key] = []
        success_flag_key = '_'.join(['success_flag',model_name[i]])
        if success_flag_key not in result:
            result[success_flag_key] = []
        min_dist_key = "_".join(["min_dist", model_name[i]])
        if min_dist_key not in result:
            result[min_dist_key] = []
        max_dist_key = "_".join(["max_dist", model_name[i]])
        if max_dist_key not in result:
            result[max_dist_key] = []
        std_dist_key = "_".join(["std_dist", model_name[i]])
        if std_dist_key not in result:
            result[std_dist_key] = []
        var_dist_key = "_".join(["var_dist", model_name[i]])
        if var_dist_key not in result:
            result[var_dist_key] = []
        avg_dist_key = "_".join(["avg_dist", model_name[i]])
        if avg_dist_key not in result:
            result[avg_dist_key] = []
        collision_key = "_".join(["collision", model_name[i]])
        if collision_key not in result:
            result[collision_key] = []
        if i == 0:
            # print("reset env")
            state, _ = env.reset()
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
            note.write("target_pos\n")
            note.write(' '.join([str(init_target_pos[0]),str(init_target_pos[1])]))
            note.close()

        else:
            # print("policy changed!")

            env.tracker.set_2d_pose(init_tracker_pos)
            env.target.set_2d_pose(init_target_pos)
            state = env.get_state()
        done = False
        Truncated = False
        epi_reward = 0
        episode_count = 0
        epi_min_dist = 0
        epi_max_dist = 0
        epi_std_dist = 0
        epi_var_dist = 0
        epi_avg_dist = 0
        epi_collisions = 0
        policy = policy_set[i]
        file_path = os.path.join(folder_name, str(episode), f"{model_name[i]}_episode_data.csv")
        header = ['x','y','yaw','if_collision','min_dist','max_dist','avg_dist','std_dist','var_dist','laser_data']
        # print("tracker pos:",env.tracker.get_2d_pose(),"target pos:",env.target.get_2d_pose(),"episode_count:",episode_count,"epi_reward:",epi_reward)
        with open(file_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(header)
        while not done:
            episode_count += 1
            batch = Batch()
            batch.obs = [state]
            batch.info = {}
            batch = policy(batch)
            next_state, reward, done, truncated, info = env.step(batch.act[0])
            epi_reward += reward
            state = next_state
            epi_min_dist += info["min_dist"]
            epi_max_dist += info["max_dist"]
            epi_std_dist += info["std_dist"]
            epi_var_dist += info["var_dist"]
            epi_avg_dist += info["avg_dist"]
            epi_collisions += info["if_collision"]
            # 写入数据
            with open(file_path, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        info["tracker_pos"][0],
                        info["tracker_pos"][1],
                        info["tracker_pos"][2],
                        info["if_collision"],
                        info["min_dist"],
                        info["max_dist"],
                        info["avg_dist"],
                        info["std_dist"],
                        info["var_dist"],
                        info["laser_data"],
                    ]
                )
        result[epi_len_key].append(episode_count)
        result[reward_key].append(epi_reward/episode_count)
        result[success_flag_key].append(info["success_flag"])
        result[min_dist_key].append(epi_min_dist / episode_count)
        result[max_dist_key].append(epi_max_dist / episode_count)
        result[avg_dist_key].append(epi_avg_dist / episode_count)
        result[std_dist_key].append(epi_std_dist / episode_count)
        result[var_dist_key].append(epi_var_dist / episode_count)
        result[collision_key].append(epi_collisions/ episode_count)
    if episode == 0:
        with open(result_file_path,mode = 'w') as f:
            writer = csv.writer(f)
            head = result.keys()
            writer.writerow(head)
    with open(result_file_path, mode="a") as f:
        writer = csv.writer(f)
        writer.writerow([result[key][-1] for key in result.keys()])
# 将字典转换为DataFrame
df = pd.DataFrame(result)
# 保存为Excel文件
df.to_excel(
    "{}/result_nav_ppo_{}_{}_{}_{}_{}.xlsx".format(
        folder_name,time.month, time.day, time.hour, time.minute, time.second
    ),
    index=False,
)
