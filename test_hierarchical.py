import gymnasium as gym
import datetime
from tianshou.data import Batch
import pandas as pd
import os
import csv
from get_test_hierarchical_policy import *
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# policy_hierarchical_nonmix = get_hierarchical_policy(
#     low_level_policy_path="./Log/join_train_track_ppo_1_18_10_29",
#     high_level_policy_path="./log/Hierarchical-v1/hierarchical_track_dqn_1_14_20_38",
# )
# policy_hierarchical_1 = get_hierarchical_policy(
#     low_level_policy_path="./Log/join_train_track_ppo_1_18_10_29",
#     high_level_policy_path="./log/Hierarchical-v1/hierarchical_track_dqn_3_11_11_38",
# )
policy_hierarchical = get_hierarchical_policy(
    low_level_policy_path="./Log/join_train_track_ppo_1_18_10_29",
    high_level_policy_path="./log/Hierarchical-v1/hierarchical_track_dqn_3_15_12_6",
)
# policy_ASTAR_hierarchical = get_ASTAR_hierarchical_policy(
#     low_level_policy_path="./Log/join_train_track_ppo_1_18_13_38",
#     high_level_policy_path="./log/Hierarchical-v1/hierarchical_track_dqn_3_5_9_4",
# )
policy_rule_based = Rule_Base_Agent(model_path="./Log/join_train_track_ppo_1_18_10_29")
# policy_NAV = get_policy_nav(path="./Log/nav_train_ppo_12_29_11_27")
policy_set = [policy_hierarchical,policy_rule_based]#, policy_hierarchical_early]
for policy in policy_set:
    policy.eval()
model_name = ["Hierarchical","Rule_Based"]#,"Hierarchical_DQN_EARLY"]
task = "Hierarchical-v1"
test_count = 0
episode_count = 0
policy_num = len(model_name)

test_episodes = 200
env = gym.make(task, headless=True, mode=True)
time = datetime.datetime.now()
folder_name = f"/media/hp/新加卷/XNW/Hiearchical_RL/test_records/hierarchical/{time.month}_{time.day}_{time.hour}_{time.minute}_{time.second}"
result = {}
os.makedirs(folder_name)
result_file_path = "{}/result_hierarchical_{}_{}_{}_{}_{}.csv".format(
    folder_name, time.month, time.day, time.hour, time.minute, time.second
)
print(f"——————————————test map:{task},compare models:{model_name}——————————————")
for episode in range(test_episodes):
    print("**********Current episode:", episode, "**********")
    epi_path = os.path.join(folder_name, str(episode))
    os.makedirs(epi_path)
    for i in range(policy_num):
        print("**********Current policy:", model_name[i], "**********")
        reward_key = "_".join(["reward", model_name[i]])
        if reward_key not in result:
            result[reward_key] = []
        # tracker_pos_x_key = "_".join(["tracker_pos_x", model_name[i]])
        # if tracker_pos_x_key not in result:
        #     result[tracker_pos_x_key] = []
        # tracker_pos_y_key = "_".join(["tracker_pos_y", model_name[i]])
        # if tracker_pos_y_key not in result:
        #     result[tracker_pos_y_key] = []
        # target_pos_x_key = "_".join(["target_pos_x", model_name[i]])
        # if target_pos_x_key not in result:
        #     result[target_pos_x_key] = []
        # target_pos_y_kgey = "_".join(["target_pos_y", model_name[i]])
        # if target_pos_y_key not in result:
        #     result[target_pos_y_key] = []
        epi_len_key = "_".join(["epi_len", model_name[i]])
        if epi_len_key not in result:
            result[epi_len_key] = []
        success_flag_key = "_".join(["success_flag", model_name[i]])
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
        collision_key = "_".join(["collision_steps", model_name[i]])
        if collision_key not in result:
            result[collision_key] = []
        lost_steps_key = "_".join(["lost_steps", model_name[i]])
        if lost_steps_key not in result:
            result[lost_steps_key] = []
        track_steps_key = "_".join(["track_steps", model_name[i]])
        if track_steps_key not in result:
            result[track_steps_key] = []
        max_lost_step_key = "_".join(["max_lost_step", model_name[i]])
        if max_lost_step_key not in result:
            result[max_lost_step_key] = []
        refind_count_key = "_".join(["refind_count", model_name[i]])
        if refind_count_key not in result:
            result[refind_count_key] = []
        max_track_step_key = "_".join(["max_track_step", model_name[i]])
        if max_track_step_key not in result:
            result[max_track_step_key] = []
        if i == 0:
            # print("reset env")
            state, _ = env.reset()
            obstacles = env.pr.script_call(
                function_name_at_script_name="get_obstacle_pos@ResizableFloor_5_26",
                script_handle_or_type=1,
            )
            init_tracker_pos = env.tracker.get_2d_pose()
            init_target_pos = env.target.get_2d_pose()
            note = open(f"{epi_path}/map.txt", mode="a")
            note.write("walls\n")
            note.write(obstacles[2][0])
            note.write("\n")
            note.write("obstacles\n")
            note.write(obstacles[3])
            note.write("\n")
            note.write("target_pos\n")
            note.write(" ".join([str(init_target_pos[0]), str(init_target_pos[1])]))
            note.close()

        else:
            # print("policy changed!")

            env.tracker.set_2d_pose(init_tracker_pos)
            env.target.set_2d_pose(init_target_pos)
            _ = env.pr.script_call(
                function_name_at_script_name="reset_signal@ResizableFloor_5_26",
                script_handle_or_type=1,
            )
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
        file_path = os.path.join(
            folder_name, str(episode), f"{model_name[i]}_episode_data.csv"
        )
        header = [
            "tracker_x",
            "tracker_y",
            "tracker_yaw",
            "target_x",
            "target_y",
            "target_yaw",
            "if_collision",
            "high_level_action",
            "min_dist",
            "max_dist",
            "avg_dist",
            "std_dist",
            "var_dist",
            "laser_data",
        ]
        # print("tracker pos:",env.tracker.get_2d_pose(),"target pos:",env.target.get_2d_pose(),"episode_count:",episode_count,"epi_reward:",epi_reward)
        with open(file_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(header)
        flag = 0
        while not done:
            # print("episode step:", episode_count)
            episode_count += 1
            batch = Batch()
            batch.obs = [state]
            batch.info = {}
            batch.flag = flag
            origin_obs = torch.tensor(batch.obs).to(config.device)
            if model_name[i]== "Nav_agent":         
                modified_obs = torch.cat([origin_obs[:,:2], torch.tensor([[0,1]]).to(config.device),origin_obs[:,2:]], dim=1)
                batch.obs = modified_obs
                
            batch = policy(batch)

            # print(batch["high_level_act"])
            tracker_pos = env.tracker.get_2d_pose()
            target_pos = env.target.get_2d_pose()
            # result[tracker_pos_x_key].append(init_tracker_pos[0])
            # result[tracker_pos_y_key].append(init_tracker_pos[1])
            # result[target_pos_x_key].append(init_target_pos[0])
            # result[target_pos_y_key].append(init_target_pos[1])
            next_state, reward, done, truncated, info = env.step(batch.act[0])
            flag = info['flag']
            if model_name[i] == "Rule_Based":
                batch["high_level_act"] = [flag]
            if model_name[i] == "Nav_agent":
                batch["high_level_act"] = [flag]
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
                        tracker_pos[0],
                        tracker_pos[1],
                        tracker_pos[2],
                        target_pos[0],
                        target_pos[1],
                        target_pos[2],
                        info["if_collision"],
                        batch['high_level_act'][0],
                        info["min_dist"],
                        info["max_dist"],
                        info["avg_dist"],
                        info["std_dist"],
                        info["var_dist"],
                        info["laser_data"],
                    ]
                )
        result[epi_len_key].append(episode_count)
        result[reward_key].append(epi_reward / episode_count)
        result[success_flag_key].append(info["success_flag"])
        result[min_dist_key].append(epi_min_dist / episode_count)
        result[max_dist_key].append(epi_max_dist / episode_count)
        result[avg_dist_key].append(epi_avg_dist / episode_count)
        result[std_dist_key].append(epi_std_dist / episode_count)
        result[var_dist_key].append(epi_var_dist / episode_count)
        result[collision_key].append(epi_collisions /episode_count)
        result[lost_steps_key].append(info["lost_steps"])
        result[track_steps_key].append(info["track_steps"])
        result[max_lost_step_key].append(info["max_lost_step"])
        result[refind_count_key].append(info["refind_count"])
        result[max_track_step_key].append(info["max_track_step"])
    if episode == 0:
        with open(result_file_path, mode="w") as f:
            writer = csv.writer(f)
            head = result.keys()
            writer.writerow(head)
    with open(result_file_path, mode="a") as f:
        writer = csv.writer(f)
        writer.writerow([result[key][-1] for key in result.keys()])
# # 将字典转换为DataFrame
# df = pd.DataFrame(result)
# # 保存为Excel文件
# df.to_excel(
#     "{}/result_nav_ppo_{}_{}_{}_{}_{}.xlsx".format(
#         folder_name, time.month, time.day, time.hour, time.minute, time.second
#     ),
#     index=False,
# )
