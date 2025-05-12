# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# import numpy as np
# import csv
# import os

# plt.style.use("seaborn")
# for epi in range(1000):
#     print("current epi:", epi)
#     policys = ["Hierarchical", "Hierarchical_new", "Nav_agent", "Rule_Based"]
#     colors = []
#     folder = "1_25_9_28_15"
#     epi = str(epi)
#     path = os.path.join("./test_records/hierarchical", folder, epi)
#     map_file = os.path.join(path, "map.txt")
#     color_name = "Set3"
#     select2 = (1, 2, 3, 4, 5, 6)  # 连续性色组图也可以从0-1之间选择
#     colors = plt.get_cmap(color_name)(select2)  # 从色组里选择颜色，我选择的是select2
#     with open(map_file, "r", encoding="utf-8") as f:
#         # read()：读取文件全部内容，以字符串形式返回结果
#         data = f.readlines()
#         walls_pos = list(map(float, data[1].split(" ")))
#         obstacles_pos = list(map(float, data[3].split(" ")))
#         target_pos = list(map(float, data[5].split(" ")))
#     obs_x = []
#     obs_y = []
#     radius = []
#     wall_x = []
#     wall_y = []
#     wall_yaw = []
#     wall_length = []
#     for i in range(0, len(obstacles_pos), 3):
#         obs_x.append(obstacles_pos[i])
#         obs_y.append(obstacles_pos[i + 1])
#         radius.append(obstacles_pos[i + 2])
#     for i in range(0, len(walls_pos), 4):
#         wall_x.append(walls_pos[i])
#         wall_y.append(walls_pos[i + 1])
#         wall_yaw.append(walls_pos[i + 2])
#         wall_length.append(walls_pos[i + 3])
#     wall_width = np.ones_like(np.array(wall_length))
#     fig = plt.figure(figsize=(6, 6))
#     ax = fig.add_subplot(111)
#     plt.scatter(x=obs_x, y=obs_y, s=np.array(radius) * 650, c="lightslategray")
#     for i in range(len(wall_yaw)):
#         ax.add_patch(
#             patches.Rectangle(
#                 (wall_x[i], wall_y[i]),
#                 wall_width[i] * 0.2,
#                 wall_length[i] * 0.5,
#                 wall_yaw[i] / np.pi * 180 - 90,
#                 edgecolor="gray",
#                 facecolor="gray",
#             )
#         )
#     plt.title("Path planed by different methods")
#     for policy in policys:
#         csv_path = os.path.join(path, f"{policy}_episode_data.csv")
#         x = []
#         y = []

#         if os.path.exists(csv_path):
#             with open(csv_path, mode="r") as f:
#                 c = 1
#                 for row in csv.reader(f):
#                     if c == 1:
#                         c -= 1
#                         continue
#                     x.append(float(row[0]))
#                     y.append(float(row[1]))
#             f.close()
#         else:
#             ValueError("*******policy data does not exist*******")

#         plt.plot(x, y, label=policy, lw=3)
#         plt.scatter(x[0], y[0], c="darkseagreen", s=360, marker="h")
#         plt.scatter(x[-1], y[-1], c="blue", s=100, marker="X")
#     plt.legend()
#     plt.scatter(target_pos[0], target_pos[1], c="red", s=360, marker="*")
#     plt.savefig(os.path.join(path, "paths.jpg"), dpi=300)
#     print("epi ", epi, "process ended")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import csv
import os
import numpy as np
from matplotlib.lines import Line2D
import matplotlib
from pathlib import Path
# Set the font family to SimHei (黑体)
plt.rcParams['font.family'] = 'STIXGeneral'
matplotlib.use("Agg")
# 设置专业论文级绘图参数
plt.style.use('seaborn-v0_8')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.dpi': 300  # 提高输出分辨率
})

# 颜色配置方案
COLOR_SCHEME = {
    "Hierarchical_Agent": "coral",  # (基础色，高亮色)
    "Rule_Based": "darkgreen",
    "KURL": "#b3974e",
    "Nav": "#5f6694",
    "Target": "#5f7604",
    "wall": "#AAB7B8",
    "obstacle": "#7D3C98",
    "follower_start": "#F1C40F",
    "follower_end": "#F1C40F",
    "target_end":"#5f7604",
    "target_start": "#5f7604",
}

# 改进后的可视化函数
def plot_agent_trajectory(ax, x, y, agent_type,skill=None):
    """绘制带有时序信息的轨迹线"""
    line = None
    # 创建连续轨迹线
    if skill == None:
        line = ax.plot(x, y, 
                    color=COLOR_SCHEME[agent_type],
                    lw=2, alpha=0.7, zorder=3)[0]
    else:
        for i in range(len(skill)):
            if skill[i] == 1:
                ax.scatter(x[i], y[i], s=2, 
                           color='r', marker='h', zorder=5)
            elif skill[i] == 0:
                ax.scatter(x[i], y[i], s=2, 
                           color='b', marker='h', zorder=5)
    # 标注起终点
    if agent_type == "Target":
        ax.scatter(x[0], y[0], s=200, 
              color=COLOR_SCHEME['target_start'], marker='h', zorder=5)
        ax.scatter(x[-1], y[-1], s=200,
                color=COLOR_SCHEME['target_end'], marker='X', zorder=5)
    else:
        ax.scatter(x[0], y[0], s=200, 
                  color=COLOR_SCHEME['follower_start'], marker='h', zorder=5)
        ax.scatter(x[-1], y[-1], s=200,
                color=COLOR_SCHEME['follower_end'], marker='X', zorder=5)
    
    return line
# 确定要读取的文件夹
folder_name = "test_records/hierarchical/1_22_23_46_30"
folder_path = os.path.join(folder_name)
folder = Path(folder_path)
iteration_num = len([file for file in folder.iterdir() if file.is_dir()])
Agents_name = ["Hierarchical_Agent", "Rule_Based"]
fig_path = os.path.join(folder_name,'figs')
if not os.path.exists(fig_path):
    os.makedirs(fig_path)
for epi in range(308,309):
    print(f"********** Current episode: {epi} **********")
    epi_folder_path = os.path.join(folder_path,str(epi))
    if not os.path.exists(epi_folder_path):
        print(f"********** folder {epi} does not exist **********")
        continue

    # 读取地图文件
    map_path = os.path.join(epi_folder_path,"map.txt")
    with open(map_path,'r') as f:
        data = f.readlines()
        walls = list(map(float,list(data[1].split(" "))))
        obstacles = list(map(float,list(data[3].split(" "))))
        taget_pos = list(map(float,list(data[5].split(" "))))
    wall_x = []
    wall_y = []
    wall_len = []
    wall_angle = []
    obstacle_x = []
    obstacle_y = []
    obstacle_radius = []
    # 获得墙体信息
    for i in range(0,len(walls),4):
        wall_x.append(walls[i])
        wall_y.append(walls[i+1])
        wall_len.append(walls[i+3])
        wall_angle.append(walls[i+2])

    # 获得障碍物信息
    for i in range(0,len(obstacles),3):
        obstacle_x.append(obstacles[i])
        obstacle_y.append(obstacles[i+1])
        obstacle_radius.append(obstacles[i+2])
    # 读取智能体轨迹
    for agent_name in Agents_name:
        print(f"Agent:{agent_name}")
        agent_csv_path = os.path.join(epi_folder_path,f"{agent_name}_episode_data.csv")
        if not os.path.exists(agent_csv_path):
            ValueError(f"********** the csv of {agent_name} does not exist **********")
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        # 优化环境元素绘制
        # 绘制墙体
        for i in range(len(wall_x)):
            ax.add_patch(
                patches.Rectangle(
                    xy = (wall_x[i], wall_y[i]),
                    width=wall_len[i],
                    height=0.2,
                    angle=wall_angle[i]*180/np.pi,
                    facecolor=COLOR_SCHEME['wall'],
                    edgecolor='k',
                    alpha=0.4,
                    zorder=1
                )
            )

        # 绘制障碍物（添加光影效果）
        obstacles = ax.scatter(obstacle_x, obstacle_y, 
                            s=np.array(obstacle_radius)*500,
                            c=COLOR_SCHEME['obstacle'],
                            alpha=0.6,
                            edgecolor='k',
                            linewidth=0.5,
                            zorder=2)
        legend_elements = [Line2D([0], [0], marker='h', color='w', label='跟踪平台运动起点',
                    markerfacecolor=COLOR_SCHEME['follower_start'], markersize=15),
                    Line2D([0], [0], marker='h', color='w', label='跟踪目标运动起点',
                    markerfacecolor=COLOR_SCHEME['target_start'], markersize=15),
                    Line2D([0], [0], marker='X', color='w', label='跟踪目标运动终点',
                    markerfacecolor=COLOR_SCHEME['target_end'], markersize=15)
                    ]

        # target = ax.scatter(taget_pos[0], taget_pos[1],
        #                     marker = "h",
        #                     s=500,
        #                     c=COLOR_SCHEME['target_start'],
        #                     alpha=0.6,
        #                     linewidth=0.5,
        #                     zorder=2)   
        tracker_x = []
        tracker_y = []
        target_x = []
        target_y = []
        skill = []
        with open(agent_csv_path,'r') as f:
            reader = csv.reader(f)
            count = 0
            for line in reader:
                if count == 0:
                    count += 1
                    continue
                count += 1
                tracker_x.append(float(line[0]))
                tracker_y.append(float(line[1]))
                target_x.append(float(line[3]))
                target_y.append(float(line[4]))
                skill.append(int(line[6]))
        # 绘制智能体轨迹
        tracker_trajectory_line = plot_agent_trajectory(ax, tracker_x, tracker_y,agent_name,skill)
        target_trajectory_line = plot_agent_trajectory(ax, target_x, target_y,'Target')

        # 添加专业图例

        legend_elements.extend([
            Line2D([0], [0], color=COLOR_SCHEME["Target"], lw=2, label=f"跟踪目标运动轨迹"),

            # Line2D([0], [0], marker='X', color='w', label=f'跟踪平台运动终点',
            #     markerfacecolor=COLOR_SCHEME[agent_name], markersize=15),
        ])

        ax.legend(handles=legend_elements, loc='upper right', frameon=True, framealpha=0.9)

        # 设置坐标轴属性
        ax.set_title("运动轨迹分析", pad=20)
        ax.set_xlabel("X轴(m)")
        ax.set_ylabel("Y轴(m)")
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.8)

        plt.savefig(f"{fig_path}/{epi}_{agent_name}.jpg", 
                bbox_inches='tight', pad_inches=0.1)
        plt.close()
