# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# import numpy as np
# import csv
# import os
# plt.style.use('seaborn')
# for epi in range(1000):
#     print("current epi:",epi)
#     policys = ["ACSTAR_ONLY","ASTAR_ONLY","CSTAR_ONLY","NAV"]
#     colors = []
#     folder = "1_20_9_11_30"
#     epi = str(epi)
#     path = os.path.join("./test_records/nav", folder,epi)
#     map_file = os.path.join(path, "map.txt")
#     color_name = "Set3"
#     select2 = (1,2,3, 4, 5, 6)  # 连续性色组图也可以从0-1之间选择
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
#     fig = plt.figure(figsize=(6,6))
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
#         csv_path = os.path.join(path,f"{policy}_episode_data.csv")
#         x = []
#         y = []
    
#         if os.path.exists(csv_path):
#             with open(csv_path,mode='r') as f:
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
#     plt.savefig(os.path.join(path,"paths.jpg"),dpi=300)
#     print("epi ", epi,"process ended")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import csv
import os
import numpy as np
from matplotlib.lines import Line2D
import matplotlib
from pathlib import Path
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
    "AC_SHARE": "coral",  # (基础色，高亮色)
    "A_SHARE": "darkgreen",
    "C_SHARE": "#b3974e",
    "AC_MLP": "#5f6694",
    "AC_MLP_SHARE": "#5f7604",
    "wall": "#AAB7B8",
    "obstacle": "#7D3C98",
    "start": "#F1C40F",
    "end": "#ddae33",
    "target": "r",
}
# 根据实际情况修改字体文件路径
plt.rcParams["font.family"] = "SimHei"
# 解决负号显示问题
plt.rcParams["axes.unicode_minus"] = False
# 改进后的可视化函数
def plot_agent_trajectory(ax, tracker_x, tracker_y, agent_type):
    """绘制带有时序信息的轨迹线"""
    # 创建连续轨迹线
    line = ax.plot(tracker_x, tracker_y, 
                   color=COLOR_SCHEME[agent_type],
                   lw=2, alpha=0.7, zorder=3)[0]
    
    # 标注起终点
    ax.scatter(tracker_x[0], tracker_y[0], s=200, 
              color=COLOR_SCHEME['start'], marker='h', zorder=5)
    ax.scatter(tracker_x[-1], tracker_y[-1], s=200,
              color=COLOR_SCHEME[agent_type], marker='X', zorder=5)
    
    return line
# 确定要读取的文件夹
folder_name = "test_records/nav/1_19_11_27_48"
folder_path = os.path.join(folder_name)
folder = Path(folder_path)
iteration_num = len([file for file in folder.iterdir() if file.is_dir()])
Agents_name = ["AC_SHARE", "AC_MLP"]
fig_path = os.path.join(folder_name,'figs')
if not os.path.exists(fig_path):
    os.makedirs(fig_path)
for epi in range(37,38):
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
    legend_elements = [Line2D([0], [0], marker='h', color='w', label='起始位置',
                markerfacecolor=COLOR_SCHEME['start'], markersize=15),
                Line2D([0], [0], marker='*', color='w', label='目标所在位置',
                markerfacecolor=COLOR_SCHEME['target'], markersize=15),
                ]

    target = ax.scatter(taget_pos[0], taget_pos[1],
                        marker = "*",
                        s=500,
                        c=COLOR_SCHEME['target'],
                        alpha=0.6,
                        linewidth=0.5,
                        zorder=2)   
    # 读取智能体轨迹
    for agent_type in Agents_name:
        print(f"Agent:{agent_type}")
        if agent_type == "AC_MLP":
            agent_name = "NAV"
        elif agent_type == "AC_MLP_SHARE":
            agent_name = "NAV_TRACK_SHRE"
        elif agent_type == "AC_SHARE":
            agent_name = "ACSTAR_ONLY"
        else:
            agent_name = agent_type
        agent_csv_path = os.path.join(epi_folder_path,f"{agent_name}_episode_data.csv")
        if not os.path.exists(agent_csv_path):
            ValueError(f"********** the csv of {agent_type} does not exist **********")
        tracker_x = []
        tracker_y = []
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
        legend_elements.extend([Line2D([0], [0], color=COLOR_SCHEME[agent_type], lw=2, label=f"{agent_type}"),
                Line2D([0], [0], marker='X', color='w', label=f'{agent_type} 路径终点',
                markerfacecolor=COLOR_SCHEME[agent_type], markersize=15),])
        # 绘制智能体轨迹
        trajectory_line = plot_agent_trajectory(ax, tracker_x, tracker_y,agent_type)

        # 添加专业图例


    ax.legend(handles=legend_elements, loc='upper right', frameon=True, framealpha=0.9)

    # 设置坐标轴属性
    ax.set_title("运动轨迹分析", pad=20)
    ax.set_xlabel("X轴(m)")
    ax.set_ylabel("Y轴(m)")
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.8)

    plt.savefig(f"{fig_path}/{epi}.jpg", 
            bbox_inches='tight', pad_inches=0.1)
    plt.close()
