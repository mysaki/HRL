import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import csv
import os

plt.style.use("seaborn")
for epi in range(1000):
    print("current epi:", epi)
    policys = ["Hierarchical", "Hierarchical_new", "Nav_agent", "Rule_Based"]
    colors = []
    folder = "1_25_9_28_15"
    epi = str(epi)
    path = os.path.join("./test_records/hierarchical", folder, epi)
    map_file = os.path.join(path, "map.txt")
    color_name = "Set3"
    select2 = (1, 2, 3, 4, 5, 6)  # 连续性色组图也可以从0-1之间选择
    colors = plt.get_cmap(color_name)(select2)  # 从色组里选择颜色，我选择的是select2
    with open(map_file, "r", encoding="utf-8") as f:
        # read()：读取文件全部内容，以字符串形式返回结果
        data = f.readlines()
        walls_pos = list(map(float, data[1].split(" ")))
        obstacles_pos = list(map(float, data[3].split(" ")))
        target_pos = list(map(float, data[5].split(" ")))
    obs_x = []
    obs_y = []
    radius = []
    wall_x = []
    wall_y = []
    wall_yaw = []
    wall_length = []
    for i in range(0, len(obstacles_pos), 3):
        obs_x.append(obstacles_pos[i])
        obs_y.append(obstacles_pos[i + 1])
        radius.append(obstacles_pos[i + 2])
    for i in range(0, len(walls_pos), 4):
        wall_x.append(walls_pos[i])
        wall_y.append(walls_pos[i + 1])
        wall_yaw.append(walls_pos[i + 2])
        wall_length.append(walls_pos[i + 3])
    wall_width = np.ones_like(np.array(wall_length))
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    plt.scatter(x=obs_x, y=obs_y, s=np.array(radius) * 650, c="lightslategray")
    for i in range(len(wall_yaw)):
        ax.add_patch(
            patches.Rectangle(
                (wall_x[i], wall_y[i]),
                wall_width[i] * 0.2,
                wall_length[i] * 0.5,
                wall_yaw[i] / np.pi * 180 - 90,
                edgecolor="gray",
                facecolor="gray",
            )
        )
    plt.title("Path planed by different methods")
    for policy in policys:
        csv_path = os.path.join(path, f"{policy}_episode_data.csv")
        x = []
        y = []

        if os.path.exists(csv_path):
            with open(csv_path, mode="r") as f:
                c = 1
                for row in csv.reader(f):
                    if c == 1:
                        c -= 1
                        continue
                    x.append(float(row[0]))
                    y.append(float(row[1]))
            f.close()
        else:
            ValueError("*******policy data does not exist*******")

        plt.plot(x, y, label=policy, lw=3)
        plt.scatter(x[0], y[0], c="darkseagreen", s=360, marker="h")
        plt.scatter(x[-1], y[-1], c="blue", s=100, marker="X")
    plt.legend()
    plt.scatter(target_pos[0], target_pos[1], c="red", s=360, marker="*")
    plt.savefig(os.path.join(path, "paths.jpg"), dpi=300)
    print("epi ", epi, "process ended")
