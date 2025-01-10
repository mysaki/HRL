import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import csv
import os
for epi in range(200):
    print("current epi:",epi)
    folder = "1_9_23_7_49"
    epi = str(epi)
    path = os.path.join("./test_records/nav", folder,epi)
    map_file = os.path.join(path, "map.txt")
    color_name = "Set3"
    select2 = (1,2,3, 4, 5, 6)  # 连续性色组图也可以从0-1之间选择
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
    fig = plt.figure(figsize=(6,6))
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
    # plt.show()
    csv_path_AC_STAR = os.path.join(path,"ACSTAR_episode_data.csv")
    csv_path_NAV = os.path.join(path, "NAV_episode_data.csv")
    csv_path_A_STAR = os.path.join(path, "ASTAR_episode_data.csv")
    csv_path_C_STAR = os.path.join(path, "CSTAR_episode_data.csv")

    csv_path_AC_STAR_RNN = os.path.join(path, "ACSTAR_RNN_episode_data.csv")
    csv_path_A_STAR_RNN = os.path.join(path, "ASTAR_RNN_episode_data.csv")
    csv_path_C_STAR_RNN = os.path.join(path, "CSTAR_RNN_episode_data.csv")

    csv_path_AC_STAR_MHA = os.path.join(path, "ACSTAR_MHA_episode_data.csv")
    csv_path_A_STAR_MHA = os.path.join(path, "ASTAR_MHA_episode_data.csv")
    csv_path_C_STAR_MHA = os.path.join(path, "CSTAR_MHA_episode_data.csv")

    csv_path_AC_STAR_ONLY = os.path.join(path, "ACSTAR_ONLY_episode_data.csv")
    csv_path_A_STAR_ONLY = os.path.join(path, "ASTAR_ONLY_episode_data.csv")
    csv_path_C_STAR_ONLY = os.path.join(path, "CSTAR_ONLY_episode_data.csv")

    AC_STAR_record_x = []
    AC_STAR_record_y = []
    NAV_record_x = []
    NAV_record_y = []
    A_STAR_record_x = []
    A_STAR_record_y = []
    C_STAR_record_x = []
    C_STAR_record_y = []

    A_STAR_RNN_record_x = []
    A_STAR_RNN_record_y = []
    C_STAR_RNN_record_x = []
    C_STAR_RNN_record_y = []
    AC_STAR_RNN_record_x = []
    AC_STAR_RNN_record_y = []

    A_STAR_MHA_record_x = []
    A_STAR_MHA_record_y = []
    C_STAR_MHA_record_x = []
    C_STAR_MHA_record_y = []
    AC_STAR_MHA_record_x = []
    AC_STAR_MHA_record_y = []

    A_STAR_ONLY_record_x = []
    A_STAR_ONLY_record_y = []
    C_STAR_ONLY_record_x = []
    C_STAR_ONLY_record_y = []
    AC_STAR_ONLY_record_x = []
    AC_STAR_ONLY_record_y = []
    if os.path.exists(csv_path_AC_STAR):
        with open(csv_path_AC_STAR,mode='r') as f:
            c = 1
            for row in csv.reader(f):
                if c == 1:
                    c -= 1
                    continue
                AC_STAR_record_x.append(float(row[0]))
                AC_STAR_record_y.append(float(row[1]))
        f.close()
    if os.path.exists(csv_path_A_STAR):
        with open(csv_path_A_STAR, mode="r") as f:
            c = 1
            for row in csv.reader(f):
                if c == 1:
                    c -= 1
                    continue
                A_STAR_record_x.append(float(row[0]))
                A_STAR_record_y.append(float(row[1]))
        f.close()
    if os.path.exists(csv_path_C_STAR):
        with open(csv_path_C_STAR, mode="r") as f:
            c = 1
            for row in csv.reader(f):
                if c == 1:
                    c -= 1
                    continue
                C_STAR_record_x.append(float(row[0]))
                C_STAR_record_y.append(float(row[1]))
        f.close()
    if os.path.exists(csv_path_NAV):
        with open(csv_path_NAV, mode="r") as f:
            c = 1
            for row in csv.reader(f):
                if c == 1:
                    c -= 1
                    continue
                NAV_record_x.append(float(row[0]))
                NAV_record_y.append(float(row[1]))
        f.close()
    if os.path.exists(csv_path_AC_STAR_RNN):
        with open(csv_path_AC_STAR_RNN, mode="r") as f:
            c = 1
            for row in csv.reader(f):
                if c == 1:
                    c -= 1
                    continue
                AC_STAR_RNN_record_x.append(float(row[0]))
                AC_STAR_RNN_record_y.append(float(row[1]))
        f.close()
    if os.path.exists(csv_path_A_STAR_RNN):
        with open(csv_path_A_STAR_RNN, mode="r") as f:
            c = 1
            for row in csv.reader(f):
                if c == 1:
                    c -= 1
                    continue
                A_STAR_RNN_record_x.append(float(row[0]))
                A_STAR_RNN_record_y.append(float(row[1]))
        f.close()
    if os.path.exists(csv_path_C_STAR_RNN):
        with open(csv_path_C_STAR_RNN, mode="r") as f:
            c = 1
            for row in csv.reader(f):
                if c == 1:
                    c -= 1
                    continue
                C_STAR_RNN_record_x.append(float(row[0]))
                C_STAR_RNN_record_y.append(float(row[1]))
        f.close()
    if os.path.exists(csv_path_AC_STAR_MHA):
        with open(csv_path_AC_STAR_MHA, mode="r") as f:
            c = 1
            for row in csv.reader(f):
                if c == 1:
                    c -= 1
                    continue
                AC_STAR_MHA_record_x.append(float(row[0]))
                AC_STAR_MHA_record_y.append(float(row[1]))
        f.close()
    if os.path.exists(csv_path_A_STAR_MHA):
        with open(csv_path_A_STAR_MHA, mode="r") as f:
            c = 1
            for row in csv.reader(f):
                if c == 1:
                    c -= 1
                    continue
                A_STAR_MHA_record_x.append(float(row[0]))
                A_STAR_MHA_record_y.append(float(row[1]))
        f.close()
    if os.path.exists(csv_path_C_STAR_MHA):
        with open(csv_path_C_STAR_MHA, mode="r") as f:
            c = 1
            for row in csv.reader(f):
                if c == 1:
                    c -= 1
                    continue
                C_STAR_MHA_record_x.append(float(row[0]))
                C_STAR_MHA_record_y.append(float(row[1]))
        f.close()
    if os.path.exists(csv_path_AC_STAR_ONLY):
        with open(csv_path_AC_STAR_ONLY, mode="r") as f:
            c = 1
            for row in csv.reader(f):
                if c == 1:
                    c -= 1
                    continue
                AC_STAR_ONLY_record_x.append(float(row[0]))
                AC_STAR_ONLY_record_y.append(float(row[1]))
        f.close()
    if os.path.exists(csv_path_A_STAR_ONLY):
        with open(csv_path_A_STAR_ONLY, mode="r") as f:
            c = 1
            for row in csv.reader(f):
                if c == 1:
                    c -= 1
                    continue
                A_STAR_ONLY_record_x.append(float(row[0]))
                A_STAR_ONLY_record_y.append(float(row[1]))
        f.close()
    if os.path.exists(csv_path_C_STAR_ONLY):
        with open(csv_path_C_STAR_ONLY, mode="r") as f:
            c = 1
            for row in csv.reader(f):
                if c == 1:
                    c -= 1
                    continue
                C_STAR_ONLY_record_x.append(float(row[0]))
                C_STAR_ONLY_record_y.append(float(row[1]))
        f.close()

    plt.title("Path planed by different methods")
    plt.plot(NAV_record_x, NAV_record_y, ":", label="NAV", lw=3, color="goldenrod")
    # plt.plot(AC_STAR_record_x, AC_STAR_record_y, "-", label="AC_STAR", lw=3,color="lightcoral")
    # plt.plot(A_STAR_record_x, A_STAR_record_y, "--", label="A_STAR", lw=3,color="tomato")
    # plt.plot(
    #     C_STAR_record_x, C_STAR_record_y, "-.", label="C_STAR", lw=3, color="sandybrown"
    # )

    # plt.plot(
    #     AC_STAR_RNN_record_x,
    #     AC_STAR_RNN_record_y,
    #     "-",
    #     label="AC_STAR_RNN",
    #     lw=3,
    #     color="olive",
    # )
    # plt.plot(
    #     A_STAR_RNN_record_x,
    #     A_STAR_RNN_record_y,
    #     "--",
    #     label="A_STAR_RNN",
    #     lw=3,
    #     color="palegreen",
    # )
    # plt.plot(
    #     C_STAR_RNN_record_x,
    #     C_STAR_RNN_record_y,
    #     "-.",
    #     label="C_STAR_RNN",
    #     lw=3,
    #     color="crimson",
    # )
    # plt.plot(
    #     AC_STAR_MHA_record_x,
    #     AC_STAR_MHA_record_y,
    #     "-",
    #     label="AC_STAR_MHA",
    #     lw=3,
    #     color="teal",
    # )
    # plt.plot(
    #     A_STAR_MHA_record_x,
    #     A_STAR_MHA_record_y,
    #     "--",
    #     label="A_STAR_MHA",
    #     lw=3,
    #     color="deepskyblue",
    # )
    # plt.plot(
    #     C_STAR_MHA_record_x,
    #     C_STAR_MHA_record_y,
    #     "-.",
    #     label="C_STAR_MHA",
    #     lw=3,
    #     color="royalblue",
    # )
    plt.plot(
        AC_STAR_ONLY_record_x,
        AC_STAR_ONLY_record_y,
        "-",
        label="AC_STAR_ONLY",
        lw=3,
        color="turquoise",
    )
    plt.plot(
        A_STAR_ONLY_record_x,
        A_STAR_ONLY_record_y,
        "--",
        label="A_STAR_ONLY",
        lw=3,
        color="greenyellow",
    )
    plt.plot(
        C_STAR_ONLY_record_x,
        C_STAR_ONLY_record_y,
        "-.",
        label="C_STAR_ONLY",
        lw=3,
        color="blueviolet",
    )
    # plt.legend(bbox_to_anchor=(1.05, 0),loc=3,borderaxespad=0)
    plt.legend()
    plt.scatter(NAV_record_x[0], NAV_record_y[0], c="darkseagreen", s=360, marker="h")
    plt.scatter(NAV_record_x[-1], NAV_record_y[-1], c="blue", s=100, marker="X")
    # plt.scatter(AC_STAR_record_x[0], AC_STAR_record_y[0], c="darkseagreen", s=360, marker="h")
    # plt.scatter(AC_STAR_record_x[-1], AC_STAR_record_y[-1], c="blue", s=100, marker="X")
    # plt.scatter(A_STAR_record_x[0], A_STAR_record_y[0], c="darkseagreen", s=360, marker="h")
    # plt.scatter(A_STAR_record_x[-1], A_STAR_record_y[-1], c="blue", s=100, marker="X")
    # plt.scatter(
    #     C_STAR_record_x[0], C_STAR_record_y[0], c="darkseagreen", s=360, marker="h"
    # )
    # plt.scatter(C_STAR_record_x[-1], C_STAR_record_y[-1], c="blue", s=100, marker="X")


    # plt.scatter(
    #     AC_STAR_MHA_record_x[0], AC_STAR_MHA_record_y[0], c="orange", s=360, marker="h"
    # )
    # plt.scatter(
    #     AC_STAR_MHA_record_x[-1], AC_STAR_MHA_record_y[-1], c="blue", s=100, marker="X"
    # )
    # plt.scatter(
    #     A_STAR_MHA_record_x[0], A_STAR_MHA_record_y[0], c="orange", s=360, marker="h"
    # )
    # plt.scatter(
    #     A_STAR_MHA_record_x[-1], A_STAR_MHA_record_y[-1], c="blue", s=100, marker="X"
    # )
    # plt.scatter(
    #     C_STAR_MHA_record_x[0], C_STAR_MHA_record_y[0], c="orange", s=360, marker="h"
    # )
    # plt.scatter(
    #     C_STAR_MHA_record_x[-1], C_STAR_MHA_record_y[-1], c="blue", s=100, marker="X"
    # )

    # plt.scatter(
    #     AC_STAR_RNN_record_x[0], AC_STAR_RNN_record_y[0], c="orange", s=360, marker="h"
    # )
    # plt.scatter(
    #     AC_STAR_RNN_record_x[-1], AC_STAR_RNN_record_y[-1], c="blue", s=100, marker="X"
    # )
    # plt.scatter(
    #     A_STAR_RNN_record_x[0], A_STAR_RNN_record_y[0], c="orange", s=360, marker="h"
    # )
    # plt.scatter(
    #     A_STAR_RNN_record_x[-1], A_STAR_RNN_record_y[-1], c="blue", s=100, marker="X"
    # )
    # plt.scatter(
    #     C_STAR_RNN_record_x[0], C_STAR_RNN_record_y[0], c="orange", s=360, marker="h"
    # )
    # plt.scatter(
    #     C_STAR_RNN_record_x[-1], C_STAR_RNN_record_y[-1], c="blue", s=100, marker="X"
    # )

    plt.scatter(
        AC_STAR_ONLY_record_x[0],
        AC_STAR_ONLY_record_y[0],
        c="orange",
        s=360,
        marker="h",
    )
    plt.scatter(
        AC_STAR_ONLY_record_x[-1],
        AC_STAR_ONLY_record_y[-1],
        c="blue",
        s=100,
        marker="X",
    )
    plt.scatter(
        A_STAR_ONLY_record_x[0], A_STAR_ONLY_record_y[0], c="orange", s=360, marker="h"
    )
    plt.scatter(
        A_STAR_ONLY_record_x[-1], A_STAR_ONLY_record_y[-1], c="blue", s=100, marker="X"
    )
    plt.scatter(
        C_STAR_ONLY_record_x[0], C_STAR_ONLY_record_y[0], c="orange", s=360, marker="h"
    )
    plt.scatter(
        C_STAR_ONLY_record_x[-1], C_STAR_ONLY_record_y[-1], c="blue", s=100, marker="X"
    )

    plt.scatter(target_pos[0], target_pos[1], c="red", s=360, marker="*")
    plt.savefig(os.path.join(path,"paths.jpg"),dpi=300)
    print("epi ", epi,"process ended")
