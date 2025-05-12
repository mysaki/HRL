import matplotlib.pyplot as plt
import numpy as np
import csv

"""
信息提取实验制图
"""
# fig,axs = plt.subplots(2,2)
# ax = plt.axes()
# angle = []
# dist = []
# angle_diff = []
# dist_diff = []
# with open('./estimation_effect.csv','r') as f:
#     content_reader = csv.DictReader(f)
#     headers = content_reader.fieldnames
#     for row in content_reader:
#         angle.append([float(row['估计角度']),float(row['实际角度'])])
#         dist.append([float(row['估计距离'])+0.245,float(row['实际距离'])])
#         angle_diff.append(float(row['角度差'])+0.82)
#         dist_diff.append(float(row['距离差'])+0.245)
# print(np.mean(angle_diff))
"""
1. 绘制估计对比曲线图
"""
# plt.subplot(2,1,1)
# plt.title("Estimation of the relative angle", fontsize=10)
# plt.xlim(xmin=0,xmax=600)
# plt.ylim(ymin=-20,ymax=45)
# plt.xlabel("Step")
# plt.ylabel("Estimation value")
# pic=plt.plot(angle)
# plt.legend(pic,["Real","Estimation"],shadow=True,fancybox="blue",fontsize=8)
# plt.subplot(2,1,2)
# plt.title("Estimation of the relative distance", fontsize=10)
# plt.xlabel("Step")
# plt.ylabel("Estimation value")
# plt.xlim(xmin=0,xmax=600)
# plt.ylim(ymin=0.2,ymax=2)
# pic=plt.plot(dist)
# plt.legend(pic,["Real","Estimation"],shadow=True,fancybox="blue",fontsize=8)
# plt.subplots_adjust(hspace=0.5)
# plt.subplots_adjust(wspace=0.5)
# plt.savefig("Estimation", dpi=600,bbox_inches='tight')
# plt.show()
"""
2. 绘制估计误差曲线图
"""
# plt.subplot(2,1,1)
# plt.title("Estimation error of the relative angle", fontsize=10)
# plt.xlim(xmin=0,xmax=600)
# plt.ylim(ymin=-2.5,ymax=1)
# plt.xlabel("Step")
# plt.ylabel("Estimation error")
# plt.plot(angle_diff)

# plt.subplot(2,1,2)
# plt.title("Estimation error of the relative distance", fontsize=10)
# plt.xlim(xmin=0,xmax=600)
# plt.ylim(ymin=-0.02,ymax=0.02)
# plt.xlabel("Step")
# plt.ylabel("Estimation error")
# plt.plot(dist_diff)
# plt.subplots_adjust(hspace=0.5)
# plt.subplots_adjust(wspace=0.5)
# # fig.text(0.5, 0.95, 'Estimation error of the relative angle and distance', ha='center', va='top', fontsize=10)
# plt.savefig("Estimation_error", dpi=600,bbox_inches='tight')

# plt.show()

"""
状态压缩实验制图
"""

"""
1. 状态压缩训练效果对比曲线图
"""
# def smooth_curve(data, weight=0.6):
#     """
#     对曲线数据进行平滑处理
#     :param data: 原始数据
#     :param weight: 平滑权重，范围在0到1之间，值越大，平滑效果越明显
#     :return: 平滑后的数据
#     """
#     smoothed = []
#     last = data[0]  # 初始化第一个数据点
#     for point in data:
#         smoothed_val = last * weight + (1 - weight) * point  # 计算平滑值
#         smoothed.append(smoothed_val)
#         last = smoothed_val
#     return smoothed
# ori_data_image=[]
# ori_data_compressed=[]
# step_image=[]
# step_compressed=[]
# color_name = 'Set3'
# select2 = (3,4) # 连续性色组图也可以从0-1之间选择
# colors = plt.get_cmap(color_name)(select2) # 从色组里选择颜色，我选择的是select2
# with open('./image.csv','r') as f:
#     content_reader = csv.DictReader(f)
#     headers = content_reader.fieldnames
#     for row in content_reader:
#         ori_data_image.append(float(row['Value']))
#         step_image.append(int(row['Step']))
# with open('./compress.csv','r') as f:
#     content_reader = csv.DictReader(f)
#     headers = content_reader.fieldnames
#     for row in content_reader:
#         ori_data_compressed.append(float(row['Value']))
#         step_compressed.append(int(row['Step']))
# smoothed_data_image = smooth_curve(ori_data_image, weight=0.5)
# upper_smoothed_data_image  = smooth_curve(np.array(smoothed_data_image )+500, weight=0.9)
# lower_smoothed_data_image  = smooth_curve(np.array(smoothed_data_image )-500, weight=0.9)
# for i in range(len(lower_smoothed_data_image )):
#     if lower_smoothed_data_image[i]<0:
#         lower_smoothed_data_image[i] = 0
# # 绘制原始数据和阴影
# plt.figure(figsize=(10, 5))
# plt.plot(step_image,smoothed_data_image , label='Image_based', linewidth=2,color=colors[1])
# # plt.fill_between(step_image,upper_smoothed_data_image ,lower_smoothed_data_image , alpha=0.3,color=colors[1])

# smoothed_data_compressed = smooth_curve(ori_data_compressed, weight=0.5)
# upper_smoothed_data_compressed  = smooth_curve(np.array(smoothed_data_compressed )+500, weight=0.5)
# lower_smoothed_data_compressed  = smooth_curve(np.array(smoothed_data_compressed )-500, weight=0.5)
# for i in range(len(lower_smoothed_data_compressed )):
#     if lower_smoothed_data_compressed[i]<0:
#         lower_smoothed_data_compressed[i] = 0
# # 绘制原始数据和阴影
# plt.plot(step_compressed,smoothed_data_compressed , label='Ours', linewidth=2,color=colors[0])
# # plt.fill_between(step_compressed,upper_smoothed_data_compressed ,lower_smoothed_data_compressed , alpha=0.3,color=colors[0])
# plt.legend(fontsize=8)
# plt.title('Episode length of the two state designs during training')
# plt.xlabel('Step')
# plt.ylabel('Episode Length')
# plt.xlim(xmin=0,xmax=1100000)

# plt.savefig("length", dpi=600,bbox_inches='tight')
# plt.show()
# color_name = 'Set3'
# select2 = (3,4) # 连续性色组图也可以从0-1之间选择
# colors = plt.get_cmap(color_name)(select2) # 从色组里选择颜色，我选择的是select2
# tracker_x = []
# tracker_y = []
# target_x = []
# target_y = []
# flag=[]
# with open('./tracker_coordnate_3.csv','r') as f:
#     content_reader = csv.DictReader(f)
#     headers = content_reader.fieldnames
#     for row in content_reader:
#         tracker_x.append(float(row['tracker_x']))
#         tracker_y.append(float(row['tracker_y']))
#         target_x.append(float(row['target_x']))
#         target_y.append(float(row['target_y']))
#         flag.append(int(row['flag']))
# # tracker_x = tracker_x[:800]
# # tracker_y = tracker_y[:800]
# # target_x = target_x[:800]
# # target_y = target_y[:800]
# plt.plot(tracker_x,tracker_y,label='tracker',color=colors[1])
# plt.plot(target_x,target_y,label='target',color=colors[0])
# plt.legend()
# plt.savefig("change_task_1", dpi=600,bbox_inches='tight')
# plt.show()

"""
2. 状态压缩泛化性能实验
"""
# plt.subplot(1,4,1)
# size = 4
# image_based = [276.43, 1334.12, 503.51, 1738.79]
# ours = [1130.40, 1279.02, 816.05, 1896.73]
# x = np.arange(size)
# # 有两种类型的数据，n设置为2
# total_width, n = 0.4, 2
# # 每种类型的柱状图宽度
# width = total_width / n

# color_name = 'Set3'
# select2 = (3,4) # 连续性色组图也可以从0-1之间选择
# colors = plt.get_cmap(color_name)(select2) # 从色组里选择颜色，我选择的是select2
# # 画柱状图
# plt.bar(x-width/2, image_based, width=width, label="Image_based", color=colors[0])
# plt.bar(x+width/2, ours, width=width, label="Ours", color=colors[1])
# # 显示图例
# # plt.legend()
# plt.title("Average Tracking Steps Per Episode")
# # 功能1
# x_labels = ["scene_4", "scene_3", "scene_2", "scene_1"]
# # 用第1组...替换横坐标x的值
# plt.xticks(x, x_labels)

# # 功能2
# # for i, j in zip(x-width/2, image_based):
# #     plt.text(i, j + 0.01, "%.2f" % j, ha="center", va="bottom", fontsize=6)
# # for i, j in zip(x + width/2, ours):
# #     plt.text(i, j + 0.01, "%.2f" % j, ha="center", va="bottom", fontsize=6)
# # plt.show()
# # plt.savefig("./average_epsode_step.jpg")
# plt.subplot(1,4,2)
# size = 4
# image_based = [140.73, 1056.49, 343.96, 1445.62]
# ours = [984.58, 1062.69, 639.95, 1648.26]
# x = np.arange(size)
# # 有两种类型的数据，n设置为2
# total_width, n = 0.4, 2
# # 每种类型的柱状图宽度
# width = total_width / n

# color_name = 'Set3'
# select2 = (3,4) # 连续性色组图也可以从0-1之间选择
# colors = plt.get_cmap(color_name)(select2) # 从色组里选择颜色，我选择的是select2
# # 画柱状图
# plt.bar(x-width/2, image_based, width=width, label="Image_based", color=colors[0])
# plt.bar(x+width/2, ours, width=width, label="Ours", color=colors[1])
# # 显示图例
# # plt.legend()
# plt.title("Average Reward Per Episode")
# # 功能1
# x_labels = ["scene_4", "scene_3", "scene_2", "scene_1"]
# # 用第1组...替换横坐标x的值
# plt.xticks(x, x_labels)

# # 功能2
# # for i, j in zip(x-width/2, image_based):
# #     plt.text(i, j + 0.01, "%.2f" % j, ha="center", va="bottom", fontsize=6)
# # for i, j in zip(x + width/2, ours):
# #     plt.text(i, j + 0.01, "%.2f" % j, ha="center", va="bottom", fontsize=6)
# # plt.show()
# # plt.savefig("./average_epsode_reward.jpg")
# plt.subplot(1,4,3)
# size = 4
# image_based = [0.509, 0.792, 0.683, 0.831]
# ours = [0.871, 0.831, 0.784, 0.869]
# x = np.arange(size)
# # 有两种类型的数据，n设置为2
# total_width, n = 0.4, 2
# # 每种类型的柱状图宽度
# width = total_width / n

# color_name = 'Set3'
# select2 = (3,4) # 连续性色组图也可以从0-1之间选择
# colors = plt.get_cmap(color_name)(select2) # 从色组里选择颜色，我选择的是select2
# # 画柱状图
# plt.bar(x-width/2, image_based, width=width, label="Image_based", color=colors[0])
# plt.bar(x+width/2, ours, width=width, label="Ours", color=colors[1])
# # 显示图例
# # plt.legend()
# plt.title("Average Reward Per Step")
# # 功能1
# x_labels = ["scene_4", "scene_3", "scene_2", "scene_1"]
# # 用第1组...替换横坐标x的值
# plt.xticks(x, x_labels)

# # 功能2
# # for i, j in zip(x-width/2, image_based):
# #     plt.text(i, j + 0.01, "%.2f" % j, ha="center", va="bottom", fontsize=6)
# # for i, j in zip(x + width/2, ours):
# #     plt.text(i, j + 0.01, "%.2f" % j, ha="center", va="bottom", fontsize=6)
# # plt.show()
# # plt.savefig("./average_step_reward.jpg")
# plt.subplot(1,4,4)
# size = 4
# image_based = [0.0, 0.22, 0.0, 0.56]
# ours = [0.0, 0.31, 0.12, 0.89]
# x = np.arange(size)
# # 有两种类型的数据，n设置为2
# total_width, n = 0.4, 2
# # 每种类型的柱状图宽度
# width = total_width / n

# color_name = 'Set3'
# select2 = (3,4) # 连续性色组图也可以从0-1之间选择
# colors = plt.get_cmap(color_name)(select2) # 从色组里选择颜色，我选择的是select2
# # 画柱状图
# plt.bar(x-width/2, image_based, width=width, label="Image_based", color=colors[0])
# plt.bar(x+width/2, ours, width=width, label="Ours", color=colors[1])
# # 显示图例
# # plt.legend()
# plt.title("Max Steps rate")
# # 功能1
# x_labels = ["scene_4", "scene_3", "scene_2", "scene_1"]
# # 用第1组...替换横坐标x的值
# plt.xticks(x, x_labels)

# # 功能2
# # for i, j in zip(x-width/2, image_based):
# #     plt.text(i, j + 0.01, "%.2f" % j, ha="center", va="bottom", fontsize=6)
# # for i, j in zip(x + width/2, ours):
# #     plt.text(i, j + 0.01, "%.2f" % j, ha="center", va="bottom", fontsize=6)
# plt.show()
# # plt.savefig("./max_step_count.jpg")


# track任务测试
# 有两种类型的数据，n设置为2
# total_width, n = 0.4, 4
# size = 3
# num_methods = 4
# # 每种类型的柱状图宽度
# width = total_width / n
# x_labels = ["","Normal", "","Medium","","Complex"] # 障碍物数量不同
# x = np.arange(size)
# AC_STAR_epi_steps = [276.43, 1334.12, 503.51]
# A_STAR_epi_steps = [1130.40, 1279.02, 816.05]
# AC_STAR_epi_rew = [140.73, 1056.49, 343.96]
# A_STAR_epi_rew = [984.58, 1062.69, 639.95]
# AC_STAR_step_rew = [0.509, 0.792, 0.683]
# A_STAR_step_rew = [0.871, 0.831, 0.784]
# AC_STAR_max_steps = [0.0, 0.22, 0.0]
# A_STAR_max_steps = [0.0, 0.31, 0.12]
# C_STAR_epi_steps = [276.43, 1334.12, 503.51]
# Track_epi_steps = [1130.40, 1279.02, 816.05]
# C_STAR_epi_rew = [140.73, 1056.49, 343.96]
# Track_epi_rew = [984.58, 1062.69, 639.95]
# C_STAR_step_rew = [0.509, 0.792, 0.683]
# Track_step_rew = [0.871, 0.831, 0.784]
# C_STAR_max_steps = [0.0, 0.22, 0.0]
# Track_max_steps = [0.0, 0.31, 0.12]
# # 创建子图
# fig, axs = plt.subplots(1, 4, figsize=(15, 4))
# color_name = 'Set3'
# select2 = (3,4,5,6) # 连续性色组图也可以从0-1之间选择
# colors = plt.get_cmap(color_name)(select2) # 从色组里选择颜色，我选择的是select2
# # 绘制子图
# axs[0].bar(x-6*width/num_methods, AC_STAR_epi_steps, width=width, label='AC-STAR', color=colors[0])
# axs[0].bar(x-2*width/num_methods, A_STAR_epi_steps, width=width, label='A-STAR', color=colors[1])
# axs[0].bar(x+2*width/num_methods, C_STAR_epi_steps, width=width, label='C-STAR', color=colors[2])
# axs[0].bar(x+6*width/num_methods, Track_epi_steps, width=width, label='Track', color=colors[3])
# axs[0].set_xticklabels(x_labels)
# axs[0].set_title("Average Tracking Steps Per Episode")
# axs[1].bar(x-6*width/num_methods, AC_STAR_epi_rew, width=width, label='AC-STAR', color=colors[0])
# axs[1].bar(x-2*width/num_methods, A_STAR_epi_rew, width=width, label='A-STAR', color=colors[1])
# axs[1].bar(x+2*width/num_methods, C_STAR_epi_rew, width=width, label='C-STAR', color=colors[2])
# axs[1].bar(x+6*width/num_methods, Track_epi_rew, width=width, label='Track', color=colors[3])
# axs[1].set_xticklabels(x_labels)
# axs[1].set_title("Average Reward Per Episode")
# axs[2].bar(x-6*width/num_methods, AC_STAR_step_rew, width=width, label='AC-STAR', color=colors[0])
# axs[2].bar(x-2*width/num_methods, A_STAR_step_rew, width=width, label='A-STAR', color=colors[1])
# axs[2].bar(x+2*width/num_methods, C_STAR_step_rew, width=width, label='C-STAR', color=colors[2])
# axs[2].bar(x+6*width/num_methods, Track_step_rew, width=width, label='Track', color=colors[3])
# axs[2].set_xticklabels(x_labels)
# axs[2].set_title("Average Reward Per Step")
# axs[3].bar(x-6*width/num_methods, AC_STAR_max_steps, width=width, label='AC-STAR', color=colors[0])
# axs[3].bar(x-2*width/num_methods, A_STAR_max_steps, width=width, label='A-STAR', color=colors[1])
# axs[3].bar(x+2*width/num_methods, C_STAR_max_steps, width=width, label='C-STAR', color=colors[2])
# axs[3].bar(x+6*width/num_methods, Track_max_steps, width=width, label='Track', color=colors[3])
# axs[3].set_xticklabels(x_labels)
# axs[3].set_title("Max Steps rate")
# # 设置图例（图例只在第一个子图中显示，其他子图隐藏图例）
# for ax in axs.flat:
#     ax.legend().remove()

# # 获取所有线条和标签
# lines, labels = fig.axes[-1].get_legend_handles_labels()

# # 创建一个全局图例，放在最底部并水平铺开
# fig.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, -0.01), ncol=4)

# # 显示图像
# plt.tight_layout(rect=[0, 0.05, 1, 1])  # 调整布局，留出底部空间
# plt.savefig("./test_track.jpg",dpi=300)


# nav任务测试
# 有两种类型的数据，n设置为2
total_width, n = 0.4, 4
size = 3
num_methods = 4
# 每种类型的柱状图宽度
width = total_width / n
x_labels = ["", "Normal", "", "Medium", "", "Complex"]  # 障碍物数量不同
x = np.arange(size)
AC_STAR_epi_len = [0, 0, 328.72]
AC_STAR_avg_dist = [0, 0, 2.44]
AC_STAR_avg_dist_std = [0, 0, 1.19]

A_STAR_epi_len = [0, 0, 431.81]
A_STAR_avg_dist = [0, 0, 2.57]
A_STAR_avg_dist_std = [0, 0, 1.21]

C_STAR_epi_len = [0, 0, 459.26]
C_STAR_avg_dist = [0, 0, 2.64]
C_STAR_avg_dist_std = [0, 0, 1.21]

NAV_epi_len = [0, 0, 286.85]
NAV_avg_dist = [0, 0, 2.55]
NAV_avg_dist_std = [0, 0,1.21]

# 创建子图
fig, axs = plt.subplots(1, 3, figsize=(15, 3))
color_name = "Set3"
select2 = (3, 4, 5, 6)  # 连续性色组图也可以从0-1之间选择
colors = plt.get_cmap(color_name)(select2)  # 从色组里选择颜色，我选择的是select2
# 绘制子图
axs[0].bar(
    x - 6 * width / num_methods,
    AC_STAR_epi_len,
    width=width,
    label="AC-STAR",
    color=colors[0],
)
axs[0].bar(
    x - 2 * width / num_methods,
    A_STAR_epi_len,
    width=width,
    label="A-STAR",
    color=colors[1],
)
axs[0].bar(
    x + 2 * width / num_methods,
    C_STAR_epi_len,
    width=width,
    label="C-STAR",
    color=colors[2],
)
axs[0].bar(
    x + 6 * width / num_methods,
    NAV_epi_len,
    width=width,
    label="Track",
    color=colors[3],
)
axs[0].set_xticklabels(x_labels)
axs[0].set_title("Average Path Lenth")
axs[1].bar(
    x - 6 * width / num_methods,
    AC_STAR_avg_dist,
    width=width,
    label="AC-STAR",
    color=colors[0],
)
axs[1].bar(
    x - 2 * width / num_methods,
    A_STAR_avg_dist,
    width=width,
    label="A-STAR",
    color=colors[1],
)
axs[1].bar(
    x + 2 * width / num_methods,
    C_STAR_avg_dist,
    width=width,
    label="C-STAR",
    color=colors[2],
)
axs[1].bar(
    x + 6 * width / num_methods,
    NAV_avg_dist,
    width=width,
    label="Track",
    color=colors[3],
)
axs[1].set_xticklabels(x_labels)
axs[1].set_title("Mean Distance")
axs[2].bar(
    x - 6 * width / num_methods,
    AC_STAR_avg_dist_std,
    width=width,
    label="AC-STAR",
    color=colors[0],
)
axs[2].bar(
    x - 2 * width / num_methods,
    A_STAR_avg_dist_std,
    width=width,
    label="A-STAR",
    color=colors[1],
)
axs[2].bar(
    x + 2 * width / num_methods,
    C_STAR_avg_dist_std,
    width=width,
    label="C-STAR",
    color=colors[2],
)
axs[2].bar(
    x + 6 * width / num_methods,
    NAV_avg_dist_std,
    width=width,
    label="Nav",
    color=colors[3],
)
axs[2].set_xticklabels(x_labels)
axs[2].set_title("Standard Deviation of The Mean Distance")
# axs[3].bar(
#     x - 6 * width / num_methods,
#     AC_STAR_max_steps,
#     width=width,
#     label="AC-STAR",
#     color=colors[0],
# )
# axs[3].bar(
#     x - 2 * width / num_methods,
#     A_STAR_max_steps,
#     width=width,
#     label="A-STAR",
#     color=colors[1],
# )
# axs[3].bar(
#     x + 2 * width / num_methods,
#     C_STAR_max_steps,
#     width=width,
#     label="C-STAR",
#     color=colors[2],
# )
# axs[3].bar(
#     x + 6 * width / num_methods,
#     Track_max_steps,
#     width=width,
#     label="Track",
#     color=colors[3],
# )
# axs[3].set_xticklabels(x_labels)
# axs[3].set_title("Max Steps rate")
# 设置图例（图例只在第一个子图中显示，其他子图隐藏图例）
for ax in axs.flat:
    ax.legend().remove()

# 获取所有线条和标签
lines, labels = fig.axes[-1].get_legend_handles_labels()

# 创建一个全局图例，放在最底部并水平铺开
fig.legend(lines, labels, loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=4)

# 显示图像
plt.tight_layout(rect=[0, 0.05, 1, 1])  # 调整布局，留出底部空间
plt.savefig("./test_nav.jpg", dpi=300)
