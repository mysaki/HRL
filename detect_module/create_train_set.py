import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
# 设置路径
data_dir = "images/LostRealImages"
output_dir = "labeled_images/real"

# # 确保输出目录存在
# os.makedirs(output_dir, exist_ok=True)

# 获取所有图像文件
image_files = [f for f in os.listdir(data_dir) if f.endswith((".png", ".jpg", ".jpeg"))]
i = 747
for image_file in image_files:
    print("On process:",i)
    # 读取图像
    img = plt.imread(os.path.join(data_dir, image_file))
    # img = img.astype(np.uint8)
    # plt.imshow(img)
    # plt.show(block=False)


    # 等待用户输入
    # label = input("输入 '1' 表示目标出现，输入 '0' 表示目标未出现: ")

    # 构建新的文件名
    new_label = "lost" #if label == "1" else "lost"
    new_image_name = f"{new_label}_{i}_real_{'.jpg'}"
    i += 1
    # plt.close()

    # 保存带标签的新图像
    plt.imsave(os.path.join(output_dir, new_image_name), img)

    # 关闭图像窗口
    # cv2.waitKey(0)  # 等待键盘输入
    # cv2.destroyAllWindows()
