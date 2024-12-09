import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
def get_flow(frame1,frame2,center,delta_x,delta_y):
    # 读取两帧图像 
    # prev_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    # next_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    prev_frame=frame1.astype(np.float32) 
    next_frame=frame2.astype(np.float32)


    mean_flow_x = 0
    mean_flow_y = 0
    # 定义指定区域的坐标(x, y, width, height) 
    x, y, w, h = int(center[0]), int(center[1]), delta_x, delta_y 
    roi = (x, y, w, h) 

    # 提取指定区域的子图像
    prev_roi = prev_frame[y:y+h, x:x+w]

    # 在指定区域内检测特征点
    feature_params = dict(maxCorners=100, qualityLevel=0.01, minDistance=10, blockSize=7)
    p0 = cv2.goodFeaturesToTrack(prev_roi, mask=None, **feature_params)

    # 将特征点坐标转换为整个图像中的坐标
    if p0 is not None:  # 检查是否检测到特征点
        p0 += np.array([x, y])

        # Lucas-Kanade 光流法参数
        lk_params = dict(winSize=(15, 15), maxLevel=2,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        prev_frame=prev_frame.astype(np.uint8)
        next_frame=next_frame.astype(np.uint8)

        # 计算光流
        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_frame, next_frame, p0, None, **lk_params)

        # 选择跟踪成功的点
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # 计算光流在 x 和 y 方向的位移
        flow_x = good_new[:, 0] - good_old[:, 0]
        flow_y = good_new[:, 1] - good_old[:, 1]

        # 计算平均位移
        mean_flow_x = np.mean(flow_x)
        mean_flow_y = np.mean(flow_y)

        # print(f"Mean flow in X direction: {mean_flow_x}")
        # print(f"Mean flow in Y direction: {mean_flow_y}")

    else:
        print("No features found in the specified region.")

    return torch.tensor([mean_flow_x,mean_flow_y])

    # # 读取图像
    # # frame1 = cv2.imread('./sim_imgs/train/observe_0.jpg')
    # # frame2 = cv2.imread('./sim_imgs/train/observe_1.jpg')
    
    # # 转换为灰度图像
    # prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    # next_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # # 计算光流
    # flow = cv2.calcOpticalFlowFarneback(prev=prev_gray, next=next_gray, flow=None, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=3, poly_sigma=1.5, flags=0)
    # # 创建一个用于可视化的空图像
    # h, w = flow.shape[:2]
    # # sum_x=0
    # # sum_y=0
    # # left_x = center[1]-delta_y if center[1]-delta_y>0 else 0
    # # right_x = center[1]+delta_y+1 if center[1]+delta_y+1<h else h
    # # left_y=center[0]-delta_x if center[0]-delta_x>0 else 0
    # # right_y = center[0]+delta_x+1 if center[0]+delta_x+1<w else w
    # # for i in range(int(left_x),int(right_x)):
    # #     for j in range(int(left_y),int(right_y)):
    # #         sum_x+=flow[i,j][0]
    # #         sum_y+=flow[i,j][1]



    # # 可视化光流
    # # for y in range(0, h, 10):
    # #     for x in range(0, w, 10):
    # #         vx, vy = flow[y, x]
    # #         cv2.line(vis, (x, y), (int(x + vx), int(y + vy)), (0, 255, 0), 1)
    # #         cv2.circle(vis, (x, y), 1, (0, 0, 255), -1)s
    # target_x= int(center[1]) if int(center[1]) <256 else 255
    # target_y= int(center[0]) if int(center[0]) <256 else 255
    # vx, vy = flow[target_x,target_y]
    # # vx, vy = sum_x/delta_x,sum_y/delta_y
    # # print(vx,vy)
    # return [vx,vy]
    # # cv2.circle(frame2, (int(center[0]),int(center[1])), 1, (0, 0, 255), -1)
    # # cv2.circle(frame2, (int(center[0] + vx), int(center[1] + vy)), 1, (0, 255, 0), -1)


                  

    # # # 显示结果
    # # plt.imshow(frame2)
    # # plt.show()
