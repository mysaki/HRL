import matplotlib.pyplot as plt
import cv2
import sys
import os
import numpy as np
import logging
from .flow import get_flow
import torch
sys.setrecursionlimit(100000) #例如这里设置为十万 

class Transfer():
    def __init__(self):
        self.init_img = None # 初始图片，用来打参考中心点
        self.target_center=[0,0]
        self.boder_box=[]
        self.click_count=0
        self.is_labeled =True
        self.candidates=[]
        self.res=5 #阈值，判断与上一个中心点的偏离是否超出允许范围
        self.step=0
        self.log()

    def log(self):
        # 第一步，创建一个logger
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)  # Log等级总开关  此时是INFO

        # 第二步，创建一个handler，用于写入日志文件
        logfile = './log.txt'
        fh = logging.FileHandler(logfile, mode='a')  # open的打开模式这里可以进行参考
        fh.setLevel(logging.INFO)  # 输出到file的log等级的开关

        # 第三步，再创建一个handler，用于输出到控制台
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)   # 输出到console的log等级的开关

        # 第四步，定义handler的输出格式（时间，文件，行数，错误级别，错误提示）
        formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # 第五步，将logger添加到handler里面
        logger.addHandler(fh)
        logger.addHandler(ch)

    def draw_point(self,event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # cv2.circle(self.init_img,(87,104),5,(255,0,0),-1)
            # cv2.circle(self.init_img,(157,186),5,(255,0,0),-1)
            if self.click_count == 0:
                cv2.circle(self.init_img,(x,y),5,(0,0,255),-1)
                self.target_center=[x,y]
                self.click_count+=1
                print("center point:",self.target_center)
            elif self.click_count == 1:
                cv2.circle(self.init_img,(x,y),5,(255,0,0),-1)
                self.boder_box.append([x,y])
                self.click_count+=1
                print("first point of boder box:",self.boder_box[0])
            elif self.click_count == 2:
                cv2.circle(self.init_img,(x,y),5,(255,0,0),-1)
                self.boder_box.append([x,y])
                print("second point of boder box:",self.boder_box[1])
                self.click_count+=1

    def numIslands(self, grid,boder_box) -> int: #递归的求法太耗时了
        def zero (grid,x,y,max_x,max_y):
            grid[x][y]=False
            self.amount+=1
            if y < self.left:
                self.left = y
            if y > self.right:
                self.right = y
            if x < self.top:
                self.top = x
            if x > self.bottom:
                self.bottom = x
            if x+1 < max_x and grid[x+1][y]==True:
                grid= zero(grid,x+1,y,max_x,max_y)
            if y+1 < max_y and grid[x][y+1]==True:
                grid= zero(grid,x,y+1,max_x,max_y)
            if x-1 >=0 and grid[x-1][y]==True:
                grid= zero(grid,x-1,y,max_x,max_y)
            if y-1 >=0 and grid[x][y-1]==True:
                grid= zero(grid,x,y-1,max_x,max_y)

            if y-1 >=0 and x-1>=0 and grid[x-1][y-1]==True:
                grid= zero(grid,x-1,y-1,max_x,max_y)

            if y-1 >=0 and x+1< max_x and grid[x+1][y-1]==True:
                grid= zero(grid,x+1,y-1,max_x,max_y)

            if y+1 < max_y and x+1< max_x and grid[x+1][y+1]==True:
                grid= zero(grid,x+1,y+1,max_x,max_y)

            if y+1 < max_y and x-1>=0 and grid[x-1][y+1]==True:
                grid= zero(grid,x-1,y+1,max_x,max_y)

            return grid

        count = 0
        self.candidates=[]
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j]==True:
                    count+=1
                    self.left=len(grid)
                    self.right=0
                    self.top=len(grid)
                    self.bottom=0
                    self.amount=0
                    # if i > boder_box[1][0] or i < boder_box[0][0] or j > boder_box[1][1] or j < boder_box[0][1]:
                    #     continue
                    grid=zero(grid,i,j,len(grid),len(grid))
                    self.candidates.append([j,i,self.amount,self.left,self.top,self.right,self.bottom])

        return count

    def get_init_target(self,img): # 打初始参考中心点
        self.init_img=img
        cv2.namedWindow('image')
        cv2.setMouseCallback('image',self.draw_point)
        while True:
            cv2.imshow('image',self.init_img)
            if cv2.waitKey(20) & 0xFF == 27:
                break
        cv2.destroyWindow('image')

    def get_info(self,img,original_img,rgb_img,epoch,task_mode):
        # if not self.is_labeled: #最开始需要进行标记
        #     # self.get_init_target(img)
        #     self.is_labeled=True
        # else:#如果已经标记过了，那就更新边界框和中心点
        #     self.get_new_info(img)
        if self.step == 0:
            self.pre_img = rgb_img
            self.pre_seg_img = img
        self.task_mode=task_mode
        self.get_new_info(img,original_img,rgb_img,epoch)

        angle_info=np.arctan2((self.target_center[0]-img.shape[0]/2)*np.tan(np.radians(34.5)),(img.shape[0]/2))
        box_area=np.abs(self.boder_box[0][0]-self.boder_box[1][0])*np.abs(self.boder_box[0][1]-self.boder_box[1][1])
        self.pre_img=rgb_img.copy()
        self.pre_seg_img = self.seg_img.copy()
        img_ratio=self.pixels/(box_area+1e-8)
        return angle_info,self.boder_box,self.target_center,self.temp_center,img_ratio

    def get_boder(self,img): # 获得像素块的边界
        self.init_img=img
        cv2.namedWindow('image')
        cv2.setMouseCallback('image',self.draw_point)
        while True:

            cv2.imshow('image',self.init_img)
            if cv2.waitKey(20) & 0xFF == 27:
                break
        cv2.destroyWindow('image')

    def calcu_dist(self,n1,n2):
        dist=np.sqrt((n1[0]-n2[0])**2+(n1[1]-n2[1])**2)
        return dist

    def calcu_area(self,boder_box):
        h=-boder_box[0][1]+boder_box[1][1]
        w=-boder_box[0][0]+boder_box[1][0]
        return h*w

    def check_info(self,old_center,old_boder,new_center,new_boder,dist,rgb_img):
        final_center=[0,0]
        final_boder=[[],[]]

        if self.calcu_area(old_boder)<100:
            h=-new_boder[0][1]+new_boder[1][1]
            w=-new_boder[0][0]+new_boder[1][0]
        else:
            h=-old_boder[0][1]+old_boder[1][1]
            w=-old_boder[0][0]+old_boder[1][0]

        self.predict_direction=get_flow(self.pre_seg_img,self.seg_img,old_center,50,50)
        if torch.isnan(self.predict_direction).any():
            self.predict_direction = torch.zeros_like(self.predict_direction)
        final_center=[old_center[0]+self.predict_direction[0]*1.0+(new_center[0]-old_center[0])*0.1,old_center[1]]
        final_boder=[[final_center[0]-w/2,final_center[1]-h/2],[final_center[0]+w/2,final_center[1]+h/2]]
        for i in range(len(final_boder)):
            for j in range(len(final_boder[0])):
                if final_boder[i][j]>255:
                    final_boder[i][j]=255
                elif final_boder[i][j]<0:
                    final_boder[i][j]=0
        return final_center,final_boder     
    def get_new_info(self,img,original_img,rgb_img,epoch):
        # logging.info("epoch - {} - step -{} ".format(epoch,self.step))
        """
        更新维护边界框以及中心点
        """
        process_img=img
        self.seg_img = img.copy()
        self.numIslands(process_img.copy(),self.boder_box) # 找到分割图像里每个像素块左上角的坐标,及每个像素块中的像素数量
        self.candidates.sort(key=lambda x: x[2],reverse=True)# 按每个像素块中的像素数量进行排序，表示该像素块是追踪目标的可能从大到小排列
        if self.candidates != []:
            if self.is_labeled == False:
                new_boder_box=[[self.candidates[0][3],self.candidates[0][4]],[self.candidates[0][5],self.candidates[0][6]]]
                self.target_center=[new_boder_box[0][0]+(new_boder_box[1][0]-new_boder_box[0][0])/2,new_boder_box[0][1]+(new_boder_box[1][1]-new_boder_box[0][1])/2]
                self.boder_box=new_boder_box
                self.temp_center=self.target_center.copy()
                self.is_labeled = True
                self.pixels=self.candidates[0][2]
            else:
                min_dist = 1e9
                old_center = self.target_center
                c = 0
                for candidate in self.candidates:
                    if c > 5:
                        break
                    c+=1               
                    new_boder_box=[[candidate[3],candidate[4]],[candidate[5],candidate[6]]]
                    # 用border_box的中心点作为新的中心点
                    new_center=[new_boder_box[0][0]+(new_boder_box[1][0]-new_boder_box[0][0])/2,new_boder_box[0][1]+(new_boder_box[1][1]-new_boder_box[0][1])/2]
                    dist=np.sqrt((new_center[0]-self.target_center[0])**2+(new_center[1]-self.target_center[1])**2)

                    if dist < min_dist:
                        min_dist = dist
                        min_center = new_center
                        min_boder_box = new_boder_box
                        self.pixels = candidate[2]
                self.temp_center = min_center
                self.target_center,self.boder_box=self.check_info(self.target_center.copy(),self.boder_box.copy(),min_center.copy(),min_boder_box.copy(),min_dist,rgb_img)
                if self.step == 0:
                    print("old_center:",old_center,"temp_center:",self.temp_center,"target_center:",self.target_center)               
                self.save_img(original_img, rgb_img, old_center, epoch)
        else:
            # print("Target not found!")
            self.target_center=[128,128]
            self.boder_box =[[90,90],[200,200]]
            self.pixels=0
        self.step+=1
    def reset(self):
        self.target_center=[150,150]
        self.temp_center=[150,150]
        self.boder_box =[[100,100],[200,200]]
        self.step = 0
        self.pixels=0
        self.init_img = None # 初始图片，用来打参考中心点
        self.pre_img = None
        # self.target_center=[0,0]
        # self.boder_box=[]
        self.click_count=0
        self.is_labeled =True
        self.candidates=[]

    def plot_img(self,img):
        pass

    def save_img(self, img, rgb_img, old_center, epoch, text_task):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale=0.5
        thickness=2
        text= str((self.target_center[0]-img.shape[0]/2)/(img.shape[0]/2))
        color=(0,0,255)
        show_img = np.transpose(img, (1, 2, 0))
        angle_info=np.arctan2((self.target_center[0]-img.shape[0]/2)*np.tan(np.radians(34.5)),(img.shape[0]/2))
        angle_info = np.degrees(angle_info)
        text_angle = str(angle_info)
        show_img=cv2.cvtColor(show_img,cv2.COLOR_GRAY2BGR)
        cv2.imwrite('./detect_result/{}/original_{}.jpg'.format(epoch,self.step),show_img*255)
        cv2.imwrite('./detect_result/{}/rgb_{}.jpg'.format(epoch,self.step),rgb_img)
        if os.path.exists('./detect_result/{}'.format(epoch)) == False:
            os.mkdir('./detect_result/{}'.format(epoch))
        cv2.rectangle(show_img,(int(self.boder_box[0][0]),int(self.boder_box[0][1])),(int(self.boder_box[1][0]),int(self.boder_box[1][1])),color=(210,200,0),thickness=5)
        cv2.putText(show_img,text, (int(self.target_center[0])+10, int(self.target_center[1])+10), font, font_scale, color, thickness)
        # cv2.putText(show_img,text_task, (50, 50), font, font_scale, (100,0,100), thickness)
        cv2.putText(show_img,text_angle, (150, 150), font, font_scale, (100,200,100), thickness)
        cv2.arrowedLine(show_img,(int(old_center[0]),int(old_center[1])),(int(self.temp_center[0]),int(self.temp_center[1])),(0,0,255),3,cv2.LINE_AA)
        cv2.circle(show_img,(int(self.target_center[0]),int(self.target_center[1])),5,(255,0,255),-1)
        cv2.arrowedLine(show_img,(int(old_center[0]),int(old_center[1])),(int(old_center[0]+self.predict_direction[0]*3),int(old_center[1]+self.predict_direction[1]*3)),(255,0,0),3,cv2.LINE_AA)
        cv2.circle(show_img,(int(old_center[0]),int(old_center[1])),5,(0,255,0),-1)
        cv2.imwrite('./detect_result/{}/{}.jpg'.format(epoch,self.step),show_img*255)


if __name__ == '__main__':
    trans=Transfer()
    for m in range(40,1001):
        img=cv2.imread('./outputs/test/seg_{}.png'.format(m),cv2.IMREAD_GRAYSCALE)
        print("img:seg_{}.png".format(m))
        angle_info,dist_info=trans.get_info(img)
        print("new_center:",trans.target_center)
        print("angle_info:",angle_info,"dist_info",dist_info)
        cv2.namedWindow('result')
        while True:
            show_img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
            cv2.circle(show_img,(int(trans.target_center[0]),int(trans.target_center[1])),5,(0,0,255),-1)
            cv2.imshow('result',show_img)
            if cv2.waitKey(20) & 0xFF == 27:
                break
        cv2.destroyWindow('result')
