from segment_anything import SamPredictor, sam_model_registry
import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
import torch
from torch import nn
import math
import random
sys.setrecursionlimit(10000000) #例如这里设置为十万 
class Model():
    def __init__(self):
        self.input_box=[]
        self.init_img=None
        self.temp_coodrs=[]
        self.click_time=0
        self.input_img=[]
        # 加载SAM模型
        model_type="vit_h"
        sam_checkpoint="sam_vit_h_4b8939.pth"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device='cuda')
        self.predictor = SamPredictor(sam)
        self.img_size=[256,256]

    def show_mask(self,mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([255,255,255])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)
    
    def get_point_prompt(self,mask):
        coords=[]
        left=len(mask[0])
        right=0
        up=len(mask)
        down=0
        for i in range(len(mask[0])):
            for j in range(len(mask)):
                if mask[j][i]==True:
                    coords.append([i,j])
                    if i<left:
                        left=i
                    if j<up:
                        up=j
                    if i>right:
                        right=i
                    if j>down:
                        down=j
        new_box=[left,up,right,down]
        new_point=np.array([[left,up]])
        sorted_coords = sorted(coords, key=lambda x: (x[0], x[1]))
        if sorted_coords !=[]:
            x=[sorted_coords[0][0]]
            y=[]
            for i in range(1,len(sorted_coords)):
                if sorted_coords[i][0] != sorted_coords[i-1][0]:
                    x.append(sorted_coords[i][0])
            x_mid=x[len(x)//2]
            for i in range(0,len(sorted_coords)):
                if sorted_coords[i][0] == x_mid:
                    while i <len(sorted_coords) and sorted_coords[i][0] == x_mid:
                        y.append(sorted_coords[i][1])
                        i+=1
                    y_mid=y[len(y)//2]                
                    break
            new_point=[x_mid,y_mid]
            return new_point
        return []
    
    def tight_box(self,mask):
        left=len(mask[0])
        right=0
        up=len(mask)
        down=0
        for i in range(len(mask[0])):
            for j in range(len(mask)):
                if mask[j][i]==True:
                    if i<left:
                        left=i
                    if j<up:
                        up=j
                    if i>right:
                        right=i
                    if j>down:
                        down=j
        new_box=[left,up,right,down]

        return np.array(new_box)

    def show_points(self,coords, labels, ax, marker_size=375):
        pos_points = coords[labels==1]
        neg_points = coords[labels==0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
     
    def draw_point(self,event,x,y,flags,param):
        global input_box
        global temp_coodrs
        global time
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click_time+=1
            cv2.circle(self.input_img,(x,y),5,(255,0,0),-1)
            if self.click_time == 1:
                self.temp_coodrs=[x,y]
            elif self.click_time == 2:
                self.temp_coodrs.extend([x,y])
                self.input_box=np.array(self.temp_coodrs)
            elif self.click_time == 3:
                self.init_point=np.array([[x,y]])
                self.click_time =0

    def show_box(self,box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    def get_init_infos(self,img):
        self.input_img=img/255
        print("请点击鼠标左键确定初始矩形框,选定后按‘esc’退出")
        while True:
            cv2.imshow('image',self.input_img)
            if cv2.waitKey(20) & 0xFF == 27:
                break
        cv2.destroyWindow('image')
        self.init_box=self.input_box
        # self.init_point = np.array([[self.input_box[0]+(self.input_box[2]-self.input_box[0])//2,self.input_box[1]+(self.input_box[3]-self.input_box[1])//2]])
        print("初始矩形框为：",self.input_box)
        print("中心点坐标为：",self.init_point)

    def create_boxs(self,box,alpha,belta,num_boxs=10):
        """
        box: original box
        alpha:缩放因子,表示允许的缩放范围
        belta:位移因子,表示允许的位移变化大小
        """
        res=[]
        scale=[]
        rate=-alpha
        while rate<=alpha:
            scale.append(rate)
            rate+=0.01
        displacement=[ i for i in range(-belta,belta,10)]
        w=box[2]-box[0]
        h=box[3]-box[1]
        for i in range(num_boxs):
            candidate_box=box.copy()
            delta_scale=random.sample(scale,1)
            delta_w=w*(np.sqrt(1-delta_scale[0])-1)
            delta_h=h*(np.sqrt(1-delta_scale[0])-1)
            # print("candidate_box:",candidate_box)
            # print("delta_W:",delta_w)
            # print("delta_H:",delta_h)
            #对矩形面积进行放缩
            candidate_box[0]-=delta_w/2 
            candidate_box[2]+=delta_w/2 
            candidate_box[1]-=delta_h/2
            candidate_box[3]+=delta_h/2 
            for j in range(4):
                if candidate_box[j]<0:
                    candidate_box[j]=0
                elif candidate_box[j] >self.img_size[1]-1:
                    candidate_box[j]=self.img_size[1]-1
            #对矩形位置进行移动
            #上下移动
            delta_y=random.sample(displacement,1)
            # print("d_1",d,self.img_size)
            # print("before:",candidate_box)
            # print("delta_y",delta_y)
            candidate_box[1]= (candidate_box[1]-delta_y) 
            candidate_box[3] = (candidate_box[3]-delta_y) 
            # print("after:",candidate_box)
            #左右移动
            delta_x=random.sample(displacement,1)
            # print("d_2",d,self.img_size)
            candidate_box[0]=(candidate_box[0]-delta_x) 
            candidate_box[2]=(candidate_box[2]-delta_x)
            for j in range(4):
                if candidate_box[j]<0:
                    candidate_box[j]=0
                elif candidate_box[j] >self.img_size[1]-1:
                    candidate_box[j]=self.img_size[1]-1

            # print("final_box:",candidate_box)
            res.append(candidate_box)
        return res

    def get_segments(self,img):
        if self.init_img == None:
            self.init_img=1
            # cv2.namedWindow('image')
            # cv2.setMouseCallback('image',model.draw_point)
            # self.get_init_infos(img)
            self.init_box=np.array([104,103,181,187])
            self.init_point=np.array([[140,157]])
            self.predictor.set_image(img)
            self.point_prompt=self.init_point
            self.prompt_box=self.init_box
            masks, scores, logits = self.predictor.predict(
            point_coords=self.point_prompt,
            point_labels=np.array([1]),
            box=self.init_box,
            multimask_output=False,
            mask_input=None,#上一帧的分割结果
            )
            self.last_mask=masks[-1].copy()
            self.last_logists=logits.copy()
            img_features=self.predictor.get_image_embedding()
            img_features=img_features.detach().cpu()
            feature_box=[int(img_features.shape[-1]*self.init_box[0]/len(img[0])),int(img_features.shape[-2]*self.init_box[1]/len(img)),math.ceil(img_features.shape[-1]*self.init_box[2]/len(img[0])),math.ceil(img_features.shape[-2]*self.init_box[3]/len(img))]
            for i in range(len(feature_box)):
                if feature_box[i]>=img_features.shape[-1]:
                    feature_box[i]=img_features.shape[-1]-1
                if feature_box[i] <0:
                    feature_box[i]=0
            temp=np.array(img_features[:,:,feature_box[1],feature_box[0]][0])
            self.e_last=temp.copy()
            count=0
            for r in range(feature_box[1],feature_box[3]):
                for c in range(feature_box[0],feature_box[2]):
                    self.e_last+=np.array(img_features[:,:,r,c][0])
                    count+=1
            self.e_last=(self.e_last-temp)/count
        else:
            boxes=self.create_boxs(box=self.init_box,alpha=0.1,belta=30,num_boxs=200)
            boxes.append(self.init_box)
            self.predictor.set_image(img)
            new_boders=[]
            e_news=[]
            img_features=self.predictor.get_image_embedding()
            img_features=img_features.detach().cpu()
            for box in boxes:
                # 找到最合适的box
                # print("box",box)
                feature_box=[int(img_features.shape[-1]*box[0]/len(img[0])),int(img_features.shape[-2]*box[1]/len(img)),math.ceil(img_features.shape[-1]*box[2]/len(img[0])),math.ceil(img_features.shape[-2]*box[3]/len(img))]
                for i in range(len(feature_box)):
                    if feature_box[i]>=img_features.shape[-1]:
                        feature_box[i]=img_features.shape[-1]-1
                    if feature_box[i] <0:
                        feature_box[i]=0
                temp=np.array(img_features[:,:,feature_box[1],feature_box[0]][0])
                e_new=temp.copy()
                count=0
                for r in range(feature_box[1],feature_box[3]):
                    for c in range(feature_box[0],feature_box[2]):
                        # print("r,c:",r,c)
                        e_new+=np.array(img_features[:,:,r,c][0])
                        count+=1
                e_new=(e_new-temp)/count
                # print("count",count)
                new_boders.append(box)
                e_news.append(e_new)
            min_theta=100
            self.best_box=self.init_box
            e_last_fuzhi=np.linalg.norm(self.e_last)
            for i in range(len(e_news)):
                d=np.dot(e_news[i],self.e_last)
                d=d/(np.linalg.norm(e_news[i])*e_last_fuzhi)
                theta=np.arccos(d)
                if theta < min_theta:
                    min_theta=theta
                    self.best_box=new_boders[i].copy()
                    self.e_last=e_news[i].copy()
            # 根据最合适的box的分割结果与上一帧分割结果的重叠部分获得point_prompt
            # masks, scores, logits = self.predictor.predict(
            #     point_coords=None,
            #     point_labels=None,
            #     box=self.best_box,
            #     multimask_output=False,
            #     mask_input=self.last_logists,#上一帧的分割结果
            #     )
            # diff_img=self.last_mask * masks[-1]
            # self.last_point_prompt=self.point_prompt.copy()
            # self.point_prompt=self.get_point_prompt(diff_img)
            self.prompt_box=np.array([min(self.best_box[0],self.init_box[0]),min(self.best_box[1],self.init_box[1]),max(self.best_box[2],self.init_box[2]),max(self.best_box[3],self.init_box[3])])
            diff_box=np.array([max(self.best_box[0],self.init_box[0]),max(self.best_box[1],self.init_box[1]),min(self.best_box[2],self.init_box[2]),min(self.best_box[3],self.init_box[3])])
            self.point_prompt=[diff_box[0]+(diff_box[2]-diff_box[0])//2,diff_box[1]+(diff_box[3]-diff_box[1])//2]
            # if self.point_prompt == []:
            #     together_img=self.last_mask.copy()
            #     self.point_prompt=self.get_point_prompt(together_img)
            # else:
            #     together_img=self.last_mask.copy()
            #     for i in range(250):
            #         for j in range(250):
            #             if self.last_mask[j][i] == True or masks[-1][j][i] == True:
            #                 together_img[j][i]=True
            # self.prompt_box=self.tight_box(together_img)
            #综合点提示和box提示获得最终的分割结果
            if self.point_prompt==[]:
                # print("No point_prompt")
                self.point_prompt=self.last_point_prompt.copy()
                self.prompt_box=self.last_box.copy()  
                masks, scores, logits = self.predictor.predict(
                    point_coords=np.array([self.point_prompt]),
                    point_labels=np.array([1]),
                    box=self.prompt_box,
                    multimask_output=True,
                    mask_input=self.last_logists,#上一帧的分割结果
                    )
            else:
                print("point_prompt:",np.array(self.point_prompt))  
                masks, scores, logits = self.predictor.predict(
                    point_coords=np.array([self.point_prompt]),
                    point_labels=np.array([1]),
                    box=self.prompt_box,
                    multimask_output=True,
                    mask_input=self.last_logists,#上一帧的分割结果
                    )
                #将当前时刻数据保存为上一步数据
                self.init_box=self.tight_box(masks[-1])
                self.last_box=self.prompt_box.copy()
                self.last_logists=logits[-1].copy()
                self.last_logists=self.last_logists.reshape([1,256,256])
                self.last_mask=masks[-1].copy()
        return self.last_logists,masks[-1],self.init_box
if __name__ == '__main__':
    
    model=Model()
    for m in range(0,2222):
        img=cv2.imread('sim_imgs/train/observe_{}.jpg'.format(m))
        logits, _,_=model.get_segments(img)
        final_img=show_img=cv2.cvtColor(logits[-1],cv2.COLOR_GRAY2BGR)
        cv2.imwrite('./retune_result/final_mask_{}.jpg'.format(m),final_img*255)
 

    # boxs=model.create_boxs(box=np.array([65,50,159,186]),alpha=0.1,belta=20,num_boxs=20)
    # img = np.zeros((512, 512, 3), np.uint8)
    # for box in boxs:
    #     cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (random.randint(0,254), random.randint(0,254), random.randint(0,254)), 2)
    # # 显示图像
    # cv2.imshow("rectangle", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #输入图片
    img=cv2.imread('sim_imgs/train/observe_0.jpg')
    cv2.namedWindow('image')
    cv2.setMouseCallback('image',model.draw_point)
    #获得初始矩形框信息
    model.predictor.set_image(img)
    old_img_features=model.predictor.get_image_embedding()
    model.get_init_infos(img)

    #获得图像的分割结果
    # masks,scores,logits=model.get_segment_results_with_points(points=model.init_point)
    masks, scores, logits = model.predictor.predict(
        point_coords=model.init_point,
        point_labels=np.array([1]),
        box=model.init_box,
        multimask_output=False,
        mask_input=None,#上一帧的分割结果
        )
    point_prompt=model.init_point
    color = np.array([255,255,255])
    h, w = masks[-1].shape[-2:]
    mask_image = masks[-1].reshape(h, w, 1) * color.reshape(1, 1, -1)
    cv2.imwrite('./masks.jpg',mask_image)
    h, w = logits[-1].shape[-2:]
    mask_image = logits[-1].reshape(h, w, 1) * color.reshape(1, 1, -1)
    cv2.imwrite('./logits.jpg',mask_image)
    last_mask=masks[-1].copy()
    last_logists=logits.copy()
    # print("begin:",last_mask.shape)
    # last_mask=last_mask.reshape([1,1,256,256]) 

    # test_img=logits[-1][model.init_box[1]:model.init_box[3],model.init_box[0]:model.init_box[2]]
    # e_last=test_img.flatten()
    # print(e_last.shape)
    # print(test_img.shape)
    # plt.imshow(test_img,cmap='gray')
    # plt.show()
    # sys.exit(0)
    # test_img=test_img[:, :, np.newaxis]
    # test_img=np.repeat(test_img, 3, axis=2)
    model.predictor.set_image(img)
    old_img_features=model.predictor.get_image_embedding()
    old_img_features=old_img_features.detach().cpu()
    feature_box=[int(old_img_features.shape[-1]*model.init_box[0]/len(img[0])),int(old_img_features.shape[-2]*model.init_box[1]/len(img)),math.ceil(old_img_features.shape[-1]*model.init_box[2]/len(img[0])),math.ceil(old_img_features.shape[-2]*model.init_box[3]/len(img))]
    for i in range(len(feature_box)):
        if feature_box[i]>=old_img_features.shape[-1]:
            feature_box[i]=old_img_features.shape[-1]-1
        if feature_box[i] <0:
            feature_box[i]=0
    # old_center=[model.init_box[0]+(model.init_box[2]-model.init_box[0])//2,model.init_box[1]+(model.init_box[3]-model.init_box[1])//2]
    
    # feature_cord=[math.ceil(old_img_features.shape[-2]*(old_center[0]/len(img[0]))),math.ceil(old_img_features.shape[-1]*(old_center[1]/len(img)))]
    # print("old_center:",old_center)
    # print("feature_cord:",feature_cord)
    temp=np.array(old_img_features[:,:,feature_box[1],feature_box[0]][0])
    e_last=temp.copy()
    count=0
    for r in range(feature_box[1],feature_box[3]):
        for c in range(feature_box[0],feature_box[2]):
            e_last+=np.array(old_img_features[:,:,r,c][0])
            count+=1
    e_last=(e_last-temp)/count
    init_box=model.init_box
    # e_last=np.array(old_img_features[:,:,feature_cord[1],feature_cord[0]][0])
    # print(e_last.shape)
    # print(e_last)
    # sys.exit(0)
    for m in range(1,2222):
        print("Processing Image:",m)
        new_img=cv2.imread('sim_imgs/train/observe_{}.jpg'.format(m))
        temp_img=cv2.imread('sim_imgs/train/observe_{}.jpg'.format(m))
        boxes=model.create_boxs(box=init_box,alpha=0.1,belta=30,num_boxs=200)
        boxes.append(init_box)
        model.predictor.set_image(new_img)
        # for box in boxes:
        #     cv2.rectangle(new_img, (box[0], box[1]), (box[2], box[3]), (random.randint(0,254), random.randint(0,254), random.randint(0,254)), 2)
        # # 显示图像
        # cv2.imshow("rectangle", new_img)
        # cv2.waitKey(0)
        # cv2.destroyWindow("rectangle")
        new_boders=[]
        e_news=[]
        new_img_features=model.predictor.get_image_embedding()
        new_img_features=new_img_features.detach().cpu()
        for box in boxes:
            # 找到最合适的box
            # print("box",box)
            feature_box=[int(new_img_features.shape[-1]*box[0]/len(new_img[0])),int(new_img_features.shape[-2]*box[1]/len(new_img)),math.ceil(new_img_features.shape[-1]*box[2]/len(new_img[0])),math.ceil(new_img_features.shape[-2]*box[3]/len(new_img))]
            for i in range(len(feature_box)):
                if feature_box[i]>=new_img_features.shape[-1]:
                    feature_box[i]=new_img_features.shape[-1]-1
                if feature_box[i] <0:
                    feature_box[i]=0
            temp=np.array(new_img_features[:,:,feature_box[1],feature_box[0]][0])
            e_new=temp.copy()
            count=0
            for r in range(feature_box[1],feature_box[3]):
                for c in range(feature_box[0],feature_box[2]):
                    # print("r,c:",r,c)
                    e_new+=np.array(new_img_features[:,:,r,c][0])
                    count+=1
            e_new=(e_new-temp)/count
            # print("count",count)
            new_boders.append(box)
            # print(e_new.shape)
            # print(e_new)
            # sys.exit()
            e_news.append(e_new)
        dist=[]
        for i in range(len(e_news)):
            # print(e_news[i].detach().cpu().numpy())
            # print(e_last.detach().cpu().numpy())
            d=np.dot(e_news[i],e_last)
            # print("new_e:",e_news[i])
            # print("last_e:",e_last)
            # print("new",np.linalg.norm(e_news[i]))
            # print("last",np.linalg.norm(e_last))
            d=d/(np.linalg.norm(e_news[i])*np.linalg.norm(e_last))
            theta=np.arccos(d)
            dist.append([np.abs(theta),new_boders[i],e_news[i]])
            # loss=criterion(torch.tensor(e_news[i]).to('cuda'),torch.tensor(e_last).to('cuda'))
            # dist.append([loss,new_boders[i],e_news[i]])
        dist.sort(key=lambda x:x[0])
        # print("distance:",dist[0][0])
        # cv2.rectangle(temp_img, (dist[0][1][0], dist[0][1][1]), (dist[0][1][2], dist[0][1][3]), (random.randint(0,254), random.randint(0,254), random.randint(0,254)), 2)
        # cv2.imshow("result", temp_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        best_box=dist[0][1].copy()
        e_last=dist[0][2].copy()
        # 根据最合适的box的分割结果与上一帧分割结果的重叠部分获得point_prompt
        masks, scores, logits = model.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=best_box,
            multimask_output=False,
            mask_input=last_logists,#上一帧的分割结果
            )
        diff_img=last_mask * masks[-1]
        show_img=cv2.cvtColor(logits[-1],cv2.COLOR_GRAY2BGR)
        last_point_prompt=point_prompt.copy()
        point_prompt=model.get_point_prompt(diff_img)
        if point_prompt == []:
            together_img=last_mask.copy()
            point_prompt=model.get_point_prompt(together_img)
        else:
            together_img=last_mask.copy()
            for i in range(250):
                for j in range(250):
                    if last_mask[j][i] == True or masks[-1][j][i] == True:
                        together_img[j][i]=True
        # print("before",model.init_box)
        prompt_box=model.tight_box(together_img)
        # print("after",model.init_box)
        img = np.zeros((250, 250), np.uint8)
        for i in range(250):
            for j in range(250):
                if diff_img[i][j]==True:
                    img[i][j]=1
                else:
                    img[i][j]=0
        show_img_diff=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        img2 = np.zeros((250, 250), np.uint8)
        for i in range(250):
            for j in range(250):
                if together_img[i][j]==True:
                    img2[i][j]=1
                else:
                    img2[i][j]=0
        show_img_together=cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
        cv2.imwrite('./retune_result/diff_mask_{}.jpg'.format(m),show_img_diff*255)
        cv2.imwrite('./retune_result/together_mask_{}.jpg'.format(m),show_img_together*255)
        cv2.imwrite('./retune_result/best_box_mask_{}.jpg'.format(m),show_img*255)
        # flag=0
        # for i in range(model.img_size[0]):
        #     for j in range(model.img_size[1]):
        #         if last_mask_result[j][i] == True and masks[-1][j][i] == True:
        #             flag=1
        #             # if random.random()>0.98:
        #             point_prompt.append([i,j])
        #             point_labels.append(1)
        #             # point_prompt=[np.array([[i,j]]),np.array([1])]
        #             break
        #         if flag==1:
        #             break
        #综合点提示和box提示获得最终的分割结果

        
        if point_prompt==[]:
            print("No point_prompt")
            point_prompt=last_point_prompt.copy()
            prompt_box=last_box.copy()  
            masks, scores, logits = model.predictor.predict(
                point_coords=np.array([point_prompt]),
                point_labels=np.array([1]),
                box=prompt_box,
                multimask_output=True,
                mask_input=last_logists,#上一帧的分割结果
                )
        else:
            print("point_prompt:",np.array(point_prompt))  
            masks, scores, logits = model.predictor.predict(
                point_coords=np.array([point_prompt]),
                point_labels=np.array([1]),
                box=prompt_box,
                multimask_output=True,
                mask_input=last_logists,#上一帧的分割结果
                )
        cv2.rectangle(new_img, (best_box[0], best_box[1]), (best_box[2], best_box[3]), (0,255,0), 2)#BGR
        cv2.rectangle(new_img, (init_box[0], init_box[1]), (init_box[2], init_box[3]), (0, 0,255), 2)
        cv2.rectangle(new_img, (prompt_box[0], prompt_box[1]), (prompt_box[2], prompt_box[3]), (255, 0,0), 2)
        cv2.circle(new_img,(point_prompt[0],point_prompt[1]),5,(255,255,0),-1)
        #将当前时刻数据保存为上一步数据
        init_box=model.tight_box(masks[-1])
        last_box=prompt_box.copy()
        last_logists=logits[-1].copy()
        last_logists=last_logists.reshape([1,256,256])

        last_mask=masks[-1].copy()
        # print("last:",last_mask.shape)
        # for i, (mask, score) in enumerate(zip(masks, scores)):
        #     plt.figure(figsize=(10,10))
        #     plt.imshow(new_img)
        #     model.show_mask(mask, plt.gca())
        #     plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        #     plt.axis('off')
        #     plt.show()  

        # print(diff_img.shape)
        final_result=logits[-1].copy()
        
        
        final_img=show_img=cv2.cvtColor(final_result,cv2.COLOR_GRAY2BGR)

        cv2.imwrite('./retune_result/prompts_{}.jpg'.format(m),new_img)
        cv2.imwrite('./retune_result/final_mask_{}.jpg'.format(m),final_img*255)
    sys.exit()
    model.get_box(masks[-1])

    print(model.candidates)
    model.candidates.sort(reverse=True,key=lambda x: x[2])
    left=model.candidates[0][3]
    top=model.candidates[0][4]
    right=model.candidates[0][5]
    buttom=model.candidates[0][6]
    new_points=np.array([[int(left+(right-left)//2),int(top+(buttom-top))]])
    img=cv2.imread('real_imgs/train/observe_1409.jpg')
    print(222)
    model.input_img=img
    #获得图像的分割结果
    masks,scores,logits=model.get_segment_results_with_box(points=new_points,last_mask_ouput=logits[-1].reshape([1,256,256])*255)
    
    color = np.array([255,255,255])
    h, w = masks[-1].shape[-2:]
    mask_image = masks[-1].reshape(h, w, 1) * color.reshape(1, 1, -1)
    cv2.imwrite('./masks_1.jpg',mask_image)
    h, w = logits[-1].shape[-2:]
    mask_image = logits[-1].reshape(h, w, 1) * color.reshape(1, 1, -1)
    cv2.imwrite('./logits_1.jpg',mask_image)

# for i, (mask, score) in enumerate(zip(masks, scores)):
#     plt.figure(figsize=(10,10))
#     plt.imshow(img)
#     show_mask(mask, plt.gca())
#     show_points(input_point, input_label, plt.gca())
#     plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
#     plt.axis('off')
#     plt.show()  
# plt.figure(figsize=(10,10))
# plt.imshow(img)
# show_points(input_point, input_label, plt.gca())
# plt.axis('on')
# plt.show()  
# masks, _, _ = predictor.predict(box=np.array([125,125,68,68]))
