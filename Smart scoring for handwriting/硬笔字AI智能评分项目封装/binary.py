

import cv2
import numpy as np
from sys import argv
import os




def mask(dir):
    mask = np.zeros((166,166,3),np.uint8)
    mask1 = np.ones((166,166,3),np.uint8)#生成一个空彩色图像
    mask1 =mask1*255
    
    cv2.circle(mask,(83,83),80,(1,1,1),-1)
    cv2.circle(mask1,(83,83),80,(0,0,0),-1)
    #注意最后一个参数-1，表示对图像进行填充，默认是不填充的，如果去掉，只有椭圆轮廓了
    # plt.imshow(mask,'gray')   
    img=cv2.imread(dir)
    img = cv2.resize(img,(166,166))
    img1=img*mask
    img1=img1+mask1   
    return img,img1     #img是dir路径下的原图像,img1是有mask的图

def mask1(dir):
    mask = np.zeros((166,166,3),np.uint8)
    mask1 = np.ones((166,166,3),np.uint8)#生成一个空彩色图像
    mask1 =mask1*255
    
    cv2.circle(mask,(83,83),80,(1,1,1),-1)
    cv2.rectangle(mask, (83, 83), (163, 163), (1, 1, 1), thickness=-1)
    cv2.circle(mask1,(83,83),80,(0,0,0),-1)
    cv2.rectangle(mask1, (83, 83), (163, 163), (0, 0, 0), thickness=-1)
    
    #注意最后一个参数-1，表示对图像进行填充，默认是不填充的，如果去掉，只有椭圆轮廓了
    # plt.imshow(mask,'gray')   
    img=cv2.imread(dir) 
    img = cv2.resize(img,(166,166))
    img1=img*mask
    img1=img1+mask1      
    return img,img1  

def seg(img):
    #img=cv2.imread(dir)
    img = cv2.resize(img,(166,166))
    img= cv2.GaussianBlur(img, (5,5),0)
    img = cv2.threshold(img[:,:,0],120,255,type=0)[1]
    
    # img=cv2.medianBlur(img, 3)
    # img =np.where(img>120,0,255.0)    
    return img   #阈值分割得到的二值图



def run(image): 
    
    dict={}
    _,img1=mask1(image)
    
    img=seg(img1)
    img_dir=os.path.dirname(image)
    name=os.path.basename(image)
    
    cv2.imwrite(img_dir+"/{}_1.jpg".format(name[:-4]),img)
    dict["binary_img"]=img_dir+"/{}_1.jpg".format(name[:-4])
    return dict

    
    
    
    
    
    
    
    

