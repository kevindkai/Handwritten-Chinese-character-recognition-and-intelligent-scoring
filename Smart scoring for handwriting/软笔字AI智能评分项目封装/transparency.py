# -*- coding: utf-8 -*-


import numpy as np
from sys import argv
import pandas as pd
import os
import cv2
import imutils
import json




def inter(img,img1):  #img是标准字的外接矩形图，img1是测试字的外接矩形图
    h,w,c=img.shape
    h1,w1,c=img1.shape
    
    img2 = imutils.resize(img1, height = h)
    img3 = imutils.resize(img1, width = w)
    m2,n2,_=img2.shape
    m3,n3,_=img3.shape
    if m2*n2<m3*n3:
        img_resize=img2
    
    else:
        img_resize=img3
    
    m,n,_=img_resize.shape
    bground=np.zeros((max(h,m),max(w,n),3))
    bground1=np.zeros((max(h,m),max(w,n),3))
    
    mb,nb,_=bground.shape
    # print(img_resize.shape,img.shape,mb,nb)
    bground[int((mb-h)/2):int((mb+h)/2),int((nb-w)/2):int((nb+w)/2),:]=img
    bground1[int((mb-m)/2):int((mb+m)/2),int((nb-n)/2):int((nb+n)/2),:]=img_resize

    bground=bground.astype(np.uint8)
    bground1=bground1.astype(np.uint8)
 
    return img_resize,bground,bground1 
    #img_resize是测试图片等比例放缩的图，bground标准字放到背景板中的图，bground1是测试字。。。。
def trans_img(bg,bg1):
    alpha_channel =bg[:,:,0] 
    bg=np.where(bg>120,0,255)
    bg=bg.astype(np.uint8)
    
    b_channel, g_channel, r_channel = cv2.split(bg)

   
    #creating a dummy alpha channel image.
    imb = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
    imb[:,:,2]=np.where(imb[:,:,2]==0,255,255)
    imb[:,:,2]=imb[:,:,2].astype(np.uint8)
    
    
    alpha_channel1 =bg1[:,:,0]
    bg1=np.where(bg1>120,0,255)
    bg1=bg1.astype(np.uint8)
    
    b_channel1, g_channel1, r_channel1 = cv2.split(bg1)

   
    #creating a dummy alpha channel image.
    imb1 = cv2.merge((b_channel1, g_channel1, r_channel1, alpha_channel1))
 

    return imb,imb1

def run(dict_hi):
    dict_final={}
    #输出透明图
    dir=dict_hi["results"]
    print("dir",dir)
    dir1=dict_hi["img_path"]
    print("dir1",dir1)
    img=os.path.basename(dir1)
    print("img",img)
    file=os.path.dirname(dir1)
    print("file",file)
    dir2=os.path.join(file,img[:-4]+"_2.jpg")
    print("dir2",dir2)
    dir3=os.path.join(file,img[:-4]+"_3.png")
    print("dir3",dir3)
    dir4=os.path.join(file,img[:-4]+"_4.png")
    dir_b=os.path.join(file,img[:-4]+"_1.jpg")
    
    img=cv2.imread(dir)
    img1=cv2.imread(dir1)
    a1,a2,a3=inter(img,img1)
    imb,imb1=trans_img(a2,a3)

    cv2.imwrite(dir3,imb)
    cv2.imwrite(dir4,imb1)
    
    dict_hi["binary"]=dir_b
    dict_hi["trans"]=[dir3,dir4]
    

    dict_final["binary"]=dir_b
    #dict_final["response"]=dict_hi["response"]
    dict_final["trans_st"]=dir3
    dict_final["trans_hand"]=dir4
    return dict_final

