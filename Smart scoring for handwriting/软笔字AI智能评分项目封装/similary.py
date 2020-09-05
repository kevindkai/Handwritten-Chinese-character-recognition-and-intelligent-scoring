# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 10:37:14 2020

@author: lvfucheng
"""
from PIL import Image
import numpy as np
import pandas as pd
from numpy import average, dot, linalg
from sys import argv
import os
import json
import threading
import imutils
import cv2
lock = threading.Lock()

# 对图片进行统一化处理
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

	

def image_similarity_vectors_via_numpy(image1, image2):

    images = [image1, image2]
    vectors = []
    norms = []
    for image in images:
        vector = []
        for pixel_tuple in image.getdata():
            vector.append(average(pixel_tuple))
        vectors.append(vector)
        # linalg=linear（线性）+algebra（代数），norm则表示范数
        # 求图片的范数
        norms.append(linalg.norm(vector, 2))
    a, b = vectors
    a_norm, b_norm = norms
    # dot返回的是点积，对二维数组（矩阵）进行计算
    res = dot(a / a_norm, b / b_norm)
    return res

threshold = 0.85 #评判的阈值
def run(dict_hi):
   # table=dict_hi['table']
    path=dict_hi["csv_path"]
   # name=os.path.basename(path)
   # dir1=os.path.dirname(path)
   # name1=name[:-4]+"_2.jpg"
   # path_p=os.path.join(dir1,name1)
    path_p = dict_hi["results"][0]
    result = {"name":[],"cosin":[],"perform":[],"response":[]}
    for i in dict_hi["results"]:
        img_s = cv2.imread(i)
        img_t = cv2.imread(path_p)

        a_0,b_s,c_t = inter(img_s,img_t)
        #height_s = img_s.shape[0]
        #height_t = img_t.shape[0]
        #weight_p = img_s.shape[1]
        #weight_d = img_t.shape[1]
        #height_rate = min(height_s,height_t)/max(height_s,height_t)
        #weight_rate = min(weight_p,weight_d)/max(weight_p,weight_d)
        #rate = min(height_s/weight_p,height_t/weight_d)/max(height_s/weight_p,height_t/weight_d)
        #prob = rate*weight_rate*height_rate
        result["name"].append(i)
        simage = Image.fromarray(np.uint8(b_s))
        timage = Image.fromarray(np.uint8(c_t))
        cosin=image_similarity_vectors_via_numpy(simage, timage)
        result["cosin"].append(cosin)
        #cosin=cosin*prob
        if cosin<threshold:
            result["response"].append("bad")
            result["perform"].append(0)
        else:
            result["response"].append("good")
            result["perform"].append(1)
    dp = pd.DataFrame(result)
    dp.to_csv(path,encoding="utf-8-sig",index=None)
    return result
    
    

    
    




