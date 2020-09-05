# -*- coding: utf-8 -*-

from PIL import Image
import numpy as np
import pandas as pd
from numpy import average, dot, linalg
from sys import argv
import os
import json
import threading
lock = threading.Lock()
# 对图片进行统一化处理
def get_thum(image, size=(116, 116), greyscale=False):
    # 利用image对图像大小重新设置, Image.ANTIALIAS为高质量的
    image = image.resize(size, Image.ANTIALIAS)
    if greyscale:
        # 将图片转换为L模式，其为灰度图，其每个像素用8个bit表示
        image = image.convert('L')
    return image
	
# 计算图片的余弦距离
def image_similarity_vectors_via_numpy(image1, image2):
    image1 = get_thum(image1)
    image2 = get_thum(image2)
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


def run(dict_hi):
    table=dict_hi['table']


    path=dict_hi["img_path"]
    name=os.path.basename(path)
    dir1=os.path.dirname(path)
    name1=name[:-4]+"_2.jpg"
        
    path_p=os.path.join(dir1,name1)
    compare=[]
    for i in dict_hi["results"]:
        pimage = Image.open(path_p)
        dimage = Image.open(i)
        height_p = np.asarray(pimage).shape[0]
        height_d = np.asarray(dimage).shape[0]
        weight_p = np.asarray(pimage).shape[1]
        weight_d = np.asarray(dimage).shape[1]
        height_rate = min(height_p,height_d)/max(height_p,height_d)
        weight_rate = min(weight_p,weight_d)/max(weight_p,weight_d)
        rate = min(height_p/weight_p,height_d/weight_d)/max(height_p/weight_p,height_d/weight_d)
        ratio = rate*weight_rate*height_rate
        cosin=image_similarity_vectors_via_numpy(pimage, dimage)
        cosin=cosin*ratio
        compare.append(cosin)
    k=np.argmax(compare)
    dict_hi["results"]=dict_hi["results"][k]
    dict_hi["threshold"]=compare[k]
    
    data = pd.read_csv(table)
    min_num = np.array(data.iloc[:,0]) #最小值阈值
    max_num = np.array(data.iloc[:,1]) #最大值阈值
    
    
    num=os.path.basename(dict_hi["results"])[3:-4]
    num=int(num)
    if compare[k]<min_num[num]:
        dict_hi["response"]="bad"
       
    elif min_num[num]<=compare[k]<=max_num[num]:
        dict_hi["response"]="median"
           
    elif compare[k]>max_num[num]:
        dict_hi["response"]="good"
    
  
    # f1 = open(path+"_final",'w')
    # f1.write(str(dict_hi))
    # f1.close()
    # print(path+"_final")
    
    
    
    path1=dict_hi['path']  #统计好坏
    
    lock.acquire(10)
    try:
        f1=open(path1,"r")
        b = f1.readlines()
        f1.close()
            
        i=int(b[0])
        j=int(b[1])
        k=int(b[2])
        if dict_hi["response"]=="good":
            i=i+1
        elif dict_hi["response"]=="median":
            j=j+1
        elif dict_hi["response"]=="bad":
            k=k+1
            
        f2=open(path1,"w")
        f2.writelines("{}\n{}\n{}".format(i,j,k))
        f2.close()
    
    finally:
        lock.release()
    
    return dict_hi
    
    
    
 
    
    




