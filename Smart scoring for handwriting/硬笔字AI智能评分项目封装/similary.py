
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


def run(dict_hi):
    table=dict_hi['table']
    path=dict_hi["img_path"]
    name=os.path.basename(path)
    dir1=os.path.dirname(path)
    name1=name[:-4]+"_2.jpg"
    path_p=os.path.join(dir1,name1)
    compare=[]
    for i in dict_hi["results"]:
        img_s = cv2.imread(i)
        img_t = cv2.imread(path_p)

        a_0,b_s,c_t = inter(img_s,img_t)
        height_s = img_s.shape[0]
        height_t = img_t.shape[0]
        weight_p = img_s.shape[1]
        weight_d = img_t.shape[1]
        height_rate = min(height_s,height_t)/max(height_s,height_t)
        weight_rate = min(weight_p,weight_d)/max(weight_p,weight_d)
        rate = min(height_s/weight_p,height_t/weight_d)/max(height_s/weight_p,height_t/weight_d)
        prob = rate*weight_rate*height_rate
        simage = Image.fromarray(np.uint8(b_s))
        timage = Image.fromarray(np.uint8(c_t))
        cosin=image_similarity_vectors_via_numpy(simage, timage)
        cosin=cosin*prob
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
    
    
    
 
    
    




