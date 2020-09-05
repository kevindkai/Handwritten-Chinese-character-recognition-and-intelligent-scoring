# -*- coding: utf-8 -*-


import cv2
import numpy as np
from sys import argv
import os 
import imutils
import json


def retange(image):
    image=image[10:664,7:664]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5),0)
    edged = cv2.Canny(blurred,20,150)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST,cv2.CHAIN_APPROX_TC89_L1)
    cnts = cnts[1] if  imutils.is_cv2()  else   cnts[0]
    x=[]
    y=[]
    for cnt in cnts:
        ares = cv2.contourArea(cnt)
        if ares<25:
            continue
        else:
            for k in cnt:
                x.append(k[0][0])
                y.append(k[0][1])
    if x and y :
        x_min=min(x)
        x_max=max(x)
        y_min=min(y)
        y_max=max(y)


        img1 = image[y_min:y_max,x_min:x_max]


    else: 
        img1 = image

        # print("该图片无法找到外接矩形")
    return image,img1         #image是二值图上画白色的矩形框,img1是裁剪的矩形框图，

def run(dict_hi):
    # base64str = base64.b64decode(argv[1])
    # path=argv[1]#字典路径
    
    # f = open(path,'r')
    # a = f.read()
    # dict_hi = eval(a)
    # f.close()
    path=dict_hi['img_path']
    name=os.path.basename(path)
    dir1=os.path.dirname(path)
    print(dir1)
    name1=name[:-4]+"_1.jpg"
        
    b_path=os.path.join(dir1,name1)
    print(b_path)
    
    dict_final={}
    
    img=cv2.imread(path)
    _,img3=retange(img)
    cv2.imwrite(dir1+"/{}_2.jpg".format(name[:-4]),img3)
    dict_final["type"] = 3
    return dict_final



