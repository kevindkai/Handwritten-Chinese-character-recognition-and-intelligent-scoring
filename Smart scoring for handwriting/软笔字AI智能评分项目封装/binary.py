# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 10:05:16 2020

@author: lvfucheng
"""

import cv2
import numpy as np
from sys import argv
import os




def seg(dir):
    img1=cv2.imread(dir)
    rows, cols, channels = img1.shape
    blank = np.zeros([rows, cols, channels], img1.dtype)
    c= cv2.addWeighted(img1, 1.5, blank,0.4 , 0) 

    #dst = cv2.adaptiveThreshold(c[:,:,0],255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,51, 25)
    dst = cv2.threshold(c[:,:,0],130,255,type=0)[1]

    dst = cv2.blur(dst, (5,5))

    dst = cv2.boxFilter(dst, -1, (5,5), normalize=1)

    dst=cv2.medianBlur(dst,5) 

    img =np.where(dst>120,0,255)  
    return img   #阈值分割得到的二值图



def run(image): 
    
    dict={}
    
    img=seg(image)
    img_dir=os.path.dirname(image)
    name=os.path.basename(image)
    
    cv2.imwrite(img_dir+"/{}_1.jpg".format(name[:-4]),img)
    dict["binary_img"]=img_dir+"/{}_1.jpg".format(name[:-4])
    return dict

    
    
    
    
    
    
    
    

