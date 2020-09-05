
from imutils.perspective import four_point_transform
import imutils
import cv2
import numpy as np
import os
def Get_cnt(input_dir):
    image = cv2.imread(input_dir)
    h=image.shape[0]
    if h>=2500:
        image = imutils.resize(image, height = 2500)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5),0)
    edged = cv2.Canny(blurred,20,150)
    cnts = cv2.findContours(edged.copy(),  cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_TC89_L1)
    temp=cnts
    cnts = cnts[1] if  imutils.is_cv2()  else   cnts[0]
    docCnt =[]
    if len(cnts) > 0:
        cnts =sorted(cnts,key=cv2.contourArea,reverse=True)
        
        for c in cnts:
            peri = cv2.arcLength(c,True)                   # 轮廓按大小降序排序
            approx = cv2.approxPolyDP(c,0.02 * peri,True)  # 获取近似的轮廓
            if len(approx) ==4:                            # 近似轮廓有四个顶点
                docCnt.append(approx)
                if len(docCnt)==2:
                    break
    return image,docCnt,cnts

def get_result_img(image,docCnt):
  result_img=[[],[]]
  docCnt1=[]
  if docCnt[0].reshape(4,2)[0,0]<docCnt[1].reshape(4,2)[0,0]:
      docCnt1.append(docCnt[0].reshape(4,2))
      docCnt1.append(docCnt[1].reshape(4,2))
  else:
      docCnt1.append(docCnt[1].reshape(4,2))
      docCnt1.append(docCnt[0].reshape(4,2))
  for i in range(2):
    result_img[i] = four_point_transform(image, docCnt1[i]) # 对原始图像进行四点透视变换
  m1,n1,c1=result_img[0].shape
  m2,n2,c2=result_img[1].shape
  ratio1=m1/n1
  ratio2=m2/n2
  
  assert (1.8>=ratio1>=1.2 or 1.8>=ratio2>=1.2) and abs(ratio1-ratio2)<=0.15 ,"无法识别，1.请不要拍摄整张纸面，只需拍摄识别区域 2.请避免在亮度过高的地方拍摄"
  assert (m2*n2)/(m1*n1)>0.80 ,"拍摄角度过大，请正视拍摄"
    
  return result_img

def qiege(result_img,output_dir):
    A={}
    imgpaths=[]
    k=0
    for i in result_img:
      i=cv2.resize(i,(317,414))
      i=cv2.medianBlur(i,1)
      for j in range(6):
        for m in range(4):
            A[j,m]=i[69*j+2:69*(j+1)-6,7+77*m:77*(m+1)-1]
            A[j,m]=cv2.normalize(A[j,m],None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
            A[j,m]=cv2.resize(A[j,m], (166,166))
            path=os.path.join(output_dir,"img{}.jpg".format(k))
            cv2.imwrite(path,A[j,m])
            k=k+1
            imgpaths.append(path)
    return imgpaths            
    
def run(request_data):
    input_dir = request_data['imagePath']
    output_dir= request_data['targetPath']
    
    image,docCnt,cnts=Get_cnt(input_dir)
    result_img=get_result_img(image,docCnt)
    imgpanths=qiege(result_img,output_dir) #切割和保存
    return imgpanths


