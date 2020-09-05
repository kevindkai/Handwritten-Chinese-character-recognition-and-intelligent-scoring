
from imutils.perspective import four_point_transform
import imutils
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import math

def Get_cnt(input_dir):
    image = cv2.imread(input_dir)
    # orig = image.copy()
    # image=image[:,:,::-1]
    
    h,w,_=image.shape
    assert 1.1<h/w<1.6,"图片尺寸有问题，请重新拍摄" 
    if h>=2500:
       image = imutils.resize(image, height = 2500)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

 
    
    blurred = cv2.GaussianBlur(gray, (5,5),0)
    edged = cv2.Canny(blurred,20,150)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST,cv2.CHAIN_APPROX_TC89_L1)
    cnts = cnts[1] if  imutils.is_cv2()  else   cnts[0]
    docCnt =[]
    
    

    if len(cnts) > 0:
        cnts =sorted(cnts,key=cv2.contourArea,reverse=True)
        for c in cnts:
            peri = cv2.arcLength(c,True)                   # 轮廓按大小降序排序
            approx = cv2.approxPolyDP(c,0.02 * peri,True)  # 获取近似的轮廓
            if len(approx) ==4:                            # 近似轮廓有四个顶点
                docCnt.append(approx)                      # 将满足的轮廓四点都保存
                
    return image,docCnt,cnts


def Get_cnt1(input_dir):
    image = cv2.imread(input_dir)
    # orig = image.copy()
    # image=image[:,:,::-1]
    
    h=image.shape[0]
    if h>=2500:
       image = imutils.resize(image, height = 2500)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    
    

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # 限制对比度的自适应阈值均衡化
    gray = clahe.apply(gray)

    
    blurred = cv2.GaussianBlur(gray, (5,5),0)
    edged = cv2.Canny(blurred,20,150)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST,cv2.CHAIN_APPROX_TC89_L1)
    cnts = cnts[1] if  imutils.is_cv2()  else   cnts[0]
    docCnt =[]
    
    

    if len(cnts) > 0:
        cnts =sorted(cnts,key=cv2.contourArea,reverse=True)
        for c in cnts:
            peri = cv2.arcLength(c,True)                   # 轮廓按大小降序排序
            approx = cv2.approxPolyDP(c,0.02 * peri,True)  # 获取近似的轮廓
            if len(approx) ==4:                            # 近似轮廓有四个顶点
                docCnt.append(approx)                      # 将满足的轮廓四点都保存
    # cv2.drawContours(image,docCnt,-1,(0,0,255),3)           
    return image,docCnt,cnts

def Get_cnt2(input_dir):
    image = cv2.imread(input_dir)
    # orig = image.copy()
    # image=image[:,:,::-1]
    
    h=image.shape[0]
    if h>=2500:
       image = imutils.resize(image, height = 2500)
    
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    lower_write=np.array([0,0,130])
    upper_write=np.array([180,30,255])
    mask = cv2.inRange(hsv, lower_write, upper_write)

  
    blurred = cv2.GaussianBlur(mask, (5,5),0)
    edged = cv2.Canny(blurred,20,150)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST,cv2.CHAIN_APPROX_TC89_L1)
    cnts = cnts[1] if  imutils.is_cv2()  else   cnts[0]
    docCnt =[]
    
    

    if len(cnts) > 0:
        cnts =sorted(cnts,key=cv2.contourArea,reverse=True)
        for c in cnts:
            peri = cv2.arcLength(c,True)                   # 轮廓按大小降序排序
            approx = cv2.approxPolyDP(c,0.02 * peri,True)  # 获取近似的轮廓
            if len(approx) ==4:                            # 近似轮廓有四个顶点
                docCnt.append(approx)                      # 将满足的轮廓四点都保存
    # cv2.drawContours(image,docCnt,-1,(0,0,255),3)           
    return image,docCnt,cnts

def Get_cnt3(input_dir):
    image = cv2.imread(input_dir)
    # orig = image.copy()
    
    
    h=image.shape[0]
    if h>=2500:
       image = imutils.resize(image, height = 2500)
       image1=image[:,:,::-1]

    gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

 
    
    blurred = cv2.GaussianBlur(gray, (5,5),0)
    edged = cv2.Canny(blurred,20,150)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST,cv2.CHAIN_APPROX_TC89_L1)
    cnts = cnts[1] if  imutils.is_cv2()  else   cnts[0]
    docCnt =[]
    
    

    if len(cnts) > 0:
        cnts =sorted(cnts,key=cv2.contourArea,reverse=True)
        for c in cnts:
            peri = cv2.arcLength(c,True)                   # 轮廓按大小降序排序
            approx = cv2.approxPolyDP(c,0.02 * peri,True)  # 获取近似的轮廓
            if len(approx) ==4:                            # 近似轮廓有四个顶点
                docCnt.append(approx)                      # 将满足的轮廓四点都保存
                
    return image,docCnt,cnts

def Get_cnt4(input_dir):
    image = cv2.imread(input_dir)
    # orig = image.copy()
    
    
    h=image.shape[0]
    if h>=2500:
       image = imutils.resize(image, height = 2500)
       image1=image[:,:,::-1]
    gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

    
    

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # 限制对比度的自适应阈值均衡化
    gray = clahe.apply(gray)

    
    blurred = cv2.GaussianBlur(gray, (5,5),0)
    edged = cv2.Canny(blurred,20,150)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST,cv2.CHAIN_APPROX_TC89_L1)
    cnts = cnts[1] if  imutils.is_cv2()  else   cnts[0]
    docCnt =[]
    
    

    if len(cnts) > 0:
        cnts =sorted(cnts,key=cv2.contourArea,reverse=True)
        for c in cnts:
            peri = cv2.arcLength(c,True)                   # 轮廓按大小降序排序
            approx = cv2.approxPolyDP(c,0.02 * peri,True)  # 获取近似的轮廓
            if len(approx) ==4:                            # 近似轮廓有四个顶点
                docCnt.append(approx)                      # 将满足的轮廓四点都保存
    # cv2.drawContours(image,docCnt,-1,(0,0,255),3)           
    return image,docCnt,cnts

def Get_cnt5(input_dir):
    image = cv2.imread(input_dir)
    # orig = image.copy()
    # image=image[:,:,::-1]
    
    h=image.shape[0]
    if h>=2500:
       image = imutils.resize(image, height = 2500)
    
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    lower_write=np.array([0,0,120])
    upper_write=np.array([180,30,255])
    mask = cv2.inRange(hsv, lower_write, upper_write)
 
    blurred = cv2.GaussianBlur(mask, (5,5),0)
    edges = cv2.Canny(blurred,20,150)
    
    lines = cv2.HoughLines(edges,1,np.pi/180,320) #这里对最后一个参数使用了经验型的值
    result = blurred
    for line in lines:
	    rho = line[0][0]  #第一个元素是距离rho
	    theta= line[0][1] #第二个元素是角度theta

	    if  (theta < (np.pi/4. )) or (theta > (3.*np.pi/4.0)): #垂直直线
		    pt1 = (int(rho/np.cos(theta)),0)               #该直线与第一行的交点
		#该直线与最后一行的焦点
		    pt2 = (int((rho-result.shape[0]*np.sin(theta))/np.cos(theta)),result.shape[0])
		    cv2.line( result, pt1, pt2, (0),3)             # 绘制一条白线
	    else:                                                  #水平直线
		    pt1 = (0,int(rho/np.sin(theta)))               # 该直线与第一列的交点
		#该直线与最后一列的交点
		    pt2 = (result.shape[1], int((rho-result.shape[1]*np.cos(theta))/np.sin(theta)))
		    cv2.line(result, pt1, pt2, (0), 3)           # 绘制一条直线
    cnts = cv2.findContours(blurred.copy(), cv2.RETR_LIST,cv2.CHAIN_APPROX_TC89_L1)
    cnts = cnts[1] if  imutils.is_cv2()  else   cnts[0]
    docCnt =[]
    if len(cnts) > 0:
        cnts =sorted(cnts,key=cv2.contourArea,reverse=True)
        for c in cnts:
            peri = cv2.arcLength(c,True)                   # 轮廓按大小降序排序
            approx = cv2.approxPolyDP(c,0.02 * peri,True)  # 获取近似的轮廓
            if len(approx) ==4:                            # 近似轮廓有四个顶点
                docCnt.append(approx) 
    
    
    
         
    return image,docCnt,cnts



def get_result_img(image,docCnt):
  result_imgs=[]
  areas=[]
  for i in docCnt:
    result_imgs.append(four_point_transform(image, i.reshape(4,2))) # 对原始图像进行四点透视变换
  for i in range(4):
    m,n,c=result_imgs[i].shape
    areas.append(m*n)
  if areas[2]/areas[0]>=0.65:
    result_img=result_imgs[2]

  else:
    result_img=result_imgs[0]
  m,n,c=result_img.shape
  
  # 从这里开始进行报错
  origin_m,origin_n,origin_c=image.shape
  # print(m*n)
  # print(origin_m*origin_n)
  judge_img=result_img[int(m*0.01):int(m*0.08),int(n*0.19):int(n*0.29),:]
  judge_img = imutils.resize(judge_img, height = 110)
  gray1=cv2.cvtColor(judge_img,cv2.COLOR_BGR2GRAY) 
  dst = cv2.adaptiveThreshold(gray1,255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,101, 1)
  element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3, 3))#形态学去噪
  dst=cv2.morphologyEx(dst,cv2.MORPH_OPEN,element)  #开运算去噪
  
  
  contours, hierarchy = cv2.findContours(dst,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)  #轮廓检测函数
  judge=False
  
  for cont in contours:
    # cv2.drawContours(judge_img,cont,-1,(0,0,255),5)
    
    ares = cv2.contourArea(cont)#计算包围性状的面积
    
    
    if 3000>ares>1000 :   # 过滤面积
        # print(ares)
        judge= True
        break
  

  assert judge,"无法识别，1.请保持纸张平整，旁边不要放置其他干扰物品 2.请避免在局部亮度过高的地方拍摄 3请避免在过暗处拍摄"
  assert 1.8>=m/n>=1.1 ,"拍摄角度过大，请正视拍照"
  assert (m*n)/(origin_m*origin_n)>0.25,'无法识别 1.请保持纸张平整，旁边不要放置其他干扰物品 2.请避免在局部亮度过高的地方拍摄 3.请避免在过暗处拍摄'
  return result_img
# 返回矫正结果
def qiege(result_img,output_dir):
    result_img=cv2.resize(result_img,(750,1000),interpolation=cv2.INTER_AREA)
    filter_img=cv2.medianBlur(result_img,1)
    img={}
    imgpaths=[]
    img[0]=filter_img[118:532,43:360]
    img[1]=filter_img[118:532,392:709]
    img[2]=filter_img[555:969,43:360]
    img[3]=filter_img[555:969,392:709]
    A={}
    k=-1
    for m in range(4):
        for i in range(6):
            for j in range(4):
                A[m,i,j]=img[m][69*i+2:69*(i+1)-6,7+77*j:77*(j+1)-1]
                A[m,i,j]=cv2.normalize(A[m,i,j],None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
                k=k+1
                A[m,i,j]=cv2.resize(A[m,i,j], (166,166),cv2.INTER_CUBIC)
                path=os.path.join(output_dir,"img{}.jpg".format(k))
                imgpaths.append(path)
                cv2.imwrite(path,A[m,i,j]) 
    return imgpaths

def qiege1(result_img,output_dir):
    result_img=cv2.resize(result_img,(750,1000),interpolation=cv2.INTER_AREA)
    filter_img=cv2.medianBlur(result_img,1)
    img={}
    imgpaths=[]
    img[0]=filter_img[118:532,43:360]
    img[1]=filter_img[118:532,392:709]
    img[2]=filter_img[555:969,43:360]
    img[3]=filter_img[555:969,392:709]
    A={}
    k=-1
    for m in range(2):
        for i in range(6):
            for j in range(4):
                A[m,i,j]=img[m][69*i+2:69*(i+1)-6,7+77*j:77*(j+1)-1]
                A[m,i,j]=cv2.normalize(A[m,i,j],None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
                k=k+1
                A[m,i,j]=cv2.resize(A[m,i,j], (166,166),cv2.INTER_CUBIC)
                path=os.path.join(output_dir,"img{}.jpg".format(k))
                imgpaths.append(path)
                cv2.imwrite(path,A[m,i,j]) 
    return imgpaths 
               
def run(request_data):
    input_dir = request_data['imagePath']
    output_dir= request_data['targetPath']

    try:
        image,docCnt,cnts=Get_cnt(input_dir)
        result_img=get_result_img(image,docCnt)
             
    except:
        try:
            image,docCnt,cnts=Get_cnt1(input_dir)
            result_img=get_result_img(image,docCnt)
                  
        except:
            image,docCnt,cnts=Get_cnt5(input_dir)
            result_img=get_result_img(image,docCnt)
    
    result_img1=result_img.copy()
    result_img1=cv2.resize(result_img1,(1000,1450))
    m,n,_=result_img1.shape
    img1=result_img1[int(m/2)+100:-200]

    gray=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(gray,(3,3),0)
    edges = cv2.Canny(img, 50, 150, apertureSize = 3)
    lines = cv2.HoughLinesP(edges, 0.2, np.pi / 180, 200,
                        minLineLength=230, maxLineGap=5) 

    try:
        k=False
        if lines.size:
            for line in lines:
                x1,y1,x2,y2=line[0]
                if abs(y1-y2)<50:
                    k=True
                    break
        assert k==True
            
        imgpanths=qiege1(result_img,output_dir)
                
    except:
        imgpanths=qiege(result_img,output_dir)
    
    
    
    return imgpanths  

