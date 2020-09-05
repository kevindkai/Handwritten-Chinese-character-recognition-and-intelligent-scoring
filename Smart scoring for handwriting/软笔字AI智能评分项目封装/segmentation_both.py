# -*- coding: utf-8 -*-


from imutils.perspective import four_point_transform
import imutils
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def contrast_brightness(image,c,b):
    blank=np.zeros_like(image,image.dtype)
    # 计算两个数组的加权和(dst = alpha*src1 + beta*src2 + gamma)
    #dst=cv.addWeighted(image,c,white,c,b)
    dst=cv2.addWeighted(image,c,blank,1-c,b)#这样才能增加对比度
    return dst
def judge_point(points):
    points=np.array(points)
    a=points.argsort()
    index=[a[0],a[len(points)-1]]
    for i,j in enumerate(a):
        if 0<i<len(a)-1:
            if points[j]-points[a[0]]<100 or points[a[len(points)-1]]-points[j]<100:
                if 15<(points[j]-points[a[i-1]])<100 or 15<(points[a[i+1]]-points[j])<100: 
                    index.append(j)
        
    return index
def draw_lines(index,points,img):
    for i in index:
        cv2.line(img, points[i][0], points[i][1], [0],3)
def Get_cnt(input_dir):
    image = cv2.imread(input_dir)
    
    h,w,_=image.shape
    assert w/h>1.2,"请横着手机拍摄照片"
    if h>2500:
        image = imutils.resize(image, height = 2500)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    
    # plt.figure(0)
    # plt.imshow(mask,"gray")
    blurred = cv2.GaussianBlur(gray, (5,5),0)
    edged = cv2.Canny(blurred,20,150)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST,cv2.CHAIN_APPROX_TC89_L1)
    cnts = cnts[1] if  imutils.is_cv2()  else   cnts[0]
    docCnt =[]
    # plt.figure(0)
    # plt.imshow(edged,"gray")
    

    if len(cnts) > 0:
        cnts =sorted(cnts,key=cv2.contourArea,reverse=True)
        for c in cnts:
            peri = cv2.arcLength(c,True)                   # 轮廓按大小降序排序
            approx = cv2.approxPolyDP(c,0.02 * peri,True)  # 获取近似的轮廓
            if len(approx) ==4:                            # 近似轮廓有四个顶点
                docCnt.append(approx)                      # 将满足的轮廓四点都保存
    # cv2.drawContours(image,[docCnt[2]],-1,(0,0,255),3)
    # plt.figure(5)
    # plt.imshow(image[:,:,::-1],"brg")        
    return image,docCnt,cnts

def Get_cnt1(input_dir):
    image = cv2.imread(input_dir)
    
    h,w,_=image.shape
    assert w/h>1.2,"请横着手机拍摄照片"
    if h>2500:
        image = imutils.resize(image, height = 2500)

    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    

    
    lower_write=np.array([10,0,100])
    upper_write=np.array([165,80,255])
    mask = cv2.inRange(hsv, lower_write, upper_write)

    
    # plt.figure(6)
    # plt.imshow(mask,"gray")
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
                
    return image,docCnt,cnts
def Get_cnt2(input_dir):
    image = cv2.imread(input_dir)
    
    h,w,_=image.shape
    assert w/h>1.2,"请横着手机拍摄照片"
    if h>2500:
        image = imutils.resize(image, height = 2500)
    
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    
    # plt.figure(0)
    # plt.imshow(mask,"gray")
    blurred = cv2.GaussianBlur(image[:,:,1], (5,5),0)
    edged = cv2.Canny(blurred,20,150)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST,cv2.CHAIN_APPROX_TC89_L1)
    cnts = cnts[1] if  imutils.is_cv2()  else   cnts[0]
    docCnt =[]
    # plt.figure(0)
    # plt.imshow(edged,"gray")
    

    if len(cnts) > 0:
        cnts =sorted(cnts,key=cv2.contourArea,reverse=True)
        for c in cnts:
            peri = cv2.arcLength(c,True)                   # 轮廓按大小降序排序
            approx = cv2.approxPolyDP(c,0.02 * peri,True)  # 获取近似的轮廓
            if len(approx) ==4:                            # 近似轮廓有四个顶点
                docCnt.append(approx)                      # 将满足的轮廓四点都保存
    # cv2.drawContours(image,[docCnt[2]],-1,(0,0,255),3)
    # plt.figure(5)
    # plt.imshow(image[:,:,::-1],"brg")        
    return image,docCnt,cnts
def Get_cnt3(input_dir):
    image = cv2.imread(input_dir)
    
    h,w,_=image.shape
    assert w/h>1.2,"请横着手机拍摄照片"
    if h>2500:
        image=cv2.resize(image,(0,0),fx=0.5,fy=0.5,interpolation=cv2.INTER_AREA)
    
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    
    # plt.figure(0)
    # plt.imshow(mask,"gray")
    blurred = cv2.GaussianBlur(image[:,:,1], (5,5),0)
    edged = cv2.Canny(blurred,20,150)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST,cv2.CHAIN_APPROX_TC89_L1)
    cnts = cnts[1] if  imutils.is_cv2()  else   cnts[0]
    docCnt =[]
    # plt.figure(0)
    # plt.imshow(edged,"gray")
    

    if len(cnts) > 0:
        cnts =sorted(cnts,key=cv2.contourArea,reverse=True)
        for c in cnts:
            peri = cv2.arcLength(c,True)                   # 轮廓按大小降序排序
            approx = cv2.approxPolyDP(c,0.02 * peri,True)  # 获取近似的轮廓
            if len(approx) ==4:                            # 近似轮廓有四个顶点
                docCnt.append(approx)                      # 将满足的轮廓四点都保存
    # cv2.drawContours(image,[docCnt[2]],-1,(0,0,255),3)
    # plt.figure(5)
    # plt.imshow(image[:,:,::-1],"brg")        
    return image,docCnt,cnts
def Get_cnt4(input_dir):
    image = cv2.imread(input_dir)
    
    h,w,_=image.shape
    assert w/h>1.2,"请横着手机拍摄照片"
    
    if h>2500:
        image=cv2.resize(image,(0,0),fx=0.8,fy=0.8,interpolation=cv2.INTER_AREA)

    # cv2.INTER_AREA
    # image = cv2.GaussianBlur(image, (5,5),0)
    # blank=np.zeros_like(image[:,:,1],image[:,:,1].dtype)
    # img=cv2.addWeighted(image[:,:,1],1.5,blank,0,0)
    # img = cv2.threshold(img,235,255,type=0)[1]
    blurred = cv2.GaussianBlur(image[:,:,1], (5,5),0)
  
    plt.figure(1)
    plt.imshow(blurred,"gray")  
    edged = cv2.Canny(blurred,20,150)
    lines = cv2.HoughLines(edged,1,np.pi/180,400) #这里对最后一个参数使用了经验型的值
    result = blurred.copy()
    plt.figure(7)
    plt.imshow(edged,"gray")  
    points_v=[]
    points_v1=[]
    points_h=[]
    points_h1=[]
    for line in lines:
        rho = line[0][0]  #第一个元素是距离rho
        theta= line[0][1] #第二个元素是角度theta

        if  (theta < (np.pi/8. )) or (theta > (7.*np.pi/8.0)): #垂直直线
            pt1 = (int(rho/np.cos(theta)),0)               #该直线与第一行的交点
            points_v1.append(pt1[0]) 
            #该直线与最后一行的焦点
            pt2 = (int((rho-result.shape[0]*np.sin(theta))/np.cos(theta)),result.shape[0])
            points_v.append([pt1,pt2])            # 绘制一条白线
        elif 3*np.pi/8. <theta < 5*np.pi/8.:        #水平直线
            pt1 = (0,int(rho/np.sin(theta)))               # 该直线与第一列的交点
            points_h1.append(pt1[1]) 
            #该直线与最后一列的交点
            pt2 = (result.shape[1], int((rho-result.shape[1]*np.cos(theta))/np.sin(theta)))
            points_h.append([pt1,pt2]) # 绘制一条直线
    index1=judge_point(points_v1)
    index2=judge_point(points_h1)
    draw_lines(index1, points_v, result)
    draw_lines(index2, points_h, result)

                 
    edged1 = cv2.Canny(result,20,150)        
    cnts = cv2.findContours(edged1.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_TC89_L1)
    cnts = cnts[1] if  imutils.is_cv2()  else   cnts[0]
    docCnt =[]

    if len(cnts) > 0:
        cnts =sorted(cnts,key=cv2.contourArea,reverse=True)
        for c in cnts:
            peri = cv2.arcLength(c,True)                   # 轮廓按大小降序排序
            approx = cv2.approxPolyDP(c,0.02 * peri,True)  # 获取近似的轮廓
            if len(approx) ==4:                            # 近似轮廓有四个顶点
                docCnt.append(approx)                      # 将满足的轮廓四点都保存
    cv2.drawContours(image,docCnt,-1,(0,0,255),3)
    plt.figure(6)
    plt.imshow(result,"gray")        
    return image,docCnt,cnts
def get_result_img_3_4(image,docCnt):
  result_imgs=[]
  areas=[]
  for i in docCnt:
    result_imgs.append(four_point_transform(image, i.reshape(4,2))) # 对原始图像进行四点透视变换
  for i in range(3):
    m,n,c=result_imgs[i].shape
    areas.append(m*n)
  if areas[1]/areas[0]>=0.65:
      if areas[2]/areas[1]>=0.65:
          result_img=result_imgs[2]
      else:
          result_img=result_imgs[1]
          
  else:
    result_img=result_imgs[0]
  m,n,c=result_img.shape
  
  # 从这里开始进行报错
  origin_m,origin_n,origin_c=image.shape
  # print(m*n)
  # print(origin_m*origin_n)
  
  judge=False
  
  img_part=result_img[int(0.018*m):int(0.08*m),10:int(0.04*n)]
  m1,n1,_=img_part.shape
  # plt.figure(3)
  # plt.imshow(img_part[:,:,::-1],"brg")
  hsv=cv2.cvtColor(img_part,cv2.COLOR_BGR2HSV)    
  lower_write=np.array([156,43,46])
  upper_write=np.array([180,255,255])
  mask = cv2.inRange(hsv, lower_write, upper_write)
  mask=mask.astype(np.int64)
    
  lower_write1=np.array([0,43,46])
  upper_write1=np.array([10,255,255])
  mask1 = cv2.inRange(hsv, lower_write1, upper_write1)
  mask1=mask1.astype(np.int64)
  mask3=mask+mask1
  mask4=np.where(mask3>255,255,mask3)
  mask4=mask4.astype(np.uint8)
  # plt.figure(2)
  # plt.imshow(mask4,"gray")
  k=np.where(mask4>200,1,0)
  total_num=np.sum(k)
  ratio=total_num/(m1*n1)
  print(ratio)
  if ratio>0.10:
      judge=True
      

    

  assert judge,"无法识别，请查看是否有遮挡物，避免光线过强或过暗"
  assert 0.8>=m/n>=0.5 ,"无法识别 1.拍摄角度过大，请正视拍照"
  assert (m*n)/(origin_m*origin_n)>0.25,'无法识别 1.请保持纸张平整，旁边不要放置其他干扰物品 2.请避免在局部亮度过高的地方拍摄 3.请避免在过暗处拍摄'
  return result_img
# 返回矫正结果
def get_result_img_4_5(image,docCnt):
  result_imgs=[]
  areas=[]
  for i in docCnt:
    result_imgs.append(four_point_transform(image, i.reshape(4,2))) # 对原始图像进行四点透视变换
  for i in range(4):
    m,n,c=result_imgs[i].shape
    areas.append(m*n)
  if areas[1]/areas[0]>=0.65:
    if areas[2]/areas[1]>=0.65:
        result_img=result_imgs[2]
    else:
        result_img=result_imgs[1]

  else:
    result_img=result_imgs[0]
  m,n,c=result_img.shape
  
  # 从这里开始进行报错
  origin_m,origin_n,origin_c=image.shape
  # print(m*n)
  # print(origin_m*origin_n)
  
  judge=False
  
  img_part=result_img[int(0.018*m):int(0.08*m),10:int(0.04*n)]
  m1,n1,_=img_part.shape
  # plt.figure(3)
  # plt.imshow(img_part[:,:,::-1],"brg")
  hsv=cv2.cvtColor(img_part,cv2.COLOR_BGR2HSV)    
  lower_write=np.array([156,43,46])
  upper_write=np.array([180,255,255])
  mask = cv2.inRange(hsv, lower_write, upper_write)
  mask=mask.astype(np.int64)
    
  lower_write1=np.array([0,43,46])
  upper_write1=np.array([10,255,255])
  mask1 = cv2.inRange(hsv, lower_write1, upper_write1)
  mask1=mask1.astype(np.int64)
  mask3=mask+mask1
  mask4=np.where(mask3>255,255,mask3)
  mask4=mask4.astype(np.uint8)
  # plt.figure(2)
  # plt.imshow(mask4,"gray")
  k=np.where(mask4>200,1,0)
  total_num=np.sum(k)
  ratio=total_num/(m1*n1)
  print(ratio)
  if ratio>0.35:
      judge=True
      
  assert judge,"无法识别，请查看是否有遮挡物，避免光线过强或过暗"
  assert 0.8>=m/n>=0.5 ,"无法识别 1.拍摄角度过大，请正视拍照"
  assert (m*n)/(origin_m*origin_n)>0.25,'无法识别 1.请保持纸张平整，旁边不要放置其他干扰物品 2.请避免在局部亮度过高的地方拍摄 3.请避免在过暗处拍摄'
  return result_img  

def qiege_3_4(result_img,output_dir):
    blurred = cv2.GaussianBlur(result_img, (5,5),0)
    m,n,_=blurred.shape
    img_4_5=blurred[:,int(0.13*n):]
    m1,n1,_=img_4_5.shape
    length1=m1//3
    length2=n1//4
    A={}
    k=-1
    imgpaths=[]
    
    for i in range(3):
        for j in range(4):
            A[i,j]=img_4_5[length1*i:length1*(i+1),length2*j:length2*(j+1)]
            A[i,j]=cv2.normalize(A[i,j],None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
            k=k+1
            path=os.path.join(output_dir,"img{}.jpg".format(k))
            cv2.imwrite(path,A[i,j])
            imgpaths.append(path)
    return imgpaths
    
def qiege_4_5(result_img,output_dir):
    blurred = cv2.GaussianBlur(result_img, (5,5),0)
    m,n,_=blurred.shape
    img_4_5=blurred[:,int(0.138*n):]
    m1,n1,_=img_4_5.shape
    length1=m1//4
    length2=n1//5
    A={}
    k=-1
    imgpaths=[]

    for i in range(4):
        for j in range(5):
            A[i,j]=img_4_5[length1*i:length1*(i+1),length2*j:length2*(j+1)]
            A[i,j]=cv2.normalize(A[i,j],None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
            k=k+1
            A[i,j]=cv2.resize(A[i,j], (664,664))
            path=os.path.join(output_dir,"img{}.jpg".format(k))
            cv2.imwrite(path,A[i,j])
            imgpaths.append(path)
    return imgpaths
    
def run(request_data): 
    input_dir = request_data['imagePath']
    output_dir= request_data['targetPath']

    try:
        image,docCnt,cnts=Get_cnt(input_dir)
        try:
            result_img=get_result_img_4_5(image,docCnt)
            imgpanths = qiege_4_5(result_img,output_dir)
            print(1)
        except:
            result_img=get_result_img_3_4(image,docCnt)
            imgpanths = qiege_3_4(result_img,output_dir)
            print(2)
    except:
        try:
            image,docCnt,cnts=Get_cnt1(input_dir)
            try:
                result_img=get_result_img_4_5(image,docCnt)
                imgpanths = qiege_4_5(result_img,output_dir)
                print(3)
            except:
                result_img=get_result_img_3_4(image,docCnt)
                imgpanths = qiege_3_4(result_img,output_dir)
                print(4)
        except:
            try:
                image,docCnt,cnts=Get_cnt2(input_dir)
                try:
                    result_img=get_result_img_4_5(image,docCnt)
                    imgpanths = qiege_4_5(result_img,output_dir)
                    print(5)
                except:
                    result_img=get_result_img_3_4(image,docCnt)
                    imgpanths = qiege_3_4(result_img,output_dir)
                    print(6)            
            except:
                try:
                    image,docCnt,cnts=Get_cnt3(input_dir)
                    try:
                        result_img=get_result_img_4_5(image,docCnt)
                        imgpanths = qiege_4_5(result_img,output_dir)
                        print(7)
                    except:
                        result_img=get_result_img_3_4(image,docCnt)
                        imgpanths = qiege_3_4(result_img,output_dir)
                        print(8)
                except:
                    image,docCnt,cnts=Get_cnt4(input_dir)
                    try:
                        result_img=get_result_img_4_5(image,docCnt)
                        imgpanths = qiege_4_5(result_img,output_dir)
                        print(9)
                    except:
                        result_img=get_result_img_3_4(image,docCnt)
                        imgpanths = qiege_3_4(result_img,output_dir)
                        print(10)               
            
    
    return imgpanths

    




