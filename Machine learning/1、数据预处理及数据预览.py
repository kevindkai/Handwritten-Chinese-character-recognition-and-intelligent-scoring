# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 14:42:34 2019

@author: kw
"""
#导入包
import pandas as pd
import numpy as np
import os
import lightgbm as lgb
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from sklearn.exceptions import NotFittedError
from itertools import chain
import pandas_profiling
#一、读取数据及数据可视化
train=pd.read_csv(r"C:\Users\kw\Desktop\train.csv",index_col=0,na_values=-999)

train.head()

train.shape

train.info() #可以快速获取数据集的简单描述

train["x"].value_counts() #获取分类型数据的类数及number

train.describe()

%matplotlib inline
train.hist(bins=50,figsize=(20,15))
plt.show() #变量可视化

report=pandas_profiling.ProfileReport(train) #生成数据报告并显示                 

profile=train.profile_report(title="Census Dataset")  #pandas-profiling EDA报告
profile.to_file(output_file=Path(r"C:\Users\kw\Desktop\census_report.html"))

#二、缺失值处理
null_all=data.isnull().sum() #统计缺失值个数
null=data[pd.isnull(data['x1'])] #查看x列有缺失值的数据
ratio=len(data[pd.isnull(data['x1'])])/len(data) #x列缺失值占比
#(1)直接删除法
newdata=data.dropna(axis=0) #删除存在缺失值的行
newdata=data.dropna(axis=0,subset=['x1','x3']) #删除x1、x3列存在缺失值的行
#(2)使用全局常量填充缺失值
newdata=data.fillna(values=2)
#(3)均值、众数、中位数填充
newdata=data.fillna(data.means(axis=1),axis=1) #均值，正态分布数据
newdata=data.fillna(data.median(axis=1),axis=1) #中位数，偏态分布数据
newdata=data.fillna(stats.mode(data,axis=1) #众数，分布差异大，个别值集中
newdata=data.fillna(method="ffill") #用前一个数据进行填充
newdata=data.fillna(method="bfill") #用后一个数据进行填充

from sklearn.preprocessing import Imputer
impute=Imputer(missing_values="NaN",strategy=["mean","median","most_frequent"],axis=0)
impute=impute.fit(data.values)
newdata=pd.DataFrame(impute.transform(data.values))
#(4)插值法、KNN填充
newdata=data.interpolate() #用前一个值和后一个值的平均值进行填充

from fancyimpute import BiScaler,KNN,NuclearNormMinimization,SoftImpute
newdata=KNN(k=3).fit_transform(data)
newdata=pd.DataFrame(newdata)
#(5)随机森林填充
from sklearn.ensemble import RandomForestRegressor
data_known=data[data.c.notnull()].as_matrix() #已知该特征
data_uknown=data[data.c.isnull()].as_matrix() #未知该特征
X=data_known[:,1:n] #X为特征属性值
Y=data_known[:,0] #Y为结果标签
rf=RandomForestRegressor(random_state=0,n_estimators=200,max_depth=3,n_jobs=-1)
rf.fit(X,Y) #训练模型
predicted=rf.predict(data_uknown[:,1:m])
data.loc[(data.c.isnull()),'c']=predicted
data

#分类数据缺失值处理(mode fill)
from sklearn.base import TransformerMixin
class CustomCategoryImputer(TransformerMixin):
    def __init__(self,cols=None):
        self.cols=cols
        
    def transform(self,df):
        X=df.copy()
        for col in self.cols:
            X[col].fillna(X[col].value_counts().index[0],inplace=True)
        return X
    
    def fit(self,*_):
        return self
cci=CustomCategoryImputer(cols=['x1','x2'])
cci.fit_transform(X)

#定量数据缺失值处理自定义
from sklearn.preprocessing import Imputer
class CustomQuantitativeImputer(TransformerMixin):
    def __init__(self,cols=None,strategy='mean'):
        self.cols=cols
        self.strategy=strategy
    
    def transform(self,df):
        X=df.copy()
        impute=Imputer(strategy=self.strategy)
        for col in self.cols:
            X[col]=impute.fit_transform(X[[col]])
        return X
    
    def fit(self,*_):
        return self

cqi=CustomQuantitativeImputer(cols=['x7'],strategy='mean')
cqi.fit_transform(X)

#三、数据处理
#1.标准化
#(1)Z分数标准化
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
data_standardized=pd.DataFrame(scaler.fit_transform(data))
#(2)max-min标准化
from sklearn.preprocessing import MinMaxScaler
min_max=MinMaxScaler()
data_min_max=pd.DataFrame(min_max.fit_transform(data))
#(3)行归一化
from sklearn.preprocessing import Normalizer
normalize=Normalizer()
data_normalized=pd.DataFrame(normalize.fit_transform(data))
#(4)对数变换
#对于具有重尾分布的数据，进行对数变换可使数据偏向正态化
newdata=np.log10(data) #底数可改
#(4)指数变换
from scipy import stats
#Box-Cox变换假定数据均是正的，首先得判断数据正负情况
data.min()
newdata=stats.boxcox(data,lambda=0)
#(5)数据的概率图
fig,(ax1,ax2,ax3)=plt.subplots(3,1)
prob1=stats.probplot(data['x1'],dist=stats.norm,plot=ax1)
ax1.set_xlabel('')
ax1.set_title('   ')
prob2=stats.probplot(data['x2'],dist=stats.norm,plot=ax2)
ax2.set_xlabel('')
ax2.set_title('   ')
prob3=stats.probplot(data['x3'],dist=stats.norm,plot=ax3)
ax3.set_xlabel('')
ax3.set_title('   ')

#2.连续型数据分箱(无监督型和有监督型)
#无监督型：等宽 + 等频 + 聚类
#(1)固定宽度分箱
newdata=np.floor_divide(data,k) #除以k进行分箱
newdata=np.floor(np.log10(data))  #通过对数函数映射到指数宽度分箱
#(2)分位数分箱
df=data.quantile([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])  #十分位数分箱
pd.qcut(data,4,labels=False) #四分位数分箱，并返回分箱序号
data=pd.Series(data)
data.quantile([0.25,0.5,0.75])
#(3)聚类分箱


#有监督型：卡方分箱法、ID3-C4.5-CART等单变量决策树算法、信用评分建模的IV最大化分箱
#(1)卡方分箱法

#(2)基于CART的决策树分箱(每个叶子节点的样本量>=总样本量的5%；内部节点再划分所需的最小样本数>=总样本量的10%)
import pandas as pd
import numpy as np
sample_set=pd.read_csv('data')
def calc_score_median(sample_set,var):
    '''
    计算相邻评分的中位数，以便进行决策树二元切分
    param sample_set:待切分样本
    param var:分割变量名称
    '''
    var_list=list(np.unique(sample_set[var]))
    var_median_list=[]
    for i in range(len(var_list)-1):
        var_median=(var_list[i]+var_list[i+1])/2
        var_median_list.append(var_median)
    return var_median_list
#var表示需要进行分箱的变量名，返回一个样本变量中位数的list
def choose_best_split(sample_set,var,min_sample):
    '''
    使用CART分类决策树选择最好的样本切分点
    返回切分点
    param sample_set:待切分样本
    param var:分割变量名称
    param min_sample:待切分样本的最小样本量（限制条件）
    '''
    #根据样本评分计算相邻不同分数的中间值
    score_median_list=calc_score_median(sample_set,var)
    median_len=len(score_median_list)
    sample_cnt=sample_set.shape[0]
    sample1_cnt=sum(sample_set['target'])

#(3)



















