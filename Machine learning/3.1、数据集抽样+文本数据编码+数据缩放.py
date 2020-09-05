# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 17:40:03 2019

@author: kw
"""
#----------(一)数据集拆分-----------------
#1.自定义拆分
import numpy as np
def split_train_test(data,test_ratio):
    shuffled_indices=np.random.permutation(len(data))
    test_set_size=int(len(data)*test_ratio)
    test_indices=shuffled_indices[:test_set_size]
    train_indices=shuffled_indices[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]
train_set,test_set=split_train_test(data,test_ratio=0.25)

#2.sklearn中的train_test_split模块
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(data,test_size=0.25,random_state=100)

##3.分层抽样(强烈推荐) data['class']:分层依据
from sklearn.model_selection import StratifiedShuffleSplit
split=StratifiedShuffleSplit(n_splits=1,test_size=0.5,random_state=100)
for train_index,test_index in split.split(data,data['class']):
    start_train_set=data.iloc[train_index]
    start_test_set=data.iloc[test_index]
    
data['class'].value_counts()/len(data)

#--------------(二)文本数据编码-------------------
#1.将文本标签转化为数字，1,2,3,4(或许顺序数据的时候可以用一下)
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
data_code=data['X1']
data_encode=encoder.fit_transform(data_code)
data_encode

#2.独热编码，转化为数值矩阵，二分类
from sklearn.preprocessing import OneHotEncoder
encoder=OneHotEncoder()
data_encode1=encoder.fit_transform(data_code).reshape(-1,1)
data_encode1
data_encode1.toarray()

#3.LabelBinarizer编码，可以从文本类别转化为整数类别，再从整数类别转化为独热向量
from sklearn.preprocessing import LabelBinarizer
encoder=LabelBinarizer()
data_encode1=encoder.fit_transform(data_code)
data_encode1

#4.自定义转化器
from sklearn.base import BaseEstimator,TransformerMixin
a,b,c,d=3,4,5,6
class CombinedAttributesAdder(BaseEstimator,TransformerMixin):
    def __init__(self,add_bedrooms_per_room=True):
        self.add_bedrooms_per_room=add_bedrooms_per_room
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        rooms_per_household=X[:,a]/X[:,b]
        population_per_household=X[:,c]/X[:,d]
        if self.add_bedrooms_per_room:
            bedrooms_per_room=X[:,b]/X[:,a]
            return np.c_[X,rooms_per_household.population_per_household,bedrooms_per_room]
        else:
            return np.c_[X,rooms_per_household,population_per_household]
attr_adder=CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs=attr_adder.transform(housing.values)

-----------(三)数据缩放-----------
#1.归一化
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
#处理数值型数据的pipeline
num_pipeline=Pipeline([
        ('imputer',Imputer(strategy='median')),
        ('attribs_adder',CombinedAttributesAdder()),
        ('std_scaler',StandardScaler())])

housing_tr=num_pipeline.fit_transform(housing)

#处理文本型数据的pipeline
from sklearn.pipeline import FeatureUnion
cat_pipeline=Pipeline([
        ('selector',DataFrameSelector(cat_attribs)),
        ('label_binarizer',LabelBinarizer())])
    
#整个pipeline流水线
full_pipeline=FeatureUnion(transformer_list=[
        ('num_pipeline',num_pipeline),
        ('cat_pipeline',cat_pipeline)])
    
result=full_pipeline.fit_transform(housing)

