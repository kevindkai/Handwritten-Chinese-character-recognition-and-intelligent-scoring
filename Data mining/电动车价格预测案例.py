#1. 导入必要的包
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from sklearn.exceptions import NotFittedError
from itertools import chain
import pandas_profiling
from sklearn.model_selection import GridSearchCV,cross_val_score,train_test_split
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import VotingClassifier

#2. 读取数据
data=pd.read_excel(r"C:\Users\kw\Desktop\比赛\train.xlsx",index_col=0)
x=data.iloc[:,:20]
y=data["price"]
df=pd.read_excel(r"C:\Users\kw\Desktop\比赛\test.xlsx",index_col=0)
df.head()
x_new=df.iloc[:,:20]

#3. training数据可视化
#数据可视化
data.head()
data.shape
data.info() #可以快速获取数据集的简单描述
data['price'].value_counts() #获取分类型数据的类数及number
data.describe()
%matplotlib inline
data.hist(bins=50,figsize=(20,15))
plt.show() #变量可视化
#report=pandas_profiling.ProfileReport(data) #生成数据报告并显示                 
#profile=data.profile_report(title="Census Dataset")  #pandas-profiling EDA报告
#profile.to_file(output_file=Path(r"C:\Users\kw\Desktop\train_report.html"))

#4.将数据集进行标准化
std=StandardScaler()
x=std.fit_transform(x)
x_new=std.transform(x_new)

#5.降维处理
pca=PCA(n_components=0.999,svd_solver='auto')
x_pca=pca.fit_transform(x)
x_recovered=pca.inverse_transform(x_pca)
x_recovered=pd.DataFrame(x_recovered)
x_new=pca.transform(x_new)
x_recover=pca.inverse_transform(x_new)
x_recover=pd.DataFrame(x_recover)

#6.拆分训练集和测试集
x_train,x_test,y_train,y_test=train_test_split(x_recovered,y,test_size=0.25,random_state=120)

#7.建模
lg=LogisticRegression(multi_class="multinomial",solver="lbfgs",C=1.0,penalty="l2")
lg.fit(x_train,y_train)
y_predict=lg.predict(x_test)
print("准确率",lg.score(x_test,y_test))
print("accuracy准确率",accuracy_score(y_test,y_predict))

RF=RandomForestClassifier(n_estimators=100)
RF.fit(x_train,y_train)
y_predict=RF.predict(x_test)
print("准确率",RF.score(x_test,y_test))
print("accuracy准确率",accuracy_score(y_test,y_predict))

knn_clf=KNeighborsClassifier(n_neighbors=9)
knn_clf=RandomForestClassifier(n_estimators=100)
knn_clf.fit(x_train,y_train)
y_predict=knn_clf.predict(x_test)
print("准确率",knn_clf.score(x_test,y_test))
print("accuracy准确率",accuracy_score(y_test,y_predict))

#8.预测
#8.预测
lg.fit(x_train,y_train)
y_predict=lg.predict(x_recover)
print("逻辑回归预测目标的分类",y_predict)

RF.fit(x_train,y_train)
y_predict=RF.predict(x_recover)
print("随机森林预测目标的分类",y_predict)

knn_clf.fit(x_train,y_train)
y_predict=knn_clf.predict(x_recover)
print("k近邻预测目标的分类",y_predict)

voting=VotingClassifier(estimators=[('lg',lg),('rf',RF),('knn',knn_clf)],voting='soft')
voting.fit(x_train,y_train)
y_predict=voting.predict(x_recover)
print("voting预测目标的分类",y_predict)
