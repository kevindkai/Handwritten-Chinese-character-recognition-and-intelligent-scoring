# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 09:23:58 2019

@author: kw
"""

#(一)、降维(PCA、KernalPCA、LLE)
#降维的两种方法：投影和流形学习
#降维：投影，线性
#流形学习：展开
#-----------1.PCA:识别出最接近数据的超平面，然后将数据投影其上(选择保留最大差异性的轴，因为它丢失的信息最少)
#编程实现(奇异值分解)
X_centered=X-X.mean(axis=0)
U,S,V=np.linalg.svd(X_centered)
c1=V.T[:,0]
c2=V.T[:,1]
W2=V.T[:,2]
X2D=X_centered.dot(w2)
#sklearn实现
from sklearn.decomposition import PCA
pca=PCA(n_components=2)
X2D=pca.fit_transform(X)
pca.components_.T[:,0] #查看第一个主成分
pca.explained_variance_ratio_ #主成分的方差解释率
##选择正确数量的维度
pca=PCA()
pca.fit(X)
cumsum=np.cumsum(pca.explained_variance_)
d=np.argmax(cumsum>=0.95)+1

pca=PCA(n_components=0.95,svd_solver='auto')
X_reduced=pca.fit_transform(X)

##数据解压回原来的维度
pca=PCA(n_components=154)
X_mnist_reduced=pca.fit_transform(X_mnist)
X_mnist_recovered=pca.inverse_transform(X_mnist_reduced)

#增量PCA(将训练集分成一个个小批量，一次给IPCA算法喂一个)
#将数据集划分为100个小批量数据集
#(1)numpy中的array_split+partial_fit
from sklearn.decomposition import IncrementalPCA
n_batches=100
inc_pca=IncrementalPCA(n_components=154)
for X_batch in np.array_split(X_mnist,n_batches):
    inc_pca.partial_fit(X_batch)
    
X_mnist_reduced=inc_pca.transform(X_mnist)

#(2)Numpy中的memmap方法
X_mm=np.memmap(filename,dtype='float32',mode='readonly',shape=(m,n))
batch_size=m//n_batches
inc_pca=IncrementalPCA(n_components=154,batch_size=batch_size)
inc_pca.fit(X_mm)

#------------2.随机PCA(降低计算复杂度)
rnd_pca=PCA(n_components=154,svd_solver='randomized')
X_reduced=rnd_pca.fit_transform(X_mnist)

#------------3.核主成分分析--------(非线性问题)-----
from sklearn.decomposition import KernelPCA
rbf_pca=KernelPCA(n_components=2,kernel='rbf',gamma=0.04)
X_reduced=rbf_pca.fit_transform(X)
#由于KPCA是一种无监督的学习算法，因此没有明显的性能指标来帮你选择最佳的核函数和超参数值。
#而降维通常是监督式学习任务的准备步骤，所以可以使用网格搜索，来找到使任务性能最佳的核和超参数。
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
clf=Pipeline([('kpca',KernelPCA(n_components=2)),
              ('log_reg',LogisticRegression())])
param_grid=[{
        "kpca_gamma":np.linspace(0.03,0.05,10),
        "kpca_kernel":['rbf','sigmoid']}]
grid_search=GridSearchCV(clf,param_grid,cv=3)
grid_search.fit(X,Y)
print(grid_search.best_params_)

##重建原像及找出误差
rbf_pca=KernelPCA(n_components=2,kernel='rbf',gamma=0.0433,fit_inverse_transform=True)
X_reduced=rbf_pca.fit_transform(X)
X_preimage=rbf_pca.inverse_transform(X_reduced)
from sklearn.metrics import mean_squared_error
mean_squared_error(X,X_preimage)

#-----------4.局部线性嵌入(LLE)----------
#概念：局部线性嵌入式另一种非常强大的非线性降维技术。不像之前的算法依赖于投影，它是
#一种流形学习技术。简单来说，LLE首先测量每个算法如何与其最近的邻居线性相关，然后为训练集寻找
#一个能最大程度保留这些局部关系的低维表示。这使得它特别擅长展开弯曲的流形，特别是没有太多噪声时。
from sklearn.manifold import LocallyLinearEmbedding
lle=LocallyLinearEmbedding(n_components=2,n_neighbors=10)
X_reduced=lle.fit_transform(X)

#------------5.其他算法-------
#(1)多维缩放算法，保持实例之间的距离，降低维度
#(2)等度量映射算法，将每个实例与其最近的邻居连接起来，创建连接图形，然后保留实例之间的这个测地距离，降低维度
#(3)T-分布随机近邻算法
#(4)LDA线性判别算法

#PCA补充版
#主成分分析可以使用相关系数矩阵，也可以使用协方差矩阵，一般情况下是使用缩放数据上的协方差矩阵
'''降维过程
(1)创建数据集的协方差矩阵
(2)计算协方差矩阵的特征值和特征向量
(3)保留前k个特征值(按特征值降序排列)
(4)用保留的特征向量乘以原数据矩阵，得到新的数据集
'''
#碎石图
plt.plot(np.cumsum(explained_variance_ratio))
plt.titile('Scree Plot')
plt.xlabel('Preincipal Component(k)')
plt.ylabel('%of Variance Explained')
#(5)LDA:基于类别可分性的分类有助于避免机器学习流水线的过拟合，也叫防止维度诅咒
'''LDA的计算步骤：
(1)计算每个类别的均值向量  使用bool索引
(2)计算类内和类间的散布矩阵
(3)使用SwSb的特征值和特征向量
(4)降序排列特征值，保留前k个特征向量
(5)使用前几个特征向量将数据投影到新空间
'''
mean_vectors=[]
for cl in [0,1,2]:
    class_mean_vector=np.mean(iris_X[iris_Y==cl],axis=0)
    mean_vectors.append(class_mean_vector)
    print(label_dict[cl],class_mean_vector)
#LDA其实是伪装成特征转换算法的分类器，它尝试用响应变量查找最优坐标系，尽可能优化类别可分性。只有存在响应变量时才可用
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda=LinearDiscriminantAnalysis(n_components=2)
X_lda=lda.fit_transform(X,Y)
#************高级流水线(复合流水线)**********
from sklearn.model_selection import GridSearchCV
params={'prepro__scale__with_std':[True,False],
        'prepro__scale__with_mean':[True,False],
        'prepro__pca__n_components':[1,2,3,4],
        'prepro__lda__n_components':[1,2] #lda的最大n_components是类别数-1
        'clf_n_neighbors':range(1,9)}
prepro=Pipeline([('scale',StandardScaler()),
                 ('pca',PCA()),
                 'lda',LinearDiscriminantAnalysis()])
pipe=Pipeline(steps=[('prepro',prepro),
                     ('clf',KNeighborsClassifier())])
grid=GridSearchCV(pipe,params)
grid.fit(X,Y)

