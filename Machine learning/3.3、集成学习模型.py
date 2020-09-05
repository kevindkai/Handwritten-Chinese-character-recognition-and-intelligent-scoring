# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 10:02:40 2019

@author: kw
"""
##----------集成学习------------
'''
(1)Bagging:随机森林
(2)Boosting:AdaBoost、GBDT、XGBoost、LightGBM、CatBoost、NGboost
(3)Stacking
(4)Blending
'''
#(二)、集成学习(bagging,boosting,stacking)
#1.投票分类器(voting)
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
lr=LogisticRegression()
RF=RandomForestClassifier()
svm_clf=SVC()
voting=VotingClassifier(estimators=[('lr',lr),
                                    ('rf',RF),
                                    ('svc',svm_clf)],
    voting='hard')
voting.fit(X_train,Y_train)
from sklearn.metrics import accuracy_score
for clf in (lr,RF,svm_clf,voting):
    clf.fit(X_train,Y_train)
    Y_pred=clf.predict(X_test)
    print(clf.__class__.__name__,accuracy_score(Y_test,Y_pred))

#如果所有分类器都能够估算出类别的概率(即有predict_proba()方法)，那么你可以将概率在所有单个分类器上平均，然后让sklearn给出
    平均概率最高的类别作为预测。这种被称为软投票法。通常来说，它比硬投票法的表现更优，因为它给予那些高度自信的投票更高的权重。
    而所有你需要做的就是用voting='soft'代替voting='hard',并确保所有的分类器都可以估算出概率。默认情况下，SVC类是不行的，
    所以你需要将其超参数probability设置为TRUE（这会导致SVC使用交叉验证来估算类别概率，减慢训练速度，并会添加predict_proba()方法）。
    
#-------2.bagging和pasting,只要将boostrap=FALSE，即是pasting
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
bag_clf=BaggingClassifier(
        DecisionTreeClassifier(),n_estimators=500,
        max_samples=100,bootstrap=True,n_jobs=-1)
bag_clf.fit(X_train,Y_train)
Y_pred=bag_clf.predict(X_test)
#对于bagging预测器，由于某些实例会被采样多次，而有些实例则可能根本不被采样。BaggingClassifier默认采用m个训练实例，
然后放回样本(bootstrap=True),m是训练集的大小。这意味着对于每个预测器来说，平均只对63%的训练实例进行采样。剩下37%未被采样
的训练实例称为包外（OOB）实例。注意，对所有的预测器来说，这是不一样的37%。
#创建BaggingClassifier时，设置oob_score=True,就可以请求在训练结束后自动进行包外评估。
bag_clf=BaggingClassifier(
        DecisionTreeClassifier(),n_estimators=500,
        bootstrap=True,n_jobs=-1,oob_score=True)
bag_clf.fit(X_train,Y_train)
bag_clf.oob_score_
from sklearn.metrics import accuracy_score
Y_pred=bag_clf.predict(X_test)
accuracy_score(Y_test,Y_pred)
bag_clf.oob_decision_function_ #包外决策函数

#-----3.Boosting(AdaBoost)------
from sklearn.ensemble import AdaBoostClassifier
ada_clf=AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=1),n_estimators=200,
        algorithm='SAMME.R',learning_rate=0.5)
ada_clf.fit(X_train,Y_train)


#-------4.Gradient Boosting------
#梯度提升：逐步在集成中添加预测器，每一个都对其前序做出改正。不同之处在于，它不是像AdaBoost那样在每个迭代中调整实例权重，
而是让新的预测器针对前有一个预测器的残差进行拟合
from sklearn.tree import DecisionTreeClassifier
DT=DecisionTreeClassifier(max_depth=2)
DT.fit(X,Y)
Y2=Y-DT.predict(X)
DT2=DecisionTreeClassifier(max_depth=2)
DT2.fit(X,Y2)
Y3=Y2-DT2.predict(X)
DT3=DecisionTreeClassifier(max_depth=2)
DT.fit(X,Y3)
Y_pred=sum(tree.predict(X_new) for tree in (DT,DT2,DT3))

#简单方法
from sklearn.ensemble import GradientBoostingClassifier
gbrt=GradientBoostingClassifier(max_depth=2,n_estimators=3,learning_rate=1)
gbrt.fit(X,Y)

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
X_train,X_val,Y_train,Y_val=train_test_split(X,Y,test_size=0.25)
gbrt=GradientBoostingClassifier(max_depth=2,n_estimators=120)
gbrt.fit(X_train,Y_train)
errors=[mean_squared_error(Y_val,Y_pred)
          for Y_pred in gbrt.staged_predict(X_val)]
bst_n_estimators=np.argmin(errors)
gbrt_best=GradientBoostingClassifier(max_depth=2,n_estimators=bst_n_estimators)
gbrt.fit(X_train,Y_train)

gbrt=GradientBoostingClassifier(max_depth=2,warm_start=True)
min_val_error=float('inf')
error_going_up=0
for n_estimators in range(1,120):
    gbrt.n_estimators=n_estimators
    gbrt.fit(X_train,Y_train)
    Y_pred=gbrt.predict(X_val)
    val_error=mean_squared_error(Y_val,Y_pred)
    if val_error<min_val_error:
        min_val_error=val_error
        error_going_up=0
    else:
        error_going_up+=1
        if error_going_up==5:
            break
#---------5堆叠法(stacking)------
from sklearn.datasets import load_iris
iris=load_iris()
X,Y=iris.data[:,1:3],iris.target
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import StackingClassifier
import numpy as np
knn_clf=KNeighborsClassifier(n_neighbors=2)
RF_clf=RandomForestClassifier(random_state=1)
baye_clf=GaussianNB()
lr=LogisticRegression()
stack_clf=StackingClassifier(classifiers=[knn_clf,RF_clf,baye_clf],meta_classifier=lr)
print('3-fold cross validation:\n')
for clf,label in zip([knn_clf,RF_clf,baye_clf,lr],["KNN","Random Forest","Naive Bayes","StackingClassifier"]):
    scores=cross_val_score(clf,X,Y,cv=3,scoring='accuracy')
    print('Accuracy:%0.2f(+/-%0.2f)[%s]'%(scores.mean(),scores.std(),label))
#---------5.堆叠法(stacking)------
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.datasets.samples_generator import make_blobs #聚类数据生成器其参数设置详见：https://blog.csdn.net/kevinelstri/article/details/52622960
 
'''创建训练的数据集'''
data, target = make_blobs(n_samples=50000, centers=2, random_state=0, cluster_std=0.60)
 
'''模型融合中使用到的各个单模型'''
clfs = [RandomForestClassifier(n_estimators=5, n_jobs=-1, criterion='gini'),
        RandomForestClassifier(n_estimators=5, n_jobs=-1, criterion='entropy'),
        ExtraTreesClassifier(n_estimators=5, n_jobs=-1, criterion='gini'),
        ExtraTreesClassifier(n_estimators=5, n_jobs=-1, criterion='entropy'),
        GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=5)]
 
'''切分一部分数据作为测试集'''
X, X_predict, y, y_predict = train_test_split(data, target, test_size=0.33, random_state=2017)
 
 
dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
dataset_blend_test = np.zeros((X_predict.shape[0], len(clfs)))
 
'''5折stacking'''
n_folds = 5
skf = list(StratifiedKFold(y, n_folds))
for j, clf in enumerate(clfs):
    '''依次训练各个单模型'''
    # print(j, clf)
    dataset_blend_test_j = np.zeros((X_predict.shape[0], len(skf)))
    for i, (train, test) in enumerate(skf):
        '''使用第i个部分作为预测，剩余的部分来训练模型，获得其预测的输出作为第i部分的新特征。'''
        # print("Fold", i)
        X_train, y_train, X_test, y_test = X[train], y[train], X[test], y[test]
        clf.fit(X_train, y_train)
        y_submission = clf.predict_proba(X_test)[:, 1]
        dataset_blend_train[test, j] = y_submission
        dataset_blend_test_j[:, i] = clf.predict_proba(X_predict)[:, 1]
    '''对于测试集，直接用这k个模型的预测值均值作为新的特征。'''
    dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)
    print("val auc Score: %f" % roc_auc_score(y_predict, dataset_blend_test[:, j]))
# clf = LogisticRegression()
clf = GradientBoostingClassifier(learning_rate=0.02, subsample=0.5, max_depth=6, n_estimators=30)
clf.fit(dataset_blend_train, y)
y_submission = clf.predict_proba(dataset_blend_test)[:, 1]
 
print("Linear stretch of predictions to [0,1]")
y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())
print("blend result")
print("val auc Score: %f" % (roc_auc_score(y_predict, y_submission)))

#----------6.bleding--------
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.datasets.samples_generator import make_blobs
 
'''创建训练的数据集'''
data, target = make_blobs(n_samples=50000, centers=2, random_state=0, cluster_std=0.60)
 
'''模型融合中使用到的各个单模型'''
clfs = [RandomForestClassifier(n_estimators=5, n_jobs=-1, criterion='gini'),
        RandomForestClassifier(n_estimators=5, n_jobs=-1, criterion='entropy'),
        ExtraTreesClassifier(n_estimators=5, n_jobs=-1, criterion='gini'),
        ExtraTreesClassifier(n_estimators=5, n_jobs=-1, criterion='entropy'),
        GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=5)]
 
'''切分一部分数据作为测试集'''
X, X_predict, y, y_predict = train_test_split(data, target, test_size=0.33, random_state=2017)
 
'''5折stacking'''
n_folds = 5
skf = list(StratifiedKFold(y, n_folds))
 
'''切分训练数据集为d1,d2两部分'''
X_d1, X_d2, y_d1, y_d2 = train_test_split(X, y, test_size=0.5, random_state=2017)
dataset_d1 = np.zeros((X_d2.shape[0], len(clfs)))
dataset_d2 = np.zeros((X_predict.shape[0], len(clfs)))
 
for j, clf in enumerate(clfs):
    '''依次训练各个单模型'''
    # print(j, clf)
    '''使用第1个部分作为预测，第2部分来训练模型，获得其预测的输出作为第2部分的新特征。'''
    # X_train, y_train, X_test, y_test = X[train], y[train], X[test], y[test]
    clf.fit(X_d1, y_d1)
    y_submission = clf.predict_proba(X_d2)[:, 1]
    dataset_d1[:, j] = y_submission
    '''对于测试集，直接用这k个模型的预测值作为新的特征。'''
    dataset_d2[:, j] = clf.predict_proba(X_predict)[:, 1]
    print("val auc Score: %f" % roc_auc_score(y_predict, dataset_d2[:, j]))
 
'''融合使用的模型'''
# clf = LogisticRegression()
clf = GradientBoostingClassifier(learning_rate=0.02, subsample=0.5, max_depth=6, n_estimators=30)
clf.fit(dataset_d1, y_d2)
y_submission = clf.predict_proba(dataset_d2)[:, 1]
 
print("Linear stretch of predictions to [0,1]")
y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())
print("blend result")
print("val auc Score: %f" % (roc_auc_score(y_predict, y_submission)))

#CatBoost-------------
'''CatBoost 主要有五个特性：
(1)无需调参即可获得较高的模型质量，采用默认参数就可以获得非常好的效果，减少在参数上花费的时间
(2)支持类别型变量，无需对非数值型特征进行预处理
(3)快速、可扩展的GPU版本，可以用基于GPU的梯度提升算法实现来训练你的模型，支持多卡并行
(4)提高准确性，提出一种全新的梯度提升机制来构建模型以减少过拟合
(5)快速预测，即便应对延时非常苛刻的任务也能够快速高效部署模型
'''
class CatBoostClassifier(iterations=None,
                         learning_rate=None,
                         depth=None,
                         l2_leaf_reg=None,
                         model_size_reg=None,
                         rsm=None,
                         loss_function=None,
                         border_count=None,
                         feature_border_type=None,
                         per_float_feature_quantization=None,                         
                         input_borders=None,
                         output_borders=None,
                         fold_permutation_block=None,
                         od_pval=None,
                         od_wait=None,
                         od_type=None,
                         nan_mode=None,
                         counter_calc_method=None,
                         leaf_estimation_iterations=None,
                         leaf_estimation_method=None,
                         thread_count=None,
                         random_seed=None,
                         use_best_model=None,
                         verbose=None,
                         logging_level=None,
                         metric_period=None,
                         ctr_leaf_count_limit=None,
                         store_all_simple_ctr=None,
                         max_ctr_complexity=None,
                         has_time=None,
                         allow_const_label=None,
                         classes_count=None,
                         class_weights=None,
                         one_hot_max_size=None,
                         random_strength=None,
                         name=None,
                         ignored_features=None,
                         train_dir=None,
                         custom_loss=None,
                         custom_metric=None,
                         eval_metric=None,
                         bagging_temperature=None,
                         save_snapshot=None,
                         snapshot_file=None,
                         snapshot_interval=None,
                         fold_len_multiplier=None,
                         used_ram_limit=None,
                         gpu_ram_part=None,
                         allow_writing_files=None,
                         final_ctr_computation_mode=None,
                         approx_on_full_history=None,
                         boosting_type=None,
                         simple_ctr=None,
                         combinations_ctr=None,
                         per_feature_ctr=None,
                         task_type=None,
                         device_config=None,
                         devices=None,
                         bootstrap_type=None,
                         subsample=None,
                         sampling_unit=None,
                         dev_score_calc_obj_block_size=None,
                         max_depth=None,
                         n_estimators=None,
                         num_boost_round=None,
                         num_trees=None,
                         colsample_bylevel=None,
                         random_state=None,
                         reg_lambda=None,
                         objective=None,
                         eta=None,
                         max_bin=None,
                         scale_pos_weight=None,
                         gpu_cat_features_storage=None,
                         data_partition=None
                         metadata=None, 
                         early_stopping_rounds=None,
                         cat_features=None, 
                         grow_policy=None,
                         min_data_in_leaf=None,
                         min_child_samples=None,
                         max_leaves=None,
                         num_leaves=None,
                         score_function=None,
                         leaf_estimation_backtracking=None,
                         ctr_history_unit=None,
                         monotone_constraints=None)

#Catboostregressor
class CatBoostRegressor(iterations=None,
                        learning_rate=None,
                        depth=None,
                        l2_leaf_reg=None,
                        model_size_reg=None,
                        rsm=None,
                        loss_function='RMSE',
                        border_count=None,
                        feature_border_type=None,
                        per_float_feature_quantization=None,
                        input_borders=None,
                        output_borders=None,
                        fold_permutation_block=None,
                        od_pval=None,
                        od_wait=None,
                        od_type=None,
                        nan_mode=None,
                        counter_calc_method=None,
                        leaf_estimation_iterations=None,
                        leaf_estimation_method=None,
                        thread_count=None,
                        random_seed=None,
                        use_best_model=None,
                        best_model_min_trees=None,
                        verbose=None,
                        silent=None,
                        logging_level=None,
                        metric_period=None,
                        ctr_leaf_count_limit=None,
                        store_all_simple_ctr=None,
                        max_ctr_complexity=None,
                        has_time=None,
                        allow_const_label=None,
                        one_hot_max_size=None,
                        random_strength=None,
                        name=None,
                        ignored_features=None,
                        train_dir=None,
                        custom_metric=None,
                        eval_metric=None,
                        bagging_temperature=None,
                        save_snapshot=None,
                        snapshot_file=None,
                        snapshot_interval=None,
                        fold_len_multiplier=None,
                        used_ram_limit=None,
                        gpu_ram_part=None,
                        pinned_memory_size=None,
                        allow_writing_files=None,
                        final_ctr_computation_mode=None,
                        approx_on_full_history=None,
                        boosting_type=None,
                        simple_ctr=None,
                        combinations_ctr=None,
                        per_feature_ctr=None,
                        ctr_target_border_count=None,
                        task_type=None,
                        device_config=None,                        
                        devices=None,
                        bootstrap_type=None,
                        subsample=None,                        
                        sampling_unit=None,
                        dev_score_calc_obj_block_size=None,
                        max_depth=None,
                        n_estimators=None,
                        num_boost_round=None,
                        num_trees=None,
                        colsample_bylevel=None,
                        random_state=None,
                        reg_lambda=None,
                        objective=None,
                        eta=None,
                        max_bin=None,
                        gpu_cat_features_storage=None,
                        data_partition=None,
                        metadata=None,
                        early_stopping_rounds=None,
                        cat_features=None,
                        grow_policy=None,
                        min_data_in_leaf=None,
                        min_child_samples=None,
                        max_leaves=None,
                        num_leaves=None,
                        score_function=None,
                        leaf_estimation_backtracking=None,
                        ctr_history_unit=None,
                        monotone_constraints=None)

##------NGBoost----------

# import packages
import pandas as pd
from ngboost.ngboost import NGBoost
from ngboost.learners import default_tree_learner
from ngboost.distns import Normal
import ngboost.scores
from MLE import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
# read the dataset
df = pd.read_csv('~/train.csv')
# feature engineering
tr, te = Nanashi_solution(df)
# NGBoost
ngb = NGBoost(Base=default_tree_learner, Dist=Normal, Score=MLE(),
natural_gradient=True,verbose=False)

ngboost = ngb.fit(np.asarray(tr.drop(['SalePrice'],1)),
np.asarray(tr.SalePrice))

y_pred_ngb = pd.DataFrame(ngb.predict(te.drop(['SalePrice'],1)))
# LightGBM
ltr = lgb.Dataset(tr.drop(['SalePrice'],1),label=tr['SalePrice'])

param = {
'bagging_freq': 5,
'bagging_fraction': 0.6,
'bagging_seed': 123,
'boost_from_average':'false',
'boost': 'gbdt',
'feature_fraction': 0.3,
'learning_rate': .01,
'max_depth': 3,
'metric':'rmse',
'min_data_in_leaf': 128,
'min_sum_hessian_in_leaf': 8,
'num_leaves': 128, 'num_threads': 8,
'tree_learner': 'serial',
'objective': 'regression',
'verbosity': -1,
'random_state':123,
'max_bin': 8,
'early_stopping_round':100
}


lgbm = lgb.train(param,ltr,num_boost_round=10000,valid_sets= [(ltr)],verbose_eval=1000)

y_pred_lgb = lgbm.predict(te.drop(['SalePrice'],1))
y_pred_lgb = np.where(y_pred>=.25,1,0)

# XGBoost
params = {
            'max_depth': 4, 'eta': 0.01,
            'objective':'reg:squarederror',
            'eval_metric': ['rmse'],
            'booster':'gbtree',
            'verbosity':0,
            'sample_type':'weighted',
            'max_delta_step':4,
            'subsample':.5,
            'min_child_weight':100,
            'early_stopping_round':50
}

dtr, dte = xgb.DMatrix(tr.drop(['SalePrice'],1),label=tr.SalePrice),
xgb.DMatrix(te.drop(['SalePrice'],1),label=te.SalePrice)

num_round = 5000
xgbst = xgb.train(params,dtr,num_round,verbose_eval=500)

y_pred_xgb = xgbst.predict(dte)

# Check the results
print('RMSE: NGBoost',
round(sqrt(mean_squared_error(X_val.SalePrice,y_pred_ngb)),4))
print('RMSE: LGBM',
round(sqrt(mean_squared_error(X_val.SalePrice,y_pred_lgbm)),4))
print('RMSE: XGBoost',
round(sqrt(mean_squared_error(X_val.SalePrice,y_pred_xgb)),4))

# see the probability distributions by visualising
Y_dists = ngb.pred_dist(X_val.drop(['SalePrice'],1))
y_range = np.linspace(min(X_val.SalePrice), max(X_val.SalePrice), 200)
dist_values = Y_dists.pdf(y_range).transpose()

# plot index 0 and 114
idx = 114
plt.plot(y_range,dist_values[idx])
plt.title(f"idx: {idx}")
plt.tight_layout()
plt.show()


'''
import numpy as np
import scipy as sp
from ngboost.distns import Normal
from ngboost.scores import MLE, CRPS
from ngboost.learners import default_tree_learner, default_linear_learner
from ngboost.distns.normal import Normal


class NGBoost(object):

    def __init__(self, Dist=Normal, Score=MLE,
                 Base=default_tree_learner, natural_gradient=True,
                 n_estimators=500, learning_rate=0.01, minibatch_frac=1.0,
                 verbose=True, verbose_eval=100, tol=1e-4):
        self.Dist = Dist
        self.Score = Score
        self.Base = Base
        self.natural_gradient = natural_gradient
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.minibatch_frac = minibatch_frac
        self.verbose = verbose
        self.verbose_eval = verbose_eval
        self.init_params = None
        self.base_models = []
        self.scalings = []
        self.tol = tol

    def pred_param(self, X, max_iter=None):
        m, n = X.shape
        params = np.ones((m, self.Dist.n_params)) * self.init_params
        for i, (models, s) in enumerate(zip(self.base_models, self.scalings)):
            if max_iter and i == max_iter:
                break
            resids = np.array([model.predict(X) for model in models]).T
            params -= self.learning_rate * resids * s
        return params

    def sample(self, X, Y, params):
        if self.minibatch_frac == 1.0:
            return np.arange(len(Y)), X, Y, params
        sample_size = int(self.minibatch_frac * len(Y))
        idxs = np.random.choice(np.arange(len(Y)), sample_size, replace=False)
        return idxs, X[idxs,:], Y[idxs], params[idxs, :]

    def fit_base(self, X, grads):
        models = [self.Base().fit(X, g) for g in grads.T]
        fitted = np.array([m.predict(X) for m in models]).T
        self.base_models.append(models)
        return fitted

    def line_search(self, resids, start, Y, scale_init=1):
        S = self.Score
        D_init = self.Dist(start.T)
        loss_init = S.loss(D_init, Y)
        scale = scale_init
        while True:
            scaled_resids = resids * scale
            D = self.Dist((start - scaled_resids).T)
            loss = S.loss(D, Y)
            norm = np.mean(np.linalg.norm(scaled_resids, axis=1))
            if not np.isnan(loss) and (loss < loss_init or norm < self.tol) and\
               np.linalg.norm(scaled_resids, axis=1).mean() < 5.0:
                break
            scale = scale * 0.5
        self.scalings.append(scale)
        return scale

    def fit(self, X, Y, X_val = None, Y_val = None, train_loss_monitor = None, val_loss_monitor = None):

        loss_list = []
        val_loss_list = []
        self.fit_init_params_to_marginal(Y)

        params = self.pred_param(X)
        if X_val is not None and Y_val is not None:
            val_params = self.pred_param(X_val)

        S = self.Score

        if not train_loss_monitor:
            train_loss_monitor = S.loss

        if not val_loss_monitor:
            val_loss_monitor = S.loss

        for itr in range(self.n_estimators):
            _, X_batch, Y_batch, P_batch = self.sample(X, Y, params)

            D = self.Dist(P_batch.T)

            loss_list += [train_loss_monitor(D, Y_batch)]
            loss = loss_list[-1]
            grads = S.grad(D, Y_batch, natural=self.natural_gradient)

            proj_grad = self.fit_base(X_batch, grads)
            scale = self.line_search(proj_grad, P_batch, Y_batch)

            params -= self.learning_rate * scale * np.array([m.predict(X) for m in self.base_models[-1]]).T

            val_loss = 0
            if X_val is not None and Y_val is not None:
                val_params -= self.learning_rate * scale * np.array([m.predict(X_val) for m in self.base_models[-1]]).T
                val_loss = val_loss_monitor(self.Dist(val_params.T), Y_val)
                val_loss_list += [val_loss]
                if len(val_loss_list) > 10 and np.mean(np.array(val_loss_list[-5:])) > \
                   np.mean(np.array(val_loss_list[-10:-5])):
                    if self.verbose:
                        print(f"== Quitting at iteration / VAL {itr} (val_loss={val_loss:.4f})")
                    break

            if self.verbose and int(self.verbose_eval) > 0 and itr % int(self.verbose_eval) == 0:
                grad_norm = np.linalg.norm(grads, axis=1).mean() * scale
                print(f"[iter {itr}] loss={loss:.4f} val_loss={val_loss:.4f} scale={scale:.4f} "
                      f"norm={grad_norm:.4f}")

            if np.linalg.norm(proj_grad, axis=1).mean() < self.tol:
                if self.verbose:
                    print(f"== Quitting at iteration / GRAD {itr}")
                break

        return self

    def fit_init_params_to_marginal(self, Y, iters=1000):
        self.init_params = self.Dist.fit(Y)
        return

    def pred_dist(self, X, max_iter=None):
        params = np.asarray(self.pred_param(X, max_iter))
        dist = self.Dist(params.T)
        return dist

    def predict(self, X):
        dist = self.pred_dist(X)
        return list(dist.loc.flatten())

    def score(self, X, Y):
        return self.Score.loss(self.pred_dist(X), Y)

    def staged_predict(self, X, max_iter=None):
        predictions = []
        m, n = X.shape
        params = np.ones((m, self.Dist.n_params)) * self.init_params
        for i, (models, s) in enumerate(zip(self.base_models, self.scalings)):
            if max_iter and i == max_iter:
                break
            resids = np.array([model.predict(X) for model in models]).T
            params -= self.learning_rate * resids * s
            dists = self.Dist(np.asarray(params).T)
            predictions.append(dists.loc.flatten())
        return predictions

    def staged_pred_dist(self, X, max_iter=None):
        predictions = []
        m, n = X.shape
        params = np.ones((m, self.Dist.n_params)) * self.init_params
        for i, (models, s) in enumerate(zip(self.base_models, self.scalings)):
            if max_iter and i == max_iter:
                break
            resids = np.array([model.predict(X) for model in models]).T
            params -= self.learning_rate * resids * s
            dists = self.Dist(np.asarray(params).T)
            predictions.append(dists)
        return predictions
'''