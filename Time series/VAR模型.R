# VAR模型是向量自回归模型的简称，是基于数据的统计性质建立的一种常用的计量经济学模型，它把系统中的每一个
# 内生变量作为系统中所有内生变量的滞后值的函数来构建模型，从而将单变量自回归模型推广到由多元时间序列
# 变量组成的“向量”自回归模型。VAR模型是处理多个相关经济指标的分析与预测最容易操作的模型之一，并且在一定
# 的条件下，多元MA和ARMA模型也可转化为VAR模型，因此近年来VAR模型受到越来越多的经济工作者的重视。
#1.平稳性检验
#2.协整检验
#3.滞后阶数的确定
#4.VAR模型的拟合
#5.脉冲响应分析
#6.VAR模型的预测
#读取数据

# 读取数据 --------------------------------------------------------------------
library(readxl)
all <- read_excel("C:/Users/kw/Desktop/all.xls", sheet = "Sheet1")
data <- all[,1:2]
head(data)
data <- data.frame(data)

# 时序图 ---------------------------------------------------------------------
x.ts <- ts(x,start = c(1997,1),end = c(2019,11),frequency = 12)
y.ts <- ts(y,start = c(1997,1),end = c(2019,11),frequency = 12)

# 平稳性检验 -------------------------------------------------------------------
library(urca) 
#或许数据需要进行log处理或者diff处理
adfx <-  ur.df(data$NPP1,type = 'trend',selectlags = 'AIC') #原假设是存在单位根
adfy <-  ur.df(data$...2,type = 'trend',selectlags = 'AIC')
summary(adfx) #是否平稳看(Value of test-statistic is)和(Critical values for test statistics)
summary(adfy) #若value值为正，则大于临界值时拒绝原假设；若value值为负，则小于临界值时拒绝原假设
#若存在单位根，则进行差分处理，然后重新检验差分后的序列
dx <- diff(x)
dy <- diff(y)
adfdx 
adfdy
#若两个内生变量都平稳或一阶差分后平稳，即一阶单整，则不能做Granger因果检验，只能做协整检验

# 协整检验 --------------------------------------------------------------------
#协整检验主要是针对非平稳的单个序列，但是它们的线性组合可能是平稳的。几个变量之间可能存在的一种长期
#均衡关系进行检验，表现为存在某个协整方程。由于所有变量都是一阶单整的，是非平稳时间序列，因此各变量
#之间可能存在协整关系，如果要对所选择的内生变量进行VAR模型的构建，需要进行协整检验，以判断各个变量
#之间是否存在长期稳定的协整关系，处理各变量之间的是否存在伪回归问题。在这使用E-G两步法协整检验。
fit <- lm(y~x)
summary(fit)
library(zoo)
library(lmtest)
dwtest(fit) #检验序列的自相关性
error <- residuals(fit) #提取残差序列
urt.res <- ur.df(error,type = 'none',selectlags = 'AIC')
summary(urt.res)



# 滞后阶数的确定及模型拟合 -----------------------------------------------------------------

library(MASS)
library(sandwich)
library(strucchange)
library(vars)
lg <- VARselect(d[,2:3])
lg
lg$selection # 滞后阶
#建立VAR模型
var <- VAR(d[,2:3],lag.max = 3,ic='AIC')
summary(var)
coef(var)
plot(var)

# 脉冲响应函数 ------------------------------------------------------------------
var.irf <- irf(var)
var.irf$ci
plot(var.irf)
#方差分解
sv <- fevd(var,n.ahead = 20)
summary(sv)
sv$NPP1
# VAR预测 -------------------------------------------------------------------

pred <- predict(var,n.ahead = 10,ci=0.95)
pred
