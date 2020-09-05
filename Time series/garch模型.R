#所有出现library函数的，都是先调用此函数才能运行，如果你的电脑没有安装此函数包，则需先安装再调用
library(readxl)
t1 <- read_excel("C:/Users/kw/Desktop/t1.xls") #导入数据，要看你存放的位置
View(t1)
t <- as.Date(t1$year) #转化为日期
gdp <- as.ts(t1$GDP)   #转化为时间序列数据
student <- as.ts(t1$students)
fund <- as.ts(t1$fund)   #转化为时间序列数据
library(zoo)       #调用时间序列包
t1.ts <- zoo(y,t)  
plot(t1.ts,xlab = "time",ylab = "volatility")  #y的时序图
t2.ts <- zoo(x,t)
plot(t2.ts,xlab = "time",ylab="financing")     #x的时序图
library(psych)    #调用描述统计函数包
describe(y)       #对y进行描述统计分析
shapiro.test(y)     #y的正态性检验
qqnorm(y);qqline(y)  #y的正态QQ图
hist(y,freq = F);lines(density(y));rug(y)   #y的直方图和概率密度曲线
describe(x) 
shapiro.test(x)       #同上y
qqnorm(x);qqline(x)
hist(x,freq = F);lines(density(x));rug(x)
library(tseries)  #调用时间序列包
adf.test(y)       #单位根检验
adf.test(x)
library(rugarch)   #调用garch模型拟合包
a <- lm(y ~ x)     #最小二乘估计得回归方程
summary(a)   #查看模型
library(car)     #进行D.W检验
durbinWatsonTest(a)
r <- residuals(a)   #得到模型的残差
acf(r,lag.max = 36,plot = F)  #残差自回归检验
library(FinTS)    #调用arch检验包
ArchTest(r)       #残差的arch检验
y <- garch(y ~x ,order = c(1,1),control = garch) #garch模型拟合，有点小问题，我待会解决
y <- c(1:3, 7, 5)
x <- c(1:3, 6:7)
( ee <- effects(lm(y ~ x)) )
c( round(ee - effects(lm(y+10 ~ I(x-3.8))), 3) )
# just the first is different




                                                              
               