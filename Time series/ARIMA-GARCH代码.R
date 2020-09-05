#加载时间序列分析所需要的包
library(fGarch)
library(rugarch)
library(TSA)
library(tseries)
library(zoo)
library(forecast)
library(psych)
library(ggplot2)
library(lattice)
library(ccgarch)
library(quantmod)
library(readxl)

waihui <- read_excel("C:/Users/kw/Desktop/waihui2.xls")  #读取数据
par(mfrow=c(1,1))                             #设置画布
par(mar=c(2,4,2,2))
t <- as.Date(waihui$date)                     #将date变量转化为日期数据
price <- as.ts(waihui$price)                  #将price变量转化为时间序列数据
price<- log(price)                            #对price取对数
library(zoo)                                  #调用时间序列包
t1.ts <- zoo(price,t)  
plot(t1.ts,xlab = "time",ylab = "ln(price)")  #画ln(price)的时序图

dt=zoo(diff(price),t)
plot(dt,xlab="time",ylab="diff ln(price)")    #画diff(ln(price))的时序图

adf <- adf.test(diff(price))                  #对差分后的对数序列（对数收益率）进行单位根检验
adf
Box.test(diff(price),type="Ljung-Box")        #对差分后的对数序列（对数收益率）进行白噪声检验
par(mfrow=c(2,1))
par(mar=c(2,4,1,2)) 
acf(diff(price))                              #画对数收益率序列的自相关图
pacf(diff(price))                             #画对数收益率序列的偏自相关图

#build ARIMA(2,1,0)--model
model=arima(price,order = c(2,1,0))           #建立ARIMA模型
summary(model)                                #查看建好的模型
model
auto.arima(price)                             #通过自动定阶函数确定模型
library(stats)
predict(model,n.ahead=5)                      #预测下一周的数据
r=residuals(model,standard=T)                 #计算模型的标准化残差值

dr=zoo(r,t)
par(mfrow=c(1,1))
plot(dr,xlab="time",ylab="residual")          #画模型残差的时序图
library(FinTS)                                #调用arch检验包
ArchTest(r) 
McLeod.Li.test(y=r)

#建立GARCH模型
r2=r^2                                        #得到残差的平方
library(fGarch)
g1=garchFit(~1+garch(1,1),data=r2,trace=F)    #建立GARCH(1,1)模型
summary(g1)
g2=garchFit(~1+garch(1,2),data = r2,trace = F)#建立GARCH(1,2)模型
summary(g2)
g3=garchFit(~1+garch(2,1),data = r2,trace = F)#建立GARCH(2,1)模型
summary(g3)
g4=garchFit(~1+garch(2,2),data = r2,trace = F)#建立GARCH(2,2)模型
summary(g4)
predict(g1)                                   #通过GARCH(1,1)模型进行预测

#计算GARCH模型的残差并进行检验
rr=residuals(g1,standardize=T)                #计算GARCH(1,1)模型的标准化残差
garch_r<- zoo(rr,t)               
par(mfrow=c(2,1))
plot(garch_r,xlab = "time",ylab = "standard residual of GARCH ") #GARCH模型标准化残差的时序图
abline(h=2,col="red")
abline(h=-2,col="red")
ArchTest(rr) 
McLeod.Li.test(y=rr)