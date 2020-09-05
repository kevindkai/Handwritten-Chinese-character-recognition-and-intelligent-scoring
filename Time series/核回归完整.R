#导入数据，变量定义
air <- read.delim("C:/Users/kw/Desktop/air.dat", header=FALSE)
names(air) <- c("OZ","RAD","TEMP","WIND")
y <- air$OZ
x <- air$WIND
plot(x,y,xlab = "WIND",ylab = "OZ")
#线性回归
model1 <- lm(y~x)
summary(model1)
model2 <- lm(y~x+I(x^2))
summary(model2)
with(air,{
  plot(x,y,xlab = "WIND",ylab = "OZ")
  abline(model1,col="red")
  abline(model2,col="blue")
})
# 核回归 ---------------------------------------------------------------------
h <- 0.5
fx.hat <- function(z, h) {
  dnorm((z - x)/h)/h
}
NWSMOOTH <- function(h, y, x) {
  n <- length(y)
  s.hat <- rep(0, n)
  for (i in 1:n) {
    a <- fx.hat(x[i], h)
    s.hat[i] <- sum(y * a/sum(a))
  }
  return(s.hat)
}
NWsmooth.val <- NWSMOOTH(h, y, x)
GMSMOOTH <- function(y, x, h) {
  n <- length(y)
  s <- c(-Inf, 0.5 * (x[-n] + x[-1]), Inf)
  s.hat <- rep(0, n)
  for (i in 1:n) {
    fx.hat <- function(z, h, x) {
      dnorm((x - z)/h)/h
    }
    a <- y[i] * integrate(fx.hat, s[i], s[i + 1], h = h, x = x[i])$value
    s.hat[i] <- sum(a)
  }
  return(s.hat)
}
GMsmooth.val <- GMSMOOTH(y, x, h)
plot(x, y, xlab = "WIND", ylab = "OZ", col = 1)
abline(model1, ylim = c(-15.5, 15.5), col = 1)
abline(model2,col=2)
lines(x, NWsmooth.val, lty = 2, col = 3)
lines(x, GMsmooth.val, lty = 3, col = 4)
letters <- c("linear model","quadratic model", "NW method", "GM method")
legend("topright", legend = letters, lty = 1:4, col = 1:4, cex = 0.7)
#基于normal核方法的不同窗宽核回归
library(stats)
library(graphics)
plot(x,y,xlab = "WIND",ylab = "OZ")
model4 <- ksmooth(x,y,kernel = "normal",bandwidth = 0.1)
model5 <- ksmooth(x,y,kernel="normal",bandwidth = 0.2)
model6 <- ksmooth(x,y,kernel="normal",bandwidth = 0.5)
model7 <- ksmooth(x,y,kernel="normal",bandwidth = 0.8)
model8 <- ksmooth(x,y,kernel="normal",bandwidth = 1)
model9 <- ksmooth(x,y,kernel="normal",bandwidth = 2)
model10 <- ksmooth(x,y,kernel="normal",bandwidth = 5)
model11 <- ksmooth(x,y,kernel="normal",bandwidth = 10)
with(air,{
  plot(x,y,xlab = "WIND",ylab = "OZ")
  lines(model4,col=1)
  lines(model5,col=2)
  lines(model6,col=3)
  lines(model7,col=4)
  lines(model8,col=5)
  lines(model9,col=6)
  lines(model10,col=7)
  lines(model11,col=8)
  letters <- c("bandwidth=0.1","bandwidth=0.2","bandwidth=0.5","bandwidth=0.8","bandwidth=1","bandwidth=2","bandwidth=5","bandwidth=10")
  legend("topright", legend = letters, lty = 1:8, col = 1:8, cex = 0.7)
})
#基于box的不同窗宽核回归
model4 <- ksmooth(x,y,kernel = "box",bandwidth = 0.1)
model5 <- ksmooth(x,y,kernel="box",bandwidth = 0.2)
model6 <- ksmooth(x,y,kernel="box",bandwidth = 0.5)
model7 <- ksmooth(x,y,kernel="box",bandwidth = 0.8)
model8 <- ksmooth(x,y,kernel="box",bandwidth = 1)
model9 <- ksmooth(x,y,kernel="box",bandwidth = 2)
model10 <- ksmooth(x,y,kernel="box",bandwidth = 5)
model11 <- ksmooth(x,y,kernel="box",bandwidth = 10)
with(air,{
  plot(x,y,xlab = "WIND",ylab = "OZ")
  lines(model4,col=1)
  lines(model5,col=2)
  lines(model6,col=3)
  lines(model7,col=4)
  lines(model8,col=5)
  lines(model9,col=6)
  lines(model10,col=7)
  lines(model11,col=8)
  letters <- c("bandwidth=0.1","bandwidth=0.2","bandwidth=0.5","bandwidth=0.8","bandwidth=1","bandwidth=2","bandwidth=5","bandwidth=10")
  legend("topright", legend = letters, lty = 1:8, col = 1:8, cex = 0.7)
})

#基于不同核方法和不同窗宽的核回归
model4 <- ksmooth(x,y,kernel = "normal",bandwidth = 0.2)
model5 <- ksmooth(x,y,kernel="box",bandwidth = 0.2)
model6 <- ksmooth(x,y,kernel="normal",bandwidth = 0.5)
model7 <- ksmooth(x,y,kernel="box",bandwidth = 0.5)
model8 <- ksmooth(x,y,kernel="normal",bandwidth = 1)
model9 <- ksmooth(x,y,kernel="box",bandwidth = 1)
model10 <- ksmooth(x,y,kernel="normal",bandwidth = 2)
model11 <- ksmooth(x,y,kernel="box",bandwidth = 2)
with(air,{
  plot(x,y,xlab = "WIND",ylab = "OZ")
  lines(model4,col=1)
  lines(model5,col=2)
  lines(model6,col=3)
  lines(model7,col=4)
  lines(model8,col=5)
  lines(model9,col=6)
  lines(model10,col=7)
  lines(model11,col=8)
  letters <- c("n_bw=0.2","b_bw=0.2","n_bw=0.5","b_bw=0.5","n_bw=1","b_bw=1","n_bw=2","b_bw=2")
  legend("topright", legend = letters, lty = 1:8, col = 1:8, cex = 0.7)
})
#构建smooth、loess、supsmu模型
model_loess <- loess.smooth(x,y)
model_sup <- supsmu(x,y)
summary(model_loess)
plot(x,y,xlab = "WIND",ylab = "OZ")
lines(model_loess,col="red")
lines(model6,col=5)
lines(model_sup,col="purple")
letters <- c("smooth","loess","supsmu")
legend("topright", legend = letters, lty = 1:3, col = c(5,"red","purple"), cex = 1)

#均分误差函数
mse<-function(ft){
  y_1<-ft$y
  MSE<-sum((y-y_1)^2)/(length(y)-1)
  MSE
}
mse(model_loess)
mse(model_sup)
mse(model8)


#dataset <- air[order(air$OZ),]
#y=dataset[,1]
#x=dataset[,4]
#x <- order(x)
#y <- order(y)
kerepan<-function(x){
  ifelse(abs(x)<=1,0.75*(1-x^2),0)
  }
local<- function(y, x, h) {
  n <- length(y)
  s<- rep(0, n)
  for (i in 1:n) {
    z<-(x - x[i])/h
    weight <- kerepan(z)
    mod <- lm(y ~ x, weights = weight)
    s[i] <- as.numeric(predict(mod, data.frame(x = x[i])))
  }
  return(s)
}
fit<- local(y, x, h=5)
plot(x,y,main = "Local Linear Kernel Estimate",xlab = "WIND",ylab = "OZ")
lines(x,fit,col="red")

#定义Epanechnikov函数
kernalEpanechnikov <- function (x)
{
  if(ncol(x)!=1)
  {
    stop('error input the data')
  }
  stdX <- sd(x)
  h<-2.34*stdX*length(x)^(-1/5)
  xPh<- abs(x/h)
  xPh[xPh <=1] <-1
  xPh[xPh>1] <- 0
  kernalX <- 0.75/h*(1-(x/h)^2)*xPh
  return(kernalX)
}
fit<-loess(model8,kernel="epan",bandwidth=5)
fithat<-predict(fit)
se<-sum(y-fithat)/111
#第二大题代码
set.seed(26)
data <- runif(n = 12000,min = 0,max = 1)
x <- matrix(data,nrow = 400,ncol = 30)
y <- x[,1]-1.5*x[,3]+0.8*x[,11]+rnorm(400,mean=0)
names(x) <- c(1:30)
head(x)
df <- cbind(x,y)
df <- data.frame(df)
model_ols <- lm(y~.,data=df)
summary(model_ols)
#lasso回归
library(glmnet)
A <- model.matrix(y~.,df)
lass_model <- glmnet(A,y,alpha=1)
plot(lass_model)
cv <- cv.glmnet(A,y,alpha=1)
plot(cv)
lambda <- cv$lambda.min
predict(lass_model,s=lambda,type="coefficients")
