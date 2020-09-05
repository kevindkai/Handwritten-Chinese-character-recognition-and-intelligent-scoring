data <- read.csv("C:/Users/kw/Desktop/clear.csv",encoding = "UTF-8")
date <- data$data_time
library(lubridate)
date <- ymd_hms(date)
date <- round_date(date,"day")
df <- table(date)
df <- data.frame(df)
t <- df$date #转化为日期
num <- as.ts(df$Freq)   #转化为时间序列数据
#rate <- log(rate)
library(zoo)       #调用时间序列包
t1.ts <- zoo(num,t)  
plot(t1.ts,xlab = "时间",ylab = "网站访问量")  #rate的时序图

acf(num, lag.max = 30)  # 绘制ACF图

# 差分
num.diff <- diff(num, differences = 2)  # 进行差分
acf(num.diff, lag.max = 30)  # 绘制差分后序列的ACF图
# 单位根检验
library(tseries)
adf.test(num)

Box.test(num, type = "Ljung-Box")  # 纯随机性检验

# BIC图
library(TSA)
# 原序列定阶
saleroom.BIC <- armasubsets(y = saleroom, nar = 5, nma = 5)
plot(saleroom.BIC)
# 差分后的序列定阶
saleroom.diff.BIC <- armasubsets(y = saleroom.diff, nar = 5, nma = 5)
plot(saleroom.diff.BIC)



# 根据BIC图定阶
library(forecast)
# 初始化
checkout <- data.frame(p = 0, d = 0, q = 0, P = 0, D = 0, 
                       Q = 0, "残差P值" = 0, "平均误差" = 0)
test_checkout <- data.frame(p = 0, d = 0, q = 0, P = 0, D = 0, 
                            Q = 0, "残差P值" = 0, "平均误差" = 0)
j <- 1

test_model <- function(p, q, P, Q){
  model <- Arima(saleroom, order = c(p, 0, q),
                 seasonal = list(order = c(P, 2, Q), period = 7))
  result <- Box.test(model$residuals, type = "Ljung-Box")
  # 预测
  sale.forecast <- forecast(model, h = 3, level = c(99.5))
  # 计算平均绝对百分误差
  error <- abs(as.numeric(sale.forecast[[4]]) - sale[29:31,3]) / sale[29:31,3]
  p.value <- round(result$p.value, 4)
  print(paste('p=', p, ';q=', q, ';P=', P,',Q=', Q, ';残差P值:',
              p.value, ';平均误差:', mean(error), collapse = ""))
  test_checkout[1,1] <- p
  test_checkout[1,2] <- 0
  test_checkout[1,3] <- q
  test_checkout[1,4] <- P
  test_checkout[1,5] <- 2
  test_checkout[1,6] <- Q
  test_checkout[1,7] <- round(result$p.value, 4)
  test_checkout[1,8] <- mean(error) 
  return(test_checkout)
}

for (p in c(0,3,4,5)) {
  if (p == 0 | p == 3) {
    for (q in 1:5) {
      for (P in c(0,1)) {
        for (Q in c(1,2,3,5)) {
          test_checkout <- test_model(p, q, P, Q)
          checkout[j, ] <- test_checkout[1, ]
          j <- j + 1
        }
      }
    }
  }
  if (p == 4) {
    for (q in 1:5) {
      if (q == 1) {
        for (Q in c(1,2,3,5)) {
          test_checkout <- test_model(p, q, 1, Q)
          checkout[j, ] <- test_checkout[1, ]
          j <- j + 1
        }
      }
      if (q != 1) {
        for (Q in c(1,2,3,5)) {
          test_checkout <- test_model(p, q, 0, Q)
          checkout[j, ] <- test_checkout[1, ]
          j <- j + 1
        }
      }
    }
  }
  if (p == 5) {
    for (q in 1:5) {
      for (Q in c(1,2,3,5)) {
        test_checkout <- test_model(p, q, 0, Q)
        checkout[j, ] <- test_checkout[1, ]
        j <- j + 1
      }
    }
  }
}
write.csv(checkout, "./tmp/checkout.csv", row.names = F)  # 导出每个模型的结果


# 取最优模型预测
model <- Arima(saleroom, order = c(0,0,1), 
               seasonal = list(order = c(0,2,2), period = 7))
summary(model)

Box.test(model$residuals, type = "Ljung-Box")  # 纯随机性检验

# 预测未来3天的销售额
sale.forecast <- forecast(model, h = 3, level = c(99.5))
plot(sale.forecast)

# 计算平均误差
error <- abs(as.numeric(sale.forecast[[4]]) - sale[29:31,3]) / sale[29:31,3]
mean(error)