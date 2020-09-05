#----使用决策树算法实现运营商客户流失预测-------
#基于某电信企业在2016年1月~2016年3月客户的短信、流量、通信情况以及客户基本信息的数据，构建决策树模型，实现对流失客户的预测
data=read.csv("C:/Users/kw/Desktop/USER_INFO_M.csv")
summary(data)
#INNET_MONTH存在异常值、AGREE_EXP_DATE存在缺失值、VIP_LVL存在异常值、ACCT_FEE存在异常值、CALL_DURA
#存在异常值、NO_ROAM_LOCAL_CALL_DURA存在异常值




#----使用K-Means算法实现运营商客户价值分析-----
#构建在网时长、信用等级、VIP等级、本月消费、通话时长这5个特征，确定K值



#----------使用ARIMA模型预测网站访问量------
"""
#需求说明：
随着流量的增大，某网站的数据信息量也在以一定的幅度增长。基于该网站2016年9月~2017年2月
每天的访问量，使用ARIMA模型预测网站未来7天的访问量。
#实现思路及步骤：
（1）导入数据，绘制原数据的时序图与自相关图，检验序列的平稳性。
（2）通过纯随机性检验，判断序列的价值。
（3）绘制BIC图进行定阶。
（4）预测未来7天的网站访问量。
"""

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

acf(num, lag.max = 30)  # 绘制ACF图"
# 差分
num.diff <- diff(num)  # 进行差分
t2.ts <- zoo(num.diff,t)  
plot(t2.ts,xlab = "时间",ylab = "一阶差分的网站访问量")  #rate的时序图
acf(num.diff, lag.max = 30)  # 绘制差分后序列的ACF图

# 单位根检验
library(tseries)

adf.test(num)
adf.test(num.diff)
Box.test(num.diff, type = "Ljung-Box")  # 纯随机性检验
acfPlot(num.diff)
pacf(num.diff)
library(forecast)
model <- auto.arima(num)
summary(model)
# model residual test
r=model$residuals
rt=zoo(r,t)
plot(rt)
Box.test(r,type="Ljung-Box",lag=1)
pred <- predict(model,n.ahead = 7)
pred$pred
# BIC图
library(TSA)
# 原序列定阶
saleroom.BIC <- armasubsets(y = saleroom, nar = 5, nma = 5)
plot(saleroom.BIC)
# 差分后的序列定阶
num.diff.BIC <- armasubsets(y = num.diff, nar = 5, nma = 5)
plot(num.diff.BIC)


#-----------使用协同过滤算法实现网站的智能推荐-------
"""
#需求说明：
基于实训1中某网站2016年9月每天的访问数据，使用基于内容的协同过滤算法实现网站的智能推荐，帮助客户发现他们
感兴趣但很难发现的网页信息。
#实现思路及步骤
（1）连接数据库，查询一个月的数据。
（2）清洗数据，提取重要特征，删除ID为空的记录。
（3）将数据转化为二元型数据，利用协同过滤算法进行建模。
（4）利用模型对原始数据集进行预测并获得推荐长度为5的结果。
"""

#根据之前建立的df数据框，我们可以知道2016年9月的总访问量数据，根据这个数据，我们可以得到详细的
count <- sum(df$Freq[1:30])
dr <- data[1:count,]
#根据推荐模型，一般为两个特征或者三个特征
dr <- dr[c("ip","page_path")]
head(dr)
#直接删除存在缺失值的样本
newdr <- na.omit(dr)
#删除重复的样本
newdr <- unique(newdr)
# 得到存在异常值的样本的index，然后将其删除
strange <- newdr$page_path[8]
index1 <- newdr$page_path==strange
newdata <- newdr[-which(index1),]
library(plyr)
library(recommenderlab)

# 将数据转换为0-1二元型数据，即模型的输入数据集
info <- as(newdata, "binaryRatingMatrix")

# 采用基于物品的协同过滤算法构建模型
info.re <- Recommender(info, method = "IBCF")

# 利用模型对原始数据集进行预测并获得推荐长度为5的结果
info.p <- predict(info.re, info, n = 5) 
print(as(info.p, "list"))


#-----------使用Aprior算法实现网站的关联分析------
"""
# 需求说明：
基于实训1中某网站的访问数据，使用Apriori算法对网站进行关联分析。
# 实现思路及步骤：
（1）基于实训3预处理后的数据，构建二元矩阵。
（2）构建关联规则模型。
（3）根据关联规则模型的置信度、网站详情表的主推度等因素，计算推荐的综合评分。
"""
library(arules)  # 导入所需库包
# 数据形式转换
dataList <- list()
for (i in unique(newdata$ip)) {
  dataList[[i]] <- newdata[which(newdata$ip == i), 2]
}
# 将数据转换为关联规则所需要的数据类型
TransRep <- as(dataList, "transactions")
# 查看转换后数据的前2行数据
inspect(TransRep[1:2])
RulesRep <- apriori(TransRep, parameter = list(support = 0.02, confidence = 0.25))

inspect(sort(RulesRep, by = "lift")[1:25])  # 按提升度从高到低查看前25条规则

# 生成关联规则
rules <- apriori(TransRep, parameter = list(support = 0.01, confidence = 0.5))
summary(rules)
# 查看提升度排名二的规则
inspect(sort(rules, by = list('lift'))[1:2])  

# 绝对数量显示
click <- data.frame(table(newdata$page_path))
click["percent(%)"] <- click$Freq/sum(click$Freq)*100 
click['hot'] <- (click$Freq-min(click$Freq))/(min(click$Freq))
head(click[order(click$Freq,decreasing = TRUE),],10)
itemFrequencyPlot(TransRep, type = 'absolute', topN = 10, horiz = T)

write(rules, "rules.csv", sep = ",", row.names = FALSE)

result <- read.csv("rules.csv", stringsAsFactors = FALSE)
# 将规则拆开
click.recom <- strsplit(result$rules, "=>")

# 去除中括号
lhs <- 0
rhs <- 0
for (i in 1:length(click.recom)) {
  lhs[i] <- gsub("[{|}+\n]|\\s", "", click.recom[[i]][1])
  rhs[i] <- gsub("[{|}+\n]|\\s", "", click.recom[[i]][2])
}

rules.new <- data.frame(lhs = lhs, rhs = rhs, support = result$support,
                        confidence = result$confidence, lift = result$lift)

write.csv(rules.new, "rules_new.csv", row.names = FALSE)  # 写出数据


# 计算综合评分
# 读取数据
rules.new <- read.csv("rules_new.csv", stringsAsFactors = FALSE)
click_volume <- click[c('Var1','Freq','hot')]

# 统计前项
rules.count <- as.data.frame(table(rules.new$lhs))
rules.count <- rules.count[order(rules.count$Freq, decreasing = TRUE), ]

# 计算每个网站所推荐的网站的综合评分
# 设A的权重a1 = 4, a2 = 6
A <- matrix(c(0, 6, 
              4, 0), 2, 2, byrow = T)
E <- c(1, 1)

# 初始化
rules.new$hots <- 0  # 热门度
rules.new$mark <- 0  # 综合评分

for (i in 1:nrow(rules.new)) {
  # 找到对应的热门度
  hots.num <- which(click_volume$Var1 == rules.new$rhs[i])
  rules.new$hots[i] <- click_volume$hot[hots.num]

  
  # 计算综合评分
  Y <- c(rules.new$hots[i],rules.new$confidence[i])
  rules.new$mark[i] <- round((E - Y) %*% A %*% t(t(Y)), 3)
}

# 对综合评分进行排序
rules.new <- rules.new[order(rules.new$mark, decreasing = TRUE), ]

write.csv(rules.new, "./tmp/recommend.csv", row.names = FALSE)  # 写出数据


# 选取后项为"芹菜炒腰花" 的数据
rules.item <- rules.new[which(rules.new$rhs == "芹菜炒腰花"), ]
write.csv(rules.item, "./tmp/rules_item.csv", row.names = FALSE)


