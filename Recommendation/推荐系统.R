#--------------推荐系统---------------------
# 读入数据
data <- read.csv("C:/Users/kw/Desktop/dataMovieLense.csv")
head(data)
prop.table(table(data$score))
summary(data$score)
newdata <- na.omit(data)
index1 <- newdata$score>5
newdata <- newdata[-which(index1),]
library(ggplot2)
# 直方图
ggplot(data,aes(x=data$score))+geom_histogram(binwidth = 0.09,color='grey')+labs(x='电影评分',title='电影评分统计')
# 饼图
ggplot(data,x=data$score,aes(x=factor(1),fill=factor(data$score)))+geom_bar(width = 1)+
  coord_polar(theta="y")+ggtitle("评分分布图")+
  labs(x="",y="")+
  guides(fill=guide_legend(title = '评分分数'))

library(reshape)
newdata <- cast(newdata,user~movie,value='score')
class(newdata)
library(recommenderlab)
class(newdata) <- 'data.frame'
newdata <- as.matrix(newdata)
newdata <- as(newdata,"realRatingMatrix")
as(newdata,'matrix')[1:5,1:3]
as(newdata,'list')[[1]][1:5]
# 采用基于物品的协同过滤算法构建模型
model1 <- Recommender(newdata[1:900], method = "IBCF")
model1
# 利用模型对原始数据集进行预测并获得推荐长度为3的结果
model1_predict <- predict(model1, newdata[901:903], n = 3) 
model1_predict
as(model1_predict,'list')
# 用户对item的评分预测
score_pre <- predict(model1,newdata[901:903],type='ratings')
score_pre
as(score_pre,'matrix')[1:3,1:3]
# 模型的评估
rmse <- function(true,predict)
{
  sqrt(mean(true-predict)^2,na.rm=T)
}
# 划分数据集
model_eval <- evaluationScheme(newdata,method='split',train=0.75,given=15,goodRating=5)
model_eval
# 分别用RANDOM、UBCF、IBCF建立预测模型
model_random <- Recommender(getData(model_eval,'train'),method='RANDOM')
model_ubcf <- Recommender(getData(model_eval,'train'),method='UBCF')
model_ibcf <- Recommender(getData(model_eval,'train'),method='IBCF')
# 分别根据每个模型预测评分
predict_random <- predict(model_random,getData(model_eval,'known'),type='ratings')
predict_ubcf <- predict(model_ubcf,getData(model_eval,'known'),type='ratings')
predict_ibcf <- predict(model_ibcf,getData(model_eval,'known'),type='ratings')

# 计算预测误差
error <- rbind(calcPredictionAccuracy(predict_random,getData(model_eval,'unknown')),
               calcPredictionAccuracy(predict_ubcf,getData(model_eval,'unknown')),
               calcPredictionAccuracy(predict_ibcf,getData(model_eval,'unknown')))
rownames(error) <- c('Random','UBCF','IBCF')
