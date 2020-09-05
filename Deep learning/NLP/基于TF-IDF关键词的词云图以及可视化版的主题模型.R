

# 基于TF-IDF关键词的词云图
library(readxl)
freq <- read_excel("C:/Users/kw/Desktop/cloud.xlsx",col_names = FALSE)
freq <- freq[c("word","num")]
library(wordcloud2)
letterCloud(freq,word = "车", color = "random-light",
            backgroundColor = "black",size = 0.3)
figPath = system.file("examples/t.png",package = "wordcloud2")
wordcloud2(freq, figPath = figPath, size = 1.5,color = "skyblue")
wordcloud2(freq,shape='star') #绘制词云


library(jiebaR)
library(jiebaRD)
library(Rwordseg)#分词处理工具
library(tm)#文本整理工具，创建矩阵
library(tmcn)
library(lda)#LDA模型包
library(LDAvis)#LDA可视化
library(topicmodels)

library(jiebaR) #加载
comments_segged<- readLines("C:/Users/kw/Desktop/c_df110.segment.2020-06-14_11_11_57.txt",encoding="UTF-8") #读取分词结果
comments <- as.list(comments_segged) #将向量转化为列表
doc.list <- strsplit(as.character(comments),split=" ") #将每行文本，按照空格分开，每行变成一个词向量，储存在列表里
term.table <- table(unlist(doc.list))
#这里有两步，unlist用于统计每个词的词频；table把结果变成一个交叉表式的factor，原理类似python里的词典，key是词，value是词频
term.table <- sort(term.table, decreasing = TRUE) #按照词频降序排列

del <- term.table < 5| nchar(names(term.table))<2   #把不符合要求的筛出来
term.table <- term.table[!del]   #去掉不符合要求的
vocab <- names(term.table)    #创建词库
get.terms <- function(x) {
  index <- match(x, vocab)  # 获取词的ID
  index <- index[!is.na(index)]  #去掉没有查到的，也就是去掉了的词
  rbind(as.integer(index - 1), as.integer(rep(1, length(index))))   #生成上图结构
}
documents <- lapply(doc.list, get.terms)
K <- 5   #主题数
G <- 5000    #迭代次数
alpha <- 0.10   
eta <- 0.02

library(lda)
set.seed(357)
fit <- lda.collapsed.gibbs.sampler(documents = documents, K = K, vocab = vocab, num.iterations = G, alpha = alpha, eta = eta, initial = NULL, burnin = 0, compute.log.likelihood = TRUE)

theta <- t(apply(fit$document_sums + alpha, 2, function(x) x/sum(x)))  #文档—主题分布矩阵
phi <- t(apply(t(fit$topics) + eta, 2, function(x) x/sum(x)))  #主题-词语分布矩阵
term.frequency <- as.integer(term.table)   #词频
doc.length <- sapply(documents, function(x) sum(x[2, ])) #每篇文章的长度，即有多少个词

library(LDAvis)
json <- createJSON(phi = phi, theta = theta,
                   doc.length = doc.length, vocab = vocab,
                   term.frequency = term.frequency)
#json为作图需要数据，下面用servis生产html文件，通过out.dir设置保存位置
serVis(json, out.dir = './vis', open.browser = FALSE)
writeLines(iconv(readLines("./vis/lda.json"), from = "GBK", to = "UTF8"), 
           file("./vis/lda.json", encoding="UTF-8"))