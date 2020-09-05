text <- readLines("C:/Users/kw/Desktop/c_df110.txt",encoding = "UTF-8")
class(text)
text <- unique(text)  # 对评论内容去重,
#write.csv(text, "C:/Users/kw/Desktop/text.csv", row.names = FALSE)
#text <- as.character(text)
# 分词
library(jiebaR)  # Version:0.9.1
cutter <- worker(type = "tag", stop_word = "C:/Users/kw/Desktop/CNstopwords.txt")
seg_word <- list()
for(i in 1:length(text)){
  seg_word[[i]] <- segment(text[i], cutter)
}
head(seg_word, 40)
# 将词语转为数据框形式，一列是词，一列是词语所在的句子ID，最后一列是词语在该句子的位置
n_word <- sapply(seg_word, length)  # 每个词条的词个数
index <- rep(1:length(seg_word), n_word)  # 每个词条有多少个词就复制多少次
#type <- rep(meidi_reviews$type, n_word)
nature <- unlist(sapply(seg_word, names))
result <- data.frame(index, unlist(seg_word), nature)
colnames(result) <- c("id", "word","nature")
head(result)
# 将每个词在每个词条的位置标记出来
n_word <- sapply(split(result,result$id), nrow)
index_word <- sapply(n_word, seq_len)
index_word <- unlist(index_word)  
result$index_word <- index_word
head(result)

# 提取含有名词类的评论数据
is_n <- subset(result, grepl("n", result$nature), "id")
result <- result[result$id %in% is_n$id, ]


# 代码 8-8

# 绘制词云
# 查看分词效果，最快捷的方式是绘制词云,基于词频的词云图
library(wordcloud2)  # Version:0.2.0
#  统计词频
word.frep <- table(result$word)
word.frep <- sort(word.frep, decreasing = TRUE)
word.frep <- data.frame(word.frep)
head(word.frep)
wordcloud2(word.frep[1:1000,], color = "random-dark")
write.csv(result, "C:/Users/kw/Desktop/word.csv", row.names = FALSE)

# 构建语料库
library(NLP)
library(tm)  # Version:0.7-1
pos.corpus <- Corpus(VectorSource(result$word))

# 词条-文档关系矩阵
pos.gxjz <- DocumentTermMatrix(pos.corpus,
                               control = list(wordLengths = c(1, Inf),
                                              bounds = list(global = 5, Inf),
                                              removeNumbers = TRUE))


# 构造主题间余弦相似度函数
library(topicmodels)
lda.k <- function(gxjz){
  # 初始化平均余弦相似度
  mean_similarity <- c()
  mean_similarity[1] = 1
  # 循环生成主题并计算主题间相似度
  for(i in 2:10){
    control <- list(burnin = 500, iter = 1000, keep = 100)
    Gibbs <- LDA(gxjz, k = i, method = "Gibbs", control = control)
    term <- terms(Gibbs, 50)  # 提取主题词
    # 构造词频向量
    word <- as.vector(term)  # 列出所有词
    freq <- table(word)  # 统计词频
    unique_word <- names(freq)
    mat <- matrix(rep(0, i * length(unique_word)),  # 行数为主题数，列数为词
                  nrow = i, ncol = length(unique_word))
    colnames(mat) <- unique_word
    
    # 生成词频向量
    for(k in 1:i){
      for(t in 1:50){
        mat[k, grep(term[t,k], unique_word)] <- mat[k, grep(term[t, k], unique_word)] + 1
      }
    }
    p <- combn(c(1:i), 2)
    l <- ncol(p)
    top_similarity <- c()
    for(j in 1:l){
      # 计算余弦相似度
      x <- mat[p[, j][1], ]
      y <- mat[p[, j][2], ]
      top_similarity[j] <- sum(x * y) / sqrt(sum(x^2) * sum(y ^ 2))
    }
    mean_similarity[i] <- sum(top_similarity) / l
    message("top_num ", i)
  }
  return(mean_similarity)
}
library(knitr)
# 计算平均主题余弦相似度
pos_k <- lda.k(pos.gxjz)
par(mfrow = c(2, 1))
plot(pos_k, type = "l")

# LDA主题分析
control <- list(burnin = 500, iter = 1000, keep = 100)
pos.gibbs <- LDA(pos.gxjz, k = 3, method = "Gibbs", control = control)


pos.termsl <- terms(pos.gibbs, 10)
pos.termsl

write.csv(pos.termsl, "C:/Users/kw/Desktop/pos_termsl.csv", row.names = FALSE)
