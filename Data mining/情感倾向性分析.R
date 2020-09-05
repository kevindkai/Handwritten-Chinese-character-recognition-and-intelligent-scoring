library(tableone)
## PS matching
library(Matching)
## Weighted analysis
library(survey)
library(reshape2)
library(ggplot2)

## 读取数据
rhc <- read.csv("rhc.csv")
dim(rhc)
head(rhc)

# 待统计的协变量：
vars <- c("age","sex","race","edu","income","ninsclas","cat1","das2d3pc","dnr1",
          "ca","surv2md1","aps1","scoma1","wtkilo1","temp1","meanbp1","resp1",
          "hrt1","pafi1","paco21","ph1","wblc1","hema1","sod1","pot1","crea1",
          "bili1","alb1","resp","card","neuro","gastr","renal","meta","hema",
          "seps","trauma","ortho","cardiohx","chfhx","dementhx","psychhx",
          "chrpulhx","renalhx","liverhx","gibledhx","malighx","immunhx",
          "transhx","amihx")
## Construct a table
tabUnmatched <- CreateTableOne(vars = vars, strata = "swang1", 
                               data = rhc, test = FALSE)
## Show table with SMD
print(tabUnmatched, smd = TRUE)



rhc$swang1 <- factor(rhc$swang1, levels = c("No RHC", "RHC"))
## Fit model
psModel <- glm(formula = swang1 ~ age + sex + race + edu + income + ninsclas +
                 cat1 + das2d3pc + dnr1 + ca + surv2md1 + aps1 + scoma1 +
                 wtkilo1 + temp1 + meanbp1 + resp1 + hrt1 + pafi1 +
                 paco21 + ph1 + wblc1 + hema1 + sod1 + pot1 + crea1 +
                 bili1 + alb1 + resp + card + neuro + gastr + renal +
                 meta + hema + seps + trauma + ortho + cardiohx + chfhx +
                 dementhx + psychhx + chrpulhx + renalhx + liverhx + gibledhx +
                 malighx + immunhx + transhx + amihx,
               family  = binomial(link = "logit"),
               data    = rhc)

## Predicted probability of being assigned to RHC
rhc$pRhc <- predict(psModel, type = "response")
head(rhc$pRhc)

## Predicted probability of being assigned to no RHC
rhc$pNoRhc <- 1 - rhc$pRhc
head(rhc$pNoRhc)

## Predicted probability of being assigned to the
## treatment actually assigned (either RHC or no RHC)
rhc$pAssign <- NA
rhc$pAssign[rhc$swang1 == "RHC"]    <- rhc$pRhc[rhc$swang1   == "RHC"]
rhc$pAssign[rhc$swang1 == "No RHC"] <- rhc$pNoRhc[rhc$swang1 == "No RHC"]
## Smaller of pRhc vs pNoRhc for matching weight
rhc$pMin <- pmin(rhc$pRhc, rhc$pNoRhc)
head(rhc$pMin)

listMatch <- Match(Tr       = (rhc$swang1 == "RHC"),      # Need to be in 0,1
                   ## logit of PS,i.e., log(PS/(1-PS)) as matching scale
                   X        = log(rhc$pRhc / rhc$pNoRhc),
                   ## 1:1 matching
                   M        = 1,
                   ## caliper = 0.2 * SD(logit(PS))
                   caliper  = 0.2,
                   replace  = FALSE,
                   ties     = TRUE,
                   version  = "fast")
## Extract matched data
rhcMatched <- rhc[unlist(listMatch[c("index.treated","index.control")]), ]

## Construct a table
tabMatched <- CreateTableOne(vars = vars, strata = "swang1", 
                             data = rhcMatched, test = FALSE)
## Show table with SMD
print(tabMatched, smd = TRUE)

## Matching weight
rhc$mw <- rhc$pMin / rhc$pAssign
## Weighted data
rhcSvy <- svydesign(ids = ~ 1, data = rhc, weights = ~ mw)

## Construct a table (This is a bit slow.)
tabWeighted <- svyCreateTableOne(vars = vars, strata = "swang1", 
                                 data = rhcSvy, test = FALSE)
## Show table with SMD
print(tabWeighted, smd = TRUE)

library(data.table)
## Construct a data frame containing variable name and SMD from all methods
dataPlot <- data.table(variable  = rownames(ExtractSmd(tabUnmatched)),
                       Unmatched = ExtractSmd(tabUnmatched),
                       Matched   = ExtractSmd(tabMatched),
                       Weighted  = ExtractSmd(tabWeighted))
colnames(dataPlot) <- c("variable","Unmatched","Matched","Weighted")
## Create long-format data for ggplot2
dataPlotMelt <- melt(data          = dataPlot,
                     id.vars       = c("variable"),
                     variable.name = "Method",
                     value.name    = "SMD")

## Order variable names by magnitude of SMD
varNames <- as.character(dataPlot$variable)[order(dataPlot$Unmatched)]

## Order factor levels in the same order
dataPlotMelt$variable <- factor(dataPlotMelt$variable,
                                levels = varNames)

## Plot using ggplot2
ggplot(data = dataPlotMelt, mapping = aes(x = variable, y = SMD,
                                          group = Method, color = Method)) +
  geom_line() +
  geom_point() +
  geom_hline(yintercept = 0.1, color = "black", size = 0.1) +
  coord_flip() +
  theme_bw() + theme(legend.key = element_blank())


## Unmatched model (unadjsuted)
glmUnmatched <- glm(formula = (death == "Yes") ~ swang1,
                    family  = binomial(link = "logit"),
                    data    = rhc)
## Matched model
glmMatched <- glm(formula = (death == "Yes") ~ swang1,
                  family  = binomial(link = "logit"),
                  data    = rhcMatched)
## Weighted model
glmWeighted <- svyglm(formula = (death == "Yes") ~ swang1,
                      family  = binomial(link = "logit"),
                      design    = rhcSvy)

## Show results together
resTogether <- list(Unmatched = ShowRegTable(glmUnmatched, printToggle = FALSE),
                    Matched   = ShowRegTable(glmMatched, printToggle = FALSE),
                    Weighted  = ShowRegTable(glmWeighted, printToggle = FALSE))
print(resTogether, quote = FALSE)