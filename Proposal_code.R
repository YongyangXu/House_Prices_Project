#Lasso regressions performs best with a cross validation RMSE-score of 0.1311 Given the fact that there is a lot of multicolinearity among the variables, this was expected. Lasso does not select a substantial number of the available variables in its model, as it is supposed to do.
#The XGBoost model also performs very well with a cross validation RMSE of 0.1319

library(knitr)
library(ggplot2)
library(plyr)
library(dplyr)
library(corrplot)
library(caret)
library(gridExtra)
library(scales)
library(Rmisc)
library(ggrepel)
library(randomForest)
library(psych)
library(xgboost)
###
#Loading and Exploring Data
###
setwd('c:/Users/xyy/Desktop/5205_R/house_prices')
train <- read.csv('train.csv', stringsAsFactors = F)
test <- read.csv('test.csv', stringsAsFactors = F)

dim(train)
str(train[,c(1:10, 81)])

#get rid of ID, test$SalePrice as NA for future decomposing in modeling
test_labels <- test$Id
test$Id <- NULL
train$Id <- NULL
test$SalePrice <- NA
all <- rbind(train, test)

###
#Exploring some of the most important variables
###

#The response variable; SalePrice
library(ggplot2)
library(dplyr)

ggplot(data=all[!is.na(all$SalePrice),], aes(x=SalePrice)) +
  geom_histogram(fill="blue", binwidth = 10000) +
  scale_x_continuous(breaks= seq(0, 800000, by=100000))
#The sale prices are right skewed. 
#This was expected as few people can afford very expensive houses.
#We can take measures before modeling.



#Concentration Filtering
concentration_percentage <- function(df) {
  concentration <- sapply(df, function(x) {
    freq <- table(x) / length(x)
    max(freq)
  })
  
  concentration <- concentration * 100 
  return(concentration)
}

all_concentration <- concentration_percentage(all)
all_concentration_df <- as.data.frame(all_concentration, stringsAsFactors = FALSE)
all_concentration_df <- all_concentration_df[order(-all_concentration_df$all_concentration), , drop = FALSE]
print(all_concentration_df)
#remove columns with concentration > 90%
columns_to_remove <- names(all_concentration[all_concentration > 90])
all <- all[, !(names(all) %in% columns_to_remove)]


### Numeric predictors
numericVars <- which(sapply(all, is.numeric)) #index vector numeric variables
numericVarNames <- names(numericVars) #saving names vector for use later on
cat('There are', length(numericVars), 'numeric variables')

all_numVar <- all[, numericVars]
cor_numVar <- cor(all_numVar, use="pairwise.complete.obs") #correlations of all numeric variables

#sort on decreasing correlations with SalePrice
cor_sorted <- as.matrix(sort(cor_numVar[,'SalePrice'], decreasing = TRUE))
#select only high corelations
library(corrplot)
CorHigh <- names(which(apply(cor_sorted, 1, function(x) abs(x)>0.5)))
cor_numVar <- cor_numVar[CorHigh, CorHigh]
corrplot.mixed(cor_numVar, tl.col="black", tl.pos = "lt")
#Markdown:It is clear that the multicollinearity is an issue. For example: the correlation between GarageCars and GarageArea is very high (0.89), and both have similar (high) correlations with SalePrice. 
#In modoling part, ran an RF model and then chose the retaining variable based on feature importance.


###
#Missing data and label encoding
###
NAcol <- which(colSums(is.na(all)) > 0)
sort(colSums(sapply(all[NAcol], is.na)), decreasing = TRUE)
#Markdown:The 1459 NAs in SalePrice match the size of the test set perfectly. Because we set them 

missing_counts <- colSums(sapply(all[NAcol], is.na))
total_rows <- nrow(all)
missing_percentage <- (missing_counts / total_rows) * 100
sort(missing_percentage, decreasing = TRUE)
#more than 93% of the observations in variables “PoolQC” “MiscFeature” and “Alley” were NA
#delect the three variables
all <- all[, !(names(all) %in% c("PoolQC", "MiscFeature", "Alley"))]


###Filling the NAs

#Fence
all$Fence[is.na(all$Fence)] <- 'None'
#all[!is.na(all$SalePrice),] %>% group_by(Fence) %>% summarise(median = median(SalePrice), counts=n())
all$Fence <- as.factor(all$Fence)
#FireplaceQu
library(plyr)
Qualities <- c('None' = 0, 'Po' = 1, 'Fa' = 2, 'TA' = 3, 'Gd' = 4, 'Ex' = 5)
all$FireplaceQu[is.na(all$FireplaceQu)] <- 'None'
all$FireplaceQu<-as.integer(revalue(all$FireplaceQu, Qualities))
table(all$Fireplaces)
#LotFrontage
for (i in 1:nrow(all)){
  if(is.na(all$LotFrontage[i])){
    all$LotFrontage[i] <- as.integer(median(all$LotFrontage[all$Neighborhood==all$Neighborhood[i]], na.rm=TRUE)) 
  }
}
#LotShape:Reg is the best
all$LotShape<-as.integer(revalue(all$LotShape, c('IR3'=0, 'IR2'=1, 'IR1'=2, 'Reg'=3)))
table(all$LotShape)
# 7 kinds of Garage variables
#GarageYrBlt
all$GarageYrBlt[is.na(all$GarageYrBlt)] <- all$YearBuilt[is.na(all$GarageYrBlt)]
#check if all 157 NAs are the same observations among the variables with 157/159 NAs
length(which(is.na(all$GarageType) & is.na(all$GarageFinish) & is.na(all$GarageCond) & is.na(all$GarageQual)))
#kable(all[!is.na(all$GarageType) & is.na(all$GarageFinish), c('GarageCars', 'GarageArea', 'GarageType', 'GarageCond', 'GarageQual', 'GarageFinish')])
all$GarageCond[2127] <- names(sort(-table(all$GarageCond)))[1] #Imputing modes.
all$GarageQual[2127] <- names(sort(-table(all$GarageQual)))[1]
all$GarageFinish[2127] <- names(sort(-table(all$GarageFinish)))[1]
#kable(all[2127, c('GarageYrBlt', 'GarageCars', 'GarageArea', 'GarageType', 'GarageCond', 'GarageQual', 'GarageFinish')])
#GarageCars and GarageArea
all$GarageCars[2577] <- 0
all$GarageArea[2577] <- 0
all$GarageType[2577] <- NA
length(which(is.na(all$GarageType) & is.na(all$GarageFinish) & is.na(all$GarageCond) & is.na(all$GarageQual)))
#GarageType
all$GarageType[is.na(all$GarageType)] <- 'No Garage'
all$GarageType <- as.factor(all$GarageType)
#table(all$GarageType)
#GarageFinish
all$GarageFinish[is.na(all$GarageFinish)] <- 'None'
Finish <- c('None'=0, 'Unf'=1, 'RFn'=2, 'Fin'=3)
all$GarageFinish<-as.integer(revalue(all$GarageFinish, Finish))
#GarageQual
all$GarageQual[is.na(all$GarageQual)] <- 'None'
all$GarageQual<-as.integer(revalue(all$GarageQual, Qualities))
#GarageCond
# all$GarageCond[is.na(all$GarageCond)] <- 'None'
# all$GarageCond<-as.integer(revalue(all$GarageCond, Qualities))
#11 variables relate to basement
#
length(which(is.na(all$BsmtQual) & is.na(all$BsmtCond) & is.na(all$BsmtExposure) & is.na(all$BsmtFinType1) & is.na(all$BsmtFinType2)))
all[!is.na(all$BsmtFinType1) & (is.na(all$BsmtCond)|is.na(all$BsmtQual)|is.na(all$BsmtExposure)|is.na(all$BsmtFinType2)), c('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2')]
all$BsmtFinType2[333] <- names(sort(-table(all$BsmtFinType2)))[1]
all$BsmtExposure[c(949, 1488, 2349)] <- names(sort(-table(all$BsmtExposure)))[1]
all$BsmtCond[c(2041, 2186, 2525)] <- names(sort(-table(all$BsmtCond)))[1]
all$BsmtQual[c(2218, 2219)] <- names(sort(-table(all$BsmtQual)))[1]
#BsmtQual
all$BsmtQual[is.na(all$BsmtQual)] <- 'None'
all$BsmtQual<-as.integer(revalue(all$BsmtQual, Qualities))
#BsmtCond
all$BsmtCond[is.na(all$BsmtCond)] <- 'None'
all$BsmtCond<-as.integer(revalue(all$BsmtCond, Qualities))
#BsmtExposure
all$BsmtExposure[is.na(all$BsmtExposure)] <- 'None'
Exposure <- c('None'=0, 'No'=1, 'Mn'=2, 'Av'=3, 'Gd'=4)
all$BsmtExposure<-as.integer(revalue(all$BsmtExposure, Exposure))
#BsmtFinType1
all$BsmtFinType1[is.na(all$BsmtFinType1)] <- 'None'
FinType <- c('None'=0, 'Unf'=1, 'LwQ'=2, 'Rec'=3, 'BLQ'=4, 'ALQ'=5, 'GLQ'=6)
all$BsmtFinType1<-as.integer(revalue(all$BsmtFinType1, FinType))
#BsmtFinType2
all$BsmtFinType2[is.na(all$BsmtFinType2)] <- 'None'
FinType <- c('None'=0, 'Unf'=1, 'LwQ'=2, 'Rec'=3, 'BLQ'=4, 'ALQ'=5, 'GLQ'=6)
all$BsmtFinType2<-as.integer(revalue(all$BsmtFinType2, FinType))
#其余全部填充0
all$BsmtFullBath[is.na(all$BsmtFullBath)] <-0
# all$BsmtHalfBath[is.na(all$BsmtHalfBath)] <-0
all$BsmtFinSF1[is.na(all$BsmtFinSF1)] <-0
all$BsmtFinSF2[is.na(all$BsmtFinSF2)] <-0
all$BsmtUnfSF[is.na(all$BsmtUnfSF)] <-0
all$TotalBsmtSF[is.na(all$TotalBsmtSF)] <-0
#Masonry veneer type, and masonry veneer area
length(which(is.na(all$MasVnrType) & is.na(all$MasVnrArea)))#check if the 23 houses with veneer area NA are also NA in the veneer type
all[is.na(all$MasVnrType) & !is.na(all$MasVnrArea), c('MasVnrType', 'MasVnrArea')]
all$MasVnrType[2611] <- names(sort(-table(all$MasVnrType)))[2] #taking the 2nd value as the 1st is 'none'
all$MasVnrType[is.na(all$MasVnrType)] <- 'None'
# all[!is.na(all$SalePrice),] %>% group_by(MasVnrType) %>% summarise(median = median(SalePrice), counts=n()) %>% arrange(median)
Masonry <- c('None'=0, 'BrkCmn'=0, 'BrkFace'=1, 'Stone'=2)#assume that simple stones and for instance wooden houses are just cheaper.
all$MasVnrType<-as.integer(revalue(all$MasVnrType, Masonry))
#MasVnrArea
all$MasVnrArea[is.na(all$MasVnrArea)] <-0
all$MSZoning[is.na(all$MSZoning)] <- names(sort(-table(all$MSZoning)))[1]#获取众数
all$MSZoning <- as.factor(all$MSZoning)
#KitchenQual
# all$KitchenQual[is.na(all$KitchenQual)] <- 'TA' #replace with most common value
# all$KitchenQual<-as.integer(revalue(all$KitchenQual, Qualities))
#Functional
# all$Functional[is.na(all$Functional)] <- names(sort(-table(all$Functional)))[1]
# all$Functional <- as.integer(revalue(all$Functional, c('Sal'=0, 'Sev'=1, 'Maj2'=2, 'Maj1'=3, 'Mod'=4, 'Min2'=5, 'Min1'=6, 'Typ'=7)))
#Exterior1st
all$Exterior1st[is.na(all$Exterior1st)] <- names(sort(-table(all$Exterior1st)))[1]
all$Exterior1st <- as.factor(all$Exterior1st)
#Electrical
# all$Electrical[is.na(all$Electrical)] <- names(sort(-table(all$Electrical)))[1]
# all$Electrical <- as.factor(all$Electrical)
#SaleType
all$SaleType[is.na(all$SaleType)] <- names(sort(-table(all$SaleType)))[1]
all$SaleType <- as.factor(all$SaleType)

# export the clean dataset
# write.csv(all, "clean_dataset.csv", row.names = FALSE)



# Label encoding/factorizing the remaining character variables

Charcol <- names(all[,sapply(all, is.character)])
Charcol

all$Foundation <- as.factor(all$Foundation)
table(all$Foundation)

all$Heating <- as.factor(all$Heating)
table(all$Heating)

all$RoofStyle <- as.factor(all$RoofStyle)
table(all$RoofStyle)

all$LandContour <- as.factor(all$LandContour)
table(all$LandContour)

all$BldgType <- as.factor(all$BldgType)
table(all$BldgType)

all$Neighborhood <- as.factor(all$Neighborhood)
table(all$Neighborhood)


#Changing some numeric variables into factors


all$MoSold <- as.factor(all$MoSold)


all$MSSubClass <- as.factor(all$MSSubClass)
#revalue for better readability
all$MSSubClass<-revalue(all$MSSubClass, c('20'='1 story 1946+', '30'='1 story 1945-', '40'='1 story unf attic', '45'='1,5 story unf', '50'='1,5 story fin', '60'='2 story 1946+', '70'='2 story 1945-', '75'='2,5 story all ages', '80'='split/multi level', '85'='split foyer', '90'='duplex all style/age', '120'='1 story PUD 1946+', '150'='1,5 story PUD all', '160'='2 story PUD 1946+', '180'='PUD multilevel', '190'='2 family conversion'))
str(all$MSSubClass)


numericVars <- which(sapply(all, is.numeric)) #index vector numeric variables
factorVars <- which(sapply(all, is.factor)) #index vector factor variables
cat('There are', length(numericVars), 'numeric variables, and', length(factorVars), 'categoric variables')

### Correlations again

all_numVar <- all[, numericVars]
cor_numVar <- cor(all_numVar, use="pairwise.complete.obs") #correlations of all numeric variables
#sort on decreasing correlations with SalePrice
cor_sorted <- as.matrix(sort(cor_numVar[,'SalePrice'], decreasing = TRUE))
#select only high corelations
CorHigh <- names(which(apply(cor_sorted, 1, function(x) abs(x)>0.5)))
cor_numVar <- cor_numVar[CorHigh, CorHigh]
corrplot.mixed(cor_numVar, tl.col="black", tl.pos = "lt", tl.cex = 0.7,cl.cex = .7, number.cex=.7)


###Finding variable importance with a quick Random Forest
library(randomForest)
set.seed(2018)
which(names(all) == "SalePrice")
quick_RF <- randomForest(x=all[1:1460,-59], y=all$SalePrice[1:1460], ntree=100,importance=TRUE)
imp_RF <- importance(quick_RF)
imp_DF <- data.frame(Variables = row.names(imp_RF), MSE = imp_RF[,1])
imp_DF <- imp_DF[order(imp_DF$MSE, decreasing = TRUE),]

ggplot(imp_DF[1:20,], aes(x=reorder(Variables, MSE), y=MSE, fill=MSE)) + geom_bar(stat = 'identity') + labs(x = 'Variables', y= '% increase MSE if variable is randomly permuted') + coord_flip() + theme(legend.position="none")

###
#Feature engineering
###
#all$TotBathrooms <- all$FullBath + (all$HalfBath*0.5) + all$BsmtFullBath + (all$BsmtHalfBath*0.5)

# Binning Neighborhood
all$NeighRich[all$Neighborhood %in% c('StoneBr', 'NridgHt', 'NoRidge')] <- 2
all$NeighRich[!all$Neighborhood %in% c('MeadowV', 'IDOTRR', 'BrDale', 'StoneBr', 'NridgHt', 'NoRidge')] <- 1
all$NeighRich[all$Neighborhood %in% c('MeadowV', 'IDOTRR', 'BrDale')] <- 0



### Dropping highly correlated variables
dropVars <- c('YearRemodAdd', 'GarageYrBlt', 'GarageArea', 'GarageCond', 'TotalBsmtSF', 'TotalRmsAbvGrd', 'BsmtFinSF1')

all <- all[,!(names(all) %in% dropVars)]
all <- all[-c(1556, 2152),] #delete because contains NAs
numericVarNames <- numericVarNames[!(numericVarNames %in% c('MSSubClass', 'MoSold', 'YrSold', 'SalePrice', 'OverallQual', 'OverallCond'))] #numericVarNames was created before having done anything

DFnumeric <- all[, names(all) %in% numericVarNames]

DFfactors <- all[, !(names(all) %in% numericVarNames)]
DFfactors <- DFfactors[, names(DFfactors) != 'SalePrice']

#Skewness and normalizing of the numeric predictors

for(i in 1:ncol(DFnumeric)){
  if (abs(skew(DFnumeric[,i]))>0.8){
    DFnumeric[,i] <- log(DFnumeric[,i] +1)
  }
}

#Normal
PreNum <- preProcess(DFnumeric, method=c("center", "scale"))
print(PreNum)
DFnorm <- predict(PreNum, DFnumeric)
dim(DFnorm)

DFdummies <- as.data.frame(model.matrix(~.-1, DFfactors))
dim(DFdummies)

combined <- cbind(DFnorm, DFdummies) #combining all (now numeric) predictors into one dataframe 

#saleprice plot
library(e1071)
qqnorm(all$SalePrice)
qqline(all$SalePrice)
skew(all$SalePrice)
#Combine Q-Q plot and skewness valuw, the skew is really high. Take a log
all$SalePrice <- log(all$SalePrice) #default is the natural logarithm, "+1" is not necessary as there are no 0's
skew(all$SalePrice)
qqnorm(all$SalePrice)
qqline(all$SalePrice)


train1 <- combined[!is.na(all$SalePrice),]
test1 <- combined[is.na(all$SalePrice),]

###
#Modelling
###

###Lasso regression model

set.seed(27042018)
my_control <-trainControl(method="cv", number=5)
lassoGrid <- expand.grid(alpha = 1, lambda = seq(0.001,0.1,by = 0.0005))

lasso_mod <- train(x=train1, y=all$SalePrice[!is.na(all$SalePrice)], method='glmnet', trControl= my_control, tuneGrid=lassoGrid) 
lasso_mod$bestTune

min(lasso_mod$results$RMSE)

lassoVarImp <- varImp(lasso_mod,scale=F)
lassoImportance <- lassoVarImp$importance

varsSelected <- length(which(lassoImportance$Overall!=0))
varsNotSelected <- length(which(lassoImportance$Overall==0))

cat('Lasso uses', varsSelected, 'variables in its model, and did not select', varsNotSelected, 'variables.')


###xgboost
xgb_grid = expand.grid(
  nrounds = 1000,
  eta = c(0.1, 0.05, 0.01),
  max_depth = c(2, 3, 4, 5, 6),
  gamma = 0,
  colsample_bytree=1,
  min_child_weight=c(1, 2, 3, 4 ,5),
  subsample=1
)
#Takes about 10-20 mins to run
xgb_caret <- train(x=train1, y=all$SalePrice[!is.na(all$SalePrice)], method='xgbTree', trControl= my_control, tuneGrid=xgb_grid) 
xgb_caret$bestTune

label_train <- all$SalePrice[!is.na(all$SalePrice)]
# put our testing & training data into two seperates Dmatrixs objects
dtrain <- xgb.DMatrix(data = as.matrix(train1), label= label_train)
dtest <- xgb.DMatrix(data = as.matrix(test1))
#In addition, I am taking over the best tuned values from the caret cross validation.
default_param<-list(
  objective = "reg:linear",
  booster = "gbtree",
  eta=0.05, #default = 0.3
  gamma=0,
  max_depth=3, #default=6
  min_child_weight=4, #default=1
  subsample=1,
  colsample_bytree=1
)
#The next step is to do cross validation to determine the best number of rounds (for the given set of parameters).
xgbcv <- xgb.cv( params = default_param, data = dtrain, nrounds = 500, nfold = 5, showsd = T, stratified = T, print_every_n = 40, early_stopping_rounds = 10, maximize = F)

#train the model using the best iteration found by cross validation
xgb_mod <- xgb.train(data = dtrain, params=default_param, nrounds = 409)

XGBpred <- predict(xgb_mod, dtest)
predictions_XGB <- exp(XGBpred) #need to reverse the log to the real values
head(predictions_XGB)

#view variable importance plot
library(Ckmeans.1d.dp) #required for ggplot clustering
mat <- xgb.importance (feature_names = colnames(train1),model = xgb_mod)
xgb.ggplot.importance(importance_matrix = mat[1:20], rel_to_first = TRUE)


