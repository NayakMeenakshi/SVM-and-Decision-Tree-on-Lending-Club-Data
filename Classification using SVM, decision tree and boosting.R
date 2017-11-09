install.packages('rpart')
install.packages('rpart.plot')
install.packages('e1071')
install.packages('caret')
install.packages('gbm')
install.packages('ggplot2')
install.packages('corrplot')
install.packages('dplyr')
install.packages('pROC')
library(pROC)
library(ggplot2)
library(corrplot)
library(rpart)
library(rpart.plot)
library(dplyr)
library(gbm)
library(e1071)
library(caret) 
set.seed(1234)
lendingclub <- read.csv("E:/AML - BUAN 6341/LoanStats3a.csv")
loandata <- lendingclub[which(lendingclub$loan_status == 'Fully Paid' | lendingclub$loan_status == 'Charged Off'),]
table(loandata$loan_status)
loan <- loandata
#Final columns to be included
loan <- loan[c('grade','annual_inc', 'term', 'home_ownership', 'emp_length', 'loan_status','open_acc','revol_bal','total_acc','dti','verification_status','pub_rec')]
head(loan)
#Creating dummies for categorical columns
for(level in unique(loan$emp_length)){
  loan[paste("emp_length", level, sep = "_")] <- ifelse(loan$emp_length == level, 1, 0)
}
for(level in unique(loan$term)){
  loan[paste("term", level, sep = "_")] <- ifelse(loan$term == level, 1, 0)
}
for(level in unique(loan$grade)){
  loan[paste("grade", level, sep = "_")] <- ifelse(loan$grade == level, 1, 0)
}
for(level in unique(loan$home_ownership)){
  loan[paste("home_ownership", level, sep = "_")] <- ifelse(loan$home_ownership == level, 1, 0)
}
for(level in unique(loan$verification_status)){
  loan[paste("verification_status", level, sep = "_")] <- ifelse(loan$verification_status == level, 1, 0)
}
loan[c("emp_length","term","grade","home_ownership","verification_status")] <- list(NULL)
safe_loans  <- loan[loan$loan_status == 'Fully Paid',]
risky_loans <- loan[loan$loan_status == 'Charged Off',]
head(safe_loans)
safe_loans$loan_status  <- 1
risky_loans$loan_status <- 0
nrow(safe_loans)
nrow(risky_loans)
safe_loans <- safe_loans[1:10000,]
nrow(safe_loans)
loans <- rbind(safe_loans, risky_loans)
dt = sort(sample(nrow(loans), nrow(loans)*.7))
train<-loans[dt,]
test<-loans[-dt,]
dim(train)
dim(test)
table(train$loan_status)
table(test$loan_status)

#Splitting training data size for learning curves
train_loan = list(loans[sort(sample(nrow(loans), nrow(loans)*.2)),],
                  loans[sort(sample(nrow(loans), nrow(loans)*.3)),],
                  loans[sort(sample(nrow(loans), nrow(loans)*.4)),],
                  loans[sort(sample(nrow(loans), nrow(loans)*.5)),],
                  loans[sort(sample(nrow(loans), nrow(loans)*.6)),],
                  loans[sort(sample(nrow(loans), nrow(loans)*.7)),]
)
summary(train_loan)
x_ax = c(20,30,40,50,60,70)
#Decision Tree
set.seed(1234)
#Splitting Metric used 'information'
tree_train <- rpart(loan_status ~., data = train,method = "class",parms = list(split = 'information'))
summary(tree_train)
rpart.plot(tree_train)
#Training Error
class.pred <- table(predict(tree_train, type="class"), train$loan_status)
Train_error <- 1-sum(diag(class.pred))/sum(class.pred)
class.pred
Train_error
#Test Error
class.pred_test <- table(predict(tree_train, test,type="class"),test$loan_status)
Test_error <- 1-sum(diag(class.pred_test))/sum(class.pred_test)
class.pred_test
Test_error
#Pruning
set.seed(1234)
printcp(tree_train)
ptree <- prune(tree_train,cp= 0.014,"CP")
rpart.plot(ptree)
#testing pruned tree on train
class.pred_prune <- table(predict(ptree, train,type="class"),train$loan_status)
prune_error <- 1-sum(diag(class.pred_prune))/sum(class.pred_prune)
class.pred_prune
prune_error
#Testing pruned tree on test
class.pred_testprune <- table(predict(ptree, test,type="class"),test$loan_status)
prune_errortest <- 1-sum(diag(class.pred_prune))/sum(class.pred_prune)
class.pred_testprune
prune_errortest

#Splitting Metric used 'gini'
tree_train <- rpart(loan_status ~., data = train,method = "class",minsplit = 2, minbucket = 1,parms = list(split = 'gini'))
summary(tree_train)
rpart.plot(tree_train)
#Training Error
class.pred <- table(predict(tree_train, type="class"), train$loan_status)
Train_error <- 1-sum(diag(class.pred))/sum(class.pred)
class.pred
Train_error
#Test Error
res <- table(predict(tree_train, test,type="class"),test$loan_status)
Test_error <- 1-sum(diag(class.pred_test))/sum(class.pred_test)
res
Test_error

#Error curve on trained data using Gini
err_rates = c()
for (i in train_loan){
  tree <- rpart(loan_status ~., data = i,method = "class",minsplit = 2, minbucket = 1,parms = list(split = 'gini'))
  predict_tree <- predict(tree,i)
  tb1 <- table(predict(tree, i,type="class"),i$loan_status)
  err_rates = c(err_rates,1-sum(diag(tb1))/sum(tb1))
}
err_rates
plot(x_ax, err_rates, ylim=c(0.3,0.4), type="l", col="green", ylab="error rate", 
     xlab="training data size(in %)", main="Gini Train Error Curve")
#Pruning
set.seed(1234)
printcp(tree_train)
plotcp(tree_train)
ptree <- prune(tree_train,cp= 0.014,"CP")
rpart.plot(ptree)
#testing pruned tree on train
class.pred_prune <- table(predict(ptree, train,type="class"),train$loan_status)
prune_error <- 1-sum(diag(class.pred_prune))/sum(class.pred_prune)
class.pred_prune
prune_error
err_rates = c()
for (i in train_loan){
  ptree <- prune(tree_train,cp= 0.014,"CP")
  predict_tree <- predict(ptree,i)
  tb1 <- table(predict(tree, i,type="class"),i$loan_status)
  err_rates = c(err_rates,1-sum(diag(tb1))/sum(tb1))
}
err_rates
plot(x_ax, err_rates, ylim=c(0.3,0.4), type="l", col="green", ylab="error rate", 
     xlab="training data size(in %)", main="Pruned Tree - Train Error Curve")
#Testing pruned tree on test
class.pred_testprune <- table(predict(ptree, test,type="class"),test$loan_status)
prune_errortest <- 1-sum(diag(class.pred_prune))/sum(class.pred_prune)
class.pred_testprune
prune_errortest


##SVM - Linear
set.seed(1234)
svm_model_1 <- svm(factor(loan_status) ~., kernel="linear",data=train)
predict_svm <- predict(svm_model_1,test)
predict_tb <- table(Predicted =predict_svm, Actual = test$loan_status )
confusionMatrix(predict_svm,test$loan_status)
error <- 1-sum(diag(predict_tb))/sum(predict_tb)
predict_tb
error
auc_linear <- roc(as.numeric(test$loan_status), as.numeric(predict_svm))
print(auc_linear)
plot(auc_linear, ylim=c(0,1), print.thres=TRUE, main=paste('AUC:',round(auc_linear$auc[[1]],3)),col = 'blue')

##SVM - Radial
#svm_tune <- tune(svm, factor(loan_status) ~.,data=train,kernel="radial", ranges=list(cost=10^(-1:2), gamma=c(.5,1,2)))
#print(svm_tune)
set.seed(1234)
svm_model_1 <- svm(factor(loan_status) ~., kernel="radial",data=train)
summary(svm_model_1)
predict_svm <- predict(svm_model_1,test)
predict_tb <- table(Predicted =predict_svm, Actual = test$loan_status )
confusionMatrix(predict_svm,test$loan_status)
error <- 1-sum(diag(predict_tb))/sum(predict_tb)
predict_tb
error
auc_radial <- roc(as.numeric(test$loan_status), as.numeric(predict_svm))
print(auc_radial)
plot(auc_radial, ylim=c(0,1), print.thres=TRUE, main=paste('AUC:',round(auc_radial$auc[[1]],3)),col = 'blue')
err_rates = c()
for (d in train_loan){
  svm_model <- svm(factor(loan_status) ~.,data=d, kernel="radial")
  predict_svm <- predict(svm_model,d)
  tb1 <- table(Predicted =predict_svm, Actual = d$loan_status )
  err_rates = c(err_rates,1-sum(diag(tb1))/sum(tb1))
}
err_rates
plot(x_ax, err_rates, ylim=c(0.0001,0.0005), type="l", col="green", ylab="error rate", 
     xlab="training data size(in %)", main="Learning Curve for SVM-Radial Kernel")

##SVM - sigmoid 
set.seed(1234)
svm_model_1 <- svm(factor(loan_status) ~., kernel="sigmoid",data=train,scale = FALSE)
predict_svm <- predict(svm_model_1,test)
predict_tb <- table(Predicted =predict_svm, Actual = test$loan_status )
confusionMatrix(predict_svm,test$loan_status)
error <- 1-sum(diag(predict_tb))/sum(predict_tb)
predict_tb
error
auc_sigmoid <- roc(as.numeric(test$loan_status), as.numeric(predict_svm))
print(auc_sigmoid)
plot(auc_sigmoid, ylim=c(0,1), print.thres=TRUE, main=paste('AUC:',round(auc_sigmoid$auc[[1]],3)),col = 'blue')


#Boosting using gbm
set.seed(1234)
boost=gbm(loan_status ~., data=train, distribution="gaussian",n.trees=2000,interaction.depth=1, shrinkage = 0.01)
summary(boost)
n.trees = seq(from=10 ,to=2000, by=100)
predmatrixtrain<-predict(boost,train,n.trees = n.trees,type='response')
predmatrix<-predict(boost,test,n.trees = n.trees,type='response')
dim(predmatrix)
#Calculating The Mean squared Test Error
boost_test_error<-with(test,apply( (predmatrix-test$loan_status)^2,2,mean))
head(boost_test_error)
#Plotting the test error vs number of trees
plot(n.trees , boost_test_error , pch=19,col="blue",xlab="Number of Trees",ylab="Test Error", main = "Perfomance of Boosting on Test Set")
min(boost_test_error)

#Boosting using xgboost
set.seed(1234)
ctr <- trainControl(method = "cv", number = 10)
boost.caret <- train(as.factor(loan_status)~., train,
                     method='xgbTree', 
                     preProc=c('center','scale'),
                     trControl=ctr)
boost.caret
plot(boost.caret)
boost.caret.pred <- predict(boost.caret, test)
confusionMatrix(boost.caret.pred,test$loan_status)
predict_tab <- table(Predicted =boost.caret.pred, Actual = test$loan_status )
xgboost_test_error <- 1-sum(diag(predict_tab))/sum(predict_tab)
xgboost_test_error
