sink("ML_real.txt")

library(randomForest)
require(caTools)
library("MatchIt")
library("caret")
library("xgboost")
library("glmnet")

start = Sys.time()

x_new<-read.csv("data_x.csv")
y_new<-read.csv("twin_pairs_Y.csv")
t_new<-read.csv("twin_pairs_T.csv")



main<-function(i){

  data_all = read.csv(paste("test/data_all_",i,".csv",sep = ""))
  data_all$w= as.factor(data_all$w)
  ATT_true = data_all$ATT[1]
  ATT_True[i]<<-ATT_true
  
  ATT_base[i]<<-mean(data_all[which(data_all$w==1),]$y_obs)-mean(data_all[which(data_all$w==0),]$y_obs)
  
  ###### rf
  rf <- randomForest(w ~ brstate+stoccfipb+mplbir+birmon+gestat10+dlivord_min+dtotord_min+
                       mager8,data=data_all)
  estimate = predict(rf, newdata=data_all[,1:8],type="prob")[,2]
  data_all$estimate=estimate
  
  m.out <- matchit(w~estimate, data=data_all,
                   method = "nearest", distance = "mahalanobis", replace=T)
  m.data <- match.data(m.out)
  m.data2 <- match.data(m.out, group = "treat")
  m.data3 <- match.data(m.out, group = "control")
  ATT_rf[i]<<-mean(m.data2$y)-mean(m.data3$y *m.data3$weights)
  
  ###### treebag
  fit.treebag <- train(w~brstate+stoccfipb+mplbir+birmon+gestat10+dlivord_min+dtotord_min+
                         mager8, data=data_all, method="treebag")
  estimate <- predict(fit.treebag, newdata = data_all[,1:8], type = "prob")[,2]
  data_all$estimate=estimate
  
  m.out <- matchit(w~estimate, data=data_all,
                   method = "nearest", distance = "mahalanobis", replace=T)
  m.data <- match.data(m.out)
  m.data2 <- match.data(m.out, group = "treat")
  m.data3 <- match.data(m.out, group = "control")
  ATT_treebag[i]<<-mean(m.data2$y)-mean(m.data3$y *m.data3$weights)
  
  ###### xgboost
  fit.xgboost <- xgboost(data = as.matrix(data_all[,1:8]), 
                         label = as.numeric(as.character(data_all$w)), 
                         max.depth = 6, eta = 0.2, nthread = 2, 
                         verbose = 0,nrounds = 1000, objective = "binary:logistic")
  estimate <- predict(fit.xgboost, newdata = as.matrix(data_all[,1:8]))
  data_all$estimate=estimate
  
  m.out <- matchit(w~estimate, data=data_all,
                   method = "nearest", distance = "mahalanobis", replace=T)
  m.data <- match.data(m.out)
  m.data2 <- match.data(m.out, group = "treat")
  m.data3 <- match.data(m.out, group = "control")
  ATT_xgboost[i]<<-mean(m.data2$y)-mean(m.data3$y *m.data3$weights)
  
  ### lasso
  cv.lasso <- cv.glmnet(as.matrix(data_all[,1:8]), data_all$w, alpha = 1, 
                        nfolds = 10, family = "binomial",type.measure = "class")
  estimate <- predict(cv.lasso,newx = as.matrix(data_all[,1:8]),
                      s= "lambda.min",type="response")
  data_all$estimate=estimate
  
  if (length(table(estimate))==1){
    ATT_lasso[i]<<-mean(data_all[which(data_all$w==1),]$y)-mean(data_all[which(data_all$w==0),]$y)
  } else{
    m.out <- matchit(w~estimate, data=data_all,
                     method = "nearest", distance = "mahalanobis", replace=T)
    m.data <- match.data(m.out)
    m.data2 <- match.data(m.out, group = "treat")
    m.data3 <- match.data(m.out, group = "control")
    ATT_lasso[i]<<-mean(m.data2$y)-mean(m.data3$y *m.data3$weights)
  }
  
}


ATT_True<-c()


ATT_base<-c()
ATT_rf<-c()
ATT_treebag<-c()
ATT_xgboost<-c()
ATT_lasso<-c()

for (i in 1:100){
  main(i)
}

output=data.frame(ATT_True, ATT_base,ATT_rf,ATT_treebag,ATT_xgboost,ATT_lasso)
write.csv(output,"result.csv")



ATT_base_bias = mean(ATT_base-ATT_True) 
ATT_base_sd = sd(ATT_base) 
ATT_base_mae = mean(abs(ATT_base-ATT_True))
ATT_base_rmse = sqrt(mean((ATT_base-ATT_True)^2))

ATT_rf_bias = mean(ATT_rf-ATT_True)  
ATT_rf_sd = sd(ATT_rf) 
ATT_rf_mae = mean(abs(ATT_rf-ATT_True))
ATT_rf_rmse = sqrt(mean((ATT_rf-ATT_True)^2))


ATT_treebag_bias = mean(ATT_treebag-ATT_True)  
ATT_treebag_sd = sd(ATT_treebag) 
ATT_treebag_mae = mean(abs(ATT_treebag-ATT_True))
ATT_treebag_rmse = sqrt(mean((ATT_treebag-ATT_True)^2))

ATT_xgboost_bias = mean(ATT_xgboost-ATT_True)  
ATT_xgboost_sd = sd(ATT_xgboost) 
ATT_xgboost_mae = mean(abs(ATT_xgboost-ATT_True))
ATT_xgboost_rmse = sqrt(mean((ATT_xgboost-ATT_True)^2))


ATT_lasso_bias = mean(ATT_lasso-ATT_True)  
ATT_lasso_sd = sd(ATT_lasso) 
ATT_lasso_mae = mean(abs(ATT_lasso-ATT_True))
ATT_lasso_rmse = sqrt(mean((ATT_lasso-ATT_True)^2))



## print the results
cat(paste("The ATT_base_bias is",ATT_base_bias))
cat(paste("\nThe ATT_base_sd is",ATT_base_sd))
cat(paste("\nThe ATT_base_mae is",ATT_base_mae))
cat(paste("\nThe ATT_base_rmse is",ATT_base_rmse))

cat(paste("\n\nThe ATT_rf_bias is",ATT_rf_bias))
cat(paste("\nThe ATT_rf_sd is",ATT_rf_sd))
cat(paste("\nThe ATT_rf_mae is",ATT_rf_mae))
cat(paste("\nThe ATT_rf_rmse is",ATT_rf_rmse))

cat(paste("\n\nThe ATT_treebag_bias is",ATT_treebag_bias))
cat(paste("\nThe ATT_treebag_sd is",ATT_treebag_sd))
cat(paste("\nThe ATT_treebag_mae is",ATT_treebag_mae))
cat(paste("\nThe ATT_treebag_rmse is",ATT_treebag_rmse))

cat(paste("\n\nThe ATT_xgboost_bias is",ATT_xgboost_bias))
cat(paste("\nThe ATT_xgboost_sd is",ATT_xgboost_sd))
cat(paste("\nThe ATT_xgboost_mae is",ATT_xgboost_mae))
cat(paste("\nThe ATT_xgboost_rmse is",ATT_xgboost_rmse))

cat(paste("\n\nThe ATT_lasso_bias is",ATT_lasso_bias))
cat(paste("\nThe ATT_lasso_sd is",ATT_lasso_sd))
cat(paste("\nThe ATT_lasso_mae is",ATT_lasso_mae))
cat(paste("\nThe ATT_lasso_rmse is",ATT_lasso_rmse))


end = Sys.time()
cat(paste("\n\nThe time_whole is",end-start))

# sink()                          # Close connection to file