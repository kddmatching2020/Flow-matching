args<-commandArgs(TRUE)

dim=10
rc=as.numeric(args[1])

n=as.numeric(args[2])
sc=as.numeric(args[3])

name=paste("ML_d_",dim,"_rc_",rc,"_n_",n,"_sc_",sc,sep="")
sink(paste(name,".txt",sep = ""))


start = Sys.time()
#install.packages("caTools")
# install.packages("caretEnsemble")
# install.packages("mlbench")
# install.packages("caret")
# install.packages("xgboost")

library(randomForest)
require(caTools)
library("MatchIt")
library("caret")
library("xgboost")
library("glmnet")



generate_data<-function(n,dim=10,rc=0.2){
  sigma <- diag(1,dim,dim)
  mu <- rep(0, dim)
  x <- data.frame(MASS::mvrnorm(n = n, mu = mu, Sigma = sigma))
  
  r = dim*rc
  r_=r/2
  if (r_ ==1){
    linear = as.vector(as.matrix(x)[,1]^2 * beta_logit[1])
  } else {
    linear = as.vector(as.matrix(x)[,1:r_]^2 %*% beta_logit[1:r_])
  }
  
  for (i in 1:(r-1)){
    for (j in (i+1):r){
      linear = linear+as.matrix(x)[,i]*as.matrix(x)[,j]*beta_logit[1]
    }
  }

  
  prob_w = as.vector(1/(1+exp(linear)))
  
  w = rbinom(n, 1, prob_w)
  y = as.vector(as.matrix(x)^2 %*% beta) + w + rnorm(n, 0, 1)

  return(cbind(x,w,prob_w,y))
}



main<-function(i){
  
  data_all = generate_data(n,dim,rc=rc)
  data_all$w= as.factor(data_all$w)
  ATT_base[i]<<-mean(data_all[which(data_all$w==1),]$y)-mean(data_all[which(data_all$w==0),]$y)
  
  glm1 <- glm(w~X1+X2+X3+X4+X5+X6+X7+X8+X9+X10, family = "binomial", data = data_all)
  estimate = glm1$fitted
  bias_logit[i]<<-mean((data_all$prob_w - estimate)^2)
  
  ###### rf
  rf <- randomForest(w ~ X1+X2+X3+X4+X5+X6+X7+X8+X9+X10,data=data_all)
  estimate = predict(rf, newdata=data_all[,1:10],type="prob")[,2]
  bias_rf[i]<<-mean((data_all$prob_w - estimate)^2)
  data_all$estimate=estimate
  
  m.out <- matchit(w~estimate, data=data_all,
                   method = "nearest", distance = "mahalanobis", replace=T)
  m.data <- match.data(m.out)
  m.data2 <- match.data(m.out, group = "treat")
  m.data3 <- match.data(m.out, group = "control")
  ATT_rf[i]<<-mean(m.data2$y)-mean(m.data3$y *m.data3$weights)


  ###### treebag
  fit.treebag <- train(w~X1+X2+X3+X4+X5+X6+X7+X8+X9+X10, data=data_all, method="treebag")
  estimate <- predict(fit.treebag, newdata = data_all[,1:10], type = "prob")[,2]
  bias_treebag[i]<<-mean((data_all$prob_w - estimate)^2)
  data_all$estimate=estimate
  
  m.out <- matchit(w~estimate, data=data_all,
                   method = "nearest", distance = "mahalanobis", replace=T)
  m.data <- match.data(m.out)
  m.data2 <- match.data(m.out, group = "treat")
  m.data3 <- match.data(m.out, group = "control")
  ATT_treebag[i]<<-mean(m.data2$y)-mean(m.data3$y *m.data3$weights)
  
  ###### xgboost
  fit.xgboost <- xgboost(data = as.matrix(data_all[,1:10]), 
                         label = as.numeric(as.character(data_all$w)), 
                         max.depth = 6, eta = 0.2, nthread = 2, 
                         verbose = 0,nrounds = 1000, objective = "binary:logistic")
  estimate <- predict(fit.xgboost, newdata = as.matrix(data_all[,1:10]))
  bias_xgboost[i]<<-mean((data_all$prob_w - estimate)^2)
  data_all$estimate=estimate
  
  m.out <- matchit(w~estimate, data=data_all,
                   method = "nearest", distance = "mahalanobis", replace=T)
  m.data <- match.data(m.out)
  m.data2 <- match.data(m.out, group = "treat")
  m.data3 <- match.data(m.out, group = "control")
  ATT_xgboost[i]<<-mean(m.data2$y)-mean(m.data3$y *m.data3$weights)
  
  ### lasso
  cv.lasso <- cv.glmnet(as.matrix(data_all[,1:10]), data_all$w, alpha = 1, 
                        nfolds = 10, family = "binomial",type.measure = "class")
  estimate <- predict(cv.lasso,newx = as.matrix(data_all[,1:10]),
                      s= "lambda.min",type="response")
  bias_lasso[i]<<-mean((data_all$prob_w - estimate)^2)
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



beta_logit <- rep(sc,dim)
#beta <- runif(dim, -2, 2)
beta = c(-1.846703052520752,-1.5344433784484863,0.25243520736694336,1.4843950271606445,0.5788190364837646,0.5631067752838135,-1.6139118671417236,-1.1415417194366455,0.6788022518157959,-1.9940435886383057)


ATT_base<-c()
bias_logit<-c()

bias_rf<-c()
bias_treebag<-c()
bias_xgboost<-c()
bias_lasso<-c()


ATT_rf<-c()
ATT_treebag<-c()
ATT_xgboost<-c()
ATT_lasso<-c()


for (i in 1:100){
  main(i)
}


ATT_base_bias = mean(ATT_base) - 1 
ATT_base_sd = sd(ATT_base) 
ATT_base_mae = mean(abs(ATT_base-1))
ATT_base_rmse = sqrt(mean((ATT_base-1)^2))

ATT_rf_bias = mean(ATT_rf) - 1 
ATT_rf_sd = sd(ATT_rf) 
ATT_rf_mae = mean(abs(ATT_rf-1))
ATT_rf_rmse = sqrt(mean((ATT_rf-1)^2))

ATT_treebag_bias = mean(ATT_treebag) - 1 
ATT_treebag_sd = sd(ATT_treebag) 
ATT_treebag_mae = mean(abs(ATT_treebag-1))
ATT_treebag_rmse = sqrt(mean((ATT_treebag-1)^2))

ATT_xgboost_bias = mean(ATT_xgboost) - 1 
ATT_xgboost_sd = sd(ATT_xgboost) 
ATT_xgboost_mae = mean(abs(ATT_xgboost-1))
ATT_xgboost_rmse = sqrt(mean((ATT_xgboost-1)^2))


ATT_lasso_bias = mean(ATT_lasso) - 1 
ATT_lasso_sd = sd(ATT_lasso) 
ATT_lasso_mae = mean(abs(ATT_lasso-1))
ATT_lasso_rmse = sqrt(mean((ATT_lasso-1)^2))


mse_logit = mean(bias_logit)

mse_rf = mean(bias_rf)
mse_treebag = mean(bias_treebag)
mse_xgboost = mean(bias_xgboost)
mse_lasso = mean(bias_lasso)


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


cat(paste("\n\n\nThe mse_logit is",mse_logit))
cat(paste("\nThe mse_rf is",mse_rf))
cat(paste("\nThe mse_treebag is",mse_treebag))
cat(paste("\nThe mse_xgboost is",mse_xgboost))
cat(paste("\nThe mse_lasso is",mse_lasso))


end = Sys.time()
cat(paste("\n\nThe time_whole is",end-start))




