# install.packages("Matching")
#install.packages("MatchIt")
#install.packages("cem")

library("Matching")
library("MatchIt")


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
  length(which(data_all$w==1))
  glm1 <- glm(w~X1+X2+X3+X4+X5+X6+X7+X8+X9+X10, family = "binomial", data = data_all)
  estimate = glm1$fitted
  
  bias_pscore[i]<<-mean((data_all$prob_w - estimate)^2)
  
  ATT_base[i]<<-mean(data_all[which(data_all$w==1),]$y)-mean(data_all[which(data_all$w==0),]$y)

  # Using MatchIt package
  
  # coarsened exact matching
  cut=list(X1=3,X2=3,X3=3,X4=3,X5=3,X6=3,X7=3,X8=3,X9=3,X10=3)
  m.out <- matchit(w ~ X1+X2+X3+X4+X5+X6+X7+X8+X9+X10, data=data_all, method = "cem",cutpoints=cut)
  m.data <- match.data(m.out)
  m.data2 <- match.data(m.out, group = "treat")
  m.data3 <- match.data(m.out, group = "control")
  ATT_exact[i]<<-mean(m.data2$y)-mean(m.data3$y *m.data3$weights)
  
  # # OLS regression
  lm1 <- lm(y~X1+X2+X3+X4+X5+X6+X7+X8+X9+X10+w, data = data_all)
  ATT_ols[i]<<-lm1$coefficients[length(lm1$coefficients)]
  
  # PSM
  m.out <- matchit(w~X1+X2+X3+X4+X5+X6+X7+X8+X9+X10, data=data_all,
  method = "nearest", distance = "logit", replace=T)
  m.data <- match.data(m.out)
  m.data2 <- match.data(m.out, group = "treat")
  m.data3 <- match.data(m.out, group = "control")
  ATT_pscore[i]<<-mean(m.data2$y)-mean(m.data3$y *m.data3$weights)
}


dim=10
rc=0.2


n=5000
sc=0.2

beta_logit <- rep(sc,dim)
#beta <- runif(dim, -2, 2)
beta = c(-1.846703052520752,-1.5344433784484863,0.25243520736694336,1.4843950271606445,0.5788190364837646,0.5631067752838135,-1.6139118671417236,-1.1415417194366455,0.6788022518157959,-1.9940435886383057)


ATT_base<-c()
ATT_exact<-c()
ATT_ols<-c()
ATT_pscore<-c()

bias_pscore<-c()


for (i in 1:10){
  main(i)
}

ATT_base_bias = mean(ATT_base) - 1 
ATT_base_sd = sd(ATT_base) 
ATT_base_mae = mean(abs(ATT_base-1))
ATT_base_rmse = sqrt(mean((ATT_base-1)^2))

ATT_exact_bias = mean(ATT_exact) - 1 
ATT_exact_sd = sd(ATT_exact) 
ATT_exact_mae = mean(abs(ATT_exact-1))
ATT_exact_rmse = sqrt(mean((ATT_exact-1)^2))

ATT_ols_bias = mean(ATT_ols) - 1 
ATT_ols_sd = sd(ATT_ols) 
ATT_ols_mae = mean(abs(ATT_ols-1))
ATT_ols_rmse = sqrt(mean((ATT_ols-1)^2))

ATT_pscore_bias = mean(ATT_pscore) - 1 
ATT_pscore_sd = sd(ATT_pscore) 
ATT_pscore_mae = mean(abs(ATT_pscore-1))
ATT_pscore_rmse = sqrt(mean((ATT_pscore-1)^2))

mse_pscore = mean(bias_pscore)
rmse_pscore = sqrt(mean(bias_pscore))


result = c(ATT_base_bias,ATT_base_sd,ATT_base_mae,ATT_base_rmse,
           ATT_exact_bias,ATT_exact_sd,ATT_exact_mae,ATT_exact_rmse,
           ATT_ols_bias,ATT_ols_sd,ATT_ols_mae,ATT_ols_rmse,
           ATT_pscore_bias,ATT_pscore_sd,ATT_pscore_mae,ATT_pscore_rmse,
           mse_pscore,rmse_pscore)

