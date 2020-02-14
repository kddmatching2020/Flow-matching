library("Matching")
library("MatchIt")

# x_new<-read.csv("data_x.csv")
# y_new<-read.csv("twin_pairs_Y.csv")
# t_new<-read.csv("twin_pairs_T.csv")


main<-function(i){

  data_all = read.csv(paste("data_all_",i,".csv",sep = ""))
  ATT_True[i]<<-data_all$ATT[1]
  ATT_base[i]<<-mean(data_all[which(data_all$w==1),]$y_obs)-mean(data_all[which(data_all$w==0),]$y_obs)
  
  # coarsened exact matching
  cut=list(brstate=3,stoccfipb=3,mplbir=3,birmon=3,gestat10=3,
           dlivord_min=3,dtotord_min=3,mager8=3)
  m.out <- matchit(w ~ brstate+stoccfipb+mplbir+birmon+gestat10+dlivord_min+dtotord_min+
                     mager8,data=data_all, method = "cem",cutpoints=cut)

  m.data <- match.data(m.out)
  m.data2 <- match.data(m.out, group = "treat")
  m.data3 <- match.data(m.out, group = "control")
  ATT_exact[i]<<-mean(m.data2$y_obs)-mean(m.data3$y_obs *m.data3$weights)
  
  # # OLS regression
  lm1 <- lm(y_obs~ brstate+stoccfipb+mplbir+birmon+gestat10+dlivord_min+dtotord_min+
              mager8+w, data = data_all)
  ATT_ols[i]<<-lm1$coefficients[length(lm1$coefficients)]
  
  # matching pscore
  m.out <- matchit(w~brstate+stoccfipb+mplbir+birmon+gestat10+dlivord_min+dtotord_min+
                     mager8,data=data_all, method = "nearest", distance = "logit", replace=T)
  
  m.data <- match.data(m.out)
  m.data2 <- match.data(m.out, group = "treat")
  m.data3 <- match.data(m.out, group = "control")
  ATT_pscore[i]<<-mean(m.data2$y_obs)-mean(m.data3$y_obs *m.data3$weights)
}



ATT_True<-c()

ATT_base<-c()
ATT_exact<-c()
ATT_ols<-c()
ATT_pscore<-c()


for (i in 1:100){
  main(i)
}

ATT_base_bias = mean(ATT_base-ATT_True) 
ATT_base_sd = sd(ATT_base) 
ATT_base_mae = mean(abs(ATT_base-ATT_True))
ATT_base_rmse = sqrt(mean((ATT_base-ATT_True)^2))

ATT_exact_bias = mean(ATT_exact-ATT_True) 
ATT_exact_sd = sd(ATT_exact) 
ATT_exact_mae = mean(abs(ATT_exact-ATT_True))
ATT_exact_rmse = sqrt(mean((ATT_exact-ATT_True)^2))

ATT_ols_bias = mean(ATT_ols-ATT_True) 
ATT_ols_sd = sd(ATT_ols) 
ATT_ols_mae = mean(abs(ATT_ols-ATT_True))
ATT_ols_rmse = sqrt(mean((ATT_ols-ATT_True)^2))

ATT_pscore_bias = mean(ATT_pscore-ATT_True) 
ATT_pscore_sd = sd(ATT_pscore) 
ATT_pscore_mae = mean(abs(ATT_pscore-ATT_True))
ATT_pscore_rmse = sqrt(mean((ATT_pscore-ATT_True)^2))



result = c(ATT_base_bias,ATT_base_sd,ATT_base_mae,ATT_base_rmse,
           ATT_exact_bias,ATT_exact_sd,ATT_exact_mae,ATT_exact_rmse,
           ATT_ols_bias,ATT_ols_sd,ATT_ols_mae,ATT_ols_rmse,
           ATT_pscore_bias,ATT_pscore_sd,ATT_pscore_mae,ATT_pscore_rmse)