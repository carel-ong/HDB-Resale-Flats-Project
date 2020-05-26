dat = read.csv('C:/Users/Carel/Files/GitHub/HDB Resale Flats/processed/data_new.csv', header=T)
attach(dat)
names(dat)
dat = na.omit(dat)
model = lm(sqrt(resale_price) ~ . - intercept, data=dat)
summary(model)
plot(predict(model), residuals(model), xlab = 'Fitted values', ylab='Residuals')
plot(hatvalues(model), ylab = "Leverage")
which.max(hatvalues(model))
par(mfrow=c(2,2))
plot(model)

#Exploratory Analysis
pairs(dat) #correlation between features, check for collinearity and non-linearity of data


#forward selection
library(leaps)
regfit.fwd = regsubsets(resale_price~. -intercept, data=dat, nvmax=19, method='forward')
fwd_summary = summary(regfit.fwd)
par(mfrow = c(2,2))
plot(fwd_summary$rss, xlab='Number of Variables', ylab='R2', type = 'l')
plot(fwd_summary$adjr2, xlab='Number of Variables', ylab='Adjusted R2', type = 'l')
which.max(fwd_summary$adjr2)
points(7, fwd_summary$adjr2[7], col='red', cex=2, pch=20)
plot(fwd_summary$cp, xlab='Number of Variables', ylab='Cp', type = 'l')
which.min(fwd_summary$cp)
points(6, fwd_summary$cp[6], col='red', cex=2, pch=20)
plot(fwd_summary$bic, xlab='Number of Variables', ylab='BIC', type = 'l')
which.min(fwd_summary$bic)
points(7, fwd_summary$bic[7], col='red', cex=2, pch=20)
#choose 7 predictors
coef(regfit.fwd, 7)

#train test split
set.seed(1)
train = sample(1:nrow(dat), nrow(dat)*0.8)

#Linear regression
model1 = lm(log(resale_price)~ storey+floor_area_sqm+remaining_lease+Distance.to.CBD+mature_estate+DBSS
            +Distance.to.nearest.MRT.station-intercept, data=dat[train,])
summary(model1)
fitted = predict(model1, newdata=dat[-train,])
resale_test = dat[-train,'resale_price']
mse = mean((fitted - resale_test)^2) #4231458796
mse

#ar(2) model
AR_fit <- arima(dat[train,], order = c(2,0,0))
predict_AR <- forecast(AR_fit, h=10)
accuracy(predict_AR, dat[-train,])


#plotting to see the forecast (optional)
predict(AR_fit, n.ahead = 10)
ts.plot(Nile, xlim = c(1871, 1980))
AR_forecast <- predict(AR_fit, n.ahead = 10)$pred
AR_forecast_se <- predict(AR_fit, n.ahead = 10)$se
points(AR_forecast, type = "l", col = 2)
points(AR_forecast - 2*AR_forecast_se, type = "l", col = 2, lty = 2)
points(AR_forecast + 2*AR_forecast_se, type = "l", col = 2, lty = 2)


#Random Forest
par=mfrow(c(1,1))
library(tree)
library(randomForest)
tree_data = tree(resale_price~. -intercept, data=dat, subset=train)
summary(tree_data)
plot(tree_data)
text(tree_data, pretty=0)
rf = randomForest(resale_price~.-intercept, data=dat, subset=train,mtry=(ncol(dat)/3), importance =TRUE)
yhat.rf = predict(rf, newdata=dat[-train,])
mean((yhat.rf - resale_test)^2) # 725346483


#Boosting
#Parameter tuning for number of trees, interaction depth separately, keeping shrinkage constant
library(gbm)

#tuning for Interaction Depth
depth1 = gbm(resale_price~. - intercept, data=dat[train,],distribution = 'gaussian', n.trees=5000,interaction.depth = 1,shrinkage=0.1, cv.folds=10)
depth2 = gbm(resale_price~. - intercept, data=dat[train,],distribution = 'gaussian', n.trees=5000,interaction.depth = 2,shrinkage=0.1, cv.folds=10)
depth3 = gbm(resale_price~. - intercept, data=dat[train,],distribution = 'gaussian', n.trees=5000,interaction.depth = 3,shrinkage=0.1, cv.folds=10)
depth4 = gbm(resale_price~. - intercept, data=dat[train,],distribution = 'gaussian', n.trees=5000,interaction.depth = 4,shrinkage=0.1, cv.folds=10)
depth5 = gbm(resale_price~. - intercept, data=dat[train,],distribution = 'gaussian', n.trees=5000,interaction.depth = 5,shrinkage=0.1, cv.folds=10)
depth6 = gbm(resale_price~. - intercept, data=dat[train,],distribution = 'gaussian', n.trees=5000,interaction.depth = 6,shrinkage=0.1, cv.folds=10)
depth7 = gbm(resale_price~. - intercept, data=dat[train,],distribution = 'gaussian', n.trees=5000,interaction.depth = 7,shrinkage=0.1, cv.folds=10)
depth8 = gbm(resale_price~. - intercept, data=dat[train,],distribution = 'gaussian', n.trees=5000,interaction.depth = 8,shrinkage=0.1, cv.folds=10)

results = depth1$cv.error
results = cbind(results, depth2$cv.error)
results = cbind(results, depth3$cv.error)
results = cbind(results, depth4$cv.error)
results = cbind(results, depth5$cv.error)
results = cbind(results, depth6$cv.error)
results = cbind(results, depth7$cv.error)
results = cbind(results, depth8$cv.error)


matplot(results, type='l', lty=1, col=1:8, ylim = c(0,(5*(10^9))),xlab='Number of Trees',ylab = 'CV Error', main='Tuning ')
legend('topright', lty=1, col=1:8, legend = c('d=1','d=2','d=3','d=4','d=5','d=6','d=7','d=8'))

min(depth8$cv.error)
which.min(depth8$cv.error)

#fit model with tuned parameters, d=8, n.trees=2000
boost = gbm(resale_price~. -intercept, data=dat[train,], distribution='gaussian',n.trees=2000,interaction.depth=8, shrinkage=0.1)
summary(boost)
yhat_boost = predict(boost, newdata=dat[-train,], n.trees=2000)
mean((yhat_boost - resale_test)^2) #780136300


#Neural Network
library(nnet)

#Standardize data 
m.train = apply(dat[train,],2,mean)
sd.train = apply(dat[train,],2,sd)
s.dat = as.data.frame(t((t(dat) - m.train)/sd.train))

#PCA
pr.out=prcomp(dat, scale=TRUE)
out=pr.out$x
plot(pr.out)
pve=100*pr.out$sdev^2/sum(pr.out$sdev^2)

#CV, use training data
k=5
fold.error = rep(0,k)
n=length(resale_price[train])
index = sample(n)
foldbreaks = c(0,floor(n/k*1:k))
cv.error.k=rep(0,7)
t=c(20,25,30,33,35,37,40)
d=0
for(i in t) 
{
  d=d+1
  for(fold in 1:k) 
  {
    curval = index[(1+foldbreaks[fold]):(foldbreaks[fold+1])]
    lm.fit = nnet(resale_price~.-intercept,data=dat[-curval,],size=i,linout
                  =TRUE,decay=0.01,MaxNWts=5000)
    yhat=predict(lm.fit,dat[curval,])*sd(dat$resale_price) + mean(dat$resale_price)
    fold.error[fold] = mean((dat[train,]$resale_price[curval]-yhat)^2)
  }
  cv.error.k[d] = mean(fold.error)
}

plot(t,cv.error.k,type='b')

result=cv.error.k
result=cbind(result,cv.error.k)

matplot(t,result,type='l',lty=1,col=1:5,ylab="CV.error",xlab="Hidden Units")
legend("topright",lty=1,col=1:5,legend=
         c("D=0.5", "D=0.35", "D=0.25", "D=0.1", "D=0.01"))

#Fitting NN with tuned paramters
nn= nnet(resale_price~.-intercept,data=s.dat[train,],size=__,linout
              =TRUE,decay=___,MaxNWts=2351)
yhat.nn=predict(nn,newdata=s.dat[-train,])*sd(dat$resale_price) + mean(dat$resale_price) #normalize back for predictions
mean((yhat.nn-dat[-train,]$resale_price)^2)


