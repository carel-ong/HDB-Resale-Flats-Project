dat = read.csv('C:/Users/Carel/Files/GitHub/HDB Resale Flats/processed/data_new.csv', header=T)
attach(dat)
names(dat)
dat = na.omit(dat)
model = lm(resale_price ~ ., data=dat)
summary(model)
plot(predict(model), residuals(model), xlab = 'Fitted values', ylab='Residuals')
plot(hatvalues(model), ylab = "Leverage")
which.max(hatvalues(model))
par(mfrow=c(2,2))
plot(model)

#Exploratory Analysis
library(car)
vif(model)
dat = subset(dat, select=-c(Adjoined.flat, Apartment, Standard, Model.A,Simplified))
names(dat)
model1 = lm(resale_price~., data=dat)
vif(model1)

#forward selection
library(leaps)
regfit.fwd = regsubsets(resale_price~., data=dat, nvmax=19, method='forward')
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
linearmodel = lm(resale_price~ storey+floor_area_sqm+remaining_lease+Distance.to.CBD+mature_estate+DBSS
            +Distance.to.nearest.MRT.station, data=dat[train,])
summary(linearmodel)
fitted = predict(model1, newdata=dat[-train,])
resale_test = dat[-train,'resale_price']
mse = mean((fitted - resale_test)^2) #3872241663
mse

#ar(2) model
dat_month = read.csv('C:/Users/Carel/Files/GitHub/HDB Resale Flats/processed/DATA_FULL.csv', header=T)
attach(dat_month)
names(dat_month)
tsdata = ts(resale_price, start=c(2017,1), frequency=12)
plot(tsdata)
components =decompose(ts(resale_price, start=c(2017,1), frequency =12))
plot(components)

acf(tsdata)
pacf(tsdata)
library("forecast")
model_ar = auto.arima(tsdata, trace=TRUE)
model_ar
#accuracy(predict_AR, dat[-train,])


#Random Forest
par=mfrow(c(1,1))
library(tree)
library(randomForest)
tree_data = tree(resale_price~., data=dat, subset=train)
summary(tree_data)
plot(tree_data)
text(tree_data, pretty=0)
rf = randomForest(resale_price~., data=dat, subset=train,mtry=(ncol(dat)/3), importance =TRUE)
yhat.rf = predict(rf, newdata=dat[-train,])
mean((yhat.rf - resale_test)^2) # 731208327, 

varImpPlot(rf)
importance(rf)
#Boosting
#Parameter tuning for number of trees, interaction depth separately, keeping shrinkage constant
library(gbm)

#tuning for Interaction Depth
depth1 = gbm(resale_price~. , data=dat[train,],distribution = 'gaussian', n.trees=5000,interaction.depth = 1,shrinkage=0.1, cv.folds=10)
depth2 = gbm(resale_price~. , data=dat[train,],distribution = 'gaussian', n.trees=5000,interaction.depth = 2,shrinkage=0.1, cv.folds=10)
depth3 = gbm(resale_price~. , data=dat[train,],distribution = 'gaussian', n.trees=5000,interaction.depth = 3,shrinkage=0.1, cv.folds=10)
depth4 = gbm(resale_price~. , data=dat[train,],distribution = 'gaussian', n.trees=5000,interaction.depth = 4,shrinkage=0.1, cv.folds=10)
depth5 = gbm(resale_price~. , data=dat[train,],distribution = 'gaussian', n.trees=5000,interaction.depth = 5,shrinkage=0.1, cv.folds=10)
depth6 = gbm(resale_price~. , data=dat[train,],distribution = 'gaussian', n.trees=5000,interaction.depth = 6,shrinkage=0.1, cv.folds=10)
depth7 = gbm(resale_price~. , data=dat[train,],distribution = 'gaussian', n.trees=5000,interaction.depth = 7,shrinkage=0.1, cv.folds=10)
depth8 = gbm(resale_price~. , data=dat[train,],distribution = 'gaussian', n.trees=5000,interaction.depth = 8,shrinkage=0.1, cv.folds=10)

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
boost = gbm(resale_price~., data=dat[train,], distribution='gaussian',n.trees=3300,interaction.depth=8, shrinkage=0.1)
summary(boost)
yhat_boost = predict(boost, newdata=dat[-train,], n.trees=3300)
mean((yhat_boost - resale_test)^2) # 737371220 for n.tress=3200



