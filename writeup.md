# Coursera PredMachLearn Writeup Assignment
## András Tajti
## 2015. 01. 25.

I uploaded the submission already once. I made two false predictions with
a _Random Forest_ prediction. 

# 1. Training
I loaded the _caret_ package, and the test data.


```r
library(caret)
# load test data:
trn <- data.table::fread("pml-training.csv", data.table=FALSE)
```
Then I took out the
 "#DIV/0" values, and checked which variables are not numeric,
  although they sould be.

```r
# taking out #DIV/0!:
trn[trn[,]=="#DIV/0"] <- 0
# which vars are characters?
nonchar_vars <- names(trn)[which(sapply(trn, class) != "numeric")]
# first four and last is really characzer, others shoud be numeric:
for( var in nonchar_vars[5:(length(nonchar_vars)-1)]){
  trn[[var]] <- as.numeric(trn[[var]])
}
```
I kept the first five and the **classe** variable as character, and
 converted the others to numeric. After that, I cleaned up the variables
 without much variance to decrease the size of the dataset. I also got rid
 of the variables which did not yield a correlation larger than 0.5 with any
 other variable.

```r
# clean near zero variance variables
NZV <- nearZeroVar(trn)
# removing NZV variables (classe is not among them)
trn <- trn[, -NZV]
#Checking covariations, I need strong ones
cors <- cor(trn[,which(sapply(trn, class)=="numeric")])
diag(cors) <- 0 # I don't need self-correlation
# get the names of correlating variables
corvars <- unique(dimnames(which(abs(cors)>.5, arr.ind=TRUE))[[1]])
trn <- trn[c(corvars, "classe")]
trn$classe <- as.factor(trn$classe)
```

I wanted to do PCA on the remaining dataset, as I suspected this number of
 features is still large for my computer, but to do that, I needed knnImpute
 to remove NA values. I tried four methods to classify the data:
- general linear models
- random forests
- neural networks
- boosted tree models

Eventually, only the random forests gave valuable results, as others gave huge
 missclassification looking at the confusion matrix via the _table()_
  function, so below in the code I commented them out.

For each model, I made **25-fold** cross validation **five** times.

NOTE: THE MODEL CREATION IS SLOW! I WILL PROVIDE THE RData FILE
 CONTAINING MY WORKSPACE.

```r
# glm
# glm_modfit <- train(classe~., method="glm", data=trn[5:length(trn)],
#                     preProcess=c("knnImpute", "pca"),
#                     trControl=trainControl(method="repeatedCV",
#                                               repeats=5))

# random forests:
rf_modfit <- train(classe~., method="rf", data=trn[5:length(trn)],
                    preProcess=c("knnImpute", "pca"),
                    trControl=trainControl(method="repeatedCV",
                                              repeats=5))
# nnet_modfit <- train(classe~., method="nnet", data=trn[5:length(trn)],
#                     preProcess=c("knnImpute", "pca"),
#                     trControl=trainControl(method="repeatedCV",
#                                               repeats=5))

# gbm_modfit <- train(classe~., method="gbm", data=trn[5:length(trn)],
#                     preProcess=c("knnImpute", "pca"),
#                     trControl=trainControl(method="repeatedCV",
#                                               repeats=5))

# compare results
# Pred_nn <- predict(nnet_modfit, trn)
# table(pred_nn, trn$classe)
Pred_rf <- predict(rf_modfit, trn)
table(pred_rf, trn)
```
# Prediction
After seeing the confusion matrix, I loaded the testing set and ran _predict()_
 on it.


```r
# load test data:
tst <- data.table::fread("pml_testing.csv", data.table=FALSE)
# predict
Pred_rf_tst <- predict(rf_modfit, newdata=tst)
# write out the files
for(i in 1:nrow(tst)){
  fname <- paste0("results/result_for_problem_", i, ".txt")
  write(as.character(Pred_rf_tst[i]), file=fname)
}
```
