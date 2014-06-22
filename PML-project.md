Human Activity Recognition
========================================================

### Summary

The raw data for this project come from a bigger project to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants that were asked to perform barbell lifts correctly and incorrectly in 5 different ways.
More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset)

Our goal is to predict the manner in which they did the exercise, this is the "classe" variable in the training set (pml-training.csv), even using any of the other variables to predict with. The prediction model predicted 20 different test cases (pml-testing.csv).

Given the high dimension of the raw data set (a data frame with 19622 observations on 160 variables) and the capacity/velocity of the laptop I did the work with, I had to build a training data set as shorter as possible keeping the model accuracy as higher as I could.  

This is how I proceeded:  
1. ***cleaned the data set***: I reduced the predictors to 53 after discarded missing values, variables with no significant variance and some variables clearly unuseful to build the model.  
2. ***built my training and testing data sets***: I build a training data set with 4907 rows (25% of the original) and a testing data set of 1/3.  
3. ***fitted the model***: to build the model I used Random forest algorithm, setting the resampling method to 4 cross-validation.  
4. ***tested the model***: with confusionMatrix() I calculated a cross-tabulation of observed and predicted classes with associated statistics. That resulted in a fairly low out of sample error as the accuracy is over 0.95.  
5. ***predicted results***: I applied the model to the 20 test cases available in the test data file.  


### Algorithm


```r
library(ggplot2)
library(lattice)
library(caret)
library(rpart)
```



```r
# reading training data set
data <- read.csv("pml-training.csv")
```



```r
# discard NA
NAsum <- apply(data, 2, function(x) {
    sum(is.na(x))
})
data2 <- data[, which(NAsum == 0)]
# discard zeroVar
nsv <- nearZeroVar(data2, saveMetrics = TRUE)
data3 <- data2[, which(nsv$nzv == FALSE)]
# discard unuseful predictors
uPredict <- grep("timestamp|X|user_name|new_window", names(data3))
data5 <- data3[, -uPredict]

ptm <- proc.time()
# training set
inTrain <- createDataPartition(y = data5$classe, p = 0.25, list = FALSE)
training <- data5[inTrain, ]
testing <- data5[-inTrain, ]
testing2 <- testing[sample(nrow(testing), size = dim(training)[1] * 2/6), ]

# model
modelFit <- train(classe ~ ., data = training, trControl = trainControl(method = "cv", 
    number = 4), method = "rf", prox = TRUE)
```

```
## Loading required package: randomForest
## randomForest 4.6-7
## Type rfNews() to see new features/changes/bug fixes.
```

```r
modelFit
```

```
## Random Forest 
## 
## 4907 samples
##   53 predictors
##    5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (4 fold) 
## 
## Summary of sample sizes: 3680, 3681, 3681, 3679 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa  Accuracy SD  Kappa SD
##   2     1         1      0.007        0.009   
##   30    1         1      0.004        0.005   
##   50    1         1      0.004        0.005   
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 27.
```

```r
confusionMatrix(testing2$classe, predict(modelFit, testing2))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 461   0   0   0   0
##          B   3 315   2   0   1
##          C   0   4 260   0   0
##          D   0   0   2 288   0
##          E   0   1   0   0 298
## 
## Overall Statistics
##                                         
##                Accuracy : 0.992         
##                  95% CI : (0.986, 0.996)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.99          
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.994    0.984    0.985    1.000    0.997
## Specificity             1.000    0.995    0.997    0.999    0.999
## Pos Pred Value          1.000    0.981    0.985    0.993    0.997
## Neg Pred Value          0.997    0.996    0.997    1.000    0.999
## Prevalence              0.284    0.196    0.161    0.176    0.183
## Detection Rate          0.282    0.193    0.159    0.176    0.182
## Detection Prevalence    0.282    0.196    0.161    0.177    0.183
## Balanced Accuracy       0.997    0.990    0.991    0.999    0.998
```

```r
# time taken
proc.time() - ptm
```

```
##    user  system elapsed 
##  606.16    5.29  621.82
```



```r
# reading testing data set
dataTest <- read.csv("pml-testing.csv")
dataTestC <- dataTest[, colnames(dataTest) %in% colnames(data5)]
# prediction
predictions2 <- predict(modelFit, newdata = dataTestC)
predictions2
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```



