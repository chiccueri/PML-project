Human Activity Recognition
========================================================


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
# training set set.seed(12345)
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
## Summary of sample sizes: 3681, 3679, 3682, 3679 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa  Accuracy SD  Kappa SD
##   2     1         1      0.006        0.008   
##   30    1         1      0.005        0.007   
##   50    1         1      0.005        0.006   
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
##          A 467   0   0   0   0
##          B   6 323   5   0   0
##          C   0   6 268   1   0
##          D   0   1   2 271   0
##          E   0   0   0   1 284
## 
## Overall Statistics
##                                        
##                Accuracy : 0.987        
##                  95% CI : (0.98, 0.992)
##     No Information Rate : 0.289        
##     P-Value [Acc > NIR] : <2e-16       
##                                        
##                   Kappa : 0.983        
##  Mcnemar's Test P-Value : NA           
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.987    0.979    0.975    0.993    1.000
## Specificity             1.000    0.992    0.995    0.998    0.999
## Pos Pred Value          1.000    0.967    0.975    0.989    0.996
## Neg Pred Value          0.995    0.995    0.995    0.999    1.000
## Prevalence              0.289    0.202    0.168    0.167    0.174
## Detection Rate          0.286    0.198    0.164    0.166    0.174
## Detection Prevalence    0.286    0.204    0.168    0.168    0.174
## Balanced Accuracy       0.994    0.985    0.985    0.995    1.000
```

```r
# time taken
proc.time() - ptm
```

```
##    user  system elapsed 
##  624.16    5.13  643.30
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



