Human Activity Recognition
========================================================


```r
# reading training data set
data <- read.csv("pml-training.csv")
```



```r
library(caret)
library(rpart)
```




```r
# options(scipen = 1)
naCol <- vector()
for (i in 1:length(data)) {
    n <- sum(is.na(data[, i]))
    r <- n/length(data[, i])
    if (r > 0.5) {
        naCol <- append(naCol, i)
    }
}
data2 <- data[, -naCol]
```



```r
# discard NAs
NAs <- apply(data, 2, function(x) {
    sum(is.na(x))
})
validData <- data[, which(NAs == 0)]
# make training set
trainIndex <- createDataPartition(y = validData$classe, p = 0.2, list = FALSE)  # 3927 rows
trainData <- validData[trainIndex, ]
# discards unuseful predictors
removeIndex <- grep("timestamp|X|user_name|new_window", names(trainData))
trainData <- trainData[, -removeIndex]
# make model
modFit <- train(trainData$classe ~ ., data = trainData, method = "rpart")
```

```
## Error: cannot allocate vector of size 205.4 Mb
```

```r
modFit
```

```
## Error: object 'modFit' not found
```



