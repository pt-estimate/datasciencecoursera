# ml_assignment
Christopher Wright  
February 28, 2016  



## Coursera Machine Learning Assignment
The objective of this assignment is to build a model that predicts whether a given user is working out correctly with estimates of the error in our model predictions.

The model training will be done with personal activity device data that has been labeled by people while working out.

The first step is to load in the two provided datasets and set the seed for reproducibility.


```r
library(caret)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
training = read.csv("/Users/chriswright/Desktop/pml-training.csv",header=T, na.strings=c("NA", "#DIV/0!"))
validation = read.csv("/Users/chriswright/Desktop/pml-testing.csv",header=T, na.strings=c("NA", "#DIV/0!"))
set.seed(471396)
```

Next, we look into the basic structure of the data.


```r
dim(training)
```

```
## [1] 19622   160
```

```r
dim(validation)
```

```
## [1]  20 160
```

```r
M = sapply(training, function(x) sum(is.na(x)))
na_counts <- M[M>0]
table(na_counts)
```

```
## na_counts
## 19216 19217 19218 19220 19221 19225 19226 19227 19248 19293 19294 19296 
##    67     1     1     1     4     1     4     2     2     1     1     2 
## 19299 19300 19301 19622 
##     1     4     2     6
```

```r
round(unique(na_counts)/dim(training)[1],2)
```

```
##  [1] 0.98 0.98 1.00 0.98 0.98 0.98 0.98 0.98 0.98 0.98 0.98 0.98 0.98 0.98
## [15] 0.98 0.98
```

###Data Cleaning
Many of the variables in the training set are missing significant portions of their observations. Due to the majority of the observations of these variables having missing values, I decide to throw these out. We also throw out variables that only store user specific data that isn't related to the study.


```r
missing = is.na(training)
na_columns = which(colSums(missing) > 15000)
training = training[, -na_columns]
training = training[,-c(1:8)]
dim(training)
```

```
## [1] 19622    52
```

```r
validation = validation[, -na_columns]
validation = validation[,-c(1:8)]
```

###Data Partitions
We next create the data partitions. Because I plan on using random forests to do my predictions, I do not bother with cross validation. It is estimated internally during training via the oob estimate.

```r
inTrain <- createDataPartition(y=training$classe, p=0.80, list=FALSE)
training_data <- training[inTrain,]
testing_data <- training[-inTrain,]
```
###Model Training and Testing
The random forest is trained against the training data and used to predict the testing data classe outcomes.

```r
library(randomForest)
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```r
model = randomForest(classe~., data=training_data, ntree=200)
model
```

```
## 
## Call:
##  randomForest(formula = classe ~ ., data = training_data, ntree = 200) 
##                Type of random forest: classification
##                      Number of trees: 200
## No. of variables tried at each split: 7
## 
##         OOB estimate of  error rate: 0.46%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 4461    1    0    1    1 0.000672043
## B   13 3018    7    0    0 0.006583278
## C    0   15 2721    2    0 0.006208912
## D    0    0   23 2545    5 0.010882239
## E    0    0    1    4 2881 0.001732502
```

```r
pred <- predict(model, newdata=testing_data)
confusionMatrix(pred, testing_data$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1116    7    0    0    0
##          B    0  751    5    0    0
##          C    0    1  679   10    0
##          D    0    0    0  632    1
##          E    0    0    0    1  720
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9936          
##                  95% CI : (0.9906, 0.9959)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9919          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9895   0.9927   0.9829   0.9986
## Specificity            0.9975   0.9984   0.9966   0.9997   0.9997
## Pos Pred Value         0.9938   0.9934   0.9841   0.9984   0.9986
## Neg Pred Value         1.0000   0.9975   0.9985   0.9967   0.9997
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2845   0.1914   0.1731   0.1611   0.1835
## Detection Prevalence   0.2863   0.1927   0.1759   0.1614   0.1838
## Balanced Accuracy      0.9988   0.9939   0.9946   0.9913   0.9992
```
###Discussion
The results of the model claim a low(<1%) OOB estimate. This is an accurate estimate of out of sample error. Additionally, the model is performant on the testing set data.
