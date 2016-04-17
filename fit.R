rm(list = ls())
setwd("F:/Users/Edwin/Documents/R/MachineLearning")
library(AppliedPredictiveModeling)
library(caret)
library(rpart)
library(rpart.plot)
library(RColorBrewer)
library(rattle)
library(randomForest)
library(plyr)
library(knitr)
# Downloading data.
url_training <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
raw_training <- "pml-training.csv"
#download.file(url=url_training, destfile=raw_training, method="curl")
url_testing <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
raw_testing <- "pml-testing.csv"
#download.file(url=url_testing, destfile=raw_testing, method="curl")

# Import the data treating empty values as NA.
training <- read.csv(raw_training, na.strings=c("NA",""), header=TRUE)
colnames_train <- colnames(training)
testing <- read.csv(raw_testing, na.strings=c("NA",""), header=TRUE)
colnames_test <- colnames(testing)

# Verify that the column names (excluding classe and problem_id) are identical in the training and test set.
all.equal(colnames_train[1:length(colnames_train)-1], colnames_test[1:length(colnames_train)-1])

#Partitioning the raw training dataset into parts for training and testing
inTrain <- createDataPartition(y=training$classe, p=0.6, list=FALSE)
myTraining <- training[inTrain, ]; myTesting <- training[-inTrain, ]
dim(myTraining); dim(myTesting)

## First, check for covariates that have virtually no variablility.

nzv <- nearZeroVar(myTraining, saveMetrics=TRUE)
myTraining <- myTraining[,nzv$nzv==FALSE]

nzv<- nearZeroVar(myTesting,saveMetrics=TRUE)
myTesting <- myTesting[,nzv$nzv==FALSE]
nzv

## Having verified that the schema of both the training and testing sets are identical (excluding the final column representing the A-E class), I decided to eliminate both NA columns and other extraneous columns.

# Count the number of non-NAs in each col.
nonNAs <- function(x) {
  as.vector(apply(x, 2, function(x) length(which(!is.na(x)))))
}

# Build vector of missing data or NA columns to drop.
colcnts <- nonNAs(myTraining)
drops <- c()
for (cnt in 1:length(colcnts)) {
  if (colcnts[cnt] < nrow(myTraining)) {
    drops <- c(drops, colnames_train[cnt])
  }
}

# Drop NA data and the first 7 columns as they're unnecessary for predicting.
myTraining <- myTraining[,!(names(myTraining) %in% drops)]
myTraining <- myTraining[,8:length(colnames(myTraining))]

myTesting <- myTesting[,!(names(myTesting) %in% drops)]
myTesting <- myTesting[,8:length(colnames(myTesting))]

testing <- testing[,!(names(testing) %in% drops)]
testing <- testing[,8:length(colnames(testing))]

# Show remaining columns.
colnames(myTraining)

# Clean variables with more than 60% NA

trainingV3 <- myTraining
for(i in 1:length(myTraining)) {
  if( sum( is.na( myTraining[, i] ) ) /nrow(myTraining) >= .7) {
    for(j in 1:length(trainingV3)) {
      if( length( grep(names(myTraining[i]), names(trainingV3)[j]) ) == 1)  {
        trainingV3 <- trainingV3[ , -j]
      }   
    } 
  }
}

# Set back to the original variable name
myTraining <- trainingV3
rm(trainingV3)

## Transform the myTesting and testing data sets

clean1 <- colnames(myTraining)
clean2 <- colnames(myTraining[, -40])  # remove the classe column
myTesting <- myTesting[clean1]         # allow only variables in myTesting that are also in myTraining
testing <- testing[clean2]             # allow only variables in testing that are also in myTraining

dim(myTesting)
dim(testing)

## Building Model with Decision Trees using myTraining Data
set.seed(123)
modFit1 <- rpart(classe ~ ., data=myTraining, method="class")
fancyRpartPlot(modFit1)

## Using training model modFit1 predict using myTesting
predict1 <- predict(modFit1, myTesting, type = "class")
conmatResults1 <- confusionMatrix(predict1, myTesting$classe)
conmatResults1

plot(conmatResults1$table, col = conmatResults1$byClass, main = paste("Decision Tree Confusion Matrix: Accuracy =", round(conmatResults1$overall['Accuracy'], 4)))

## Building Model with Random Forest using myTraining Data
set.seed(123)
modFit2 <- randomForest(classe ~ ., data=myTraining)

## Using training model modFit2 predict using myTesting
predict2 <- predict(modFit2, myTesting)
conmatResults2 <- confusionMatrix(predict2, myTesting$classe)
conmatResults2

plot(conmatResults2$table, col = conmatResults2$byClass, main = paste("Random Forest Confusion Matrix: Accuracy =", round(conmatResults2$overall['Accuracy'], 4)))

##Prediction with Generalized Boosted Regression

set.seed(123)
modFit3 <- train(classe ~ ., data=myTraining, method = "gbm", verbose = FALSE)

predict3 <- predict(modFit3, myTesting)
conmatResults3 <- confusionMatrix(predict3, myTesting$classe)
conmatResults3
plot(modFit3, ylim=c(0.9, 1))

## Predicting Results on the Test Data
## Random Forests gave an Accuracy in the myTesting dataset of 99.43%, which was more accurate that what I got from the Decision Trees or GBM. The expected out-of-sample error is 100-99.89 = 0.11%.

prediction2 <- predict(modFit2, testing, type = "class")
prediction2