## Machine Learning on the coursera class project
## about data from accelerometers.
## Data was collected as part of the human activity recognition research  
## with the aim of predicting how well a sport activity is performed. 

library(caret)
library(dplyr)
library(parallel)
library(doParallel)

######################################################################################
## Data preprocessing

## loading train data
trainURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
train <- read.csv(trainURL,colClasses = "character")

## casting data to numeric
for (i in 1:(ncol(train)-1)) {
        train[,i] <- as.numeric(train[,i])
}

## number of NA values for each variable
NA_number <- sapply(train,function(x) length(which(is.na(x)))) 

## variables containing too many NA values are allmost constant
## hence they could be drop
## dropping variables with more than 2/3 of the number of training observations
drop_vars <- names(train)[NA_number > (2*ncol(train))/3]

## adding non sense variables as name of the activity subject
drop_vars <- c(drop_vars,
               "X","user_name","raw_timestamp_part_1",
               "raw_timestamp_part_2","cvtd_timestamp",
               "kurtosis_yaw_belt","skewness_yaw_belt",
               "kurtosis_yaw_dumbbell","amplitude_yaw_dumbbell",
               "skewness_yaw_dumbbell","amplitude_yaw_belt",
               "new_window","num_window")
train.processed <- train[,!colnames(train)%in%drop_vars]
str(train[,colnames(train)%in%drop_vars])

train.processed <- mutate(train.processed,classe=as.factor(classe))

## droping NA cases
train.processed <- train.processed[complete.cases(train.processed),]

## dimension of resulting processed data
dim(train.processed)

## normalizing data
train.stand <- scale(select(train.processed,-classe))
train.stand <- data.frame(cbind(classe=train.processed$classe,as.data.frame(train.stand)))
str(train.stand)

###################################################################################
## performing random forest
train <- createDataPartition(train.stand$classe,
                             p = 0.005,
                             list = FALSE)
train

fitControl <- trainControl(method = "cv",
                           number = 5)
initialTime <- Sys.time()
forest <- train(classe~.,
                data=train.stand[train,],
                method="rf",
                ntree=45,
                trControl=fitControl)
performanceTime <- Sys.time() - initialTime
performanceTime

forest.predict <- predict(forest,newdata = train.stand[-train,])
confusionMatrix(forest.predict,train.stand$classe[-train])