## Improving prediction for the project

library(caret)
library(dplyr)

## Loading train data
trainURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
train <- read.csv(trainURL,colClasses = "character")

## Loading validation data
validationURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
validation <- read.csv(validationURL,colClasses = "character")

## adding classe dummy variable
validation <- mutate(validation,classe="validation")
validation <- select(validation,-problem_id)

## merge train and validation to perform eual data processing
train.processed <- rbind(train,validation)

## casting data to numeric
for (i in 1:(ncol(train.processed)-1)) {
        train.processed[,i] <- as.numeric(train.processed[,i])
}

## number of NA values for each variable
NA_number <- sapply(train.processed,function(x) length(which(is.na(x)))) 

## variables containing too many NA values are allmost constant
## hence they could be drop
## dropping variables with more than 2/3 of the number of training observations
drop_vars <- names(train.processed)[NA_number > (2*nrow(train.processed))/3]

## adding non sense variables as name of the activity subject
drop_vars <- c(drop_vars,
               "X","user_name","raw_timestamp_part_1",
               "raw_timestamp_part_2","cvtd_timestamp",
               "kurtosis_yaw_belt","skewness_yaw_belt",
               "kurtosis_yaw_dumbbell","amplitude_yaw_dumbbell",
               "skewness_yaw_dumbbell","amplitude_yaw_belt",
               "new_window","num_window")
train.processed <- train.processed[,!colnames(train.processed)%in%drop_vars]
dim(train.processed)

train.processed <- mutate(train.processed,classe=as.factor(classe))

## droping NA cases
train.processed <- train.processed[complete.cases(train.processed),]

## dimension of resulting processed data
dim(train.processed)

## normalizing data
data.stand <- scale(select(train.processed,-classe))
data.stand <- data.frame(cbind(classe=train.processed$classe,as.data.frame(data.stand)))

dim(data.stand)

########################################################################################

train.stand <- filter(data.stand,classe!="validation")

## removing validation classe factor with cero cases
train.stand <- mutate(train.stand,classe=as.character(classe))
train.stand <- mutate(train.stand,classe=as.factor(classe))

intrain <- createDataPartition(1:nrow(train.stand),
                             p = 0.8,
                             list = FALSE)
tail(intrain)

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

predict(forest,newdata = filter(data.stand,classe=="validation"))

