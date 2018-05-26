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

processData <- function(data) {
        ## casting data to numeric
        for (i in 1:(ncol(data)-1)) {
                data[,i] <- as.numeric(data[,i])
        }
        
        ## number of NA values for each variable
        NA_number <- sapply(data,function(x) length(which(is.na(x)))) 
        
        ## variables containing too many NA values are allmost constant
        ## hence they could be drop
        ## dropping variables with more than 2/3 of the number of training observations
        drop_vars <- names(data)[NA_number > (2*ncol(data))/3]
        
        ## adding non sense variables as name of the activity subject
        drop_vars <- c(drop_vars,
                       "X","user_name","raw_timestamp_part_1",
                       "raw_timestamp_part_2","cvtd_timestamp",
                       "kurtosis_yaw_belt","skewness_yaw_belt",
                       "kurtosis_yaw_dumbbell","amplitude_yaw_dumbbell",
                       "skewness_yaw_dumbbell","amplitude_yaw_belt",
                       "new_window","num_window")
        data.processed <- data[,!colnames(data)%in%drop_vars]
        str(data[,colnames(data)%in%drop_vars])
        
        data.processed <- mutate(data.processed,classe=as.factor(classe))
        
        ## droping NA cases
        data.processed <- data.processed[complete.cases(data.processed),]
        
        ## dimension of resulting processed data
        dim(data.processed)
        
        ## normalizing data
        data.stand <- scale(select(data.processed,-classe))
        data.stand <- data.frame(cbind(classe=data.processed$classe,as.data.frame(data.stand)))
        
        data.stand
}

train.stand <- processData(train)

#write.csv(train.stand,
#          file = "data/accelerometersProcessed.csv",
#          row.names = FALSE)

###################################################################################
## performing random forest
train <- createDataPartition(train.stand$classe,
                             p = 0.8,
                             list = FALSE)
tail(train)

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

####################################################################################

total <- 1:nrow(train.stand)
test <- total[!total %in% train]

## number of simulations        
nsims <- 100
accuracy <- numeric(nsims)
for (i in 1:nsims) {
        test20 <- sample(test,20)
        forest.predict <- predict(forest,newdata=train.stand[test20,])
        accuracy[i] <- confusionMatrix(forest.predict,train.stand$classe[test20])$overall[1]  
}
accuracy
jpeg("figures/accuracyVariation.jpeg")
plot(accuracy,type="l",col="steelblue",font.lab=2,lwd=2)
dev.off()

######################################################################

validationURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
validation <- read.csv(validationURL,colClasses = "character")

selected.vars <- names(train.stand)
selected.vars <- selected.vars[!selected.vars%in%"classe"]
selected.vars

valid.processed <- validation[,selected.vars]
for (i in 1:ncol(valid.processed)) {
        valid.processed[,i] <- as.numeric(valid.processed[,i])
}
valid.processed <- scale(valid.processed)
valid.processed <- as.data.frame(valid.processed)

preds <- predict(forest, newdata = valid.processed)
preds

