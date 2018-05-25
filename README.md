# Machine Learning Coursera Project

Contain the project of the Machine Learning class of Data Science Coursera specialization. The test consist on applying machine learning techniques to predict body exercise patterns based on accelerometers of several comercial devices. You can see the full implementation <a href="https://caramirezal.github.io/dataScience/machineLearningCourseraProject.html">here</a>.

The next code implements a random forest to the body activity data. The
data is partitioned to 80/20 in train/test of the total observations. 
A 5 cross fold iin the train set was performed to train the algorithm
and 45 trees were constructed. Then a confusion matrix was constructed
for testing to calculate the model accuracy.

```
set.seed(333)

## performing random forest
train <- createDataPartition(train.stand$classe,
                             p = 0.8,
                             list = FALSE)
#tail(train)

fitControl <- trainControl(method = "cv",
                           number = 5)
initialTime <- Sys.time()
forest <- train(classe~.,
                data=train.stand[train,],
                method="rf",
                ntree=45,
                trControl=fitControl)
performanceTime.forest <- Sys.time() - initialTime
#performanceTime.forest

forest.predict <- predict(forest,newdata = train.stand[-train,])
forest.accuracy <- confusionMatrix(forest.predict,train.stand$classe[-train])$overall[1] 
confusionMatrix(forest.predict,train.stand$classe[-train])
```

