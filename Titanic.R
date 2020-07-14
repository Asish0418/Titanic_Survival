#Titanic case study.
#The dataset is available online for free, here we have loaded the dataset before so i will be just using the view command.
View(titanic_data)
#Or read it from the .csv file!
titanic_data <- read.csv("C:/Users/Asish nayak/Desktop/Data files/titanic (1).csv")
#Take a look into the data using head command.
head(titanic_data)

#The starting step of any datascience problem is to divide the data into train and test sets.
#We need to be careful that the proportions of survived as a variable that we'll be going to predict, has to same in both training as well as testing set.
set.seed(1)
library(caret)
idx <- createDataPartition(y = titanic_data$Survived, times = 1, p = 2/3, list = FALSE)
idx
#We see that idx has 594 row numbers, which we'll put into the training set.
train <- titanic_data[idx,]
test <- titanic_data[-idx,]
#See the structure of data spread and the typre of variables in train set using 'str' function
str(train)
#We see that the columns sex, name, embarked, cabin, ticket are defined as factors, we need to change in to characters.
titanic_data$Name <- as.character(titanic_data$Name)
titanic_data$Sex <- as.character(titanic_data$Sex)
titanic_data$Ticket <- as.character(titanic_data$Ticket)
titanic_data$Cabin <- as.character(titanic_data$Cabin)
titanic_data$Embarked <- as.character(titanic_data$Embarked)
str(titanic_data)
#We see that it's changed now.

#For this analysis, we will be going to work with the following columns : Survived, Pclass, Sex, Age, SibSp, Parch, Fare, and Embarked.
#So we will select only these columns for the working model. Use 'dplyr' library to do so.
library(dplyr)
train <- train %>% select(Survived, Pclass, Sex, Age, SibSp, Parch, Fare, Embarked)

#Now lets try to find some summary statistics on the training data set.
summary(train)
#We have to now data type of few columns to factors for better working.
train[,c("Survived", "Pclass", "Sex", "Embarked")] <- lapply(train[,c("Survived", "Pclass", "Sex", "Embarked")], as.factor)
#We see that age column had NA values, to check whether other columns contains NA values or not, use 'colsums' function.
colSums(is.na(train))
#We see that age column has 113 missing values, so we'll use the summary stats like mean, mode to impute the missing values into the column.
mean_age <- mean(train$Age, na.rm = TRUE)
mean_age
train$Age[is.na(train$Age)] <- mean_age
#We have successfully imputed the values for the NAs.

#Now lets compare the independent numeric variables to the dependent variable 'Survived', by using the boxplots!
ggplot(train, aes(x=Survived, y=Age)) + geom_boxplot()
#We see that the mean age for both survived and non survived is similar but the one's who didn't survived has a lot of outliers.
ggplot(train, aes(x=Survived, y=Fare)) + geom_boxplot()
#It looks like those who suvived, has paid more substantially than the one's who didn't survived.
ggplot(train, aes(x=Survived, y=Pclass)) + geom_boxplot()
#We also see that a lot of passengers survived who were in in Pclass=1, and Pclass being a categorical data, this plot doesn't have much information.
#so use stacked barplots to relate the categorical variable with the survived column.
ggplot(train, aes(x=Pclass, fill=Survived)) + geom_bar(position = "fill")
#We see that a lot of people survived being in the Pclass=1.

#Lets calculate some similar feature engineering in the testing set as we did for the training set so that we can make predictions on equal data.
test <- test %>% select(Survived, Pclass, Sex, Age, SibSp, Parch, Fare, Embarked)
test[,c("Survived", "Pclass", "Sex", "Embarked")] <- lapply(test[,c("Survived", "Pclass", "Sex", "Embarked")], as.factor)
summary(test)
#We see that Age here also contains 64 missing values, lets impute these too.
mean_age_test <- mean(test$Age, na.rm = TRUE)
mean_age_test
test$Age[is.na(test$Age)] <- mean_age_test

#We will now set a logistic regression model in the training set.
glm_model <- glm(Survived ~ ., data = train, family = binomial(link = "logit"))
#We'll now predict the model on the test data.
library(broom)
pred_test <- glm_model %>% augment(newdata = test, type.predict = "response") 

#Lets make two benchmark matrix, Accuracy and the AUC for the model on the test set using a thershold of 0.5 to classify observations as survived or non survived.
pred_test %>% mutate(survived_hat = if_else(.fitted > 0.5,1,0)) %>% summarise(acc = mean(survived_hat==Survived))
#We see that the accuracy comes out to be 81.5%.
library(pROC)
roc(response = test$Survived, predictor = pred_test$.fitted)
#We see that the area under the curve is 86.31%.

prop.table(table(test$Survived))
#We see that if we would've classified all passengers as not survived, our accuracy would,ve been 61.9%.
#So our model preforms better than just classifying all values as the majority.
