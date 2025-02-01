library(dplyr)
library(caret)
library(corrr)
library(e1071) 
library(class) 
library(infotheo)
library(glmnet)
library(MASS)
library(RWeka)
library(glmnet)
library(stats)
library(FSelector)
library(xgboost)
library(recipes)
library(themis)
library(pROC)


############################ Preprocessing####################################
dataset <- read.csv('project_dataset_5K.csv')
targetVariable <- dataset$Class
dataset <- dplyr::select(dataset, -Class)

dataset$I_DATE <- as.Date(with(dataset, paste0(IMONTH, "-", IDAY, "-", IYEAR)), "%m-%d-%Y")

# Reorder columns
dataset <- dataset[, c("FMONTH", "I_DATE", setdiff(names(dataset), c("FMONTH", "I_DATE")))]

columns_to_remove <- c("FMONTH", "IMONTH", "IDAY", "IYEAR", "DISPCODE", "SEQNO", "_PSU", "CTELENM1",
                       "STATERE", "CELLSEX", "SEXVAR", "PSATIME", "CSRVINST", "IMFVPLA", "QSTVER",
                       "_STSTR", "_STRWT", "_RAWRAKE", "_WT2RAKE", "_CLLCPWT", "_DUALCOR",
                       "_LLCPWT2", "_LLCPWT", "_RFSEAT3","X_PSU")

columns_to_remove <- trimws(columns_to_remove)

dataset <- dataset[, !names(dataset) %in% columns_to_remove]


threshold <- 0.5 * nrow(dataset)
dataset <- dataset[, colSums(is.na(dataset)) < threshold]
dataset$Class <- targetVariable


impute_with_distribution <- function(column) {
  freq <- table(column[!is.na(column)])
  
  probs <- freq / sum(freq)
  
  imputed_values <- sample(x = names(probs), size = sum(is.na(column)), replace = TRUE, prob = probs)
  
  if(is.numeric(column)) {
    imputed_values <- as.numeric(imputed_values)
  } else if(is.factor(column)) {
    imputed_values <- as.factor(imputed_values)
  } 
  
  column[is.na(column)] <- imputed_values
  
  return(column)
}

imputed_dataset <- lapply(dataset, impute_with_distribution)
imputed_dataset <- as.data.frame(imputed_dataset)
class_variable <- imputed_dataset$Class

imputed_dataset_without_class <- imputed_dataset[, !(names(imputed_dataset) %in% c("Class"))]

numeric_logical_dataset <- imputed_dataset_without_class[sapply(imputed_dataset_without_class, function(x) is.numeric(x) || is.logical(x))]

cov_matrix <- cov(numeric_logical_dataset, use = "complete.obs")
columns_with_zero_cov = colSums(abs(cov_matrix)) == 0
numeric_logical_dataset_filtered <- numeric_logical_dataset[ , !columns_with_zero_cov]

numeric_logical_dataset_filtered$Class <- class_variable
head(numeric_logical_dataset_filtered)

dataset<-numeric_logical_dataset_filtered


dataset_without_class <- dplyr::select(dataset, -Class)
correlation_matrix <- cor(dataset_without_class)

highly_correlated <- findCorrelation(correlation_matrix, cutoff = 0.8, verbose = TRUE)

dataset_reduced <- dataset_without_class[-highly_correlated]

dataset_reduced$Class <- class_variable

dataset<- dataset_reduced

dataset$Class <- as.factor(dataset$Class)

X <- dataset[, sapply(dataset, is.numeric)]

variances <- apply(X, 2, var)

threshold <- quantile(variances, 0.25)
selected_features <- names(variances[variances > threshold])

selected_features <- c(selected_features, "Class")
selected_dataset <- dataset[, selected_features]

df_final <- selected_dataset %>% 
  mutate(across(where(~ n_distinct(.) > 9), scale))

head(df_final)

print(dim(selected_dataset))
print(names(selected_dataset))
write.csv(selected_dataset, "Preprocessed_data.csv", row.names = FALSE)


######################## Preprocessing Done ########################

df <- read.csv("Preprocessed_data.csv")
dim(df)
df$Class <- as.factor(df$Class)

################# Splitting into train and test#######################
set.seed(123)

splitIndex <- createDataPartition(df$Class, p = 0.70, list = FALSE, times = 1)

train_data <- df[splitIndex,]
test_data <- df[-splitIndex,]

print(dim(train_data))
print(dim(test_data))

table(train_data$Class)
table(test_data$Class)
barplot(table(train_data$Class))
write.csv(train_data, "train_data.csv", row.names = FALSE)
write.csv(test_data, "test_data.csv", row.names = FALSE)

## Oversampling with LASSO ##


####################### Random Oversampling ##################
minority_class <- ifelse(mean(train_data$Class == "Y") < 0.5, "Y", "N")
minority_data <- filter(train_data, Class == minority_class)

majority_count <- sum(train_data$Class != minority_class)
minority_count <- nrow(minority_data)
oversample_amount <- majority_count - minority_count

additional_samples <- minority_data[sample(nrow(minority_data), size = oversample_amount, replace = TRUE), ]

balanced_train_data <- rbind(train_data, additional_samples)
table(balanced_train_data$Class)
barplot(table(balanced_train_data$Class))


################## feature seelction using LASSO ############################

x <- model.matrix(Class ~ . - 1, data=train_data) 
y <- as.numeric(train_data$Class) - 1  

set.seed(123)
cv_fit <- cv.glmnet(x, y, family="binomial", alpha=1) 

optimal_coefs <- coef(cv_fit, s = "lambda.min")

coefs_dense <- as.vector(optimal_coefs)

coefs_dense <- coefs_dense[-1]  

feature_names <- rownames(optimal_coefs)[-1]

selected_features_lasso <- feature_names[coefs_dense != 0]
#print(selected_features)
selected_features_with_target <- c(selected_features_lasso, "Class")

train_data_lasso_os <- balanced_train_data[, selected_features_with_target, drop = FALSE]
test_data_lasso_os <- test_data[, selected_features_with_target, drop = FALSE]

dim(train_data_lasso_os)
dim(test_data_lasso_os)

############################## Classification with Oversampling - LASSO ########################
# 1. lda
set.seed(123)
ldaModel <- lda(Class~ ., data = train_data_lasso_os) 
ldaModel
pred <- predict(ldaModel, test_data_lasso_os)
performance_measures_LDA_os_lasso  <- confusionMatrix(data=pred$class, 
                                         reference = test_data_lasso_os$Class)
performance_measures_LDA_os_lasso

lda_probabilities <- pred$posterior[,2]
roc_lda <- roc(as.numeric(test_data_lasso_os$Class)-1, lda_probabilities) 
auc_lda <- auc(roc_lda)
print(paste("AUC for LDA:", auc_lda))


# 2. Naive Bayesian
set.seed(123)
nb_model <- naiveBayes(Class ~ ., data = train_data_lasso_os)
nb_predictions <- predict(nb_model, test_data_lasso_os)

conf_matrix_nb_os_lasso <- confusionMatrix(nb_predictions, test_data_lasso_os$Class)
print(conf_matrix_nb_os_lasso)
nb_prob <- predict(nb_model, test_data_lasso_os, type = "raw")

prob_positive_class <- nb_prob[,2]

actual_classes <- as.numeric(test_data_lasso_os$Class) - 1
roc_result <- roc(actual_classes, prob_positive_class)
auc_value <- auc(roc_result)

print(auc_value)


# 3. GBM
set.seed(123)
ctrl <- trainControl(method = "CV",
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE,
                     savePredictions = TRUE)
gbmGrid <- expand.grid(interaction.depth = c(1, 3, 5, 7, 9),
                       n.trees = (2:10)*65,
                       shrinkage = c(.01),
                       n.minobsinnode = 3)

set.seed(123)
gbmFit <- caret::train(x = train_data_lasso_os[, -35], 
                       y = train_data_lasso_os$Class,
                       method = "gbm",
                       tuneGrid = gbmGrid,
                       metric = "ROC",
                       verbose = FALSE,
                       trControl = ctrl)
gbmFit
plot(gbmFit)

pred <- predict(gbmFit, train_data_lasso_os)
cm_pca_os <- caret::confusionMatrix(pred, train_data_lasso_os$Class)
cm_pca_os

gbm_prob <- predict(gbmFit, newdata = test_data_lasso_os[, -35], type = "prob")
prob_positive_class <- gbm_prob[,2]

actual_classes <- as.numeric(test_data_lasso_os$Class) - 1
roc_result <- roc(actual_classes, prob_positive_class)
auc_value <- auc(roc_result)
print(auc_value)

plot(roc_result, main="ROC Curve for GBM Model")

# 4. XGBoost
xgb_control = trainControl(
  method = "cv", number = 10,
  summaryFunction = twoClassSummary,
  classProbs = TRUE,
  savePredictions = TRUE
)

xgbGrid <- expand.grid(
  nrounds = seq(from = 50, to = 300, by = 50),
  eta = c(0.05, 0.1, 0.3),
  max_depth = c(1, 2, 3),
  gamma = c(0, 1, 2),
  colsample_bytree = 1,
  min_child_weight = c(1, 3),
  subsample = c(0.5, 0.7)
)

set.seed(123)
xgbModel <- caret::train(x = train_data_lasso_os[, -35], 
                         y = train_data_lasso_os$Class,
                         method = "xgbTree",
                         tuneGrid = xgbGrid,
                         metric = "ROC",
                         verbose = FALSE,
                         trControl = xgb_control)
xgbModel
plot(xgbModel)

pred <- predict(xgbModel, test_data_lasso_os)
cm_cg_os_lasso <- caret::confusionMatrix(pred, test_data_lasso_os$Class)
cm_cg_os_lasso

xgb_prob <- predict(xgbModel, newdata = test_data_lasso_os[, -35], type = "prob")

prob_positive_class <- xgb_prob[,2]
actual_classes <- as.numeric(test_data_lasso_os$Class) - 1

roc_result <- roc(actual_classes, prob_positive_class)
auc_value <- auc(roc_result)
print(auc_value)
plot(roc_result, main="ROC Curve for XGBoost Model")

# 5. Logistic regression 

train_data_lasso_os$Class <- ifelse(train_data_lasso_os$Class ==  "Y", 1, 0)
test_data_lasso_os$Class <- ifelse(test_data_lasso_os$Class ==  "Y", 1, 0)

train_data_lasso_os$Class <- as.factor(train_data_lasso_os$Class)
test_data_lasso_os$Class <- as.factor(test_data_lasso_os$Class)

logitModel <- glm(Class ~ ., data = train_data_lasso_os, family = "binomial") 
options(scipen=999)
summary(logitModel)

logitModel.pred <- predict(logitModel, test_data_lasso_os[, -35], type = "response")

data.frame(actual = test_data_lasso_os$Class[1:34], predicted = logitModel.pred[1:34])

pred <- factor(ifelse(logitModel.pred >= 0.5, 1, 0))
pred
performance_measures_lr_os_lasso  <- confusionMatrix(data=pred, 
                                         reference = as.factor(test_data_lasso_os$Class))
performance_measures_lr_os_lasso


actual_classes <- as.numeric(as.character(test_data_lasso_os$Class)) - 1
roc_result <- roc(actual_classes, logitModel.pred)

plot(roc_result, main="ROC Curve for Logistic Regression Model")
auc_value <- auc(roc_result)
print(paste("AUC value:", auc_value))


# 6. Elastic Net Regularized Logistic Regression
x_train <- model.matrix(Class ~ . -1, data = train_data_lasso_os) 
y_train <- as.numeric(train_data_lasso_os$Class)   
x_test <- model.matrix(Class ~ . -1, data = test_data_lasso_os)

set.seed(123)
cv_fit <- cv.glmnet(x_train, y_train, alpha=0.51, family="binomial") 

best_lambda <- cv_fit$lambda.min

predictions_prob_lasso_os <- predict(cv_fit, newx=x_test, s=best_lambda, type="response")

predictions_lasso_os <- ifelse(predictions_prob_lasso_os > 0.51, "1", "0")

predictions_lasso_os <- factor(predictions_lasso_os, levels=c("0", "1"), labels=levels(train_data_lasso_os$Class))

conf_matrix_lasso_en_os <- confusionMatrix(predictions_lasso_os, test_data_lasso_os$Class)
print(conf_matrix_lasso_en_os)


actual_classes <- as.numeric(as.character(test_data_lasso_os$Class)) - 1

roc_result <- roc(actual_classes, predictions_prob_lasso_os[,1])

plot(roc_result, main="ROC Curve for Elastic Net Model")
auc_value <- auc(roc_result)
print(paste("AUC value:", auc_value))


##################### Feature Selection with PCA #####################
df <- read.csv("preprocessed_data.csv")
df$Class <- as.factor(df$Class)

################# Splitting into train and test###########
set.seed(123)

splitIndex <- createDataPartition(df$Class, p = 0.70, list = FALSE, times = 1)

train_data <- df[splitIndex,]
test_data <- df[-splitIndex,]

print(dim(train_data))
print(dim(test_data))

table(train_data$Class)
table(test_data$Class)

barplot(table(train_data$Class))

#######################v Random Oversampling ##################
minority_class <- ifelse(mean(train_data$Class == "Y") < 0.5, "Y", "N")
minority_data <- filter(train_data, Class == minority_class)

majority_count <- sum(train_data$Class != minority_class)
minority_count <- nrow(minority_data)
oversample_amount <- majority_count - minority_count

additional_samples <- minority_data[sample(nrow(minority_data), size = oversample_amount, replace = TRUE), ]

balanced_train_data <- rbind(train_data, additional_samples)
table(balanced_train_data$Class)


######################### PCA ############################
pc <- prcomp(balanced_train_data[, -73], center = TRUE, scale = TRUE)
summary(pc)

train_pca_os <- predict(pc, balanced_train_data)
train_pca_os <- data.frame(train_pca_os, balanced_train_data[73])
test_pca_os <- predict(pc, test_data)
test_pca_os <- data.frame(test_pca_os, test_data[73])


################### Classification ######################
# 1. LDA

set.seed(123)
ldaModel <- lda(Class~ ., data = train_pca_os[c(1:66, 73)]) 
ldaModel
pred <- predict(ldaModel, test_pca_os)
performance_measures_pca_os  <- confusionMatrix(data=pred$class, 
                                         reference = test_pca_os$Class)
performance_measures_pca_os


lda_probabilities <- pred$posterior[,2]
roc_lda <- roc(as.numeric(test_pca_os$Class)-1, lda_probabilities) 
auc_lda <- auc(roc_lda)
print(paste("AUC for LDA:", auc_lda))

# 2. Naive Bayes
nb_model <- naiveBayes(Class ~ ., data = train_pca_os[c(1:66, 73)])
nb_predictions <- predict(nb_model, test_pca_os)

conf_matrix_pca_os <- confusionMatrix(nb_predictions, test_pca_os$Class)

print(conf_matrix_pca_os)

nb_prob <- predict(nb_model, test_pca_os, type = "raw")

prob_positive_class <- nb_prob[,2]

actual_classes <- as.numeric(test_pca_os$Class) - 1
roc_result <- roc(actual_classes, prob_positive_class)
auc_value <- auc(roc_result)

print(auc_value)
# 3. GBM

set.seed(123)
ctrl <- trainControl(method = "CV",
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE,
                     savePredictions = TRUE)
gbmGrid <- expand.grid(interaction.depth = c(1, 3, 5, 7, 9),
                       n.trees = (2:10)*65,
                       shrinkage = c(.01),
                       n.minobsinnode = 3)

set.seed(123)
gbmFit <- caret::train(x = train_pca_os[, c(1:50), -ncol(train_pca_os)], 
                       y = train_pca_os$Class,
                       method = "gbm",
                       tuneGrid = gbmGrid,
                       metric = "ROC",
                       verbose = FALSE,
                       trControl = ctrl)
gbmFit
plot(gbmFit)

pred <- predict(gbmFit, test_pca_os)
cm_pca_os <- caret::confusionMatrix(pred, test_pca_os$Class)
cm_pca_os

gbm_prob <- predict(gbmFit, newdata = test_pca_os[, -ncol(train_pca_os)], type = "prob")
prob_positive_class <- gbm_prob[,2]

actual_classes <- as.numeric(test_pca_os$Class) - 1
roc_result <- roc(actual_classes, prob_positive_class)
auc_value <- auc(roc_result)
print(auc_value)

plot(roc_result, main="ROC Curve for GBM Model")

# 4.  XGBoost
xgb_control = trainControl(
  method = "cv", number = 10,
  summaryFunction = twoClassSummary,
  classProbs = TRUE,
  savePredictions = TRUE
)

xgbGrid <- expand.grid(
  nrounds = seq(from = 50, to = 300, by = 50),
  eta = c(0.05, 0.1, 0.3),
  max_depth = c(1, 2, 3),
  gamma = c(0, 1, 2),
  colsample_bytree = 1,
  min_child_weight = c(1, 3),
  subsample = c(0.5, 0.7)
)

set.seed(123)
xgbModel <- caret::train(x = train_pca_os[,c(1:50), -ncol(train_pca_os)], 
                         y = train_pca_os$Class,
                         method = "xgbTree",
                         tuneGrid = xgbGrid,
                         metric = "ROC",
                         verbose = FALSE,
                         trControl = xgb_control)
xgbModel
plot(xgbModel)

pred <- predict(xgbModel, test_pca_os)
cm_xg_pca_os <- caret::confusionMatrix(pred, test_pca_os$Class)
cm_xg_pca_os

xgb_prob <- predict(xgbModel, newdata = test_pca_os[, -ncol(train_pca_os)], type = "prob")

prob_positive_class <- xgb_prob[,2]
actual_classes <- as.numeric(test_pca_os$Class) - 1

roc_result <- roc(actual_classes, prob_positive_class)
auc_value <- auc(roc_result)
print(auc_value)
plot(roc_result, main="ROC Curve for XGBoost Model")


# 5. Logistic regression
train_pca_os$Class <- as.factor(train_pca_os$Class)
test_pca_os$Class <- as.factor(test_pca_os$Class)
logit_model <- glm(Class ~ ., data = train_pca_os, family = "binomial")

predictions_logit <- predict(logit_model, newdata = test_pca_os, type = "response")
predictions_class_logit <- ifelse(predictions_logit > 0.5, "Y", "N")

conf_matrix_logit_pca_os <- confusionMatrix(factor(predictions_class_logit, levels = levels(test_pca_os$Class)), test_pca_os$Class)
print(conf_matrix_logit_pca_os)

actual_classes <- ifelse(test_pca_os$Class == "Y", 1, 0)
actual_classes <- as.numeric(actual_classes)
roc_obj <- roc(actual_classes, predictions_logit)
plot(roc_obj, main="ROC Curve for Logistic Regression Model")

auc_value <- auc(roc_obj)
print(paste("AUC value:", auc_value))

# 6. Elastic Net Regularized Logistic Regression

library(glmnet)

x_train_pca <- model.matrix(Class ~ . - 1, data = train_pca_os)
y_train_pca <- ifelse(train_pca_os$Class == "Y", 1, 0)

x_test_pca <- model.matrix(Class ~ . - 1, data = test_pca_os)

set.seed(123)
cv_model_enet <- cv.glmnet(x_train_pca, y_train_pca, family = "binomial", alpha = 0.51) 

best_lambda <- cv_model_enet$lambda.min

predictions_prob_enet <- predict(cv_model_enet, newx = x_test_pca, s = best_lambda, type = "response")
predictions_class_enet <- ifelse(predictions_prob_enet > 0.51, "Y", "N")

conf_matrix_enet_pca_os <- confusionMatrix(factor(predictions_class_enet, levels = levels(test_pca_os$Class)), test_pca_os$Class)
print(conf_matrix_enet_pca_os)


actual_classes <- ifelse(test_pca_os$Class == "Y", 1, 0)
roc_result <- roc(actual_classes, predictions_prob_enet)

plot(roc_result, main="ROC Curve for Elastic Net Model")
auc_value <- auc(roc_result)
print(paste("AUC value:", auc_value))


################################ Oversampling Info Gain ########################
df <- read.csv("preprocessed_data.csv")
df$Class <- as.factor(df$Class)

################# Splitting into train and test###########
set.seed(123)

splitIndex <- createDataPartition(df$Class, p = 0.70, list = FALSE, times = 1)

train_data <- df[splitIndex,]
test_data <- df[-splitIndex,]

print(dim(train_data))
print(dim(test_data))

table(train_data$Class)
table(test_data$Class)

barplot(table(train_data$Class))

#######################v Random Oversampling ##################
minority_class <- ifelse(mean(train_data$Class == "Y") < 0.5, "Y", "N")
minority_data <- filter(train_data, Class == minority_class)

majority_count <- sum(train_data$Class != minority_class)
minority_count <- nrow(minority_data)
oversample_amount <- majority_count - minority_count

additional_samples <- minority_data[sample(nrow(minority_data), size = oversample_amount, replace = TRUE), ]

balanced_train_data <- rbind(train_data, additional_samples)
table(balanced_train_data$Class)


########### feature selection info gain ################
info.gain <- information.gain(Class ~ ., balanced_train_data)
info.gain <- cbind(rownames(info.gain), data.frame(info.gain, row.names=NULL))
names(info.gain) <- c("Attribute", "Info Gain")
sorted.info.gain <- info.gain[order(-info.gain$`Info Gain`), ]
sorted.info.gain

threshold <- 0.002
selected_attributes <- sorted.info.gain[sorted.info.gain$`Info Gain` > threshold, ]

selected_attributes_vector <- as.vector(selected_attributes$Attribute)

if(!"Class" %in% selected_attributes_vector) {
  selected_attributes_vector <- c(selected_attributes_vector, "Class")
}

train_ig_os <- balanced_train_data[, colnames(balanced_train_data) %in% selected_attributes_vector, drop = FALSE]
test_ig_os <- test_data[, colnames(balanced_train_data) %in% selected_attributes_vector, drop = FALSE]
dim(train_ig_os)
dim(test_ig_os)

####################### Classification #################################

# 1. LDA 

set.seed(123)
ldaModel <- lda(Class~ ., data = train_ig_os) 
ldaModel
pred <- predict(ldaModel, test_ig_os)
performance_measures_lda_ig_os  <- confusionMatrix(data=pred$class, 
                                         reference = test_ig_os$Class)
performance_measures_lda_ig_os

lda_probabilities <- pred$posterior[,2]
roc_lda <- roc(as.numeric(test_ig_os$Class)-1, lda_probabilities) 
auc_lda <- auc(roc_lda)
print(paste("AUC for LDA:", auc_lda))

# 2. Naive Bayesian
nb_model <- naiveBayes(Class ~ ., data = train_ig_os)
nb_predictions <- predict(nb_model, test_ig_os)

conf_matrix_nb_ig_os <- confusionMatrix(nb_predictions, test_ig_os$Class)
print(conf_matrix_nb_ig_os)

nb_prob <- predict(nb_model, train_ig_os, type = "raw")

prob_positive_class <- nb_prob[,2]

actual_classes <- as.numeric(train_ig_os$Class) - 1
roc_result <- roc(actual_classes, prob_positive_class)
auc_value <- auc(roc_result)
auc_value

# 3. GBM

set.seed(123)
ctrl <- trainControl(method = "CV",
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE,
                     savePredictions = TRUE)
gbmGrid <- expand.grid(interaction.depth = c(1, 3, 5, 7, 9),
                       n.trees = (2:10)*65,
                       shrinkage = c(.01),
                       n.minobsinnode = 3)

set.seed(123)
gbmFit <- caret::train(x = train_ig_os[, -39], 
                       y = train_ig_os$Class,
                       method = "gbm",
                       tuneGrid = gbmGrid,
                       metric = "ROC",
                       verbose = FALSE,
                       trControl = ctrl)
gbmFit
plot(gbmFit)

pred <- predict(gbmFit, test_ig_os)
cm_gbm_ig_os <- caret::confusionMatrix(pred, test_ig_os$Class)
cm_gbm_ig_os

gbm_probabilities <- predict(gbmFit, test_ig_os, type = "prob")[,2]
roc_gbm <- roc(response = test_ig_os$Class, predictor = gbm_probabilities)
auc_gbm <- auc(roc_gbm)
print(auc_gbm)


# 4. XGBoost
xgb_control = trainControl(
  method = "cv", number = 12,
  summaryFunction = twoClassSummary,
  classProbs = TRUE,
  savePredictions = TRUE
)

xgbGrid <- expand.grid(
  nrounds = seq(from = 50, to = 300, by = 50),
  eta = c(0.05, 0.1, 0.3),
  max_depth = c(1, 2, 3),
  gamma = c(0, 1, 2),
  colsample_bytree = 1,
  min_child_weight = c(1, 3),
  subsample = c(0.5, 0.7)
)

set.seed(123)
xgbModel <- caret::train(x = train_ig_os[, -39], 
                         y = train_ig_os$Class,
                         method = "xgbTree",
                         tuneGrid = xgbGrid,
                         metric = "ROC",
                         verbose = FALSE,
                         trControl = xgb_control)
xgbModel
plot(xgbModel)

pred <- predict(xgbModel, test_ig_os)
cm_xg_ig_os <- caret::confusionMatrix(pred, test_ig_os$Class)
cm_xg_ig_os

xgb_probabilities <- predict(xgbModel, test_ig_os, type = "prob")[,2]
roc_xgb <- roc(response = test_ig_os$Class, predictor = xgb_probabilities)
auc_xgb <- auc(roc_xgb)
print(auc_xgb)

# 5. Logistic regression 
train_ig_os$Class <- ifelse(train_ig_os$Class ==  "Y", 1, 0)
test_ig_os$Class <- ifelse(test_ig_os$Class ==  "Y", 1, 0)

train_ig_os$Class <- as.factor(train_ig_os$Class)
test_ig_os$Class <- as.factor(test_ig_os$Class)


logitModel <- glm(Class ~ ., data = train_ig_os, family = "binomial") 
options(scipen=999)
summary(logitModel)
logitModel.pred <- predict(logitModel, test_ig_os[, -41], type = "response")

data.frame(actual = test_ig_os$Class[1:40], predicted = logitModel.pred[1:40])

pred <- factor(ifelse(logitModel.pred >= 0.5, 1, 0))
pred
performance_measures_smote_ig  <- confusionMatrix(data=pred, 
                                                  reference = as.factor(test_ig_os$Class))
performance_measures_smote_ig
logitModel.probabilities <- predict(logitModel, test_ig_os[, -41], type = "response")
roc_logit <- roc(response = as.numeric(test_ig_os$Class), predictor = logitModel.probabilities)
auc_logit <- auc(roc_logit)
print(auc_logit)

# 6. Elastic Net Regularized Logistic Regression
x_train <- model.matrix(Class ~ . - 1, data = train_ig_os) 
y_train <- as.numeric(train_ig_os$Class) - 1  

x_test <- model.matrix(Class ~ . - 1, data = test_ig_os)

set.seed(123) 
cv_model <- cv.glmnet(x_train, y_train, family = "binomial", alpha = 0.51578)

best_lambda <- cv_model$lambda.min

predictions_prob <- predict(cv_model, newx = x_test, s = best_lambda, type = "response")

predictions <- ifelse(predictions_prob > 0.51578, 1, 0)
predictions_factor_en_ig_os <- factor(predictions, levels = c(0, 1), labels = levels(test_ig_os$Class))

confusionMatrix(predictions_factor_en_ig_os, test_ig_os$Class)


actual_classes <- as.numeric(as.character(test_ig_os$Class)) - 1
roc_obj <- roc(actual_classes, predictions_prob)

plot(roc_obj, main = "ROC Curve for Elastic Net Model")
auc_value <- auc(roc_obj)
print(paste("AUC value:", auc_value))

################### Oversampling complete, will implement SMOTE now ############

#### Smote and PCA #### 


df <- read.csv("preprocessed_data.csv")
df$Class <- as.factor(df$Class)

################# Splitting into train and test###########
set.seed(123)

splitIndex <- createDataPartition(df$Class, p = 0.70, list = FALSE, times = 1)

train_data <- df[splitIndex,]
test_data <- df[-splitIndex,]

print(dim(train_data))
print(dim(test_data))

table(train_data$Class)
table(test_data$Class)

barplot(table(train_data$Class))

########################### Balancing train data using SMOTE ####################
balanced_recipe <- recipe(Class ~ ., data = train_data) %>%
  step_smote(Class) %>%
  prep()
balanced_train_data <- bake(balanced_recipe, new_data = NULL)
table(balanced_train_data$Class)
barplot(table(balanced_train_data$Class))

######################### PCA ############################

pc <- prcomp(balanced_train_data[, -73], center = TRUE, scale = TRUE)
summary(pc)
train_pca <- predict(pc, balanced_train_data)
train_pca <- data.frame(train_pca, balanced_train_data[73])
test_pca <- predict(pc, test_data)
test_pca <- data.frame(test_pca, test_data[73])

############# Classification ################
# 1. lda
set.seed(123)
ldaModel <- lda(Class~ ., data = train_pca[c(1:60, 73)]) 
ldaModel
pred <- predict(ldaModel, test_pca)
performance_measures_lda_smote_pca  <- confusionMatrix(data=pred$class, 
                                         reference = test_pca$Class)
performance_measures_lda_smote_pca

lda_probabilities <- pred$posterior[,2]
roc_lda <- roc(as.numeric(test_pca$Class)-1, lda_probabilities) 
auc_lda <- auc(roc_lda)
print(paste("AUC for LDA:", auc_lda))

# 2. Naive Bayes
nb_model <- naiveBayes(Class ~ ., data = train_pca[c(1:60, 73)])
nb_predictions <- predict(nb_model, test_pca)

conf_matrix_nb_pca_smote <- confusionMatrix(nb_predictions, test_pca$Class)
print(conf_matrix_nb_pca_smote)

nb_probabilities <- predict(nb_model, test_pca, type = "raw")

test_pca$Class <- factor(test_pca$Class, levels = c("N", "Y"))
roc_curve_nb <- roc(response = test_pca$Class, predictor = nb_probabilities[, "Y"])
auc_nb <- auc(roc_curve_nb)

print(auc_nb)


#3. GBM

set.seed(123)
ctrl <- trainControl(method = "CV",
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE,
                     savePredictions = TRUE)
gbmGrid <- expand.grid(interaction.depth = c(1, 3, 5, 7, 9),
                       n.trees = (2:10)*65,
                       shrinkage = c(.01),
                       n.minobsinnode = 3)

set.seed(123)
gbmFit <- caret::train(x = train_pca[, c(1:55), -ncol(train_pca)], 
                       y = train_pca$Class,
                       method = "gbm",
                       tuneGrid = gbmGrid,
                       metric = "ROC",
                       verbose = FALSE,
                       trControl = ctrl)
gbmFit
plot(gbmFit)

pred <- predict(gbmFit, test_pca)
cm_gb_smote_pca <- caret::confusionMatrix(pred, test_pca$Class)
cm_gb_smote_pca

gbm_probabilities <- predict(gbmFit, test_pca, type = "prob")[,2]
roc_gbm <- roc(response = test_pca$Class, predictor = gbm_probabilities)
auc_gbm <- auc(roc_gbm)
print(auc_gbm)


#4. XGBoost
xgb_control = trainControl(
  method = "cv", number = 10,
  summaryFunction = twoClassSummary,
  classProbs = TRUE,
  savePredictions = TRUE
)

xgbGrid <- expand.grid(
  nrounds = seq(from = 50, to = 300, by = 50),
  eta = c(0.05, 0.1, 0.3),
  max_depth = c(1, 2, 3),
  gamma = c(0, 1, 2),
  colsample_bytree = 1,
  min_child_weight = c(1, 3),
  subsample = c(0.5, 0.7)
)

set.seed(123)
xgbModel <- caret::train(x = train_pca[, c(1:55), -ncol(train_pca)], 
                         y = train_pca$Class,
                         method = "xgbTree",
                         tuneGrid = xgbGrid,
                         metric = "ROC",
                         verbose = FALSE,
                         trControl = xgb_control)
xgbModel
plot(xgbModel)

pred <- predict(xgbModel, test_pca)
cm_xg_smote_pca <- caret::confusionMatrix(pred, test_pca$Class)
cm_xg_smote_pca

xgb_probabilities <- predict(xgbModel, test_pca, type = "prob")[,2]
roc_xgb <- roc(response = test_pca$Class, predictor = xgb_probabilities)
auc_xgb <- auc(roc_xgb)
print(auc_xgb)


# 5. Logistic Regression  
train_pca$Class <- ifelse(train_pca$Class ==  "Y", 1, 0)
test_pca$Class <- ifelse(test_pca$Class ==  "Y", 1, 0)

train_pca$Class <- as.factor(train_pca$Class)
test_pca$Class <- as.factor(test_pca$Class)

logitModel <- glm(Class ~ ., data = train_pca, family = "binomial") 
options(scipen=999)
summary(logitModel)

logitModel.pred <- predict(logitModel, test_pca[, -73], type = "response")

data.frame(actual = test_pca$Class[1:72], predicted = logitModel.pred[1:72])

pred <- factor(ifelse(logitModel.pred >= 0.5, 1, 0))
pred
performance_measure_lr_smote_pca  <- confusionMatrix(data=pred, 
                                         reference = as.factor(test_pca$Class))
performance_measure_lr_smote_pca

logitModel.probabilities <- predict(logitModel, test_pca[, -73], type = "response")
roc_logit <- roc(response = as.numeric(test_pca$Class), predictor = logitModel.probabilities)
auc_logit <- auc(roc_logit)
print(auc_logit)



# 6. Elastic Net Regularized Logistic Regression
x_train <- model.matrix(Class ~ . -1, data = train_pca) 
y_train <- as.numeric(train_pca$Class)   
x_test <- model.matrix(Class ~ . -1, data = test_pca)


set.seed(123)
cv_fit <- cv.glmnet(x_train, y_train, alpha=0.52, family="binomial")

best_lambda <- cv_fit$lambda.min
predictions_prob_pca_smote <- predict(cv_fit, newx=x_test, s=best_lambda, type="response")

predictions_pca_smote <- ifelse(predictions_prob_pca_smote > 0.52, "1", "0")

predictions_pca_smote <- factor(predictions_pca_smote, levels=c("0", "1"), labels=levels(train_pca$Class))

conf_matrix_pca_smote <- confusionMatrix(predictions_pca_smote, test_pca$Class)
print(conf_matrix_pca_smote)


########## Smote info Gain ###########

df <- read.csv("selected_dataset.csv")

df$Class <- as.factor(df$Class)


################# Splitting into train and test###########
set.seed(123)

splitIndex <- createDataPartition(df$Class, p = 0.70, list = FALSE, times = 1)

train_data <- df[splitIndex,]
test_data <- df[-splitIndex,]

print(dim(train_data))
print(dim(test_data))

table(train_data$Class)
table(test_data$Class)

barplot(table(train_data$Class))

#################### SMote for Balancing ################

balanced_recipe <- recipe(Class ~ ., data = train_data) %>%
  step_smote(Class) %>%
  prep()

balanced_train_data <- bake(balanced_recipe, new_data = NULL)

table(balanced_train_data$Class)

################### Info Gain ######################

library(FSelector)

# information gain
info.gain <- information.gain(Class ~ ., df)
info.gain <- cbind(rownames(info.gain), data.frame(info.gain, row.names=NULL))
names(info.gain) <- c("Attribute", "Info Gain")
sorted.info.gain <- info.gain[order(-info.gain$`Info Gain`), ]
sorted.info.gain

threshold <- 0.002
selected_attributes <- sorted.info.gain[sorted.info.gain$`Info Gain` > threshold, ]


selected_attributes_vector <- as.vector(selected_attributes$Attribute)
if(!"Class" %in% selected_attributes_vector) {
  selected_attributes_vector <- c(selected_attributes_vector, "Class")
}

train_ig <- balanced_train_data[, colnames(balanced_train_data) %in% selected_attributes_vector, drop = FALSE]
test_ig <- test_data[, colnames(test_data) %in% selected_attributes_vector, drop = FALSE]

table(train_ig$Class)
table(test_ig$Class)

##########################. Classifiers ###########################

# 1. lda
library(MASS)
set.seed(123)
ldaModel <- lda(Class~ ., data = train_ig) 
ldaModel
pred <- predict(ldaModel, test_ig)
performance_measures_smote_ig  <- confusionMatrix(data=pred$class, 
                                         reference = test_ig$Class)
performance_measures_smote_ig

lda_probabilities <- pred$posterior[,2]
roc_lda <- roc(as.numeric(test_ig$Class)-1, lda_probabilities) 
auc_lda <- auc(roc_lda)
print(paste("AUC for LDA:", auc_lda))

# 2. Naive Bayesian
nb_model <- naiveBayes(Class ~ ., data = train_ig)
nb_predictions <- predict(nb_model, test_ig)

conf_matrix_smote_ig <- confusionMatrix(nb_predictions, test_ig$Class)

print(conf_matrix_smote_ig)

nb_probabilities <- predict(nb_model, test_ig, type = "raw")
test_ig$Class <- factor(test_ig$Class, levels = c("N", "Y"))
roc_curve_nb <- roc(response = test_ig$Class, predictor = nb_probabilities[, "Y"])

auc_value <- auc(roc_curve_nb)
print(auc_value)


# 3. GBM

set.seed(123)
ctrl <- trainControl(method = "CV",
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE,
                     savePredictions = TRUE)
gbmGrid <- expand.grid(interaction.depth = c(1, 3, 5, 7, 9),
                       n.trees = (2:10)*65,
                       shrinkage = c(.01),
                       n.minobsinnode = 3)

set.seed(123)
gbmFit <- caret::train(x = train_ig[, -27], 
                       y = train_ig$Class,
                       method = "gbm",
                       tuneGrid = gbmGrid,
                       metric = "ROC",
                       verbose = FALSE,
                       trControl = ctrl)
gbmFit
plot(gbmFit)

pred <- predict(gbmFit, test_ig)
cm_smote_ig <- caret::confusionMatrix(pred, test_ig$Class)
cm_smote_ig

gbm_probabilities <- predict(gbmFit, test_ig, type = "prob")[,2]
roc_gbm <- roc(response = test_ig$Class, predictor = gbm_probabilities)
auc_gbm <- auc(roc_gbm)
print(auc_gbm)

# 4. XGBoost
xgb_control = trainControl(
  method = "cv", number = 10,
  summaryFunction = twoClassSummary,
  classProbs = TRUE,
  savePredictions = TRUE
)

xgbGrid <- expand.grid(
  nrounds = seq(from = 50, to = 300, by = 50),
  eta = c(0.05, 0.1, 0.3),
  max_depth = c(1, 2, 3),
  gamma = c(0, 1, 2),
  colsample_bytree = 1,
  min_child_weight = c(1, 3),
  subsample = c(0.5, 0.7)
)

set.seed(123)
xgbModel <- caret::train(x = train_ig[, -27], 
                         y = train_ig$Class,
                         method = "xgbTree",
                         tuneGrid = xgbGrid,
                         metric = "ROC",
                         verbose = FALSE,
                         trControl = xgb_control)
xgbModel
plot(xgbModel)

pred <- predict(xgbModel, test_ig)
cm_smote_ig <- caret::confusionMatrix(pred, test_ig$Class)
cm_smote_ig

xgb_probabilities <- predict(xgbModel, test_ig, type = "prob")[,2]
roc_xgb <- roc(response = test_ig$Class, predictor = xgb_probabilities)
auc_xgb <- auc(roc_xgb)
print(auc_xgb)

# 5. Logistic regression 
train_ig$Class <- ifelse(train_ig$Class ==  "Y", 1, 0)
test_ig$Class <- ifelse(test_ig$Class ==  "Y", 1, 0)

train_ig$Class <- as.factor(train_ig$Class)
test_ig$Class <- as.factor(test_ig$Class)


logitModel <- glm(Class ~ ., data = train_ig, family = "binomial") 
options(scipen=999)
summary(logitModel)
logitModel.pred <- predict(logitModel, test_ig[, -27], type = "response")

data.frame(actual = test_ig$Class[1:26], predicted = logitModel.pred[1:26])

pred <- factor(ifelse(logitModel.pred >= 0.5, 1, 0))
pred
performance_measures_smote_ig  <- confusionMatrix(data=pred, 
                                         reference = as.factor(test_ig$Class))
performance_measures_smote_ig
logitModel.probabilities <- predict(logitModel, test_ig[, -73], type = "response")
roc_logit <- roc(response = as.numeric(test_ig$Class), predictor = logitModel.probabilities)
auc_logit <- auc(roc_logit)
print(auc_logit)

# 6. Elastic Net Regularized Logistic Regression

x_train <- model.matrix(Class ~ . -1, data = train_ig) 
y_train <- as.numeric(train_ig$Class)   
x_test <- model.matrix(Class ~ . -1, data = test_ig)


set.seed(123)
cv_fit <- cv.glmnet(x_train, y_train, alpha=0.5, family="binomial") 

best_lambda <- cv_fit$lambda.min

predictions_prob_lasso <- predict(cv_fit, newx=x_test, s=best_lambda, type="response")

predictions_lasso <- ifelse(predictions_prob_lasso > 0.5, "1", "0")

predictions_lasso <- factor(predictions_lasso, levels=c("0", "1"), labels=levels(train_ig$Class))

conf_matrix_lasso_smote_ig <- confusionMatrix(predictions_lasso, test_ig$Class)
print(conf_matrix_lasso_smote_ig)

predictions_prob_lasso_ordered <- ifelse(predictions_prob_lasso > 0.51, 1, 0)
roc_lasso_en <- roc(response = as.numeric(test_ig$Class), predictor = predictions_prob_lasso_ordered)
auc_lasso_en <- auc(roc_lasso_en)
print(auc_lasso_en)


#### SMOTE LASSO ###

df <- read.csv("selected_dataset.csv")
df$Class <- as.factor(df$Class)


################# Splitting into train and test###########
set.seed(123)

splitIndex <- createDataPartition(df$Class, p = 0.70, list = FALSE, times = 1)

train_data <- df[splitIndex,]
test_data <- df[-splitIndex,]

print(dim(train_data))
print(dim(test_data))

table(train_data$Class)
table(test_data$Class)

barplot(table(train_data$Class))


############################## SMOTE for data balancing #######################
balanced_recipe <- recipe(Class ~ ., data = train_data) %>%
  step_smote(Class) %>%
  prep()

balanced_train_data <- bake(balanced_recipe, new_data = NULL)
table(balanced_train_data$Class)

################## feature seelction using LASSO ############################
x <- model.matrix(Class ~ . - 1, data=train_data) 
y <- as.numeric(train_data$Class) - 1  

set.seed(123)  
cv_fit <- cv.glmnet(x, y, family="binomial", alpha=1) 

optimal_coefs <- coef(cv_fit, s = "lambda.min")

coefs_dense <- as.vector(optimal_coefs)

coefs_dense <- coefs_dense[-1]  

feature_names <- rownames(optimal_coefs)[-1]

selected_features_lasso <- feature_names[coefs_dense != 0]

#print(selected_features)

selected_features_with_target <- c(selected_features_lasso, "Class")

train_data_lasso <- balanced_train_data[, selected_features_with_target, drop = FALSE]
test_data_lasso <- test_data[, selected_features_with_target, drop = FALSE]


dim(train_data_lasso)
dim(test_data_lasso)

############################## Classification ########################

# 1. lda
set.seed(123)
ldaModel <- lda(Class~ ., data = train_data_lasso) 
ldaModel
pred <- predict(ldaModel, test_data_lasso)
performance_measures_smote_lasso  <- confusionMatrix(data=pred$class, 
                                         reference = test_data_lasso$Class)
performance_measures_smote_lasso

lda_probabilities <- pred$posterior[,2]
roc_lda <- roc(as.numeric(test_data_lasso$Class)-1, lda_probabilities) 
auc_lda <- auc(roc_lda)
print(paste("AUC for LDA:", auc_lda))


# 2. Naive Bayesian with LAsso 
nb_model <- naiveBayes(Class ~ ., data = train_data_lasso)
nb_predictions <- predict(nb_model, test_data_lasso)

conf_matrix_smote_lasso <- confusionMatrix(nb_predictions, test_data_lasso$Class)
print(conf_matrix_smote_lasso)

nb_probabilities <- predict(nb_model, test_data_lasso, type = "raw")

roc_curve <- roc(as.numeric(as.character(test_data_lasso$Class)), nb_probabilities[,2])
auc_value <- auc(roc_curve)
print(auc_value)

# 3. GBM

set.seed(123)
ctrl <- trainControl(method = "CV",
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE,
                     savePredictions = TRUE)
gbmGrid <- expand.grid(interaction.depth = c(1, 3, 5, 7, 9),
                       n.trees = (2:10)*65,
                       shrinkage = c(.01),
                       n.minobsinnode = 3)

set.seed(123)
gbmFit <- caret::train(x = train_data_lasso[, -36], 
                       y = train_data_lasso$Class,
                       method = "gbm",
                       tuneGrid = gbmGrid,
                       metric = "ROC",
                       verbose = FALSE,
                       trControl = ctrl)
gbmFit
plot(gbmFit)

pred <- predict(gbmFit, test_data_lasso)
cm_smote_lasso <- caret::confusionMatrix(pred, test_data_lasso$Class)
cm_smote_lasso

gbm_probabilities <- predict(gbmFit, test_data_lasso, type = "prob")[,2]
roc_gbm <- roc(response = test_data_lasso$Class, predictor = gbm_probabilities)
auc_gbm <- auc(roc_gbm)
print(auc_gbm)

# 4. XGBoost


xgb_control = trainControl(
  method = "cv", number = 10,
  summaryFunction = twoClassSummary,
  classProbs = TRUE,
  savePredictions = TRUE
)

xgbGrid <- expand.grid(
  nrounds = seq(from = 50, to = 300, by = 50),
  eta = c(0.05, 0.1, 0.3),
  max_depth = c(1, 2, 3),
  gamma = c(0, 1, 2),
  colsample_bytree = 1,
  min_child_weight = c(1, 3),
  subsample = c(0.5, 0.7)
)

set.seed(123)
xgbModel <- caret::train(x = train_data_lasso[, -36], 
                         y = train_data_lasso$Class,
                         method = "xgbTree",
                         tuneGrid = xgbGrid,
                         metric = "ROC",
                         verbose = FALSE,
                         trControl = xgb_control)
xgbModel
plot(xgbModel)

pred <- predict(xgbModel, test_data_lasso)
cm_smote_lasso <- caret::confusionMatrix(pred, test_data_lasso$Class)
cm_smote_lasso

xgb_probabilities <- predict(xgbModel, test_data_lasso, type = "prob")[,2]
roc_xgb <- roc(response = test_data_lasso$Class, predictor = xgb_probabilities)
auc_xgb <- auc(roc_xgb)
print(auc_xgb)

# 5. Logistic regression 
train_data_lasso$Class <- ifelse(train_data_lasso$Class ==  "Y", 1, 0)
test_data_lasso$Class <- ifelse(test_data_lasso$Class ==  "Y", 1, 0)

train_data_lasso$Class <- as.factor(train_data_lasso$Class)
test_data_lasso$Class <- as.factor(test_data_lasso$Class)

logitModel <- glm(Class ~ ., data = train_data_lasso, family = "binomial") 
options(scipen=999)
summary(logitModel)

logitModel.pred <- predict(logitModel, test_data_lasso[, -73], type = "response")

data.frame(actual = test_data_lasso$Class[1:30], predicted = logitModel.pred[1:30])

pred <- factor(ifelse(logitModel.pred >= 0.5, 1, 0))
pred
performance_measures_smote_lasso  <- confusionMatrix(data=pred, 
                                         reference = as.factor(test_data_lasso$Class))
performance_measures_smote_lasso

logitModel.probabilities <- predict(logitModel, test_data_lasso[, -73], type = "response")

roc_logit <- roc(response = as.numeric(test_data_lasso$Class), predictor = logitModel.probabilities)
auc_logit <- auc(roc_logit)
print(auc_logit)


# 6. Elastic Net Regularized Logistic Regression
x_train <- model.matrix(Class ~ . -1, data = train_data_lasso)
y_train <- as.numeric(train_data_lasso$Class)   
x_test <- model.matrix(Class ~ . -1, data = test_data_lasso)

set.seed(123)
cv_fit <- cv.glmnet(x_train, y_train, alpha=0.51, family="binomial") 

best_lambda <- cv_fit$lambda.min

predictions_prob_lasso <- predict(cv_fit, newx=x_test, s=best_lambda, type="response")

predictions_lasso <- ifelse(predictions_prob_lasso > 0.51, "1", "0")

predictions_lasso <- factor(predictions_lasso, levels=c("0", "1"), labels=levels(train_data_lasso$Class))

conf_matrix_lasso_smote_lasso <- confusionMatrix(predictions_lasso, test_data_lasso$Class)
print(conf_matrix_lasso_smote_lasso)

predictions_prob_lasso_ordered <- ifelse(predictions_prob_lasso > 0.51, 1, 0)

roc_lasso_en <- roc(response = as.numeric(test_data_lasso$Class), predictor = predictions_prob_lasso_ordered)
auc_lasso_en <- auc(roc_lasso_en)
print(auc_lasso_en)


