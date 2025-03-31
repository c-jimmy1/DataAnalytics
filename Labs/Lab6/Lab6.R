################################
# SVM Regression Lab on NY House Dataset
# Predicting Price from Square Footage
################################

# Load required libraries
library(readr)
library(ggplot2)
library(e1071)
library(caret)
library(cv)  # used for cross-validation functions if needed

## 1. Load and Explore the Data
NY_House_Dataset <- read_csv("NY-House-Dataset.csv")
dataset <- NY_House_Dataset

# Check column names and inspect data
names(dataset)
str(dataset)

# Plot raw data and log-transformed data for visualization
ggplot(dataset, aes(x = PROPERTYSQFT, y = PRICE)) +
  geom_point() +
  ggtitle("Price vs. Square Footage (Raw)") +
  xlab("Property Square Footage") + ylab("Price")

ggplot(dataset, aes(x = log10(PROPERTYSQFT), y = log10(PRICE))) +
  geom_point() +
  ggtitle("Log10(Price) vs. Log10(Square Footage)") +
  xlab("log10(Property Square Footage)") + ylab("log10(Price)")

dataset.sub0 <- dataset[-which(dataset$PROPERTYSQFT == 2184.207862),]

## 3. Build Regression Models Using log10 transformation
# 3A. Linear Regression Model
lin.mod <- lm(log10(PRICE) ~ log10(PROPERTYSQFT), data = dataset.sub0)
summary(lin.mod)

# Plot the fitted linear regression line on the log-transformed data
ggplot(dataset.sub0, aes(x = log10(PROPERTYSQFT), y = log10(PRICE))) +
  geom_point() +
  stat_smooth(method = "lm", col = "blue") +
  ggtitle("Linear Regression on Log Transformed Data")

# 3B. SVM Regression Model - Linear Kernel
svm.mod.linear <- svm(log10(PRICE) ~ log10(PROPERTYSQFT), data = dataset.sub0, kernel = "linear")
summary(svm.mod.linear)

# Predict on the same dataset (or you could split train/test)
svm.pred.linear <- predict(svm.mod.linear, dataset.sub0)

# Plot SVM (linear) fitted curve
ggplot(dataset.sub0, aes(x = log10(PROPERTYSQFT), y = log10(PRICE))) +
  geom_point() +
  geom_line(aes(y = svm.pred.linear), col = "green") +
  ggtitle("SVM Regression (Linear Kernel)")

# 3C. SVM Regression Model - Radial Kernel
svm.mod.radial <- svm(log10(PRICE) ~ log10(PROPERTYSQFT), data = dataset.sub0, kernel = "radial")
summary(svm.mod.radial)

svm.pred.radial <- predict(svm.mod.radial, dataset.sub0)

# Plot SVM (radial) fitted curve
ggplot(dataset.sub0, aes(x = log10(PROPERTYSQFT), y = log10(PRICE))) +
  geom_point() +
  geom_line(aes(y = svm.pred.radial), col = "red") +
  ggtitle("SVM Regression (Radial Kernel)")

tuned.svm <- tune.svm(log10(PRICE) ~ log10(PROPERTYSQFT), data = dataset.sub0, 
                      kernel = "radial", 
                      gamma = 10^seq(-3, 2, 1), 
                      cost = 10^seq(-3, 2, 1), 
                      tune.control = tune.control(cross = 5))
print(tuned.svm)
# Update the model using the best gamma and cost:
svm.mod.radial.opt <- svm(log10(PRICE) ~ log10(PROPERTYSQFT), data = dataset.sub0, 
                          kernel = "radial", gamma = tuned.svm$best.parameters$gamma, 
                          cost = tuned.svm$best.parameters$cost)
svm.pred.radial.opt <- predict(svm.mod.radial.opt, dataset.sub0)

## 4. Split Data for Evaluation and Calculate Error Metrics
set.seed(123)
train.indexes <- sample(nrow(dataset.sub0), 0.75 * nrow(dataset.sub0))
train <- dataset.sub0[train.indexes, ]
test  <- dataset.sub0[-train.indexes, ]

# Evaluate using each model on the test set

# A. Linear Model
lm.mod <- lm(log10(PRICE) ~ log10(PROPERTYSQFT), data = train)
lm.pred <- predict(lm.mod, newdata = test)
err.lm <- lm.pred - log10(test$PRICE)
mae.lm <- mean(abs(err.lm))
mse.lm <- mean(err.lm^2)
rmse.lm <- sqrt(mse.lm)

# B. SVM with Linear Kernel
svm.lin.mod <- svm(log10(PRICE) ~ log10(PROPERTYSQFT), data = train, kernel = "linear")
svm.lin.pred <- predict(svm.lin.mod, newdata = test)
err.svm.lin <- svm.lin.pred - log10(test$PRICE)
mae.svm.lin <- mean(abs(err.svm.lin))
mse.svm.lin <- mean(err.svm.lin^2)
rmse.svm.lin <- sqrt(mse.svm.lin)

# C. SVM with Radial Kernel (using optimized parameters if desired)
svm.rad.mod <- svm(log10(PRICE) ~ log10(PROPERTYSQFT), data = train, 
                   kernel = "radial", gamma = tuned.svm$best.parameters$gamma, 
                   cost = tuned.svm$best.parameters$cost)
svm.rad.pred <- predict(svm.rad.mod, newdata = test)
err.svm.rad <- svm.rad.pred - log10(test$PRICE)
mae.svm.rad <- mean(abs(err.svm.rad))
mse.svm.rad <- mean(err.svm.rad^2)
rmse.svm.rad <- sqrt(mse.svm.rad)

# Print the error metrics for each model
cat("Linear Model Errors:\n")
cat("MAE =", mae.lm, "\nMSE =", mse.lm, "\nRMSE =", rmse.lm, "\n\n")

cat("SVM Linear Kernel Errors:\n")
cat("MAE =", mae.svm.lin, "\nMSE =", mse.svm.lin, "\nRMSE =", rmse.svm.lin, "\n\n")

cat("SVM Radial Kernel Errors:\n")
cat("MAE =", mae.svm.rad, "\nMSE =", mse.svm.rad, "\nRMSE =", rmse.svm.rad, "\n\n")

## 5. Monte Carlo Cross Validation to Estimate Average Errors

# Define a function to calculate errors from a given model and test set
calc_errors <- function(model, test_data, response_col) {
  pred <- predict(model, newdata = test_data)
  err <- pred - test_data[[response_col]]
  mae <- mean(abs(err))
  mse <- mean(err^2)
  rmse <- sqrt(mse)
  return(c(MAE = mae, MSE = mse, RMSE = rmse))
}

# Set number of iterations for CV
k_iter <- 100
errors_lm   <- matrix(NA, nrow = k_iter, ncol = 3)
errors_svmL <- matrix(NA, nrow = k_iter, ncol = 3)
errors_svmR <- matrix(NA, nrow = k_iter, ncol = 3)

for(i in 1:k_iter) {
  # Random split
  idx <- sample(nrow(dataset.sub0), 0.75 * nrow(dataset.sub0))
  train_cv <- dataset.sub0[idx, ]
  test_cv  <- dataset.sub0[-idx, ]
  
  # Train models on the training set
  lm_cv <- lm(log10(PRICE) ~ log10(PROPERTYSQFT), data = train_cv)
  svm_lin_cv <- svm(log10(PRICE) ~ log10(PROPERTYSQFT), data = train_cv, kernel = "linear")
  svm_rad_cv <- svm(log10(PRICE) ~ log10(PROPERTYSQFT), data = train_cv, 
                    kernel = "radial", gamma = tuned.svm$best.parameters$gamma, 
                    cost = tuned.svm$best.parameters$cost)
  
  # Evaluate on test set (use log10(PRICE) as response)
  errors_lm[i, ]   <- calc_errors(lm_cv, test_cv, "log10(PRICE)" = predict(lm_cv, newdata = test_cv) - log10(test_cv$PRICE)) # not used directly
  # Alternatively, calculate errors manually:
  lm_pred_cv <- predict(lm_cv, newdata = test_cv)
  err <- lm_pred_cv - log10(test_cv$PRICE)
  errors_lm[i, ] <- c(mean(abs(err)), mean(err^2), sqrt(mean(err^2)))
  
  svm_lin_pred_cv <- predict(svm_lin_cv, newdata = test_cv)
  err_lin <- svm_lin_pred_cv - log10(test_cv$PRICE)
  errors_svmL[i, ] <- c(mean(abs(err_lin)), mean(err_lin^2), sqrt(mean(err_lin^2)))
  
  svm_rad_pred_cv <- predict(svm_rad_cv, newdata = test_cv)
  err_rad <- svm_rad_pred_cv - log10(test_cv$PRICE)
  errors_svmR[i, ] <- c(mean(abs(err_rad)), mean(err_rad^2), sqrt(mean(err_rad^2)))
}

# Compute average error metrics over all iterations 
avg_errors_lm   <- colMeans(errors_lm)
avg_errors_svmL <- colMeans(errors_svmL)
avg_errors_svmR <- colMeans(errors_svmR)

cat("Monte Carlo CV - Average Error Metrics:\n")
cat("Linear Model: MAE =", avg_errors_lm[1], " MSE =", avg_errors_lm[2], " RMSE =", avg_errors_lm[3], "\n")
cat("SVM Linear Kernel: MAE =", avg_errors_svmL[1], " MSE =", avg_errors_svmL[2], " RMSE =", avg_errors_svmL[3], "\n")
cat("SVM Radial Kernel: MAE =", avg_errors_svmR[1], " MSE =", avg_errors_svmR[2], " RMSE =", avg_errors_svmR[3], "\n")
