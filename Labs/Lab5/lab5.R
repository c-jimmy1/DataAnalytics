library(readr)
library(ggplot2)
library(e1071)
library(caret)

# Read wine data; adjust file path if needed.
wine <- read.csv("wine.data", header = FALSE)
colnames(wine) <- c("Type", "Alcohol", "MalicAcid", "Ash", "Alcalinity", 
                    "Magnesium", "TotalPhenols", "Flavanoids", "NonflavanoidPhenols", 
                    "Proanthocyanins", "ColorIntensity", "Hue", "OD280_OD315", "Proline")

# Choose a subset of features (example: Alcohol, Flavanoids, ColorIntensity, and Hue)
features <- c("Alcohol", "Flavanoids", "ColorIntensity", "Hue")
wine_subset <- wine[, c("Type", features)]
wine_subset$Type <- as.factor(wine_subset$Type)

# Split the data into training and test sets.
set.seed(123)
trainIndex <- createDataPartition(wine_subset$Type, p = 0.75, list = FALSE)
wineTrain <- wine_subset[trainIndex, ]
wineTest  <- wine_subset[-trainIndex, ]

# Tune SVM with linear kernel
tune.out.linear <- tune.svm(Type ~ ., data = wineTrain, kernel = "linear", 
                            cost = 2^(seq(-5, 5, 2)))
best_linear <- tune.out.linear$best.model

# Predict on the test set and view performance
pred_linear <- predict(best_linear, newdata = wineTest)
cm_linear <- confusionMatrix(pred_linear, wineTest$Type)
print(cm_linear)

# Tune SVM with radial kernel
tune.out.radial <- tune.svm(Type ~ ., data = wineTrain, kernel = "radial", 
                            cost = 2^(seq(-5, 5, 2)), gamma = 2^(seq(-5, 5, 2)))
best_radial <- tune.out.radial$best.model

# Predict on the test set and view performance
pred_radial <- predict(best_radial, newdata = wineTest)
cm_radial <- confusionMatrix(pred_radial, wineTest$Type)
print(cm_radial)

# Train kNN classifier using 10-fold cross-validation
set.seed(123)
knnFit <- train(Type ~ ., data = wineTrain, method = "knn",
                tuneLength = 10, trControl = trainControl(method = "cv"))
pred_knn <- predict(knnFit, newdata = wineTest)
cm_knn <- confusionMatrix(pred_knn, wineTest$Type)
print(cm_knn)

# Read NY housing data
NY_data <- read_csv("NY-House-Dataset.csv")

# Optionally inspect the data
str(NY_data)

# Split the data into training and test sets.
set.seed(123)
trainIndex <- createDataPartition(NY_data$PRICE, p = 0.75, list = FALSE)
NY_train <- NY_data[trainIndex, ]
NY_test  <- NY_data[-trainIndex, ]

# Train SVM regression model
svr_model <- svm(PRICE ~ PROPERTYSQFT, data = NY_train)
svr_pred  <- predict(svr_model, newdata = NY_test)

# Plot predicted vs. real price
ggplot(data.frame(Real = NY_test$PRICE, Predicted = svr_pred), aes(x = Real, y = Predicted)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE, color = "blue") +
  ggtitle("SVM Regression: Predicted vs. Real Price") +
  xlab("Real Price") +
  ylab("Predicted Price") +
  geom_abline(intercept = 0, slope = 1, col = "red", linetype = "dashed")

# Train linear regression model
lm_model <- lm(PRICE ~ PROPERTYSQFT, data = NY_train)
lm_pred  <- predict(lm_model, newdata = NY_test)

# Plot predicted vs. real price for the linear model
ggplot(data.frame(Real = NY_test$PRICE, Predicted = lm_pred), aes(x = Real, y = Predicted)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE, color = "blue") +
  ggtitle("Linear Regression: Predicted vs. Real Price") +
  xlab("Real Price") +
  ylab("Predicted Price") +
  geom_abline(intercept = 0, slope = 1, col = "red", linetype = "dashed")

# Calculate residuals
svr_resid <- NY_test$PRICE - svr_pred
lm_resid  <- NY_test$PRICE - lm_pred

# Plot residuals for SVM regression
par(mfrow = c(1, 2))  # set up side-by-side plots

plot(svr_pred, svr_resid, 
     main = "SVM Regression Residuals", 
     xlab = "Predicted Price", 
     ylab = "Residuals")
abline(h = 0, col = "red", lwd = 2)

# Plot residuals for linear regression
plot(lm_pred, lm_resid, 
     main = "Linear Regression Residuals", 
     xlab = "Predicted Price", 
     ylab = "Residuals")
abline(h = 0, col = "red", lwd = 2)

par(mfrow = c(1, 1))  # reset plot layout

