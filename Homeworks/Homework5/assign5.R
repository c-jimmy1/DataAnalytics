library(dplyr)
library(ggplot2)
library(e1071)       # For Naïve Bayes
library(class)       # For k-NN
library(randomForest)

data <- read.csv("NYC_Citywide_Annualized_Calendar_Sales_Update_20241107.csv", stringsAsFactors = FALSE)
data[data == ""] <- NA

manhattan.data <- data %>% filter(toupper(BOROUGH) == "MANHATTAN")

summary(manhattan_data$SALE_PRICE)

# ===================== 1B =====================

# Convert sale price to numeric (remove any commas if present)
manhattan.data$SALE.PRICE <- as.numeric(gsub(",", "", manhattan.data$SALE.PRICE))

# Summary and histogram of SALE.PRICE
summary(manhattan.data$SALE.PRICE)
ggplot(manhattan.data, aes(x = SALE.PRICE)) +
  geom_histogram(binwidth = 500000, fill = "skyblue", color = "black") +
  labs(title = "Histogram of Sale Price (Manhattan)", x = "Sale Price", y = "Frequency")

# Boxplot to visually identify outliers in SALE.PRICE
ggplot(manhattan.data, aes(y = SALE.PRICE)) +
  geom_boxplot(fill = "orange", outlier.color = "red") +
  labs(title = "Boxplot of Sale Price (Manhattan)", y = "Sale Price")

# Identify outlier values using boxplot stats
bp <- boxplot(manhattan.data$SALE.PRICE, plot = FALSE)
outliers <- bp$out
cat("Identified outlier sale prices (Manhattan):\n")
print(outliers)

# =================== 1C =======================
# Remove observations with missing or zero sale price
manhattan.data <- manhattan.data %>% filter(!is.na(SALE.PRICE) & SALE.PRICE > 0)

# Fit a multiple linear regression model
# (Assume that the dataset has columns: GROSS.SQUARE.FEET, YEAR.BUILT, TOTAL.UNITS)
model1 <- lm(SALE.PRICE ~ GROSS.SQUARE.FEET + YEAR.BUILT + TOTAL.UNITS, data = manhattan.data)
summary(model1)

# Test the model on a subset: e.g., the neighborhood "CHELSEA"
# Ensure that the NEIGHBORHOOD variable is a factor
manhattan.data$NEIGHBORHOOD <- as.factor(manhattan.data$NEIGHBORHOOD)
chelsea.data <- manhattan.data %>% filter(NEIGHBORHOOD == "CHELSEA")

# Predict sale prices for the Chelsea subset
chelsea.predictions <- predict(model1, newdata = chelsea.data)
chelsea.residuals <- chelsea.data$SALE.PRICE - chelsea.predictions

# Plot predicted vs. actual sale prices
ggplot(chelsea.data, aes(x = chelsea.predictions, y = SALE.PRICE)) +
  geom_point(color = "blue") +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  labs(title = "Predicted vs. Actual Sale Price (Chelsea)", x = "Predicted Sale Price", y = "Actual Sale Price")

# Plot residuals vs. predicted sale prices
ggplot(chelsea.data, aes(x = chelsea.predictions, y = chelsea.residuals)) +
  geom_point(color = "darkgreen") +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(title = "Residuals vs. Predicted Sale Price (Chelsea)", x = "Predicted Sale Price", y = "Residuals")


# ================ 1D =========================
set.seed(32)

# Remove rows with missing values in SALE.PRICE, GROSS.SQUARE.FEET, YEAR.BUILT, TOTAL.UNITS, and NEIGHBORHOOD
manhattan.data <- manhattan.data %>% 
  filter(!is.na(SALE.PRICE) & !is.na(GROSS.SQUARE.FEET) & 
           !is.na(YEAR.BUILT) & !is.na(TOTAL.UNITS) & !is.na(NEIGHBORHOOD))

# Ensure GROSS.SQUARE.FEET is numeric
manhattan.data$GROSS.SQUARE.FEET <- as.numeric(gsub(",", "", manhattan.data$GROSS.SQUARE.FEET))

# Split into training (70%) and testing (30%) sets
train.indices <- sample(1:nrow(manhattan.data), size = 0.7 * nrow(manhattan.data))
train.data <- manhattan.data[train.indices, ]
test.data <- manhattan.data[-train.indices, ]

# Naïve Bayes Model
nb.model <- naiveBayes(NEIGHBORHOOD ~ SALE.PRICE + GROSS.SQUARE.FEET, data = train.data, laplace = 1)
pred.nb <- predict(nb.model, test.data)
conf.nb <- table(test.data$NEIGHBORHOOD, pred.nb)
print("Confusion Matrix - Naive Bayes:")
print(conf.nb)

# Helper function to calculate precision and recall for a specific class
calc.metrics <- function(cm, class) {
  true.positive <- cm[class, class]
  false.positive <- sum(cm[, class]) - true.positive
  false.negative <- sum(cm[class, ]) - true.positive
  precision <- true.positive / (true.positive + false.positive)
  recall <- true.positive / (true.positive + false.negative)
  c(Precision = precision, Recall = recall)
}
# Example: metrics for "CHELSEA"
metrics.nb <- calc.metrics(conf.nb, "CHELSEA")
print("Naive Bayes Metrics for CHELSEA:")
print(metrics.nb)

# k-Nearest Neighbors (k-NN)
train.knn <- train.data[, c("SALE.PRICE", "GROSS.SQUARE.FEET")]
test.knn <- test.data[, c("SALE.PRICE", "GROSS.SQUARE.FEET")]
train.knn.scaled <- scale(train.knn)
test.knn.scaled <- scale(test.knn, center = attr(train.knn.scaled, "scaled:center"), 
                         scale = attr(train.knn.scaled, "scaled:scale"))
knn.pred <- knn(train.knn.scaled, test.knn.scaled, cl = train.data$NEIGHBORHOOD, k = 5)
conf.knn <- table(test.data$NEIGHBORHOOD, knn.pred)
print("Confusion Matrix - k-NN:")
print(conf.knn)
metrics.knn <- calc.metrics(conf.knn, "CHELSEA")
print("k-NN Metrics for CHELSEA:")
print(metrics.knn)

# Random Forest Model
rf.model <- randomForest(NEIGHBORHOOD ~ SALE.PRICE + GROSS.SQUARE.FEET, data = train.data, ntree = 100)
pred.rf <- predict(rf.model, test.data)
conf.rf <- table(test.data$NEIGHBORHOOD, pred.rf)
print("Confusion Matrix - Random Forest:")
print(conf.rf)
metrics.rf <- calc.metrics(conf.rf, "CHELSEA")
print("Random Forest Metrics for CHELSEA:")
print(metrics.rf)


