library(dplyr)
library(ggplot2)
library(e1071)       # For Naïve Bayes
library(class)       # For k-NN
library(randomForest)


data <- read.csv("NYC_Citywide_Annualized_Calendar_Sales_Update_20241107.csv", stringsAsFactors = FALSE)


# ====================== 2A ===========================

queens.data <- data %>% filter(toupper(BOROUGH) == "QUEENS")

# Convert SALE.PRICE and GROSS.SQUARE.FEET to numeric in Queens data
queens.data$SALE.PRICE <- as.numeric(gsub(",", "", queens.data$SALE.PRICE))
queens.data$GROSS.SQUARE.FEET <- as.numeric(gsub(",", "", queens.data$GROSS.SQUARE.FEET))

# Filter out rows with missing or zero sale price and missing predictors in Queens
queens.data <- queens.data %>% 
  filter(!is.na(SALE.PRICE) & SALE.PRICE > 0 & 
           !is.na(GROSS.SQUARE.FEET) & !is.na(YEAR.BUILT) & !is.na(TOTAL.UNITS))

# Convert sale price to numeric (remove any commas if present)
queens.data$SALE.PRICE <- as.numeric(gsub(",", "", queens.data$SALE.PRICE))

# Summary and histogram of SALE.PRICE
summary(queens.data$SALE.PRICE)
ggplot(queens.data, aes(x = SALE.PRICE)) +
  geom_histogram(binwidth = 500000, fill = "skyblue", color = "black") +
  labs(title = "Histogram of Sale Price (Manhattan)", x = "Sale Price", y = "Frequency")

# Boxplot to visually identify outliers in SALE.PRICE
ggplot(queens.data, aes(y = SALE.PRICE)) +
  geom_boxplot(fill = "orange", outlier.color = "red") +
  labs(title = "Boxplot of Sale Price (Manhattan)", y = "Sale Price")

# outlier values using boxplot stats
bp <- boxplot(queens.data$SALE.PRICE, plot = FALSE)
outliers <- bp$out
cat("Identified outlier sale prices (Manhattan):\n")
print(outliers)

# Remove observations with missing or zero sale price
queens.data <- queens.data %>% filter(!is.na(SALE.PRICE) & SALE.PRICE > 0)

# Fit a multiple linear regression model
model1 <- lm(SALE.PRICE ~ GROSS.SQUARE.FEET + YEAR.BUILT + TOTAL.UNITS, data = queens.data)
summary(model1)

queens.data$NEIGHBORHOOD <- as.factor(queens.data$NEIGHBORHOOD)
astoria.data <- queens.data %>% filter(NEIGHBORHOOD == "ASTORIA")

# Predict sale prices for the astoria subset
astoria.predictions <- predict(model1, newdata = astoria.data)
astoria.residuals <- astoria.data$SALE.PRICE - astoria.predictions

# Plot predicted vs. actual sale prices
ggplot(astoria.data, aes(x = astoria.predictions, y = SALE.PRICE)) +
  geom_point(color = "blue") +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  labs(title = "Predicted vs. Actual Sale Price (Astoria)", x = "Predicted Sale Price", y = "Actual Sale Price")

# Plot residuals vs. predicted sale prices
ggplot(astoria.data, aes(x = astoria.predictions, y = astoria.residuals)) +
  geom_point(color = "darkgreen") +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(title = "Residuals vs. Predicted Sale Price (Astoria)", x = "Predicted Sale Price", y = "Residuals")

# ============================ 2B ===========================
set.seed(32)
queens.data <- queens.data %>% 
  filter(!is.na(SALE.PRICE) & !is.na(GROSS.SQUARE.FEET) & 
           !is.na(YEAR.BUILT) & !is.na(TOTAL.UNITS) & !is.na(NEIGHBORHOOD))

# Split into training (70%) and testing (30%) sets
train.indices <- sample(1:nrow(queens.data), size = 0.7 * nrow(queens.data))
train.data <- queens.data[train.indices, ]
test.data <- queens.data[-train.indices, ]

rf.model <- randomForest(NEIGHBORHOOD ~ SALE.PRICE + GROSS.SQUARE.FEET, data = train.data, ntree = 100)
pred.rf <- predict(rf.model, test.data)
conf.rf <- table(test.data$NEIGHBORHOOD, pred.rf)
print("Confusion Matrix - Random Forest:")
print(conf.rf)
metrics.rf <- calc.metrics(conf.rf, "ASTORIA")
print("Random Forest Metrics for ASTORIA:")
print(metrics.rf)


