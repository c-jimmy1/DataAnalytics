library("ggplot2")
library("readr")

## read dataset
NY_House_Dataset <- read_csv("NY-House-Dataset.csv")

dataset <- NY_House_Dataset

ggplot(dataset, aes(x = log10(PROPERTYSQFT), y = log10(PRICE))) +
  geom_point()

## filter data
dataset <- dataset[dataset$PRICE<195000000,]

dataset <- dataset[dataset$PROPERTYSQFT!=2184.207862,]

dataset$PROPERTYSQFT[dataset$BROKERTITLE=="Brokered by Douglas Elliman - 575 Madison Ave"][85]

## column names
names(dataset)

## fit linear model
lmod <- lm(PRICE~PROPERTYSQFT, data = dataset)

lmod <- lm(log10(PRICE)~log10(PROPERTYSQFT), data = dataset)

## print model output
summary(lmod)

## scatter plot of 2 variables
plot(PRICE~PROPERTYSQFT, data = dataset)
abline(lmod)

plot(log10(PRICE)~log10(PROPERTYSQFT), data = dataset)
abline(lmod)

## scatter plot of 2 variables
ggplot(dataset, aes(x = PROPERTYSQFT, y = PRICE)) +
  geom_point()

ggplot(dataset, aes(x = PROPERTYSQFT, y = PRICE)) +
  geom_point() +
  stat_smooth(method = "lm", col="red")

ggplot(dataset, aes(x = log10(PROPERTYSQFT), y = log10(PRICE))) +
  geom_point() +
  stat_smooth(method = "lm", col="red")


dataset <- dataset[dataset$BEDS > 0 & dataset$BATH > 0, ]

# Create new columns for log–transformed Price and PropertySqFt.
dataset$logPRICE <- log10(dataset$PRICE)
dataset$logSqFt  <- log10(dataset$PROPERTYSQFT)

# ---- Model 1: Using only PropertySqFt (transformed) as predictor ----
mod1 <- lm(logPRICE ~ logSqFt, data = dataset)
cat("===== Model 1: log10(PRICE) ~ log10(PROPERTYSQFT) =====\n")
print(summary(mod1))

# Since there is only one predictor, it is by definition the most significant.
most_sig_mod1 <- "logSqFt"

# Plotting the relationship for Model 1:
library(ggplot2)
ggplot(dataset, aes(x = logSqFt, y = logPRICE)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE, color = "red") +
  labs(title = "Model 1: log10(PRICE) vs log10(PROPERTYSQFT)",
       x = "log10(PropertySqFt)",
       y = "log10(PRICE)")

# Residual plot for Model 1:
plot(mod1$fitted.values, mod1$residuals,
     main = "Model 1 Residuals vs Fitted",
     xlab = "Fitted values",
     ylab = "Residuals")
abline(h = 0, col = "red", lwd = 2)

# ---- Model 2: Using PropertySqFt and Beds as predictors ----
mod2 <- lm(logPRICE ~ logSqFt + BEDS, data = dataset)
cat("\n===== Model 2: log10(PRICE) ~ log10(PROPERTYSQFT) + BEDS =====\n")
print(summary(mod2))

# Determine the most significant predictor (excluding the intercept)
coeffs_mod2 <- summary(mod2)$coefficients[-1, ]  # drop intercept row
most_sig_mod2 <- rownames(coeffs_mod2)[which.min(coeffs_mod2[, "Pr(>|t|)"])]

# Plot the most significant predictor vs logPRICE with best–fit line:
ggplot(dataset, aes_string(x = most_sig_mod2, y = "logPRICE")) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE, color = "red") +
  labs(title = paste("Model 2: log10(PRICE) vs", most_sig_mod2),
       x = most_sig_mod2,
       y = "log10(PRICE)")

# Residual plot for Model 2:
plot(mod2$fitted.values, mod2$residuals,
     main = "Model 2 Residuals vs Fitted",
     xlab = "Fitted values",
     ylab = "Residuals")
abline(h = 0, col = "red", lwd = 2)

# ---- Model 3: Using PropertySqFt, Beds, and Bath as predictors ----
mod3 <- lm(logPRICE ~ logSqFt + BEDS + BATH, data = dataset)
cat("\n===== Model 3: log10(PRICE) ~ log10(PROPERTYSQFT) + BEDS + BATH =====\n")
print(summary(mod3))

# Determine the most significant predictor (excluding the intercept)
coeffs_mod3 <- summary(mod3)$coefficients[-1, ]
most_sig_mod3 <- rownames(coeffs_mod3)[which.min(coeffs_mod3[, "Pr(>|t|)"])]

# Plot the most significant predictor vs logPRICE with best–fit line:
ggplot(dataset, aes_string(x = most_sig_mod3, y = "logPRICE")) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE, color = "red") +
  labs(title = paste("Model 3: log10(PRICE) vs", most_sig_mod3),
       x = most_sig_mod3,
       y = "log10(PRICE)")

# Residual plot for Model 3:
plot(mod3$fitted.values, mod3$residuals,
     main = "Model 3 Residuals vs Fitted",
     xlab = "Fitted values",
     ylab = "Residuals")
abline(h = 0, col = "red", lwd = 2)

