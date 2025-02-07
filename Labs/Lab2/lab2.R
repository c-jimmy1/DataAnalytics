
df <- read.csv("NY-House-Dataset.csv", stringsAsFactors = FALSE)

df$PRICE <- as.numeric(df$PRICE)
df$BEDS  <- as.numeric(df$BEDS)
df$BATH  <- as.numeric(df$BATH)
df$PROPERTYSQFT <- as.numeric(df$PROPERTYSQFT)

# Remove rows with NAs or zero/negative values in key columns
df_clean <- subset(df,
                   !is.na(PRICE) & PRICE > 0 &
                     !is.na(PROPERTYSQFT) & PROPERTYSQFT > 0 &
                     !is.na(BEDS) & !is.na(BATH))

# Inspect summaries
summary(df_clean[, c("PRICE", "BEDS", "BATH", "PROPERTYSQFT")])

# Remove any extreme outliers
Q1_price <- quantile(df_clean$PRICE, 0.25)
Q3_price <- quantile(df_clean$PRICE, 0.75)
IQR_price <- Q3_price - Q1_price
upper_price <- Q3_price + 3 * IQR_price

Q1_sqft <- quantile(df_clean$PROPERTYSQFT, 0.25)
Q3_sqft <- quantile(df_clean$PROPERTYSQFT, 0.75)
IQR_sqft <- Q3_sqft - Q1_sqft
upper_sqft <- Q3_sqft + 3 * IQR_sqft

df_clean <- subset(df_clean,
                   PRICE <= upper_price & PROPERTYSQFT <= upper_sqft)


#Fitting the 3 models
# Model 1: PRICE ~ PROPERTYSQFT
model1 <- lm(PRICE ~ PROPERTYSQFT, data = df_clean)

# Model 2: PRICE ~ PROPERTYSQFT + BEDS
model2 <- lm(PRICE ~ PROPERTYSQFT + BEDS, data = df_clean)

# Model 3: PRICE ~ PROPERTYSQFT + BEDS + BATH
model3 <- lm(PRICE ~ PROPERTYSQFT + BEDS + BATH, data = df_clean)

cat("\n=== Model 1 Summary ===\n")
summary(model1)

cat("\n=== Model 2 Summary ===\n")
summary(model2)

cat("\n=== Model 3 Summary ===\n")
summary(model3)


# === Model 1 Plots ===
par(mfrow = c(1, 2))   # 2 plots side-by-side

# (a) Scatter plot + best-fit line
plot(df_clean$PROPERTYSQFT, df_clean$PRICE,
     xlab = "PropertySqFt", ylab = "Price",
     main = "Model 1: Price vs PropertySqFt")
abline(model1, col = "red", lwd = 2)

# (b) Residuals vs Fitted
plot(model1, which = 1, main = "Model 1: Residuals vs Fitted")

# === Model 2 Plots ===
par(mfrow = c(1, 2))

# (a) Scatter plot vs. PROPERTYSQFT
plot(df_clean$PROPERTYSQFT, df_clean$PRICE,
     xlab = "PropertySqFt", ylab = "Price",
     main = "Model 2: Price vs PropertySqFt")
# To draw a best-fit line for multiple predictors, you can do:
# Predict y-hat from model2 for a grid of PROPERTYSQFT values
new_sqft <- data.frame(PROPERTYSQFT = seq(min(df_clean$PROPERTYSQFT),
                                          max(df_clean$PROPERTYSQFT),
                                          length.out = 100),
                       BEDS = mean(df_clean$BEDS, na.rm = TRUE)) # fix BEDS
pred_vals <- predict(model2, newdata = new_sqft)
lines(new_sqft$PROPERTYSQFT, pred_vals, col="red", lwd=2)

# (b) Residuals vs Fitted
plot(model2, which = 1, main = "Model 2: Residuals vs Fitted")

# === Model 3 Plots ===
par(mfrow = c(1, 2))

# (a) Again, let's pick PROPERTYSQFT for plotting
plot(df_clean$PROPERTYSQFT, df_clean$PRICE,
     xlab = "PropertySqFt", ylab = "Price",
     main = "Model 3: Price vs PropertySqFt")
# For a line, fix BEDS & BATH to their means
new_sqft2 <- data.frame(
  PROPERTYSQFT = seq(min(df_clean$PROPERTYSQFT),
                     max(df_clean$PROPERTYSQFT),
                     length.out = 100),
  BEDS = mean(df_clean$BEDS, na.rm = TRUE),
  BATH = mean(df_clean$BATH, na.rm = TRUE)
)
pred_vals2 <- predict(model3, newdata = new_sqft2)
lines(new_sqft2$PROPERTYSQFT, pred_vals2, col="red", lwd=2)

# (b) Residuals vs Fitted
plot(model3, which = 1, main = "Model 3: Residuals vs Fitted")

cat("\n=== AIC Comparison ===\n")
AIC(model1, model2, model3)

cat("\n=== BIC Comparison ===\n")
BIC(model1, model2, model3)

cat("\n=== R-squared ===\n")
cat("Model1 R^2:", summary(model1)$r.squared, "\n")
cat("Model2 R^2:", summary(model2)$r.squared, "\n")
cat("Model3 R^2:", summary(model3)$r.squared, "\n")

# Use these metrics (and your domain knowledge) to decide which model is best
############################################################
