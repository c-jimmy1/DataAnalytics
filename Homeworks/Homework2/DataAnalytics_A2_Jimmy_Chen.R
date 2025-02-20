library(class)

# Load Data
EPI_data <- read.csv("epi_results_2024_pop_gdp.csv")

# Create subset regions
region1 <- subset(EPI_data, region == "Eastern Europe")
region2 <- subset(EPI_data, region == "Global West")


# Remove NAs from each region's EPI.new values
EPI.region1 <- region1$EPI.new[ !is.na(region1$EPI.new) ]
EPI.region2 <- region2$EPI.new[ !is.na(region2$EPI.new) ]

# ====Variable Distributions====

# Histogram & Density for Region 1 (Eastern Europe)
hist(EPI.region1,
     probability = TRUE,           # so y-axis is a density
     main = "EPI.new Distribution: Eastern Europe",
     xlab = "EPI.new",
     col = "lightblue",
     border = "white")
lines(density(EPI.region1),        # overlay density curve
      col = "blue",
      lwd = 2)

# Histogram & Density for Region 2 (Global West)
hist(EPI.region2,
     probability = TRUE,
     main = "EPI.new Distribution: Global West",
     xlab = "EPI.new",
     col = "pink",
     border = "white")
lines(density(EPI.region2),
      col = "red",
      lwd = 2)

# QQ Plot for Region 1 (Eastern Europe)
qqnorm(EPI.region1,
       main = "QQ Plot of EPI.new (Eastern Europe) vs Normal",
       pch  = 19,
       col  = "blue")
qqline(EPI.region1,
       col = "red",
       lwd = 2)

# QQ Plot for Region 2 (Global West)
qqnorm(EPI.region2,
       main = "QQ Plot of EPI.new (Global West) vs Normal",
       pch  = 19,
       col  = "darkgreen")
qqline(EPI.region2,
       col = "red",
       lwd = 2)


# ====Linear Models====

# 1. Two Variables, Full Dataset
m1 <- lm(EPI.new ~ log(population) + log(gdp), data = EPI_data)
summary(m1)
plot(log(EPI_data$gdp), EPI_data$EPI.new, main="EPI.new vs log(gdp)")
abline(lm(EPI.new ~ log(gdp), data = EPI_data), col="red")
plot(m1$residuals, main="Residuals of Model 1")

m2 <- lm(BDH.new ~ log(population) + log(gdp), data = EPI_data)
summary(m2)
plot(log(EPI_data$gdp), EPI_data$BDH.new, main="BDH.new vs log(gdp)")
abline(lm(BDH.new ~ log(gdp), data = EPI_data), col="red")
plot(m2$residuals, main="Residuals of Model 2")

# 2. Repeat with 1 Region
regionEE <- subset(EPI_data, region == "Eastern Europe")

m1_EE <- lm(EPI.new ~ log(population) + log(gdp), data = regionEE)
summary(m1_EE)
plot(log(regionEE$gdp), regionEE$EPI.new, main="EPI.new vs log(gdp) - Eastern Europe")
abline(lm(EPI.new ~ log(gdp), data = regionEE), col="red")
plot(m1_EE$residuals, main="Residuals of Model 1 - Eastern Europe")

m2_EE <- lm(BDH.new ~ log(population) + log(gdp), data = regionEE)
summary(m2_EE)
plot(log(regionEE$gdp), regionEE$BDH.new, main="BDH.new vs log(gdp) - Eastern Europe")
abline(lm(BDH.new ~ log(gdp), data = regionEE), col="red")
plot(m2_EE$residuals, main="Residuals of Model 2 - Eastern Europe")

# The first two models are the better choice using the full dataset because it has a better fitting line. The better fit
# depends on higher R-squared and lower residuals, in which the full dataset excels in for both variable models, compared to the Europe one.
# There is a higher R-squared and generally lower residuals for the full dataset.


# ====Classification====
# 1. Subset data for 2 regions and 3 variables

df1 <- subset(EPI_data, region %in% c("Eastern Europe", "Global West"),
              select = c("region","EPI.new","BDH.new","ECO.new"))
df1 <- na.omit(df1)
df1$region <- factor(df1$region)

set.seed(123)
idx1 <- sample(1:nrow(df1), 0.7*nrow(df1))
train1 <- df1[idx1,]
test1  <- df1[-idx1,]

trainX1 <- train1[, c("EPI.new","BDH.new","ECO.new")]
trainY1 <- train1$region
testX1  <- test1[, c("EPI.new","BDH.new","ECO.new")]
testY1  <- test1$region

# Try different k values:
for(kval in c(1,3,5)) {
  pred1 <- knn(trainX1, testX1, cl = trainY1, k = kval)
  cat("\nModel1 (k=",kval,") Confusion Matrix:\n", sep="")
  print(table(Predicted=pred1, Actual=testY1))
  acc1 <- mean(pred1 == testY1)
  cat("Accuracy =", acc1, "\n")
}

# 2. Repeat with 3 other variables
df2 <- subset(EPI_data, region %in% c("Eastern Europe", "Global West"),
              select = c("region","MHP.new","TBN.new","PAE.new"))
df2 <- na.omit(df2)
df2$region <- factor(df2$region)

set.seed(123)
idx2 <- sample(1:nrow(df2), 0.7*nrow(df2))
train2 <- df2[idx2,]
test2  <- df2[-idx2,]

trainX2 <- train2[, c("MHP.new","TBN.new","PAE.new")]
trainY2 <- train2$region
testX2  <- test2[, c("MHP.new","TBN.new","PAE.new")]
testY2  <- test2$region

for(kval in c(1,3,5)) {
  pred2 <- knn(trainX2, testX2, cl = trainY2, k = kval)
  cat("\nModel2 (k=",kval,") Confusion Matrix:\n", sep="")
  print(table(Predicted=pred2, Actual=testY2))
  acc2 <- mean(pred2 == testY2)
  cat("Accuracy =", acc2, "\n")
}

# The models with "EPI.new","BDH.new","ECO.new" as the 3 variables is better than the models with "MHP.new","TBN.new","PAE.new".
# The model1 models have a higher accuracy than Model 2. The first set of variables appears more predictive for classifying the region.
#