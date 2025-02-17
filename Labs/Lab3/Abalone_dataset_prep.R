###################
##### Abalone #####
###################

# read dataset
abalone <- read.csv("abalone_dataset.csv")
dataset <- abalone

## add new column age.group with 3 values based on the number of rings 
dataset$age.group <- cut(dataset$rings, br=c(0,8,11,35), labels = c("young", 'adult', 'old'))

## alternative way of setting age.group
dataset$age.group[dataset$rings<=8] <- "young"
dataset$age.group[dataset$rings>8 & dataset$rings<=11] <- "adult"
dataset$age.group[dataset$rings>11 & dataset$rings<=35] <- "old"

#########################
### Exercise 1: kNN  ###
#########################

set.seed(123)
index <- sample(1:nrow(dataset), 0.7 * nrow(dataset))
train <- dataset[index, ]
test <- dataset[-index, ]

features1 <- c("length", "diameter", "height")
features2 <- c("whole_weight", "shucked_wieght", "viscera_wieght", "shell_weight")

train_X1 <- train[, features1]
test_X1  <- test[, features1]
train_X2 <- train[, features2]
test_X2  <- test[, features2]

train_y <- train$age.group
test_y  <- test$age.group

library(class)
pred1 <- knn(train_X1, test_X1, train_y, k=5)
pred2 <- knn(train_X2, test_X2, train_y, k=5)

table(pred1, test_y)
table(pred2, test_y)

acc1 <- sum(pred1 == test_y) / length(test_y)
acc2 <- sum(pred2 == test_y) / length(test_y)

if(acc1 >= acc2) {
  best_features <- features1
  train_X <- train_X1
  test_X  <- test_X1
} else {
  best_features <- features2
  train_X <- train_X2
  test_X  <- test_X2
}

ks <- 1:20
acc <- numeric(length(ks))
for(i in seq_along(ks)) {
  pred <- knn(train_X, test_X, train_y, k=ks[i])
  acc[i] <- sum(pred == test_y) / length(test_y)
}
best_k <- ks[which.max(acc)]
best_k

best_pred <- knn(train_X, test_X, train_y, k=best_k)
table(best_pred, test_y)

##############################
### Exercise 2: K-Means     ###
##############################

data_cluster <- dataset[, best_features]
data_cluster_scaled <- scale(data_cluster)

wss <- sapply(1:10, function(k) { kmeans(data_cluster_scaled, centers=k, nstart=10)$tot.withinss })
plot(1:10, wss, type="b", xlab="Number of clusters K", ylab="Total within-clusters sum of squares")

library(cluster)
sil_width <- numeric(9)
for(k in 2:10) {
  km <- kmeans(data_cluster_scaled, centers=k, nstart=10)
  ss <- silhouette(km$cluster, dist(data_cluster_scaled))
  sil_width[k-1] <- mean(ss[, 3])
}
optimal_k <- which.max(sil_width) + 1
optimal_k

km_final <- kmeans(data_cluster_scaled, centers=optimal_k, nstart=25)
plot(data_cluster_scaled[,1:2], col=km_final$cluster, pch=19, xlab=best_features[1], ylab=best_features[2])
points(km_final$centers[,1:2], col=1:optimal_k, pch=8, cex=2)

