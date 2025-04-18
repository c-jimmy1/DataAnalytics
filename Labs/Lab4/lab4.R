###############################################################################
# 1. Setup & Data Loading
###############################################################################


library(ggplot2)
library(caret)
library(dplyr)

# Load the Wine dataset (13 predictors + 1 class)
wine <- read.csv("wine.data", header = FALSE)

# Add meaningful column names:
colnames(wine) <- c("Class",
                    "Alcohol",
                    "Malic_Acid",
                    "Ash",
                    "Alcalinity_of_Ash",
                    "Magnesium",
                    "Total_Phenols",
                    "Flavanoids",
                    "Nonflavanoid_Phenols",
                    "Proanthocyanins",
                    "Color_Intensity",
                    "Hue",
                    "OD280_OD315",
                    "Proline")

# Convert Class to a factor:
wine$Class <- as.factor(wine$Class)

###############################################################################
# 2. PCA on the Original Dataset
###############################################################################

# We do PCA on the numeric variables (all except "Class")
wine_numeric <- wine %>% select(-Class)

# Run PCA with scaling
pca_result <- prcomp(wine_numeric, center = TRUE, scale. = TRUE)

# Summary of PCA (Proportion of variance, etc.)
summary(pca_result)

# Extract the scores (PCs) for each observation
pca_scores <- as.data.frame(pca_result$x)

###############################################################################
# 3. Plot Dataset Using the 1st and 2nd Principal Components
###############################################################################

pca_plot_data <- data.frame(PC1 = pca_scores$PC1,
                            PC2 = pca_scores$PC2,
                            Class = wine$Class)

ggplot(pca_plot_data, aes(x = PC1, y = PC2, color = Class)) +
  geom_point(size = 2) +
  ggtitle("Wine Dataset on the First Two Principal Components")

###############################################################################
# 4. Identify Variables that Contribute the Most to the 1st PC
###############################################################################
loadings_pc1 <- pca_result$rotation[, 1]  # loadings for PC1

# Sort variables by absolute loading (descending)
abs_loadings_pc1 <- sort(abs(loadings_pc1), decreasing = TRUE)
abs_loadings_pc1

# Show them with signs and original order as well:
loadings_pc1[order(abs(loadings_pc1), decreasing = TRUE)]

# Suppose we decide to drop a certain number of variables with the smallest 
# absolute loadings to PC1 (for demonstration let's drop the 3 smallest).

# Identify the 3 least important variables for PC1
least_important_vars <- names(tail(abs_loadings_pc1, 3))
least_important_vars

###############################################################################
# 5. Drop the least contributing variables and re-run PCA
###############################################################################

# Remove the least contributing variables
wine_reduced <- wine_numeric %>% select(-all_of(least_important_vars))

# PCA on the reduced dataset
pca_reduced <- prcomp(wine_reduced, center = TRUE, scale. = TRUE)
summary(pca_reduced)

###############################################################################
# 6. Train a Classifier (kNN) on the Original Dataset
###############################################################################

set.seed(123)
train_index <- createDataPartition(wine$Class, p = 0.7, list = FALSE)

train_data <- wine[train_index, ]
test_data  <- wine[-train_index, ]

# kNN model using caret (with 5-fold CV for hyperparameter tuning)
model_knn_original <- train(
  Class ~ ., 
  data = train_data, 
  method = "knn",
  tuneLength = 5,
  trControl = trainControl(method = "cv", number = 5)
)

model_knn_original

# Predict on test set
pred_original <- predict(model_knn_original, newdata = test_data)

# Confusion matrix
conf_mat_original <- confusionMatrix(pred_original, test_data$Class)
conf_mat_original

###############################################################################
# 7. Train a Classifier using the First 3 Principal Components
###############################################################################

# Extract the first 3 PCs from the full PCA:
wine_pcs <- as.data.frame(pca_result$x[, 1:3])
wine_pcs$Class <- wine$Class

# Split according to the same indices as before
train_pcs <- wine_pcs[train_index, ]
test_pcs  <- wine_pcs[-train_index, ]

# Train kNN on the first 3 PC scores
model_knn_pcs <- train(
  Class ~ .,
  data = train_pcs,
  method = "knn",
  tuneLength = 5,
  trControl = trainControl(method = "cv", number = 5)
)

model_knn_pcs

# Predict on the test PCs
pred_pcs <- predict(model_knn_pcs, newdata = test_pcs)

# Confusion matrix
conf_mat_pcs <- confusionMatrix(pred_pcs, test_pcs$Class)
conf_mat_pcs

###############################################################################
# 8. Compare Both Models: Contingency Tables & Precision/Recall/F1
###############################################################################

compute_multiclass_metrics <- function(cm) {
  # Convert the confusion matrix table to a data frame
  cm_df <- as.data.frame(cm$table)
  
  # The actual class labels come from cm$levels, not cm$reference
  classes <- cm$levels
  
  # Make sure Prediction/Reference columns are factors with these levels
  cm_df$Prediction <- factor(cm_df$Prediction, levels = classes)
  cm_df$Reference  <- factor(cm_df$Reference,  levels = classes)
  
  # Convert Freq to numeric
  cm_df$Freq <- as.numeric(as.character(cm_df$Freq))
  
  # Calculate metrics per class
  metrics_list <- lapply(classes, function(cl) {
    # True Positives: predicted = cl & reference = cl
    TP <- sum(cm_df$Freq[cm_df$Prediction == cl & cm_df$Reference == cl])
    
    # False Positives: predicted = cl, reference != cl
    FP <- sum(cm_df$Freq[cm_df$Prediction == cl & cm_df$Reference != cl])
    
    # False Negatives: predicted != cl, reference = cl
    FN <- sum(cm_df$Freq[cm_df$Prediction != cl & cm_df$Reference == cl])
    
    precision <- TP / (TP + FP)
    recall    <- TP / (TP + FN)
    
    # Handle potential divide-by-zero for F1
    f1 <- if ((precision + recall) == 0) 0 else 2 * precision * recall / (precision + recall)
    
    data.frame(
      Class = cl,
      Precision = precision,
      Recall = recall,
      F1 = f1
    )
  })
  
  # Combine the class-level results
  metrics_df <- do.call(rbind, metrics_list)
  
  # Macro-average
  macro_precision <- mean(metrics_df$Precision, na.rm = TRUE)
  macro_recall    <- mean(metrics_df$Recall,    na.rm = TRUE)
  macro_f1        <- mean(metrics_df$F1,        na.rm = TRUE)
  
  # Return a list with two parts
  list(
    by_class = metrics_df,
    macro_avg = data.frame(
      Precision = macro_precision,
      Recall    = macro_recall,
      F1        = macro_f1
    )
  )
}


# Metrics for original model
metrics_original <- compute_multiclass_metrics(conf_mat_original)
metrics_original$by_class
metrics_original$macro_avg

# Metrics for PCA-based model
metrics_pcs <- compute_multiclass_metrics(conf_mat_pcs)
metrics_pcs$by_class
metrics_pcs$macro_avg

# Examine and compare
cat("=== Original Data Model: Macro-Averaged Metrics ===\n")
print(metrics_original$macro_avg)
cat("\n=== PCA (3 PCs) Model: Macro-Averaged Metrics ===\n")
print(metrics_pcs$macro_avg)