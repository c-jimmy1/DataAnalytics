library(tidyverse)
library(tidymodels)
library(skimr)
library(janitor)
library(GGally)
library(patchwork)
theme_set(theme_minimal())

# ─────────────────────────── Exploratory Data Analysis ─────────────────────────── #

data_raw <- read_csv("ObesityDataSet_raw_and_data_sinthetic.csv") %>% clean_names()

# Derive BMI (response for regression) and convert factors
data <- data_raw %>% 
  mutate(bmi = weight / height^2) %>% 
  mutate(across(where(is.character), as.factor))

glimpse(data)

# Quick completeness check
skim_without_charts(data)


# Numerical Distributions
num_dist_plot <- data %>% 
  select(age, weight, height, bmi) %>% 
  pivot_longer(everything()) %>% 
  ggplot(aes(value)) +
  facet_wrap(~name, scales = "free") +
  geom_histogram(bins = 30)
num_dist_plot

# Categorical Balance
cat_balance_plot <- data %>% 
  count(n_obeyesdad) %>% 
  ggplot(aes(fct_reorder(n_obeyesdad, n), n)) +
  geom_col() +
  coord_flip() +
  labs(y = "Count", x = NULL)
cat_balance_plot

# Correlation matrix (numeric predictors)
num_vars <- data %>% 
  select(where(is.numeric)) %>%
  select(-one_of(c("n_obeyesdad")))

# Compute Pearson correlations
corr_mat <- round(cor(num_vars, use = "pairwise.complete.obs"), 2)

# Plot as a heat-map
corr_plot <- corr_mat %>% 
  as.data.frame() %>% 
  rownames_to_column("var1") %>% 
  pivot_longer(-var1, names_to = "var2", values_to = "cor") %>% 
  ggplot(aes(var1, var2, fill = cor)) +
  geom_tile(color = "white") +
  geom_text(aes(label = cor), size = 3) +
  scale_fill_gradient2(limits = c(-1, 1), 
                       midpoint = 0, 
                       low = "#B2182B", mid = "white", high = "#2166AC") +
  labs(title = "Pearson correlation matrix", x = NULL, y = NULL) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# ─────────────────────────── Model Development ─────────────────────────── #
# Model recipe
reg_recipe <- recipe(bmi ~ ., data = data) %>% 
  step_rm(n_obeyesdad) %>%                 # drop the categorical target
  step_dummy(all_nominal_predictors()) %>% 
  step_zv(all_predictors()) %>% 
  step_normalize(all_numeric_predictors())

# linear regression
# Workflow & fit
lm_wf <- workflow() %>% 
  add_recipe(reg_recipe) %>% 
  add_model(linear_reg() %>% set_engine("lm"))

lm_fit <- lm_wf %>% fit(data)

aug <- augment(lm_fit, new_data = data)   # gives .pred and .resid
metrics(aug, truth = bmi, estimate = .pred)

resid_vs_fitted_plot <- ggplot(aug, aes(.pred, .resid)) +                # <- use .pred
  geom_point(alpha = .6) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(title = "Residuals vs Fitted",
       x = "Fitted values (.pred)",
       y = "Residuals")

qq_plot <- ggplot(aug, aes(sample = .resid)) +
  stat_qq() + stat_qq_line() +
  labs(title = "Normal Q-Q Plot of Residuals")

resid_hist_plot <- ggplot(aug, aes(.resid)) +
  geom_histogram(bins = 30) +
  labs(title = "Distribution of Residuals")

# Multinominal Logistic Classification
# Recipes
cls_rec <- recipe(n_obeyesdad ~ ., data = data) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_zv(all_predictors()) %>% 
  step_normalize(all_numeric_predictors()) %>% 
  update_role(bmi, new_role = "predictor")           # keep BMI as predictor

# Model spec & tuning grid
multi_spec <- multinom_reg(penalty = tune()) %>% 
  set_engine("nnet") %>% 
  set_mode("classification")

cls_wf <- workflow() %>% add_recipe(cls_rec) %>% add_model(multi_spec)

cls_folds <- vfold_cv(data, v = 5, strata = n_obeyesdad)
cls_grid  <- grid_regular(penalty(), levels = 15)

cls_tuned <- tune_grid(
  cls_wf,
  resamples = cls_folds,
  grid      = cls_grid,
  metrics   = metric_set(accuracy, kap, mn_log_loss)
)

best_cls <- select_best(cls_tuned, metric = "accuracy")
# Train/test split & final fit
split <- initial_split(data, strata = n_obeyesdad, prop = 0.80)
train <- training(split);  test <- testing(split)

cls_final_wf  <- finalize_workflow(cls_wf, best_cls)
cls_final_fit <- cls_final_wf %>% fit(train)

# Test-set predictions
test_preds <- predict(cls_final_fit, test, type = "prob") %>% 
  bind_cols(predict(cls_final_fit, test)) %>% 
  bind_cols(test %>% select(n_obeyesdad))
summary(test_preds)

library(test_predslibrary(yardstick)
library(forcats)

## Confusion-matrix heat-map (blue→red)
cm_plot <- test_preds %>% 
  conf_mat(truth = n_obeyesdad, estimate = .pred_class) %>% 
  autoplot(type = "heatmap") +
  scale_fill_gradient2(
    low      = "#2166AC",  # blue
    mid      = "white",
    high     = "#B2182B",  # red
    midpoint = median(conf_mat(test_preds, n_obeyesdad,
                               .pred_class)$table),
    name     = "Count") +
  labs(title = "Multinomial Logistic – Confusion Matrix") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
cm_plot

## Coefficient importance top 20 predictors
## helper: coefficient matrix → long data-frame
coef_long <- extract_fit_engine(cls_final_fit) |>
  coef() |>
  # coef() gives a matrix: rows = outcome classes, cols = predictors
  t() |>                                   # we want term down rows
  as.data.frame() |>
  rownames_to_column("term") |>            # predictor names
  pivot_longer(-term,
               names_to  = "class",
               values_to = "estimate") |>
  group_by(term) |>
  summarise(mean_abs = mean(abs(estimate)), .groups = "drop")

## plot the 20 strongest absolute coefficients
coef_plot <- coef_long |>
  slice_max(mean_abs, n = 20) |>
  mutate(term = forcats::fct_reorder(term, mean_abs)) |>
  ggplot(aes(mean_abs, term)) +
  geom_col(fill = "#2B60DE") +
  labs(
    x     = "|coefficient| (avg across classes)",
    y     = NULL,
    title = "Top 20 Predictors – Multinomial Log-Reg"
  )

coef_plot  

if (!dir.exists("plots")) dir.create("plots")

ggsave("plots/numeric_distributions.png",  num_dist_plot,  width = 7, height = 5, dpi = 300)
ggsave("plots/categorical_balance.png",    cat_balance_plot, width = 6, height = 4, dpi = 300)
ggsave("plots/correlation_matrix.png",     corr_plot,        width = 7, height = 6, dpi = 300)
ggsave("plots/residuals_vs_fitted.png",    resid_vs_fitted_plot, width = 6, height = 4, dpi = 300)
ggsave("plots/qq_residuals.png",           qq_plot,          width = 5, height = 5, dpi = 300)
ggsave("plots/residual_hist.png",          resid_hist_plot,  width = 6, height = 4, dpi = 300)
ggsave("plots/confusion_matrix.png",       cm_plot,          width = 5, height = 5, dpi = 300)
ggsave("plots/coef_importance.png",        coef_plot,        width = 6, height = 5, dpi = 300)

