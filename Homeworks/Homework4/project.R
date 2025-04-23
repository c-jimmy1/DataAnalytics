library(tidyverse)
library(janitor)
library(skimr)
library(patchwork)
library(fs)
library(zoo)
library(scales)
library(GGally)
library(tidymodels)
library(ranger)
library(vip)


log_msg <- function(txt) {
  message(sprintf("[%s] %s", format(Sys.time(), "%Y‑%m‑%d %H:%M:%S"), txt))
}

dir_create("plots")

df_raw <- read_csv("wdi_indicators_1960_2023.csv", na = c("NA", ""))

# Indicator columns used in the project
indicator_cols <- c(
  "ny_gdp_pcap_cd",     # GDP per capita (current US$)
  "si_pov_dday",        # Poverty headcount ratio $1.90/day
  "sp_dyn_le00_in",     # Life expectancy at birth (years)
  "sp_dyn_imrt_in",     # Infant mortality (per 1,000 live births)
  "sh_xpd_chex_pc_cd",  # Current health expenditure per capita
  "se_adt_litr_zs",     # Adult literacy rate (% ages 15+)
  "se_prm_enrr"         # Primary school enrolment ratio (% gross)
).



# ===============================================================
# 2. Preliminary Analysis
# ===============================================================
df_analysis <- df_raw %>%
  clean_names() %>% 
  filter(year >= 2000, year <= 2023) %>% 
  mutate(across(all_of(indicator_cols), as.numeric))

## 2·A   Quick numeric overview ---------------------------------
skim(df_analysis %>% select(year, iso3c, all_of(indicator_cols)))

## 2·B   Combined distribution plots -----------------------------
df_long <- df_analysis %>% 
  select(all_of(indicator_cols)) %>% 
  pivot_longer(everything(),
               names_to = "indicator",
               values_to = "value") %>%drop_na(value)

p_hist <- ggplot(df_long, aes(value)) +
  geom_histogram(bins = 30) +
  facet_wrap(~indicator, scales = "free") +
  labs(title = "Histograms of WDI Indicators (2000 to 2023)",
       x = NULL, y = "Count") +
  theme_minimal()

p_box  <- ggplot(df_long, aes(y = value)) +
  geom_boxplot() +
  facet_wrap(~indicator, scales = "free") +
  labs(title = "Box plots of WDI Indicators (2000 to 2023)",
       y = NULL, x = NULL) +
  theme_minimal()

combined_panel <- p_hist / p_box    # stack the two grids
ggsave("plots/indicator_distributions_panel.png",
       combined_panel, width = 12, height = 14, dpi = 300)
print(combined_panel)

missing_pct_analysis <- summarise(
  df_analysis,
  across(all_of(indicator_cols), ~ mean(is.na(.))*100)
)
print(missing_pct_analysis)


# ===============================================================
# 3. Exploratory Analysis, Data Cleaning & Preparation
# ===============================================================

# Median imputation to replace missing vals with median value of respective col
df_clean <- df_analysis %>% 
  group_by(country) %>% 
  mutate(across(all_of(indicator_cols),
                ~ if_else(is.na(.),
                          median(., na.rm = TRUE),
                          .))) %>% 
  ungroup()


# interpolate any gaps left
df_interp <- df_clean %>% 
  arrange(country, year) %>% 
  group_by(country) %>% 
  mutate(across(all_of(indicator_cols),
                ~ na.approx(., x = year, na.rm = FALSE))) %>% 
  ungroup()


df_interp_renamed <- df_interp %>%
  rename(
    gdp_per_capita_usd       = ny_gdp_pcap_cd,
    poverty_headcount        = si_pov_dday,
    life_expectancy          = sp_dyn_le00_in,
    infant_mortality         = sp_dyn_imrt_in,
    health_exp_per_capita    = sh_xpd_chex_pc_cd,
    adult_literacy_rate      = se_adt_litr_zs,
    primary_school_enrolment = se_prm_enrr
  )

# 3D. Apply log and standardize
df_trans <- df_interp_renamed %>%
  mutate(
    log_gdp_per_capita        = log10(gdp_per_capita_usd + 1),
    log_health_exp_per_capita = log10(health_exp_per_capita + 1)
  ) %>%
  mutate(across(
    c(
      log_gdp_per_capita,
      log_health_exp_per_capita,
      poverty_headcount,
      life_expectancy,
      infant_mortality,
      adult_literacy_rate,
      primary_school_enrolment
    ),
    ~ scale(.)[, 1],
    .names = "{.col}_z"
  ))

# new plots post cleaning:
df_trans_long <- df_trans %>%
  select(contains("_z")) %>%
  pivot_longer(everything(),
               names_to = "indicator",
               values_to = "value") %>%
  drop_na(value)

# Histograms of standardized indicators
p_hist_clean <- ggplot(df_trans_long, aes(value)) +
  geom_histogram(bins = 30, fill = "steelblue", color = "white") +
  facet_wrap(~indicator, scales = "free") +
  labs(title = "Histograms of Standardized WDI Indicators (2000–2023)",
       x = NULL, y = "Count") +
  theme_minimal()

# Boxplots of standardized indicators
p_box_clean <- ggplot(df_trans_long, aes(y = value)) +
  geom_boxplot(fill = "darkorange", alpha = 0.7) +
  facet_wrap(~indicator, scales = "free") +
  labs(title = "Boxplots of Standardized WDI Indicators (2000–2023)",
       y = NULL, x = NULL) +
  theme_minimal()

# Combine and save
combined_panel_clean <- p_hist_clean / p_box_clean

ggsave("plots/indicator_distributions_cleaned_panel.png",
       combined_panel_clean, width = 12, height = 14, dpi = 300)

print(combined_panel_clean)


# Correlation Matrix
library(GGally)
df_corr <- df_trans %>% 
  select(ends_with("_z")) %>% 
  drop_na()

# Create correlation matrix plot using ggcorr
p_corr <- ggcorr(df_corr, 
                 label = TRUE, 
                 label_size = 3,
                 hjust = 0.75, 
                 layout.exp = 2,
                 low = "red", mid = "white", high = "blue") +
  labs(title = "Correlation Matrix of Standardized WDI Indicators")

# Save to file
ggsave("plots/correlation_matrix_ggcorr.png", 
       plot = p_corr, 
       width = 10, height = 8, dpi = 300)

print(p_corr)

# Summary Statistical Stats:
summary_indicator_cols <- c(
  "gdp_per_capita_usd",
  "poverty_headcount",
  "life_expectancy",
  "infant_mortality",
  "health_exp_per_capita",
  "adult_literacy_rate",
  "primary_school_enrolment"
)

summary_stats <- df_interp_renamed %>%
  select(country, year, all_of(summary_indicator_cols)) %>%
  pivot_longer(cols = all_of(summary_indicator_cols), names_to = "indicator", values_to = "value") %>%
  group_by(indicator) %>%
  summarise(
    count = sum(!is.na(value)),
    mean  = mean(value, na.rm = TRUE),
    sd    = sd(value, na.rm = TRUE),
    min   = min(value, na.rm = TRUE),
    q1    = quantile(value, 0.25, na.rm = TRUE),
    median= median(value, na.rm = TRUE),
    q3    = quantile(value, 0.75, na.rm = TRUE),
    max   = max(value, na.rm = TRUE),
    .groups = "drop"
  )

print(summary_stats)

write_csv(summary_stats, "plots/summary_stats_cleaned_indicators.csv")
cat("Summary stats saved to 'plots/summary_stats_cleaned_indicators.csv'\n")


# ===============================================================
# 4. Model Development and Application of model(s)
# ===============================================================
# 4·A: Split & Resamples
df_model <- df_trans %>%
  select(country, year,
         log_gdp_per_capita_z,
         log_health_exp_per_capita_z,
         poverty_headcount_z,
         life_expectancy_z,
         infant_mortality_z,
         adult_literacy_rate_z,
         primary_school_enrolment_z) %>%
  drop_na()

split  <- initial_split(df_model, prop = .8)
train  <- training(split)
test   <- testing(split)
folds  <- group_vfold_cv(train, group = country, v = 5)

# 4·B: Recipe (predict log_gdp_per_capita_z)
rec <- recipe(log_gdp_per_capita_z ~ ., data = train) %>%
  update_role(country, year, new_role = "id")

# 4·C: Linear Regression
log_msg("▶ Starting linear‑model CV for GDP …")
lm_wf   <- workflow(rec, linear_reg() %>% set_engine("lm"))
lm_fit  <- lm_wf %>% fit_resamples(folds, control = control_resamples(save_workflow = TRUE))
lm_final <- lm_wf %>% fit(train)
lm_metrics <- collect_metrics(lm_fit)
log_msg(sprintf("✓ Linear‑model CV finished (rmse = %.3f)", lm_metrics %>% filter(.metric=="rmse") %>% pull(mean)))
print(lm_metrics)

lm_test <- lm_final %>%
  predict(test) %>%
  bind_cols(test) %>%
  metrics(truth = log_gdp_per_capita_z, estimate = .pred)

lm_aug <- lm_final %>%
  predict(test) %>%
  bind_cols(test) %>%
  mutate(
    resid     = log_gdp_per_capita_z - .pred,
    resid_std = scale(resid)[,1]
  )

p_resid <- ggplot(lm_aug, aes(.pred, resid)) +
  geom_point(alpha = .3) +
  geom_hline(yintercept = 0, lty = 2) +
  labs(title = "Linear Model – Residuals vs Fitted (GDP)",
       x = "Fitted (.pred)", y = "Residuals (z)") +
  theme_minimal()

ggsave("plots/lm_residuals_vs_fitted_gdp.png", p_resid, width = 6, height = 4, dpi = 300)
print(p_resid)

p_qq <- ggplot(lm_aug, aes(sample = resid_std)) +
  stat_qq(alpha = .4) + stat_qq_line() +
  labs(title = "Linear Model – Normal Q‑Q (GDP)") +
  theme_minimal()

ggsave("plots/lm_qq_plot_gdp.png", p_qq, width = 6, height = 4, dpi = 300)
print(p_qq)

# 4·D: Random Forest
rf_spec <- rand_forest(trees = 1000, mtry = tune(), min_n = tune()) %>%
  set_engine("ranger", importance = "permutation") %>%
  set_mode("regression")

rf_wf <- workflow(rec, rf_spec)

rf_grid <- grid_regular(
  mtry(range = c(2, 6)),
  min_n(range = c(3, 20)),
  levels = 4
)

log_msg("▶ Tuning random‑forest for GDP …")
rf_tuned <- tune_grid(rf_wf, folds, grid = rf_grid, metrics = metric_set(rmse, rsq))
rf_best  <- select_best(rf_tuned, metric = "rmse")
log_msg(sprintf("✓ RF tuning done — best rmse = %.3f", rf_best$mean))

rf_final <- finalize_workflow(rf_wf, rf_best) %>%
  fit(train)

rf_metrics <- rf_tuned %>% collect_metrics()
print(n = 32, rf_metrics)

rf_test <- rf_final %>%
  predict(test) %>%
  bind_cols(test) %>%
  metrics(truth = log_gdp_per_capita_z, estimate = .pred)

p_vip <- rf_final %>%
  extract_fit_parsnip() %>%
  vip::vi() %>%
  ggplot(aes(Importance, reorder(Variable, Importance))) +
  geom_col() +
  coord_flip() +
  labs(title = "Random‑Forest – Variable Importance for GDP") +
  theme_minimal()

ggsave("plots/rf_variable_importance_gdp.png", p_vip, width = 7, height = 4, dpi = 300)
print(p_vip)

# 4·E: Compare Models
compare_tbl <- bind_rows(
  lm_test   %>% mutate(model = "Linear"),
  rf_test   %>% mutate(model = "Random‑Forest")
) %>%
  select(model, .metric, .estimate) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)

print(compare_tbl)
write_csv(compare_tbl, "plots/model_comparison_metrics_gdp.csv")

# 4·F: Coefficients (Linear Model)
lm_parsnip  <- extract_fit_parsnip(lm_final)
tidy_coeffs <- tidy(lm_parsnip, conf.int = TRUE)
print(tidy_coeffs)

threshold <- median(test$log_gdp_per_capita_z, na.rm = TRUE)

# 4·G: Confusion Matrices (High vs Low GDP)
threshold   <- median(test$log_gdp_per_capita_z, na.rm = TRUE)

# pull raw prediction vectors once
lm_pred_vec <- predict(lm_final, test) %>% pull(.pred)
rf_pred_vec <- predict(rf_final, test) %>% pull(.pred)

class_tbl <- test %>%
  mutate(
    actual   = factor(
      if_else(log_gdp_per_capita_z > threshold, "High", "Low"),
      levels = c("Low","High")
    ),
    pred_lin = factor(
      if_else(lm_pred_vec > threshold,        "High", "Low"),
      levels = c("Low","High")
    ),
    pred_rf  = factor(
      if_else(rf_pred_vec > threshold,        "High", "Low"),
      levels = c("Low","High")
    )
  )

cm_lin <- class_tbl %>% conf_mat(truth = actual, estimate = pred_lin)
cm_rf  <- class_tbl %>% conf_mat(truth = actual, estimate = pred_rf)

print(cm_lin)
print(cm_rf)

