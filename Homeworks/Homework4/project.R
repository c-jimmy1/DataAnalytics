# 1. Load libraries
library(tidyverse)
library(janitor)
library(skimr)     # quick “skim” tables
library(naniar)    # missing‑value graphics
library(GGally)    # scatter‑matrix
library(countrycode)


# Data Cleaning
df_raw <- read_csv("wdi_indicators_1960_2023.csv", na = c("NA", ""))

# Clean up names to snake case for easier processing
df <- df_raw %>% clean_names()

# Filter data to just 2000 to 2023, as it has the least amt of missing values
df <- df %>% filter(year >= 2000, year <= 2023)


skim(df %>% select(year, iso3c,
                   ny_gdp_pcap_cd, sp_dyn_le00_in,
                   sp_dyn_imrt_in, sh_xpd_chex_pc_cd,
                   se_adt_litr_zs, se_prm_enrr, si_pov_dday))

# Check that all data is numeric
indicator_cols <- c("ny_gdp_pcap_cd","si_pov_dday",
                    "sp_dyn_le00_in","sp_dyn_imrt_in",
                    "sh_xpd_chex_pc_cd","se_adt_litr_zs","se_prm_enrr")
df <- df %>%
  mutate(across(all_of(indicator_cols), as.numeric))

# Assess & handle missingness
# Drop any indicator with >50% missing values
missing_pct <- df %>%
  summarise(across(all_of(indicator_cols),
                   ~ mean(is.na(.)) * 100))

#  - For the rest, impute country‑level median
df <- df %>%
  group_by(country) %>%
  mutate(across(all_of(indicator_cols),
                ~ if_else(is.na(.), median(., na.rm=TRUE), .))) %>%
  ungroup()


# 9. Save cleaned data
write_csv(df, "wdi_clean_wide_2000_23.csv")

