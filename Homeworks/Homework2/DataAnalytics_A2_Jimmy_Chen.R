epi_data <- read.csv("epi2024results06022024.csv")

attach(epi_data)

region1_data <- subset(epi_data, country == "Asia")
region2_data <- subset(epi_data, country == "Europe")

cat("Rows in Region1 subset:", nrow(region1_data), "\n")
cat("Rows in Region2 subset:", nrow(region2_data), "\n")