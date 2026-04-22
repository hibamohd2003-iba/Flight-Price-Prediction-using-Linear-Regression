# ============================================================
# Flight Price Prediction - Analysis using Linear Regression
# Dataset: Flight_Price_Prediction.csv
# ============================================================

# ---- 1. Load Libraries ----
library(ggplot2)
library(dplyr)
library(tidyr)
library(corrplot)
library(scales)
library(caret)
library(gridExtra)

# ---- 2. Load Data ----
df <- read.csv("flight-data.csv", stringsAsFactors = TRUE)
df$X <- NULL  # Remove index column

cat("Dataset Dimensions:", nrow(df), "rows x", ncol(df), "cols\n")
cat("Column Names:", paste(names(df), collapse = ", "), "\n")
cat("Missing Values:\n")
print(colSums(is.na(df)))

# ---- 3. EDA: Q1 - Does price vary with Airlines (same route)? ----
# Delhi to Mumbai as example route
route_df <- df %>%
  filter(source_city == "Delhi", destination_city == "Mumbai")

p1 <- ggplot(route_df, aes(x = reorder(airline, price, median), y = price, fill = airline)) +
  geom_boxplot(alpha = 0.8, outlier.alpha = 0.3, outlier.size = 0.5) +
  scale_y_continuous(labels = comma) +
  scale_fill_brewer(palette = "Set2") +
  labs(
    title = "Flight Price by Airline (Delhi → Mumbai)",
    subtitle = "Does price vary across airlines on the same route?",
    x = "Airline",
    y = "Ticket Price (INR)",
    fill = "Airline"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold"),
    legend.position = "none",
    axis.text.x = element_text(angle = 20, hjust = 1)
  )

ggsave("plot1.png", plot = p1, width = 8, height = 5, dpi = 150)
cat("Saved plot1.png\n")

# ---- 4. EDA: Q2 - Price when bought 1-2 days before departure ----
last_minute_df <- df %>%
  mutate(booking_group = case_when(
    days_left <= 2  ~ "1-2 Days Before",
    days_left <= 7  ~ "3-7 Days Before",
    days_left <= 14 ~ "8-14 Days Before",
    days_left <= 30 ~ "15-30 Days Before",
    TRUE            ~ "30+ Days Before"
  )) %>%
  mutate(booking_group = factor(booking_group, levels = c(
    "1-2 Days Before", "3-7 Days Before", "8-14 Days Before",
    "15-30 Days Before", "30+ Days Before"
  )))

p2 <- ggplot(last_minute_df, aes(x = booking_group, y = price, fill = booking_group)) +
  geom_boxplot(alpha = 0.8, outlier.alpha = 0.2, outlier.size = 0.4) +
  scale_y_continuous(labels = comma) +
  scale_fill_brewer(palette = "RdYlGn", direction = -1) +
  labs(
    title = "Ticket Price vs. Days Before Departure",
    subtitle = "Are last-minute tickets more expensive?",
    x = "Booking Window",
    y = "Ticket Price (INR)",
    fill = "Booking Window"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold"),
    legend.position = "none",
    axis.text.x = element_text(angle = 15, hjust = 1)
  )

ggsave("plot2.png", plot = p2, width = 8, height = 5, dpi = 150)
cat("Saved plot2.png\n")

# ---- 5. EDA: Q3 - Departure & Arrival Time vs Price ----
time_order <- c("Early_Morning", "Morning", "Afternoon", "Evening", "Night", "Late_Night")

avg_by_time <- df %>%
  mutate(departure_time = factor(departure_time, levels = time_order),
         arrival_time   = factor(arrival_time,   levels = time_order)) %>%
  group_by(departure_time, arrival_time) %>%
  summarise(avg_price = mean(price, na.rm = TRUE), .groups = "drop")

p3 <- ggplot(avg_by_time, aes(x = departure_time, y = arrival_time, fill = avg_price)) +
  geom_tile(color = "white") +
  scale_fill_gradient(low = "#fff7bc", high = "#d73027", labels = comma) +
  labs(
    title = "Average Ticket Price by Departure & Arrival Time",
    subtitle = "Heatmap of mean price across time-of-day combinations",
    x = "Departure Time",
    y = "Arrival Time",
    fill = "Avg Price (INR)"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold"),
    axis.text.x = element_text(angle = 25, hjust = 1)
  )

ggsave("plot3.png", plot = p3, width = 8, height = 5, dpi = 150)
cat("Saved plot3.png\n")

# ---- 6. Price Distribution & Outlier Removal ----
p4_before <- ggplot(df, aes(x = price)) +
  geom_histogram(bins = 80, fill = "#4575b4", alpha = 0.8, color = "white") +
  scale_x_continuous(labels = comma) +
  scale_y_continuous(labels = comma) +
  labs(title = "Price Distribution (Before Outlier Removal)",
       x = "Price (INR)", y = "Frequency") +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"))

# IQR method
Q1 <- quantile(df$price, 0.25)
Q3 <- quantile(df$price, 0.75)
IQR_val <- Q3 - Q1
lower <- Q1 - 1.5 * IQR_val
upper <- Q3 + 1.5 * IQR_val

df_clean <- df %>% filter(price >= lower & price <= upper)
cat(sprintf("Rows before outlier removal: %d\n", nrow(df)))
cat(sprintf("Rows after outlier removal:  %d\n", nrow(df_clean)))
cat(sprintf("Outliers removed: %d\n", nrow(df) - nrow(df_clean)))

p4_after <- ggplot(df_clean, aes(x = price)) +
  geom_histogram(bins = 80, fill = "#1a9641", alpha = 0.8, color = "white") +
  scale_x_continuous(labels = comma) +
  scale_y_continuous(labels = comma) +
  labs(title = "Price Distribution (After Outlier Removal)",
       x = "Price (INR)", y = "Frequency") +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"))

p4 <- grid.arrange(p4_before, p4_after, ncol = 2)
ggsave("plot4.png", plot = p4, width = 12, height = 5, dpi = 150)
cat("Saved plot4.png\n")

# ---- 7. Feature Engineering - One-Hot Encoding ----
df_model <- df_clean

# Encode categorical variables
cat_cols <- c("airline", "source_city", "departure_time", "stops",
              "arrival_time", "destination_city", "class")

for (col in cat_cols) {
  dummies <- model.matrix(~ 0 + df_model[[col]])
  colnames(dummies) <- paste0(col, "_", levels(df_model[[col]]))
  df_model <- cbind(df_model, dummies)
  df_model[[col]] <- NULL
}

# Remove flight column (high cardinality ID)
df_model$flight <- NULL

# ---- 8. Train/Test Split (80:20) ----
set.seed(42)
train_idx <- createDataPartition(df_model$price, p = 0.8, list = FALSE)
train_data <- df_model[train_idx, ]
test_data  <- df_model[-train_idx, ]
cat(sprintf("Train size: %d | Test size: %d\n", nrow(train_data), nrow(test_data)))

# ---- 9. Model M1 - Linear Regression with ALL features ----
m1 <- lm(price ~ ., data = train_data)
m1_pred <- predict(m1, newdata = test_data)

m1_r2   <- cor(test_data$price, m1_pred)^2
m1_rmse <- sqrt(mean((test_data$price - m1_pred)^2))
m1_adj_r2 <- summary(m1)$adj.r.squared

cat("\n--- Model M1 (All Features) ---\n")
cat(sprintf("R2:        %.4f\n", m1_r2))
cat(sprintf("Adj R2:    %.4f\n", m1_adj_r2))
cat(sprintf("RMSE:      %.2f\n", m1_rmse))

# ---- 10. Select Top 5 Features by Coefficient Magnitude ----
coefs <- coef(m1)
coefs <- coefs[!is.na(coefs)]
coefs <- coefs[names(coefs) != "(Intercept)"]
top5 <- names(sort(abs(coefs), decreasing = TRUE))[1:5]
cat("\nTop 5 Features:", paste(top5, collapse = ", "), "\n")

# ---- 11. Model M2 - Linear Regression with Top 5 features ----
formula_m2 <- as.formula(paste("price ~", paste(paste0("`", top5, "`"), collapse = " + ")))
m2 <- lm(formula_m2, data = train_data)
m2_pred <- predict(m2, newdata = test_data)

m2_r2   <- cor(test_data$price, m2_pred)^2
m2_rmse <- sqrt(mean((test_data$price - m2_pred)^2))
m2_adj_r2 <- summary(m2)$adj.r.squared

cat("\n--- Model M2 (Top 5 Features) ---\n")
cat(sprintf("R2:        %.4f\n", m2_r2))
cat(sprintf("Adj R2:    %.4f\n", m2_adj_r2))
cat(sprintf("RMSE:      %.2f\n", m2_rmse))

cat("\n--- Comparison M1 vs M2 ---\n")
cat(sprintf("R2    : M1 = %.4f | M2 = %.4f\n", m1_r2, m2_r2))
cat(sprintf("Adj R2: M1 = %.4f | M2 = %.4f\n", m1_adj_r2, m2_adj_r2))
cat(sprintf("RMSE  : M1 = %.2f | M2 = %.2f\n", m1_rmse, m2_rmse))

# ---- 12. Actual vs Predicted Plot (M1) ----
pred_df <- data.frame(Actual = test_data$price, Predicted = m1_pred)

p5 <- ggplot(pred_df[sample(nrow(pred_df), 5000), ], aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.3, color = "#4575b4", size = 0.8) +
  geom_abline(slope = 1, intercept = 0, color = "red", linewidth = 1.2) +
  scale_x_continuous(labels = comma) +
  scale_y_continuous(labels = comma) +
  labs(
    title = "Actual vs Predicted Flight Price (M1 - All Features)",
    subtitle = sprintf("R² = %.4f | RMSE = %.0f", m1_r2, m1_rmse),
    x = "Actual Price (INR)",
    y = "Predicted Price (INR)"
  ) +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"))

ggsave("plot5.png", plot = p5, width = 8, height = 6, dpi = 150)
cat("Saved plot5.png\n")

# ---- 13. Feature Importance Plot ----
top10 <- names(sort(abs(coefs), decreasing = TRUE))[1:10]
coef_df <- data.frame(
  Feature = top10,
  Coefficient = coefs[top10]
)

p6 <- ggplot(coef_df, aes(x = reorder(Feature, abs(Coefficient)),
                           y = Coefficient, fill = Coefficient > 0)) +
  geom_col(alpha = 0.85) +
  coord_flip() +
  scale_fill_manual(values = c("TRUE" = "#1a9641", "FALSE" = "#d73027"),
                    labels = c("Negative", "Positive")) +
  labs(
    title = "Top 10 Features by Coefficient Magnitude",
    subtitle = "Linear Regression M1 coefficients",
    x = "Feature",
    y = "Coefficient Value",
    fill = "Direction"
  ) +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"))

ggsave("plot6.png", plot = p6, width = 9, height = 5, dpi = 150)
cat("Saved plot6.png\n")

cat("\n============================================================\n")
cat("Analysis Complete! All plots saved.\n")
cat("============================================================\n")
