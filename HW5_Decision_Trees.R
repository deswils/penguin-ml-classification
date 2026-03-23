# DeShan Wilson
# CSCE 587
# Homework 5 — Decision Trees in R
#
# Predicts penguin species using island, flipper length, culmen length,
# culmen depth, and body mass via rpart decision trees.
# Compares two models with different complexity parameters (cp) and
# minimum split sizes (minsplit).

library(tidyverse)
library(rpart)

# Load penguins dataset
penguins <- read.csv("penguins.csv")


# =============================================================================
# PART A: Build Model 1 (minsplit = 2, cp = 0.001) — less pruned, deeper tree
# =============================================================================

model <- rpart(
  species ~ island + flipper_length_mm + culmen_length_mm + culmen_depth_mm + body_mass_g,
  data    = penguins,
  control = rpart.control(minsplit = 2, cp = 0.001)
)


# =============================================================================
# PART B: Plot Model 1
# =============================================================================

plot(model, uniform = TRUE,
     main   = "Penguin Species Decision Tree (minsplit = 2, cp = 0.001)",
     margin = 0.1)
text(model, use.n = TRUE, all = TRUE, cex = 0.7)


# =============================================================================
# PART C: Predict species for a specific penguin using Model 1
#
# Values: island = Biscoe, flipper_length_mm = 180, culmen_length_mm = 43,
#         culmen_depth_mm = 18, body_mass_g = 4000
# Prediction: Adelie
# =============================================================================

new_penguin <- data.frame(
  island            = "Biscoe",
  flipper_length_mm = 180,
  culmen_length_mm  = 43,
  culmen_depth_mm   = 18,
  body_mass_g       = 4000
)

predict(model, new_penguin, type = "class")
# Predicted species: Adelie


# =============================================================================
# PART D: Build Model 2 (minsplit = 5, cp = 0.01) — more pruned, simpler tree
# =============================================================================

model_2 <- rpart(
  species ~ island + flipper_length_mm + culmen_length_mm + culmen_depth_mm + body_mass_g,
  data    = penguins,
  control = rpart.control(minsplit = 5, cp = 0.01)
)

# Plot Model 2
plot(model_2, uniform = TRUE,
     main   = "Penguin Species Decision Tree (minsplit = 5, cp = 0.01)",
     margin = 0.1)
text(model_2, use.n = TRUE, all = TRUE, cex = 0.7)


# =============================================================================
# PART E: Specificity for Chinstrap using Model 2
#
# Specificity = TN / (TN + FP)
# For Chinstrap: TN = all non-Chinstrap penguins correctly predicted as non-Chinstrap
#                FP = non-Chinstrap penguins incorrectly predicted as Chinstrap
# =============================================================================

# Generate predictions from model_2 (not model)
predictions_2 <- predict(model_2, penguins, type = "class")

# Build confusion matrix from model_2 predictions
conf_mat_2 <- table(Predicted = predictions_2, Actual = penguins$species)
print(conf_mat_2)

# Chinstrap specificity from model_2 confusion matrix
# Chinstrap is the 2nd column/row in the matrix
chinstrap_TN <- sum(conf_mat_2[-2, -2])  # all non-Chinstrap correctly classified
chinstrap_FP <- sum(conf_mat_2[2, -2])   # non-Chinstrap predicted as Chinstrap

chinstrap_specificity_2 <- chinstrap_TN / (chinstrap_TN + chinstrap_FP)
cat("Chinstrap Specificity (Model 2):", round(chinstrap_specificity_2, 4), "\n")

# Chinstrap specificity = TN / (TN + FP) = 274 / (274 + 0) = 1.0
