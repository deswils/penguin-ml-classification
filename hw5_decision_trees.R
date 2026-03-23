# =============================================================================
# Penguin Species Classification: Decision Trees with rpart
# =============================================================================
#
# Dataset: Palmer Penguins (palmerpenguins package)
#   333 complete records across 3 species: Adelie, Chinstrap, Gentoo
#   Features: island, culmen_length_mm, culmen_depth_mm,
#             flipper_length_mm, body_mass_g
#
# Compares two rpart decision tree models with different complexity parameters:
#   model   — minsplit = 2,  cp = 0.001  (deep, less pruned)
#   model_2 — minsplit = 5,  cp = 0.01   (shallower, more pruned)
#
# Higher cp and larger minsplit produce a simpler tree that is less likely
# to overfit but may miss finer decision boundaries.
# =============================================================================


# =============================================================================
# 1. SETUP
# =============================================================================

library(tidyverse)
library(rpart)

if (!requireNamespace("palmerpenguins", quietly = TRUE))
  install.packages("palmerpenguins", repos = "http://cran.us.r-project.org")
library(palmerpenguins)

data(penguins)

# Rename bill_* to culmen_* to match original CSV column naming convention
# and drop rows with missing values
penguins <- penguins %>%
  rename(
    culmen_length_mm = bill_length_mm,
    culmen_depth_mm  = bill_depth_mm
  ) %>%
  drop_na()

glimpse(penguins)

# Create figures directory if it doesn't exist
if (!dir.exists("figures")) dir.create("figures")


# =============================================================================
# 2. MODEL 1 — Deep Tree (minsplit = 2, cp = 0.001)
#
# Low cp and minsplit allow the tree to grow deep, splitting on small gains.
# More complex but may overfit to training data.
# =============================================================================

model <- rpart(
  species ~ island + culmen_length_mm + culmen_depth_mm +
            flipper_length_mm + body_mass_g,
  data    = penguins,
  control = rpart.control(minsplit = 2, cp = 0.001)
)

# Plot Model 1
png("figures/tree_model1.png", width = 1000, height = 700)
plot(model, uniform = TRUE,
     main   = "Decision Tree — minsplit = 2, cp = 0.001",
     margin = 0.1)
text(model, use.n = TRUE, all = TRUE, cex = 0.7)
dev.off()


# =============================================================================
# 3. PREDICT WITH MODEL 1
#
# Example penguin: island = Biscoe, flipper = 180mm, culmen length = 43mm,
#                  culmen depth = 18mm, body mass = 4000g
# Predicted species: Adelie
# =============================================================================

new_penguin <- data.frame(
  island            = "Biscoe",
  culmen_length_mm  = 43,
  culmen_depth_mm   = 18,
  flipper_length_mm = 180,
  body_mass_g       = 4000
)

cat("\n--- Model 1 prediction for example penguin ---\n")
print(predict(model, new_penguin, type = "class"))
# Expected: Adelie


# =============================================================================
# 4. MODEL 2 — Pruned Tree (minsplit = 5, cp = 0.01)
#
# Higher cp and minsplit prune the tree more aggressively.
# Simpler, more interpretable, and more likely to generalize.
# =============================================================================

model_2 <- rpart(
  species ~ island + culmen_length_mm + culmen_depth_mm +
            flipper_length_mm + body_mass_g,
  data    = penguins,
  control = rpart.control(minsplit = 5, cp = 0.01)
)

# Plot Model 2
png("figures/tree_model2.png", width = 1000, height = 700)
plot(model_2, uniform = TRUE,
     main   = "Decision Tree — minsplit = 5, cp = 0.01",
     margin = 0.1)
text(model_2, use.n = TRUE, all = TRUE, cex = 0.7)
dev.off()


# =============================================================================
# 5. SPECIFICITY FOR CHINSTRAP — MODEL 2
#
# Generate predictions from model_2 and build confusion matrix.
# Specificity = TN / (TN + FP)
# For Chinstrap: TN = non-Chinstrap correctly classified as non-Chinstrap
#                FP = non-Chinstrap incorrectly classified as Chinstrap
# =============================================================================

preds_2   <- predict(model_2, penguins, type = "class")
conf_mat_2 <- table(Predicted = preds_2, Actual = penguins$species)

cat("\n--- Model 2 Confusion Matrix ---\n")
print(conf_mat_2)

# Chinstrap is the 2nd level — extract TN and FP using -2 indexing
chinstrap_TN <- sum(conf_mat_2[-2, -2])  # non-Chinstrap correctly classified
chinstrap_FP <- sum(conf_mat_2[2, -2])   # non-Chinstrap predicted as Chinstrap

chinstrap_specificity <- chinstrap_TN / (chinstrap_TN + chinstrap_FP)
cat(sprintf("\nChinstrap Specificity (Model 2): %.4f\n", chinstrap_specificity))
# Expected: 1.0 (no non-Chinstrap penguins misclassified as Chinstrap)
