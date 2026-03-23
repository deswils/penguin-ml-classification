# =============================================================================
# Penguin Species Classification: Naïve Bayes with Stratified 10-Fold CV
# =============================================================================
#
# Dataset: Palmer Penguins (penguins.csv)
#   344 penguin records across 3 species: Adelie (151), Chinstrap (68), Gentoo (124)
#   Features: island, culmen_length_mm, culmen_depth_mm, flipper_length_mm, body_mass_g, sex
#
# Method: Naïve Bayes classifier with stratified 10-fold cross-validation.
#   Stratification ensures each fold preserves the species class distribution,
#   giving a more reliable estimate of out-of-sample classification performance
#   than a single random train/test split.
#
# Sections:
#   1. Setup
#   2. Stratified fold index construction
#   3. 10-fold CV loop & aggregate confusion matrix
#   4. Per-class sensitivity & specificity
#   5. Robustness check: why a single split overfits
# =============================================================================


# =============================================================================
# 1. SETUP
# =============================================================================

library(tidyverse)
library(e1071)

penguins <- read_csv("penguins.csv")
glimpse(penguins)

# Species counts — needed to size the stratified folds
table(penguins$species)
# Adelie: 151 (rows 1-151), Chinstrap: 68 (rows 152-219), Gentoo: 124 (rows 220-342)


# =============================================================================
# 2. STRATIFIED FOLD INDEX CONSTRUCTION
#
# For each species, randomly sample the largest multiple of 10 that fits within
# the species count, then arrange those indices into a 10-row array.
# Each row = one fold's test indices for that species.
#
# Species counts and fold sizes:
#   Adelie:    151 records -> 150 used -> 10 folds x 15
#   Chinstrap:  68 records ->  60 used -> 10 folds x  6
#   Gentoo:    124 records -> 120 used -> 10 folds x 12
# =============================================================================

set.seed(587)
adelie_folds    <- array(sample(1:151,   150), dim = c(10, 15))

set.seed(587)
chinstrap_folds <- array(sample(152:219,  60), dim = c(10,  6))

set.seed(587)
gentoo_folds    <- array(sample(220:342, 120), dim = c(10, 12))


# =============================================================================
# 3. 10-FOLD CV LOOP & AGGREGATE CONFUSION MATRIX
#
# Each iteration:
#   - Test set:  ith row of each species' fold array (combined across species)
#   - Train set: all records NOT in the test set
#   - Model:     Naïve Bayes on features (columns 2-7), predicting species
#
# Confusion matrix is accumulated across all 10 folds for an aggregate
# picture of classification performance across the full dataset.
# =============================================================================

conf_mat <- NULL

for (i in 1:10) {
  test_idx   <- c(adelie_folds[i, ], chinstrap_folds[i, ], gentoo_folds[i, ])
  test_set   <- penguins[test_idx, ]
  train_set  <- penguins[-test_idx, ]

  model <- naiveBayes(train_set[, 2:7], train_set$species)
  preds <- predict(model, test_set[, 2:7])

  fold_conf <- table(Predicted = preds, Actual = test_set$species)

  if (is.null(conf_mat)) conf_mat <- fold_conf
  else conf_mat <- conf_mat + fold_conf
}

cat("\n--- Aggregate Confusion Matrix (10-Fold Stratified CV) ---\n")
print(conf_mat)

# Save confusion matrix as figure
png("figures/confusion_matrix.png", width = 600, height = 500)
par(mar = c(5, 6, 4, 2))
image(t(conf_mat[nrow(conf_mat):1, ]),
      axes  = FALSE,
      col   = colorRampPalette(c("white", "steelblue"))(20),
      main  = "Naïve Bayes Confusion Matrix (10-Fold CV)")
axis(1, at = seq(0, 1, length = ncol(conf_mat)), labels = colnames(conf_mat))
axis(2, at = seq(0, 1, length = nrow(conf_mat)), labels = rev(rownames(conf_mat)), las = 1)
for (r in 1:nrow(conf_mat)) {
  for (cc in 1:ncol(conf_mat)) {
    text(x = (cc - 1) / (ncol(conf_mat) - 1),
         y = 1 - (r - 1) / (nrow(conf_mat) - 1),
         labels = conf_mat[r, cc], cex = 1.5)
  }
}
dev.off()


# =============================================================================
# 4. PER-CLASS SENSITIVITY & SPECIFICITY
#
# For a multiclass confusion matrix, treat each class as a one-vs-rest problem:
#   Sensitivity = TP / (TP + FN)  — how well we detect the true positives
#   Specificity = TN / (TN + FP)  — how well we reject true negatives
# =============================================================================

calc_metrics <- function(conf, class_idx) {
  TP <- conf[class_idx, class_idx]
  FN <- sum(conf[-class_idx, class_idx])   # actual = class, predicted ≠ class
  FP <- sum(conf[class_idx, -class_idx])   # predicted = class, actual ≠ class
  TN <- sum(conf[-class_idx, -class_idx])  # actual ≠ class, predicted ≠ class

  sensitivity <- TP / (TP + FN)
  specificity <- TN / (TN + FP)

  list(sensitivity = round(sensitivity, 4),
       specificity = round(specificity, 4))
}

species_names <- colnames(conf_mat)

cat("\n--- Per-Class Sensitivity & Specificity ---\n")
for (i in seq_along(species_names)) {
  m <- calc_metrics(conf_mat, i)
  cat(sprintf("%-12s  Sensitivity: %.4f  |  Specificity: %.4f\n",
              species_names[i], m$sensitivity, m$specificity))
}

# Expected results:
#   Adelie:    Sensitivity ~0.993  |  Specificity ~0.972
#   Chinstrap: Sensitivity ~0.917  |  Specificity ~0.996
#   Gentoo:    Sensitivity ~1.000  |  Specificity ~1.000


# =============================================================================
# 5. ROBUSTNESS CHECK: WHY A SINGLE RANDOM SPLIT OVERFITS
#
# A classmate's approach: set.seed(901), randomly sample 10 records as test set.
# This yields 100% accuracy — but the result is misleading because:
#   - The test set is too small (10 of 344 records) to be reliable
#   - The split is not stratified (e.g., 8 of 10 test records may be Gentoo)
#   - Training on 332 of 342 records means the model has seen nearly all the data
#   - A different seed would yield very different accuracy estimates
#
# Stratified 10-fold CV avoids all of these problems.
# =============================================================================

set.seed(901)
test_idx_single  <- sample(1:342, 10)
train_single     <- penguins[-test_idx_single, ]
test_single      <- penguins[test_idx_single, ]

model_single     <- naiveBayes(train_single[, 2:7], train_single$species)
preds_single     <- predict(model_single, test_single[, 2:7])
conf_single      <- table(Predicted = preds_single, Actual = test_single$species)

cat("\n--- Single Random Split Confusion Matrix (seed = 901, n_test = 10) ---\n")
print(conf_single)

cat("\nSpecies distribution in this test set:\n")
print(table(test_single$species))

# Likely result: all 10 classified correctly, but 8+ records are one species.
# This does NOT generalize — stratified 10-fold CV is the right approach.
