# =============================================================================
# Penguin Species Clustering: K-Means & Hierarchical Clustering
# =============================================================================
#
# Dataset: Palmer Penguins (palmerpenguins package)
#   344 penguin records across 3 species: Adelie, Chinstrap, Gentoo
#   Features used: culmen_length_mm, culmen_depth_mm, flipper_length_mm, body_mass_g
#
# Methods:
#   1. K-Means Clustering (raw and normalized data)
#   2. Hierarchical Clustering (Ward's D2 linkage)
#
# Key concept: K-means uses Euclidean distance, so features with larger scales
# (e.g. body_mass_g) dominate unless data is normalized. This script compares
# clustering results before and after min-max normalization.
# =============================================================================


# =============================================================================
# 1. SETUP
# =============================================================================

library(tidyverse)

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

# Numeric features used for clustering
numeric_features <- c("culmen_length_mm", "culmen_depth_mm",
                       "flipper_length_mm", "body_mass_g")

# Create figures directory if it doesn't exist
if (!dir.exists("figures")) dir.create("figures")


# =============================================================================
# 2. K-MEANS ON RAW DATA
#
# Use the elbow method (WCSS plot) to select k.
# WCSS = Within-Cluster Sum of Squares — lower is better, but adding more
# clusters always reduces WCSS. The "elbow" is where marginal gains diminish.
# =============================================================================

# --- Elbow plot: k = 1 through 20 ---
wcss_raw <- numeric(20)
for (k in 1:20) {
  set.seed(1801)
  wcss_raw[k] <- sum(kmeans(penguins[, numeric_features], centers = k)$withinss)
}

png("figures/elbow_raw.png", width = 800, height = 500)
plot(1:20, wcss_raw, type = "b",
     xlab = "Number of Clusters (k)",
     ylab = "Within-Cluster Sum of Squares (WCSS)",
     main = "Elbow Method — Raw Data",
     pch  = 19, col = "steelblue")
abline(v = 3, lty = 2, col = "red")
dev.off()

# Elbow is at k = 3, consistent with 3 known penguin species

# --- K-means with k = 3 ---
set.seed(1801)
km_raw <- kmeans(penguins[, numeric_features], centers = 3, nstart = 15)

png("figures/kmeans_raw.png", width = 800, height = 800)
plot(penguins[, numeric_features],
     col  = km_raw$cluster,
     main = "K-Means Clustering (k = 3) — Raw Data",
     pch  = 19, cex = 0.6)
dev.off()


# =============================================================================
# 3. NORMALIZATION & K-MEANS ON NORMALIZED DATA
#
# Min-max normalization rescales each feature to [0, 1].
# This prevents body_mass_g (range ~2700-6300g) from dominating Euclidean
# distance calculations relative to smaller-range features.
# =============================================================================

min_max_norm <- function(x) (x - min(x, na.rm = TRUE)) /
                             (max(x, na.rm = TRUE) - min(x, na.rm = TRUE))

penguins_scaled <- as.data.frame(lapply(penguins[, numeric_features], min_max_norm))

summary(penguins_scaled)  # all columns should now be in [0, 1]

# --- Elbow plot: normalized data ---
wcss_scaled <- numeric(20)
for (k in 1:20) {
  set.seed(1801)
  wcss_scaled[k] <- sum(kmeans(penguins_scaled, centers = k)$withinss)
}

png("figures/elbow_normalized.png", width = 800, height = 500)
plot(1:20, wcss_scaled, type = "b",
     xlab = "Number of Clusters (k)",
     ylab = "Within-Cluster Sum of Squares (WCSS)",
     main = "Elbow Method — Normalized Data",
     pch  = 19, col = "steelblue")
abline(v = 3, lty = 2, col = "red")
dev.off()

# --- K-means with k = 3 on normalized data ---
set.seed(1801)
km_scaled <- kmeans(penguins_scaled, centers = 3, nstart = 15)

png("figures/kmeans_normalized.png", width = 800, height = 800)
plot(penguins_scaled,
     col  = km_scaled$cluster,
     main = "K-Means Clustering (k = 3) — Normalized Data",
     pch  = 19, cex = 0.6)
dev.off()

# Note: Clusters differ between raw and normalized results because body_mass_g
# no longer dominates Euclidean distance after normalization. The normalized
# clusters are more evenly influenced by all four features.


# =============================================================================
# 4. HIERARCHICAL CLUSTERING (Ward's D2)
#
# Ward's method minimizes total within-cluster variance at each merge step,
# producing compact, roughly equal-sized clusters. Applied to normalized data.
# =============================================================================

# --- Distance matrix (Euclidean) ---
dist_matrix <- dist(penguins_scaled, method = "euclidean")

# --- Hierarchical clustering ---
peng_hclust <- hclust(dist_matrix, method = "ward.D2")

# --- Dendrogram with 3-cluster solution outlined ---
png("figures/dendrogram.png", width = 1000, height = 600)
plot(peng_hclust,
     main   = "Hierarchical Clustering Dendrogram (Ward's D2)",
     xlab   = "",
     sub    = "",
     labels = FALSE,
     hang   = -1)
rect.hclust(peng_hclust, k = 3, border = "red")
dev.off()

# --- Cut tree at k = 3 and plot ---
hclust_labels <- cutree(peng_hclust, k = 3)

png("figures/hclust_clusters.png", width = 800, height = 800)
plot(penguins_scaled,
     col  = hclust_labels,
     main = "Hierarchical Clustering (k = 3, Ward's D2) — Normalized Data",
     pch  = 19, cex = 0.6)
dev.off()


# =============================================================================
# 5. COMPARISON: K-MEANS VS HIERARCHICAL CLUSTERING
#
# Both methods recover 3 clusters on normalized data.
# K-means is faster but sensitive to initialization (mitigated with nstart = 15).
# Hierarchical clustering is deterministic and provides a full dendrogram,
# but is more computationally expensive for large datasets.
# =============================================================================

cat("K-Means cluster sizes (normalized):\n")
print(table(km_scaled$cluster))

cat("\nHierarchical cluster sizes (k = 3):\n")
print(table(hclust_labels))

cat("\nCluster assignment agreement table:\n")
print(table(KMeans = km_scaled$cluster, HClust = hclust_labels))
