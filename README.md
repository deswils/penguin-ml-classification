# Palmer Penguins: Comparative ML Classification in R

A comparative machine learning study applying unsupervised and supervised classification methods to the Palmer Penguins dataset. Three scripts cover the full spectrum from clustering through probabilistic classification to decision trees, using the same dataset throughout to make results directly comparable.

## Dataset

**Palmer Penguins** — 333 complete records (344 total, 11 dropped for missing values) across 3 species (Adelie, Chinstrap, Gentoo) from islands in the Palmer Archipelago, Antarctica.

Data is loaded directly from the [`palmerpenguins`](https://allisonhorst.github.io/palmerpenguins/) R package — no CSV download required. Column names follow the original data convention (`culmen_length_mm`, `culmen_depth_mm`) rather than the package defaults (`bill_length_mm`, `bill_depth_mm`).

| Feature | Type | Description |
|---|---|---|
| `species` | Categorical | Adelie / Chinstrap / Gentoo (target) |
| `island` | Categorical | Biscoe / Dream / Torgersen |
| `culmen_length_mm` | Numeric | Bill length (mm) |
| `culmen_depth_mm` | Numeric | Bill depth (mm) |
| `flipper_length_mm` | Numeric | Flipper length (mm) |
| `body_mass_g` | Numeric | Body mass (g) |
| `sex` | Categorical | Male / Female |

---

## Scripts

### `hw2_clustering.R` — Unsupervised Learning
K-means and hierarchical clustering on the four numeric features.

**Key concepts demonstrated:**
- Elbow method (WCSS) for selecting k
- Effect of min-max normalization on k-means results — body mass dominates Euclidean distance on raw data, normalized clustering is more balanced across features
- Hierarchical clustering with Ward's D2 linkage and dendrogram visualization
- Side-by-side comparison of k-means vs. hierarchical cluster assignments

**Figures generated:**
| File | Description |
|---|---|
| `figures/elbow_raw.png` | WCSS elbow plot — raw data |
| `figures/elbow_normalized.png` | WCSS elbow plot — normalized data |
| `figures/kmeans_raw.png` | 4D scatter matrix — raw k-means clusters |
| `figures/kmeans_normalized.png` | 4D scatter matrix — normalized k-means clusters |
| `figures/dendrogram.png` | Ward's D2 dendrogram with k=3 highlighted |
| `figures/hclust_clusters.png` | 4D scatter matrix — hierarchical clusters |

---

### `hw4_naive_bayes.R` — Supervised Learning: Probabilistic Classification
Naïve Bayes classifier with stratified 10-fold cross-validation.

**Key concepts demonstrated:**
- Stratified fold construction — preserves species class balance across folds for reliable CV estimates
- Aggregate confusion matrix across all 10 folds
- Per-class sensitivity and specificity using one-vs-rest decomposition
- Robustness comparison: stratified 10-fold CV vs. a single random train/test split — illustrates why a single split with a small test set produces misleadingly high accuracy

**Figures generated:**
| File | Description |
|---|---|
| `figures/confusion_matrix.png` | Heatmap of aggregate confusion matrix |

**Results (10-fold CV):**
| Species | Sensitivity | Specificity |
|---|---|---|
| Adelie | ~0.993 | ~0.972 |
| Chinstrap | ~0.917 | ~0.996 |
| Gentoo | ~1.000 | ~1.000 |

---

### `hw5_decision_trees.R` — Supervised Learning: Decision Trees  
`rpart` decision trees with comparison of two complexity parameter settings.

**Key concepts demonstrated:**
- Decision tree construction with `rpart` using `minsplit` and `cp` controls
- Effect of pruning: higher `cp` / larger `minsplit` produces a simpler, more generalized tree
- Per-class specificity from programmatically generated confusion matrix

**Models compared:**
| Model | `minsplit` | `cp` | Behavior |
|---|---|---|---|
| `model` | 2 | 0.001 | Deep, less pruned |
| `model_2` | 5 | 0.01 | Shallower, more pruned |

---

## Packages

```r
palmerpenguins, tidyverse, e1071, rpart
```

All packages are installed automatically if not already present. No data download required — the dataset is loaded directly from the `palmerpenguins` package.

## How to Run

1. Clone the repo
2. Run each script independently in RStudio or via `Rscript` — a `figures/` directory is created automatically and all plots are saved there

```bash
Rscript hw2_clustering.R
Rscript hw4_naive_bayes.R
Rscript hw5_decision_trees.R
```

## Reference

Horst A.M., Hill A.P., & Gorman K.B. (2020). *palmerpenguins: Palmer Archipelago (Antarctica) penguin data*. R package version 0.1.0.
