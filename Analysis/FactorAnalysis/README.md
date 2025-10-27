# Optimization Metrics Analysis

This project analyzes optimization solver performance metrics using **Principal Component Analysis (PCA)** and **Factor Analysis (FA)**.  
The goal is to identify hidden patterns in solver behavior and reduce multiple correlated metrics into a single, interpretable score.

# Data Description

- **lbfgs_week6.csv**: Input dataset containing solver runs and performance metrics.  
- **FA_func.pdf**: Notes explaining the mathematical formulation of Factor Analysis.  
- **scree_plot.png**: PCA scree plot showing explained variance by each component.  

### Output Data (`data/output_data/`)

- **mean_std.csv**: Mean and standard deviation for each metric after standardization.  
- **x_scaled.csv**: Standardized metrics (each column scaled to mean 0 and variance 1).  
- **pca_explained_variance.csv**: Explained and cumulative variance from PCA.  
- **fa_loadings_matrix.csv**: Factor loadings showing how each metric contributes to the latent factor.  
- **fa_scores_with_score.csv**: Original metrics and computed factor scores (`FA_F1`, `FA_Score`).  
