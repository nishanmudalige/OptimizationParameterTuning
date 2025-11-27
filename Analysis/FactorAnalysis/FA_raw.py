import numpy as np
import pandas as pd
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler
import main
# load data
df = pd.read_csv("./data/lbfgs_week6.csv")

df_filtered = main.filter_init_runs(df)
df_transformed = df_filtered.copy()

metric_cols = ["nvmops", "neval_obj", "neval_grad", "num_iter", "mem"]
X_raw = df_filtered[metric_cols].to_numpy()
df_transformed[metric_cols] = np.log1p(df_transformed[metric_cols].clip(lower=0))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# --------------------------
# 3. Fit FA on scaled data
# --------------------------
fa = FactorAnalysis(n_components=1, random_state=0)
F_sklearn = fa.fit_transform(X_scaled)  # sklearn's own scores

# --------------------------
# 4. Manual computation using the formula from FA_func.pdf
# --------------------------
W = fa.components_          # (k, d), k = number of latent factors, d = number of observed variables
Psi = fa.noise_variance_    # (d,)
mu = fa.mean_               # (d,)

# Prepare terms
Wpsi = W / Psi              # elementwise divide, shape (k, d)
I = np.eye(W.shape[0])      # Identity Matrix (k, k)
G = np.linalg.inv(I + Wpsi @ W.T)  # (k, k)

# Center data
Xc = X_scaled - mu          # shape (n, d)

# Manual computation
F_manual_scaled = (Xc @ Wpsi.T) @ G  # (n, k)

# --------------------------
# 5. Compare results
# --------------------------
print("Sklearn FA (scaled) scores head:")
print(F_sklearn[:5])
print("\nManual FA (scaled) scores head:")
print(F_manual_scaled[:5])

# --------------------------
# 6. Try the same with raw (non-scaled) data
# --------------------------
fa_raw = FactorAnalysis(n_components=1, random_state=0)
F_sklearn_raw = fa_raw.fit_transform(X_raw)

W_raw = fa_raw.components_
Psi_raw = fa_raw.noise_variance_
mu_raw = fa_raw.mean_

Wpsi_raw = W_raw / Psi_raw
I_raw = np.eye(W_raw.shape[0])
G_raw = np.linalg.inv(I_raw + Wpsi_raw @ W_raw.T)
Xc_raw = X_raw - mu_raw

F_manual_raw = (Xc_raw @ Wpsi_raw.T) @ G_raw

print("\nSklearn FA (non-scaled) scores head:")
print(F_sklearn_raw[:5])
print("\nManual FA (non-scaled) scores head:")
print(F_manual_raw[:5])

# --------------------------
# 7. Optional: Compare magnitudes
# --------------------------
#print("\nStd of FA scores (scaled):", F_manual_scaled.std())
#print("Std of FA scores (non-scaled):", F_manual_raw.std())
