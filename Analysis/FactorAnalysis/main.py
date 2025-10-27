"""
FA/PCA pipeline for optimization-metrics CSVs.

Steps
------
1) Load CSV
2) Drop init runs (is_init_run == True)
3) Standardize metric columns
4) Generate PCA variance report
5) Perform 1-factor FA
6) Export:
   - data/mean_std.csv
   - data/x_scaled.csv
   - data/pca_explained_variance.csv
   - data/fa_loadings_matrix.csv
   - data/fa_scores_with_score.csv
"""

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import StandardScaler

# ------------ Configuration ------------

INPUT_PATH = Path("./data/lbfgs_week6.csv")  # load data
OUTPUT_DIR = Path("./output_data")

METRIC_COLS = ["nvmops", "neval_obj", "neval_grad", "num_iter", "mem"]
ID_COL_CANDIDATES = ["status", "name", "solver", "mem", "nvar"]


# ------------ Core Functions ------------

def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    df = pd.read_csv(path)
    logging.info("Loaded %s with shape %s", path.name, df.shape)
    return df


def filter_init_runs(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where is_init_run == 'true' (case-insensitive)."""
    if "is_init_run" not in df.columns:
        return df.copy()
    mask = df["is_init_run"].astype(str).str.lower() != "true"
    return df.loc[mask].copy()


def choose_id_cols(df: pd.DataFrame, candidates: List[str]) -> List[str]:
    return [c for c in candidates if c in df.columns]


def check_required(df: pd.DataFrame, required: List[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required metric columns: {missing}")


def standardize(df: pd.DataFrame, metric_cols: List[str]) -> Tuple[np.ndarray, StandardScaler]:
    X = df[metric_cols].to_numpy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def pca_report(X_scaled: np.ndarray) -> pd.DataFrame:
    pca = PCA().fit(X_scaled)
    explained = pca.explained_variance_ratio_
    cumulative = explained.cumsum()
    return pd.DataFrame({
        "PC": [f"PC{i + 1}" for i in range(len(explained))],
        "explained": explained,
        "cumulative": cumulative
    })


def run_factor_analysis(X_scaled: np.ndarray, n_components: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    fa = FactorAnalysis(n_components=n_components, random_state=0)
    F = fa.fit_transform(X_scaled)  # Latent Factor Score Matrix, formula in ./data/FA_func.pdf
    loadings = fa.components_  # Loading martix
    return F, loadings


def build_scores(df: pd.DataFrame, id_cols: List[str], metric_cols: List[str], F: np.ndarray) -> pd.DataFrame:
    if F.shape[0] != len(df):
        raise ValueError("Row mismatch between FA scores and dataframe.")
    result = df[id_cols + metric_cols].copy()
    result["FA_F1"] = F[:, 0]  # Extract the first column, only 1 factor score
    result["FA_Score"] = result["FA_F1"]
    return result


def save_outputs(outdir: Path, metric_cols: List[str], scaler: StandardScaler,
                 X_scaled: np.ndarray, pca_df: pd.DataFrame,
                 fa_loadings: np.ndarray, scores_df: pd.DataFrame) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    """Convert to csv files."""

    # Save standardization summary
    pd.DataFrame({
        "mean": scaler.mean_,
        "std": scaler.scale_
    }, index=metric_cols).to_csv(outdir / "mean_std.csv")

    # Save standardized matrix
    pd.DataFrame(X_scaled, columns=metric_cols).to_csv(outdir / "x_scaled.csv", index=False)

    # Save PCA variance table
    pca_df.to_csv(outdir / "pca_explained_variance.csv", index=False)

    # Save FA loadings and scores
    pd.DataFrame(fa_loadings, columns=metric_cols, index=["Factor1"]).to_csv(outdir / "fa_loadings_matrix.csv")
    scores_df.to_csv(outdir / "fa_scores_with_score.csv", index=False)

    logging.info("Saved outputs to %s", outdir)


# ------------ Main ------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")  # just to keep track

    df = load_csv(INPUT_PATH)
    df = filter_init_runs(df)
    check_required(df, METRIC_COLS)

    id_cols = choose_id_cols(df, ID_COL_CANDIDATES)

    X_scaled, scaler = standardize(df, METRIC_COLS)
    pca_df = pca_report(X_scaled)
    F, loadings = run_factor_analysis(X_scaled, n_components=1)
    scores_df = build_scores(df, id_cols, METRIC_COLS, F)

    save_outputs(OUTPUT_DIR, METRIC_COLS, scaler, X_scaled, pca_df, loadings, scores_df)

    logging.info("Pipeline complete. Top cumulative variance: %.2f%%", pca_df['cumulative'].iloc[0] * 100)


if __name__ == "__main__":
    main()
