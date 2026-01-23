"""
Scalability Analysis using FA Scores

This script loads fa_scores_with_score.csv and produces:
    1) FA_Score distribution for scalable vs non-scalable
    2) FA_Score vs nvar scatter (colored by scalability)
    3) Solver-wise scaling visualization (FA_Score vs nvar grouped by solver)

Outputs all figures into ./scalability_output/
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------- Configuration ----------------

INPUT_PATH = Path("./output_data3/fa_scores_with_score.csv")  # ← change if needed
OUTDIR = Path("./scalability_output")


# ---------------- Core Plotting Functions ----------------

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def plot_fa_score_distribution(df):
    """Draw FA_Score distribution for scalable vs non-scalable problems."""
    plt.figure(figsize=(7, 4))
    sns.boxplot(data=df, x="is_scalable", y="FA_Score", palette="Set2")
    plt.title("FA_Score Distribution by Scalability")
    plt.savefig(OUTDIR / "fa_score_boxplot.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    sns.violinplot(data=df, x="is_scalable", y="FA_Score", palette="Set2")
    plt.title("FA_Score Violin Plot by Scalability")
    plt.savefig(OUTDIR / "fa_score_violin.png", dpi=200)
    plt.close()


def plot_scaling_vs_nvar(df):
    """Scatter: FA_Score vs nvar, colored by scalability."""
    plt.figure(figsize=(6, 4))
    sns.scatterplot(
        data=df, x="nvar", y="FA_Score",
        hue="is_scalable", palette="Set1", s=20, alpha=0.7
    )
    plt.title("Scaling Behavior: FA_Score vs nvar")
    plt.savefig(OUTDIR / "fa_score_vs_nvar.png", dpi=200)
    plt.close()


def plot_solver_wise(df):
    """FA_Score vs nvar per solver."""
    # Only plot if multiple solvers exist
    if df["solver"].nunique() > 1:
        g = sns.lmplot(
            data=df, x="nvar", y="FA_Score",
            hue="is_scalable", col="solver",
            col_wrap=3, height=3,
            scatter_kws={"s": 12, "alpha": 0.6},
            line_kws={"linewidth": 1.2}
        )
        plt.savefig(OUTDIR / "solver_wise_scaling.png", dpi=200)
        plt.close()


# ---------------- Main ----------------

def main():
    ensure_dir(OUTDIR)

    print("Loading data...")
    df = pd.read_csv(INPUT_PATH)

    # Ensure boolean consistency
    df["is_scalable"] = df["is_scalable"].astype(str).str.lower().isin(["true", "1"])

    print(f"Loaded {df.shape[0]} rows.")

    # 1) Distribution plots
    print("Plotting FA_Score distribution...")
    plot_fa_score_distribution(df)

    # 2) Scatter plot scaling
    print("Plotting FA_Score vs nvar...")
    if "nvar" in df.columns:
        plot_scaling_vs_nvar(df)
    else:
        print("Warning: nvar column not found, skipping scaling plot.")

    # 3) Solver-wise comparison
    print("Plotting solver-wise scaling visualization...")
    if "solver" in df.columns:
        plot_solver_wise(df)
    else:
        print("Warning: solver column not found, skipping solver-wise plot.")

    print(f"All plots saved to {OUTDIR.resolve()}")


if __name__ == "__main__":
    main()
