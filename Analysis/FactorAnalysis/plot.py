"""
Plot distributions (raw + log) for optimization metric columns.

This script is independent from FA/PCA pipeline and can be run separately.

Outputs:
--------
1) /plots_metrics/<metric>_raw.png
2) /plots_metrics/<metric>_log.png
3) /plots_metrics/metric_distribution_summary.csv
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------- Configuration ----------------

INPUT_PATH = Path("./data/complete_dataset_as_of_Nov6.csv")
OUTDIR = Path("./plots_metrics")

METRIC_COLS = ["neval_obj", "neval_grad", "num_iter", "mem", "time"]


# ---------------- Helper Functions ----------------

def load_and_filter(path: Path) -> pd.DataFrame:
    """Load CSV and drop init-runs."""
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path)

    if "is_init_run" in df.columns:
        df = df[df["is_init_run"].astype(str).str.lower() != "true"]

    return df


def plot_one_metric(df: pd.DataFrame, col: str, outdir: Path):
    """Plot raw + log-transformed histograms for the given metric."""

    x = df[col].dropna()
    log_x = np.log1p(x)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3))

    # raw
    sns.histplot(x, bins=40, kde=True, ax=axes[0],
                 color="salmon", edgecolor="black", alpha=0.6)
    axes[0].set_title(f"{col} (raw)")

    # log
    sns.histplot(log_x, bins=40, kde=True, ax=axes[1],
                 color="seagreen", edgecolor="black", alpha=0.6)
    axes[1].set_title(f"{col} (log-transformed)")

    plt.tight_layout()
    fig.savefig(outdir / f"{col}_raw_log.png", dpi=200)
    plt.close(fig)


def compute_summary(df: pd.DataFrame, metric_cols: list) -> pd.DataFrame:
    rows = []

    for col in metric_cols:
        x = df[col].dropna()
        log_x = np.log1p(x)

        rows.append({
            "metric": col,
            "skew_raw": x.skew(),
            "kurt_raw": x.kurt(),
            "skew_log": log_x.skew(),
            "kurt_log": log_x.kurt()
        })

    return pd.DataFrame(rows)


# ---------------- Main ----------------

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    df = load_and_filter(INPUT_PATH)

    # --- Generate plots ---
    for col in METRIC_COLS:
        print(f"Plotting distributions for {col} ...")
        plot_one_metric(df, col, OUTDIR)

    # --- Summary statistics ---
    summary_df = compute_summary(df, METRIC_COLS)
    summary_df.to_csv(OUTDIR / "metric_distribution_summary.csv", index=False)

    print(f"Completed. Plots saved to: {OUTDIR}")


if __name__ == "__main__":
    main()
