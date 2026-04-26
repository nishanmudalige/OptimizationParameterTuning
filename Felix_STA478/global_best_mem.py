import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# ─────────────────────────────────────────────
#  Paths & constants
# ─────────────────────────────────────────────
PLOTS_PATH   = os.path.join(".", "global_mem_plots")
TIMEOUT_S    = 3600
PENALTY_TIME = TIMEOUT_S * 2

os.makedirs(PLOTS_PATH, exist_ok=True)

# ─────────────────────────────────────────────
#  Load & filter
#  Keep only (problem, nvar) pairs that have all 100 mem values
#  so every mem's average is computed over the same set of problems.
# ─────────────────────────────────────────────
df = pd.read_csv("./datasets/new_raw.csv")
df["mem"] = pd.to_numeric(df["mem"], errors="coerce")

df["mem_count"] = df.groupby(["problem", "nvar"])["mem"].transform("count")
df = df[df["mem_count"] == 100].drop(columns="mem_count")

n_instances = df[["problem", "nvar"]].drop_duplicates().__len__()
print(f"Complete instances: {n_instances}, {df['problem'].nunique()} problems, {len(df)} rows")

# Fill missing elapsed time with penalty
df["stats_elapsed_time"] = pd.to_numeric(df["stats_elapsed_time"], errors="coerce")
n_missing = df["stats_elapsed_time"].isna().sum()
df["stats_elapsed_time"] = df["stats_elapsed_time"].fillna(PENALTY_TIME)
print(f"Missing stats_elapsed_time → {PENALTY_TIME}s: {n_missing} rows")

# ─────────────────────────────────────────────
#  Plot config
# ─────────────────────────────────────────────
METRICS = {
    "stats_elapsed_time": {
        "ylabel": "time (s)",
        "color":  "blue",
        "label":  "Average solve time",
    },
    "timed_bytes": {
        "ylabel": "MB",
        "color":  "orange",
        "label":  "Average heap allocation",
    },
    "total_alloc": {
        "ylabel": "allocation (MB)",
        "color":  "purple",
        "label":  "Average total allocation",
    },
    "neval_obj": {
        "ylabel": "count",
        "color":  "teal",
        "label":  "Average objective evaluations",
    },
    "neval_grad": {
        "ylabel": "count",
        "color":  "brown",
        "label":  "Average gradient evaluations",
    },
}

subtitle = f"n={n_instances} complete instances, DNF → {PENALTY_TIME}s"

# ─────────────────────────────────────────────
#  Helper: annotate the minimum point
# ─────────────────────────────────────────────
def annotate_minimum(ax, mean_series):
    min_mem     = mean_series.idxmin()
    min_val     = mean_series.loc[min_mem]
    value_range = mean_series.max() - mean_series.min()

    ax.scatter(min_mem, min_val, color="red", marker="o", zorder=5)
    ax.annotate(
        f"Min\nmem={min_mem}\n{min_val:.2f}",
        xy=(min_mem, min_val),
        xytext=(min_mem + 10, min_val + value_range * 0.15),
        ha="center", color="red", fontsize=9,
        arrowprops=dict(arrowstyle="->", color="red", lw=1.2),
    )

# ─────────────────────────────────────────────
#  Aggregate plots (mean ± 1 SEM, zoomed y-axis)
# ─────────────────────────────────────────────
print()
for metric, cfg in METRICS.items():
    if metric not in df.columns:
        print(f"Skipping '{metric}' — column not found")
        continue

    df[metric] = pd.to_numeric(df[metric], errors="coerce")

    grouped = df.groupby("mem")[metric]
    mean    = grouped.mean().sort_index()
    std     = grouped.std().sort_index()
    count   = grouped.count().sort_index()
    sem     = std / np.sqrt(count)  # standard error — tighter than raw std

    # Zoom: tight around the mean curve with a small padding
    padding  = (mean.max() - mean.min()) * 0.3
    y_bottom = max(0, mean.min() - padding)
    y_top    = mean.max() + padding

    fig, ax = plt.subplots()
    ax.plot(mean.index, mean.values, color=cfg["color"], linewidth=2)
    ax.fill_between(
        mean.index,
        np.maximum(mean.values - sem.values, 0),
        mean.values + sem.values,
        color=cfg["color"], alpha=0.25, label="±1 SEM",
    )
    annotate_minimum(ax, mean)

    ax.set_xlabel("mem")
    ax.set_ylabel(cfg["ylabel"])
    ax.set_title(f"{cfg['label']} vs mem\n({subtitle})")
    ax.set_ylim(bottom=y_bottom, top=y_top)
    ax.legend(fontsize=9)
    ax.grid(True)
    fig.tight_layout()

    out_path = os.path.join(PLOTS_PATH, f"aggregate_mem_vs_{metric}.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: aggregate_mem_vs_{metric}.pdf")

# ─────────────────────────────────────────────
#  Success rate
# ─────────────────────────────────────────────
total_runs_per_mem  = df.groupby("mem").size()
solved_runs_per_mem = df[df["status"] == "first_order"].groupby("mem").size()
success_rate        = (solved_runs_per_mem / total_runs_per_mem).fillna(0).sort_index()

best_mem  = success_rate.idxmax()
best_rate = success_rate.loc[best_mem]

fig, ax = plt.subplots()
ax.plot(success_rate.index, success_rate.values, color="green", linewidth=2)
ax.scatter(best_mem, best_rate, color="red", marker="o", zorder=5)
ax.annotate(
    f"Max\nmem={best_mem}\nrate={best_rate:.3f}",
    xy=(best_mem, best_rate),
    xytext=(best_mem + 10, best_rate - 0.025),
    ha="center", color="red", fontsize=9,
    arrowprops=dict(arrowstyle="->", color="red", lw=1.2),
)
ax.set_xlabel("mem")
ax.set_ylabel("success rate")
ax.set_title(f"L-BFGS success rate vs mem\n({subtitle})")
ax.set_ylim(0, 1.05)
ax.grid(True)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS_PATH, "mem_vs_success_rate.pdf"), bbox_inches="tight")
plt.close(fig)
print("Saved: mem_vs_success_rate.pdf")

print("\nDone.")