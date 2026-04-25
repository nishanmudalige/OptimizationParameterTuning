import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# ─────────────────────────────────────────────
#  Paths & constants
# ─────────────────────────────────────────────
PLOTS_PATH    = os.path.join(".", "global_mem_plots")
TIMEOUT_S     = 3600          # solver timeout (1 hour)
PENALTY_TIME  = TIMEOUT_S * 2 # penalty for DNF runs (2 hours)

os.makedirs(PLOTS_PATH, exist_ok=True)

# ─────────────────────────────────────────────
#  Load data
# ─────────────────────────────────────────────
df = pd.read_csv("./datasets/raw.csv")
df["mem"] = pd.to_numeric(df["mem"], errors="coerce")

all_mem_values = sorted(df["mem"].dropna().unique())
n_mem_values   = len(all_mem_values)
print(f"Unique mem values: {n_mem_values}")

# ─────────────────────────────────────────────
#  Build two working dataframes
#
#  df_clean   — only (problem, nvar) pairs where every mem value
#               produced a result. Used for all metrics except
#               elapsed time, so comparisons are apples-to-apples.
#
#  df_elapsed — full grid of every (problem, nvar) × every mem,
#               with missing elapsed times filled in as PENALTY_TIME.
#               This way "didn't finish" is treated as slow, not absent.
# ─────────────────────────────────────────────

# df_clean: drop any instance that is missing even one mem result
runs_per_instance = df.groupby(["problem", "nvar"]).size()
complete_instances = runs_per_instance[runs_per_instance == n_mem_values].index

df_clean = (
    df.set_index(["problem", "nvar"])
      .loc[complete_instances]
      .reset_index()
)

print(f"  Before filtering: {df['problem'].nunique():>4} problems, {len(df):>6} rows")
print(f"  After filtering:  {df_clean['problem'].nunique():>4} problems, {len(df_clean):>6} rows")
print(f"  Dropped {len(df) - len(df_clean)} rows from incomplete instances")

# df_elapsed: cross join every instance with every mem, then left-join
#             actual results so missing cells show up as NaN → PENALTY_TIME
all_instances = df[["problem", "nvar"]].drop_duplicates()
full_grid = all_instances.merge(pd.DataFrame({"mem": all_mem_values}), how="cross")

df_elapsed = full_grid.merge(
    df[["problem", "nvar", "mem", "stats_elapsed_time"]],
    on=["problem", "nvar", "mem"],
    how="left",
)
df_elapsed["stats_elapsed_time"] = pd.to_numeric(
    df_elapsed["stats_elapsed_time"], errors="coerce"
)

n_missing = df_elapsed["stats_elapsed_time"].isna().sum()
df_elapsed["stats_elapsed_time"] = df_elapsed["stats_elapsed_time"].fillna(PENALTY_TIME)
print(f"\nElapsed-time grid: {n_missing} missing entries penalised → {PENALTY_TIME}s")

# ─────────────────────────────────────────────
#  Plot config
#  Each key matches a column name in the dataframe.
#  source="elapsed" uses df_elapsed; source="clean" uses df_clean.
# ─────────────────────────────────────────────
METRICS = {
    "stats_elapsed_time": {
        "ylabel": "time (s)",
        "color":  "blue",
        "label":  "Average solve time",
        "source": "elapsed",   # uses penalty-imputed full grid
    },
    "timed_bytes": {
        "ylabel": "bytes",
        "color":  "orange",
        "label":  "Average heap allocation",
        "source": "clean",
    },
    "total_alloc": {
        "ylabel": "allocation (MB)",
        "color":  "purple",
        "label":  "Average total allocation",
        "source": "clean",
    },
    "neval_obj": {
        "ylabel": "count",
        "color":  "teal",
        "label":  "Average objective evaluations",
        "source": "clean",
    },
    "neval_grad": {
        "ylabel": "count",
        "color":  "brown",
        "label":  "Average gradient evaluations",
        "source": "clean",
    },
}

n_complete     = len(complete_instances)
n_all          = len(all_instances)

# ─────────────────────────────────────────────
#  Helper: annotate the minimum point on a line plot
# ─────────────────────────────────────────────
def annotate_minimum(ax, mean_series):
    min_mem = mean_series.idxmin()
    min_val = mean_series.loc[min_mem]
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
#  Per-metric aggregate plots (mean ± 1 std across instances)
# ─────────────────────────────────────────────
print()
for metric, cfg in METRICS.items():
    # Pick the right source dataframe
    if cfg["source"] == "elapsed":
        df_plot  = df_elapsed
        n_label  = n_all
        subtitle = f"all instances, DNF → {PENALTY_TIME}s"
    else:
        if metric not in df_clean.columns:
            print(f"Skipping '{metric}' — column not found in dataset")
            continue
        df_plot  = df_clean.copy()
        df_plot[metric] = pd.to_numeric(df_plot[metric], errors="coerce")
        n_label  = n_complete
        subtitle = "complete instances only"

    mean = df_plot.groupby("mem")[metric].mean().sort_index()
    std  = df_plot.groupby("mem")[metric].std().sort_index()

    fig, ax = plt.subplots()
    ax.plot(mean.index, mean.values, color=cfg["color"], linewidth=2)
    ax.fill_between(
        mean.index,
        mean.values - std.values,
        mean.values + std.values,
        color=cfg["color"], alpha=0.15, label="±1 std",
    )
    annotate_minimum(ax, mean)

    ax.set_xlabel("mem")
    ax.set_ylabel(cfg["ylabel"])
    ax.set_title(f"{cfg['label']} vs mem\n(n={n_label}, {subtitle})")
    ax.legend(fontsize=9)
    ax.grid(True)
    fig.tight_layout()

    out_path = os.path.join(PLOTS_PATH, f"aggregate_mem_vs_{metric}.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: aggregate_mem_vs_{metric}.pdf")

# ─────────────────────────────────────────────
#  Success rate plot  (uses the original unfiltered df
#  so the denominator reflects all attempted runs)
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
ax.set_title("L-BFGS success rate vs mem (all instances)")
ax.set_ylim(0, 1.05)
ax.grid(True)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS_PATH, "mem_vs_success_rate.pdf"), bbox_inches="tight")
plt.close(fig)
print("Saved: mem_vs_success_rate.pdf")

print("\nDone.")