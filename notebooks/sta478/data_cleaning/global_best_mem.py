import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

PLOTS_PATH = os.path.join(".", "global_mem_plots")
os.makedirs(PLOTS_PATH, exist_ok=True)

df = pd.read_csv("./datasets/final.csv")
df["mem"] = pd.to_numeric(df["mem"], errors="coerce")

# --- Filter: keep only (problem, nvar) where ALL mems succeeded ---
all_mems = df["mem"].nunique()
print(f"Total unique mem values: {all_mems}")

success_counts = df.groupby(["problem", "nvar"]).size()
complete_instances = success_counts[success_counts == all_mems].index
df_clean = df.set_index(["problem", "nvar"]).loc[complete_instances].reset_index()

print(f"Before filtering: {df['problem'].nunique()} problems, {len(df)} rows")
print(f"After filtering:  {df_clean['problem'].nunique()} problems, {len(df_clean)} rows")
print(f"Dropped: {len(df) - len(df_clean)} rows from incomplete instances")

# --- Config ---
AGGREGATE_ATTRS = {
    "stats_elapsed_time": {"ylabel": "time (s)",       "color": "blue",   "label": "Average solve time"},
    "timed_bytes":        {"ylabel": "bytes",           "color": "orange", "label": "Average heap allocation"},
    "total_alloc":        {"ylabel": "allocation (MB)", "color": "purple", "label": "Average total allocation"},
    "neval_obj":          {"ylabel": "count",           "color": "teal",   "label": "Average objective evaluations"},
    "neval_grad":         {"ylabel": "count",           "color": "brown",  "label": "Average gradient evaluations"},
}

n_complete = len(complete_instances)

# --- Aggregate plots ---
for attr, cfg in AGGREGATE_ATTRS.items():
    if attr not in df_clean.columns:
        print(f"Skipping {attr} — column not found")
        continue

    df_clean[attr] = pd.to_numeric(df_clean[attr], errors="coerce")
    grouped = df_clean.groupby("mem")[attr]
    mean = grouped.mean().sort_index()
    std = grouped.std().sort_index()

    fig, ax = plt.subplots()
    ax.plot(mean.index, mean.values, color=cfg["color"], linewidth=2)
    ax.fill_between(mean.index, mean.values - std.values, mean.values + std.values,
                    color=cfg["color"], alpha=0.15, label="±1 std")

    # Annotate min
    min_mem = mean.idxmin()
    min_val = mean.loc[min_mem]
    ax.scatter(min_mem, min_val, color="red", marker="o", zorder=5)
    ax.annotate(
        f"Min\nmem={min_mem}\n{min_val:.2f}",
        xy=(min_mem, min_val),
        xytext=(min_mem + 10, min_val + (mean.max() - mean.min()) * 0.15),
        ha="center", color="red", fontsize=9,
        arrowprops=dict(arrowstyle="->", color="red", lw=1.2))

    ax.set_xlabel("mem")
    ax.set_ylabel(cfg["ylabel"])
    ax.set_title(f"{cfg['label']} vs mem (n={n_complete} complete instances)")
    ax.legend(fontsize=9)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_PATH, f"aggregate_mem_vs_{attr}.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: aggregate_mem_vs_{attr}.pdf")

# --- Success rate (use ORIGINAL unfiltered df) ---
total_per_mem = df.groupby("mem").size()
solved_per_mem = df[df["status"] == "first_order"].groupby("mem").size()
success_rate = (solved_per_mem / total_per_mem).fillna(0).sort_index()

fig, ax = plt.subplots()
ax.plot(success_rate.index, success_rate.values, color="green", linewidth=2)
max_mem = success_rate.idxmax()
max_sr = success_rate.loc[max_mem]
ax.scatter(max_mem, max_sr, color="red", marker="o", zorder=5)
ax.annotate(
    f"Max\nmem={max_mem}\nrate={max_sr:.3f}",
    xy=(max_mem, max_sr),
    xytext=(max_mem + 10, max_sr - 0.025),
    ha="center", color="red", fontsize=9,
    arrowprops=dict(arrowstyle="->", color="red", lw=1.2))
ax.set_xlabel("mem")
ax.set_ylabel("success rate")
ax.set_title("L-BFGS success rate vs mem (all instances)")
ax.set_ylim(0, 1.05)
ax.grid(True)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS_PATH, "mem_vs_success_rate.pdf"), bbox_inches="tight")
plt.close(fig)
print("Saved: mem_vs_success_rate.pdf")

print("Done.")