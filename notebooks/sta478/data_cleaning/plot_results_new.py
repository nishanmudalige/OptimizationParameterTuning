"""
=============================================================================
PLOTTING SCRIPT — mem vs solver attributes
=============================================================================
Run with: python plot_results.py
Or in tmux: python plot_results.py | tee plot.log
=============================================================================
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime


# =============================================================================
# CONFIG
# =============================================================================
DATA_PATH = "./datasets/final.csv"
PLOTS_PATH = os.path.join(".", "plots")

# --- Direct attributes to plot against mem ---
ATTRIBUTES = {
    "stats_elapsed_time",
    "timed_bytes",
    "timed_time",
    "neval_obj",
    "neval_grad",
    "total_alloc",
}

# --- Derived metrics: (name, numerator, denominator) ---
# These create new columns like time_per_obj_eval = stats_elapsed_time / neval_obj
DERIVED_METRICS = [
    ("time_per_obj_eval",  "stats_elapsed_time", "neval_obj"),
    ("time_per_grad_eval", "stats_elapsed_time", "neval_grad"),
    ("bytes_per_obj_eval", "timed_bytes",        "neval_obj"),
    ("bytes_per_grad_eval","timed_bytes",        "neval_grad"),
]

# --- Log-scaled y-axis ---
LOG_SCALED_ATTRS = {
    "stats_elapsed_time",
    "timed_time",
    "timed_bytes",
    "total_alloc",
    "time_per_obj_eval",
    "time_per_grad_eval",
    "bytes_per_obj_eval",
    "bytes_per_grad_eval",
}

# --- Units for axis labels ---
UNITS = {
    "stats_elapsed_time":   "seconds",
    "timed_bytes":          "bytes",
    "timed_time":           "seconds",
    "neval_obj":            "count",
    "neval_grad":           "count",
    "total_alloc":          "MB",
    "time_per_obj_eval":    "sec/eval",
    "time_per_grad_eval":   "sec/eval",
    "bytes_per_obj_eval":   "bytes/eval",
    "bytes_per_grad_eval":  "bytes/eval",
}

# --- Plot style ---
plt.rcParams.update({
    "figure.figsize": (8, 5),
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
})

MEM_TICKS = [1, 20, 40, 60, 80, 100]


# =============================================================================
# HELPERS
# =============================================================================
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def safe_log_scale(ax, y):
    """Apply log scale only if data has positive values."""
    if (y > 0).any():
        ax.set_yscale("log")
        return True
    return False


def make_plot(problem_df, x_col, y_col, title, xlabel, ylabel, save_path, use_log=False):
    """Single reusable plot function."""
    fig, ax = plt.subplots()
    data = problem_df.sort_values(x_col)
    ax.plot(data[x_col], data[y_col], marker="o", linewidth=1.5, markersize=5)
    ax.set_title(title, fontsize=11, pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(1, 100)
    ax.set_xticks(MEM_TICKS)

    if use_log:
        safe_log_scale(ax, data[y_col].dropna())

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def make_overlay_plot(problem_df, x_col, y_cols, title, xlabel, ylabel, save_path, use_log=False):
    """Multiple y columns on same axes."""
    fig, ax = plt.subplots()
    data = problem_df.sort_values(x_col)
    for y_col in y_cols:
        if y_col in data.columns and data[y_col].notna().any():
            ax.plot(data[x_col], data[y_col], marker="o", label=y_col, linewidth=1.5, markersize=4)

    ax.set_title(title, fontsize=11, pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(1, 100)
    ax.set_xticks(MEM_TICKS)
    ax.legend(fontsize=9)

    if use_log:
        all_y = pd.concat([data[c] for c in y_cols if c in data.columns], ignore_index=True).dropna()
        if len(all_y) > 0:
            safe_log_scale(ax, all_y)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# LOAD DATA
# =============================================================================
log("Loading data...")
df = pd.read_csv(DATA_PATH)
df = df[df["stats_elapsed_time"].notna()].copy().reset_index(drop=True)
log(f"Loaded {len(df)} rows")

# --- Compute derived metrics ---
for metric_name, num, den in DERIVED_METRICS:
    if num in df.columns and den in df.columns:
        df[metric_name] = df[num] / df[den].replace(0, np.nan)

ALL_PLOT_ATTRS = ATTRIBUTES | {m[0] for m in DERIVED_METRICS if m[0] in df.columns}
log(f"Plotting attributes: {sorted(ALL_PLOT_ATTRS)}")

groups = list(df.groupby(["name", "nvar"]))
total = len(groups)
log(f"Total problem instances: {total}")

os.makedirs(PLOTS_PATH, exist_ok=True)


# =============================================================================
# GENERATE PLOTS
# =============================================================================
for idx, (uid, problem_df) in enumerate(groups):
    problem_name = uid[0]
    nvar = uid[1]
    status = problem_df["status"].iloc[0] if "status" in problem_df.columns else "unknown"
    solver = problem_df["source_solver"].iloc[0] if "source_solver" in problem_df.columns else "unknown"
    base_title = f"{problem_name} ({nvar} vars) – {status} / {solver}"
    folder = os.path.join(PLOTS_PATH, f"{problem_name}_({nvar})")

    if (idx + 1) % 50 == 0 or (idx + 1) == total:
        log(f"Progress: {idx+1}/{total} ({100*(idx+1)/total:.0f}%)")

    # --- 1. Individual attribute plots ---
    for attr in ALL_PLOT_ATTRS:
        if attr in problem_df.columns and problem_df[attr].notna().any():
            unit = UNITS.get(attr, "")
            ylabel = f"{attr} ({unit})" if unit else attr
            make_plot(
                problem_df, "mem", attr,
                title=f"{base_title}\nmem vs {attr}",
                xlabel="mem",
                ylabel=ylabel,
                save_path=os.path.join(folder, f"mem_vs_{attr}.pdf"),
                use_log=(attr in LOG_SCALED_ATTRS),
            )

    # --- 2. Overlay: neval_obj + neval_grad on same plot ---
    eval_cols = [c for c in ["neval_obj", "neval_grad"] if c in problem_df.columns]
    if eval_cols:
        make_overlay_plot(
            problem_df, "mem", eval_cols,
            title=f"{base_title}\nmem vs function evaluations",
            xlabel="mem",
            ylabel="count",
            save_path=os.path.join(folder, "mem_vs_evals_overlay.pdf"),
            use_log=False,
        )

    # --- 3. Overlay: time vs bytes (dual nature of cost) ---
    time_cols = [c for c in ["stats_elapsed_time", "timed_time"] if c in problem_df.columns]
    if time_cols:
        make_overlay_plot(
            problem_df, "mem", time_cols,
            title=f"{base_title}\nmem vs time",
            xlabel="mem",
            ylabel="seconds",
            save_path=os.path.join(folder, "mem_vs_times_overlay.pdf"),
            use_log=True,
        )

    # --- 4. Overlay: per-eval costs ---
    per_eval_cols = [c for c in ["time_per_obj_eval", "time_per_grad_eval"] if c in problem_df.columns]
    if per_eval_cols:
        make_overlay_plot(
            problem_df, "mem", per_eval_cols,
            title=f"{base_title}\nmem vs cost per evaluation",
            xlabel="mem",
            ylabel="sec/eval",
            save_path=os.path.join(folder, "mem_vs_per_eval_cost_overlay.pdf"),
            use_log=True,
        )


# =============================================================================
# SUMMARY GRID — one plot per attribute, all problems overlaid
# =============================================================================
log("")
log("Generating summary grid plots (all problems overlaid)...")

summary_folder = os.path.join(PLOTS_PATH, "_summary")
os.makedirs(summary_folder, exist_ok=True)

# Sample up to 50 problems to avoid unreadable plots
sample_groups = groups[:50] if len(groups) > 50 else groups

for attr in sorted(ALL_PLOT_ATTRS):
    if attr not in df.columns:
        continue

    fig, ax = plt.subplots(figsize=(10, 6))
    for uid, problem_df in sample_groups:
        data = problem_df.sort_values("mem")
        if attr in data.columns and data[attr].notna().any():
            ax.plot(data["mem"], data[attr], alpha=0.3, linewidth=0.8)

    ax.set_title(f"All problems – mem vs {attr}", fontsize=12)
    ax.set_xlabel("mem")
    unit = UNITS.get(attr, "")
    ax.set_ylabel(f"{attr} ({unit})" if unit else attr)
    ax.set_xlim(1, 100)
    ax.set_xticks(MEM_TICKS)

    if attr in LOG_SCALED_ATTRS:
        safe_log_scale(ax, df[attr].dropna())

    plt.tight_layout()
    plt.savefig(os.path.join(summary_folder, f"all_mem_vs_{attr}.pdf"), bbox_inches="tight")
    plt.close(fig)

log(f"Summary plots saved to {summary_folder}/")


# =============================================================================
# DONE
# =============================================================================
total_plots = len(groups) * (len(ALL_PLOT_ATTRS) + 3) + len(ALL_PLOT_ATTRS)
log(f"DONE — ~{total_plots} plots saved to {PLOTS_PATH}/")
