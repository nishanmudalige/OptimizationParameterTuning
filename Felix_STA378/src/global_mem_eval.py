import pandas as pd
import matplotlib.pyplot as plt
import os

RESULT_PATH = os.path.join("./", "Julia Notebook", "results")
PLOTS_PATH = os.path.join(".", "Julia Notebook", "global_mem_plots")
OMITTED_ATTRIBUTES = ["status", "name", "solver", "mem", "nvar", "is_init_run"]
os.makedirs(PLOTS_PATH, exist_ok=True)

lbfgs_df = pd.read_csv(os.path.join(RESULT_PATH, "complete_lbfgs.csv"))
lbfgs_df["time"] = lbfgs_df["time"].clip(upper=300)
scalable_df = pd.read_csv(os.path.join(RESULT_PATH, "complete_scalable.csv"))
scalable_df["time"] = scalable_df["time"].clip(upper=600)

lbfgs_plots = os.path.join(PLOTS_PATH, "lbfgs")
scalable_plots = os.path.join(PLOTS_PATH, "scalable")
os.makedirs(lbfgs_plots, exist_ok=True)
os.makedirs(scalable_plots, exist_ok=True)

for out_dir, df in [(lbfgs_plots, lbfgs_df), (scalable_plots, scalable_df)]:
    df = df[df["is_init_run"] == False].copy()
    df["status"] = df["status"].map({
        "first_order": 1,
        "unbounded": 1,
        "max_time": 0
    })
    df["mem"] = pd.to_numeric(df["mem"], errors="coerce")
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    if out_dir == lbfgs_plots:
        df["penalized_time"] = df["time"] + (df["status"] == 0) * 300
    else:
        df["penalized_time"] = df["time"] + (df["status"] == 0) * 600

    for attribute in df.columns:
        if attribute not in OMITTED_ATTRIBUTES:
            attribute_mean = df.groupby("mem")[attribute].mean().sort_index()
            fig, ax = plt.subplots()
            ax.clear()
            ax.plot(attribute_mean.index, attribute_mean.values, color="blue", linewidth=2)
            min_mem = attribute_mean.idxmin()
            max_mem = attribute_mean.idxmax()
            min_attribute = attribute_mean.loc[min_mem]
            max_attribute = attribute_mean.loc[max_mem]
            ax.scatter(min_mem, min_attribute, color="red", marker="o")

            ax.annotate(
            f"Min\nmem={min_mem}\n{attribute}={min_attribute:.2f}",
            xy=(min_mem, min_attribute),
            xytext=(min_mem + (max_mem - min_mem) / 2, min_attribute + (max_attribute - min_attribute)/ 2),
            ha="center", color="red", fontsize=10,
            arrowprops=dict(arrowstyle="->", color="red", lw=1.2))
            ax.set_xlabel("mem")
            ax.set_ylabel(attribute)
            ax.set_title(f"Average {attribute} across each value of mem")
            ax.grid(True)
            fig.savefig(os.path.join(out_dir, f"mem_VS_{attribute}.pdf"), bbox_inches="tight")
            plt.close(fig)
        elif attribute == "status":
            attribute_total = df.groupby("mem")[attribute].sum().sort_index()      
            fig, ax = plt.subplots()
            ax.clear()
            ax.plot(attribute_total.index, attribute_total.values, color="blue", linewidth=2)
            min_mem = attribute_total.idxmin()
            max_mem = attribute_total.idxmax()
            min_attribute = attribute_total.loc[min_mem]
            max_attribute = attribute_total.loc[max_mem]
            ax.scatter(max_mem, max_attribute, color="red", marker="o")

            ax.annotate(
            f"Max\nmem={max_mem}\n{attribute}={max_attribute:.2f}",
            xy=(max_mem, max_attribute),
            xytext=(max_mem + 20, max_attribute - (max_attribute - min_attribute)/ 2),
            ha="center", color="red", fontsize=10,
            arrowprops=dict(arrowstyle="->", color="red", lw=1.2))
            ax.set_xlabel("mem")
            ax.set_ylabel(attribute)
            ax.set_title(f"Number of problems solved w.r.t to mem")
            ax.grid(True)
            fig.savefig(os.path.join(out_dir, f"mem_VS_{attribute}.pdf"), bbox_inches="tight")
            plt.close(fig)

            