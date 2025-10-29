import pandas as pd
import matplotlib.pyplot as plt
import os

RESULT_PATH = os.path.join("./", "Julia Notebook", "results")
PLOTS_PATH = os.path.join(".", "Julia Notebook", "global_mem_plots")
OMITTED_ATTRIBUTES = ["status", "name", "solver", "mem", "nvar", "is_init_run"]
os.makedirs(PLOTS_PATH, exist_ok=True)

lbfgs_df = pd.read_csv(os.path.join(RESULT_PATH, "complete_lbfgs.csv"))
scalable_df = pd.read_csv(os.path.join(RESULT_PATH, "complete_scalable.csv"))

lbfgs_plots = os.path.join(PLOTS_PATH, "lbfgs")
scalable_plots = os.path.join(PLOTS_PATH, "scalable")
os.makedirs(lbfgs_plots, exist_ok=True)
os.makedirs(scalable_plots, exist_ok=True)

for out_dir, df in [(lbfgs_plots, lbfgs_df), (scalable_plots, scalable_df)]:
    df = df[df["is_init_run"] == False].copy()
    df = df[df["is_init_run"] == False].copy()
    df["mem"] = pd.to_numeric(df["mem"], errors="coerce")
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
            f"Min\nmem={min_mem}\ntime={min_attribute:.2f}",
            xy=(min_mem, min_attribute),
            xytext=(min_mem + (max_mem - min_mem) / 2, min_attribute + (max_attribute - min_attribute)/ 2),
            ha="center", color="red", fontsize=10,
            arrowprops=dict(arrowstyle="->", color="red", lw=1.2))
            ax.set_xlabel("mem")
            ax.set_ylabel(attribute)
            ax.set_title(f"Average {attribute} across each value of mem")
            ax.grid(True)

            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"mem_VS_{attribute}.pdf"), bbox_inches="tight")
            plt.close(fig)
            