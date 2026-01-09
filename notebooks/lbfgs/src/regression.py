import pandas as pd
import matplotlib.pyplot as plt
import os

RESULT_PATH = os.path.join("./", "Julia Notebook", "results")
PLOTS_PATH = os.path.join(".", "Julia Notebook", "global_mem_plots")
os.makedirs(PLOTS_PATH, exist_ok=True)

lbfgs_df = pd.read_csv(os.path.join(RESULT_PATH, "complete_lbfgs.csv"))
scalable_df = pd.read_csv(os.path.join(RESULT_PATH, "complete_scalable.csv"))

lbfgs_plots = os.path.join(PLOTS_PATH, "lbfgs")
scalable_plots = os.path.join(PLOTS_PATH, "scalable")
os.makedirs(lbfgs_plots, exist_ok=True)
os.makedirs(scalable_plots, exist_ok=True)

for out_dir, df in [(lbfgs_plots, lbfgs_df), (scalable_plots, scalable_df)]:
    df = df[df["is_init_run"] == False].copy()
    df["mem"] = pd.to_numeric(df["mem"], errors="coerce")

    time_mean = df.groupby("mem")["time"].mean().sort_index()
    memory_mean = df.groupby("mem")["memory"].mean().sort_index()

    fig, ax = plt.subplots()
    ax.plot(time_mean.index, time_mean.values, color="blue", linewidth=2)
    min_mem = time_mean.idxmin()
    min_time = time_mean.loc[min_mem]
    ax.scatter(min_mem, min_time, color="red", marker="o")

    ax.annotate(
    f"Min\nmem={min_mem}\ntime={min_time:.2f}",
    xy=(min_mem, min_time),
    xytext=(min_mem + 10, min_time + 1),
    ha="center", color="red", fontsize=10,
    arrowprops=dict(arrowstyle="->", color="red", lw=1.2))
    ax.set_xlabel("mem")
    ax.set_ylabel("time (s)")
    ax.set_title("Average time across each value of mem")
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "Mem_VS_Time.pdf"), bbox_inches="tight")


    ax.clear()
    ax.plot(memory_mean.index, memory_mean.values, color="orange", linewidth=2)
    min_mem = memory_mean.idxmin()
    min_memory = memory_mean.loc[min_mem]
    ax.scatter(min_mem, min_memory, color="red", marker="o")
    ax.annotate(
    f"Min\nmem={min_mem}\ntime={min_memory:.2f}",
    xy=(min_mem, min_memory),
    xytext=(min_mem + 10, min_memory + 30),
    ha="center", color="red", fontsize=10,
    arrowprops=dict(arrowstyle="->", color="red", lw=1.2))
    ax.set_xlabel("mem")
    ax.set_ylabel("Memory (MB)")
    ax.set_title("Average memory across each value of mem")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "Mem_VS_Memory.pdf"), bbox_inches="tight")

    plt.close(fig)






