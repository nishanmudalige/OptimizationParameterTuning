import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv("./results/lbfgs.csv")
df["memory"] = df["memory"] / (1024 * 1024) # change the data from byte to MB

OMITTED_ATTRIBUTES = {"status", "name", "solver", "mem", "nvar"}
LOG_SCALE_ATTRS = {"time", "init_eval_obj_time", "init_eval_grad_time"}

UNITS = {
    "time": "seconds",
    "memory": "MB",
    "num_iter": "#iterations",
    "nvmops": "#operations",
    "neval_obj": "#evaluations",
    "neval_grad": "#evaluations",
    "init_eval_obj_time" : "seconds",
    "init_eval_grad_time": "seconds",
}

os.makedirs("./plots", exist_ok=True)

for problem_name, problem_df in df.groupby("name"):
    print("Saving plot for " + problem_name)
    status = problem_df["status"].iloc[0]
    solver = problem_df["solver"].iloc[0]
    nvar = problem_df["nvar"].iloc[0]
    for attribute in problem_df.columns:
        if attribute not in OMITTED_ATTRIBUTES:
            ax = problem_df.plot(x="mem", y=attribute, kind="line", marker="o", title = f"{problem_name} ({nvar} variables) – {status} with {solver}")
            ax.set_xlabel("mem")
            ax.set_ylabel(f"{attribute} ({UNITS.get(attribute, '')})")
            ax.legend().remove()
            if attribute in LOG_SCALE_ATTRS:
                ax.set_yscale("log")
            plt.tight_layout()
            os.makedirs(f"./plots/{problem_name}", exist_ok=True)
            plt.savefig(f"./plots/{problem_name}/mem_vs_{attribute}.pdf", bbox_inches="tight")
            plt.close()
