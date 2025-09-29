import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv("./results/lbfgs.csv")

OMITTED_ATTRIBUTES = {"status", "name", "solver", "mem", "nvar", "init_eval_obj_time", "init_eval_grad_time"}

UNITS = {
    "time": "seconds",
    "memory": "MB",
    "num_iter": "iterations",
    "nvmops": "ops",
    "neval_obj": "#evals",
    "neval_grad": "#evals",
}

os.makedirs("./plots", exist_ok=True)

for problem_name, problem_df in df.groupby("name"):
    print("Saving plot for " + problem_name)
    for attribute in problem_df.columns:
        if attribute not in OMITTED_ATTRIBUTES:
            ax = problem_df.plot(x="mem", y=attribute, kind="line", marker="o", title=f"{problem_name} - mem vs {attribute}")
            ax.set_xlabel("mem (MB)")
            ax.set_yscale("log")
            ax.set_ylabel(f"{attribute} ({UNITS.get(attribute, '')})")
            ax.legend().remove()
            plt.tight_layout()
            os.makedirs(f"./plots/{problem_name}", exist_ok=True)
            plt.savefig(f"./plots/{problem_name}/mem_vs_{attribute}.pdf", bbox_inches="tight")
            plt.close()
