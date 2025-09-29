import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv("./results/lbfgs.csv")

OMITTED_ATTRIBUTES = {"status", "name", "solver", "mem", "nvar", "init_eval_obj_time", "init_eval_grad_time"}

os.makedirs("./plots", exist_ok=True)

for problem_name, problem_df in df.groupby("name"):
    for attribute in problem_df.columns:
        if attribute not in OMITTED_ATTRIBUTES:
            print("Making plot for " + problem_name)
            problem_df.plot(x="mem", y=attribute, kind="line", title=f"{problem_name} - {attribute}")
            plt.tight_layout()
            os.makedirs(f"./plots/{problem_name}", exist_ok=True)
            plt.savefig(f"./plots/{problem_name}/mem_vs_{attribute}.pdf")
            plt.close()
