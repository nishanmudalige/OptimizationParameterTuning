import pandas as pd
import matplotlib.pyplot as plt
import os


OMITTED_ATTRIBUTES = {"status", "name", "solver", "mem", "nvar", "is_init_run"}
LOG_SCALE_ATTRS = { "time", 
                    "init_eval_obj_time", 
                    "init_eval_obj_mem",
                    "init_eval_obj_alloc",
                    "init_eval_grad_time", 
                    "init_eval_grad_time",
                    "init_eval_grad_alloc",
                    "num_iter",
                    "neval_grad", 
                    "neval_obj"}

UNITS = {
    "time": "seconds",
    "memory": "MB",
    "num_iter": "#iterations",
    "nvmops": "#operations",
    "neval_obj": "#evaluations",
    "neval_grad": "#evaluations",
    "init_eval_obj_time" : "seconds",
    "init_eval_obj_mem" : "MB",
    "init_eval_obj_alloc" : "allocations",
    "init_eval_grad_time": "seconds",
    "init_eval_grad_mem" : "MB",
    "init_eval_grad_alloc" : "allocations"
}

PLOTS_PATH = os.path.join(".", "Julia Notebook", "plots")
RESULTS_PATH = os.path.join(".", "Julia Notebook", "results")

df = pd.read_csv(os.path.join(RESULTS_PATH, "complete_lbfgs.csv"))
df = df[df["is_init_run"] == False]

os.makedirs(PLOTS_PATH, exist_ok=True)

for problem_name, problem_df in df.groupby("name"):
    print("Making plots for " + problem_name)
    status = problem_df["status"].iloc[0]
    solver = problem_df["solver"].iloc[0]
    nvar = problem_df["nvar"].iloc[0]

    ## mem vs attributes
    for attribute in problem_df.columns:
        if attribute not in OMITTED_ATTRIBUTES:
            ax = problem_df.plot(x="mem", y=attribute, kind="line", marker="o", title = f"{problem_name} ({nvar} variables) – {status} with {solver}")
            ax.set_xlabel("mem")
            ax.set_ylabel(f"{attribute} ({UNITS.get(attribute, '')})")
            ax.legend().remove()
            if attribute in LOG_SCALE_ATTRS:
                ax.set_yscale("log")
            plt.tight_layout()
            os.makedirs(os.path.join(PLOTS_PATH, problem_name), exist_ok=True)
            plt.savefig(os.path.join(PLOTS_PATH, problem_name, f"mem_vs_{attribute}.pdf"), bbox_inches="tight")
            plt.close()
    
    time_df = problem_df["time"]
    for attribute in ["num_iter", "neval_obj", "neval_grad"]:
        problem_df["temp"] = time_df / problem_df[attribute]
        ax = problem_df.plot(x="mem", y="temp", kind="line", marker="o", title = f"{problem_name} ({nvar} variables) – {status} with {solver}")
        ax.set_xlabel("mem")
        ax.set_ylabel(f"{attribute} ({UNITS.get("time", '')} per evaluation)")
        ax.legend().remove()
        if attribute in LOG_SCALE_ATTRS:
            ax.set_yscale("log")
        plt.tight_layout()
        os.makedirs(os.path.join(PLOTS_PATH, problem_name), exist_ok=True)
        plt.savefig(os.path.join(PLOTS_PATH, problem_name, f"mem_vs_time_per_{attribute}_.pdf"), bbox_inches="tight")
        plt.close()

