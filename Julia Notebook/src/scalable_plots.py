import pandas as pd
import matplotlib.pyplot as plt
import os


OMITTED_ATTRIBUTES = {"status", "name", "solver", "mem", "nvar", "is_init_run"}
LOG_SCALE_ATTRS = { "time", 
                    "init_eval_obj_time", 
                    "init_eval_grad_time", 
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
    "init_eval_grad_time": "seconds",
}

PLOTS_PATH = os.path.join(".", "Julia Notebook", "scalable_plots")
RESULTS_PATH = os.path.join(".", "Julia Notebook", "results")

df = pd.read_csv(os.path.join(RESULTS_PATH, "lbfgs_scalable.csv"))
df["memory"] = pd.to_numeric(df["memory"], errors="coerce") / (1024*1024)
df = df[df["is_init_run"] == False].copy()

NUMERIC_COLS = {"mem", "nvar", "time", "memory", "num_iter", "nvmops",
                "neval_obj", "init_eval_obj_time", "neval_grad", "init_eval_grad_time"}
for col in NUMERIC_COLS:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

os.makedirs(PLOTS_PATH, exist_ok=True)

for problem_name, problem_df in df.groupby("name"):
    print("Making plots for " + problem_name)
    status = problem_df["status"].iloc[0]
    solver = problem_df["solver"].iloc[0]
    nvar = problem_df["nvar"].iloc[0]

    ## mem vs attributes
    for attribute in problem_df.columns:
        if attribute not in OMITTED_ATTRIBUTES:
            # skip if there's no numeric data to plot for this attribute
            if problem_df[attribute].dropna().empty or not pd.api.types.is_numeric_dtype(problem_df[attribute]):
                continue
            ax = problem_df.plot(x="mem", y=attribute, kind="line", marker="o",
                                 title=f"{problem_name} ({nvar} variables) – {status} with {solver}")
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
        # skip if division produced only NaNs
        if problem_df["temp"].dropna().empty:
            continue
        ax = problem_df.plot(x="mem", y="temp", kind="line", marker="o",
                             title=f"{problem_name} ({nvar} variables) – {status} with {solver}")
        ax.set_xlabel("mem")
        ax.set_ylabel(f"{attribute} ({UNITS.get('time', '')} per evaluation)")
        ax.legend().remove()
        if attribute in LOG_SCALE_ATTRS:
            ax.set_yscale("log")
        plt.tight_layout()
        os.makedirs(os.path.join(PLOTS_PATH, problem_name), exist_ok=True)
        plt.savefig(os.path.join(PLOTS_PATH, problem_name, f"mem_vs_time_per_{attribute}_.pdf"), bbox_inches="tight")
        plt.close()