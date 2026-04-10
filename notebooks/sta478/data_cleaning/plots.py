import pandas as pd
import matplotlib.pyplot as plt
import os

ATTRIBUTES = {"stats_elapsed_time",
              "timed_bytes",
              "timed_time"}


LOG_SCALED_ATTRS = {"stats_elapsed_time",
                    "timed_time"}

UNITS = {"stats_elapsed_time" : "seconds",
         "timed_bytes" : "bytes",
         "time_time" : "seconds",
        }

PLOTS_PATH = os.path.join(".", "plots")

df = pd.read_csv("./datasets/final.csv")
df = df[df["stats_elapsed_time"].notna()].copy().reset_index()

os.makedirs(PLOTS_PATH, exist_ok=True)

for uid, problem_df in df.groupby(["name", "nvar"]):
    problem_name = uid[0]
    nvar = uid[1]
    print(f"Making plots for {problem_name} with nvar={nvar}")
    status = problem_df["status"].iloc[0]
    solver = problem_df["source_solver"].iloc[0]

    ## mem vs attributes
    for attribute in problem_df.columns:
        if attribute in ATTRIBUTES:
            ax = problem_df.plot(x="mem", y=attribute, kind="line", marker="o", title = f"{problem_name} ({nvar} variables) – {status} with {solver}")
            ax.set_xlabel("mem")
            ax.set_ylabel(f"{attribute} ({UNITS.get(attribute, '')})")
            ax.legend().remove()
            ax.set_xlim(1, 100)
            ax.set_xticks([1, 20, 40, 60, 80, 100])
            if attribute in LOG_SCALED_ATTRS:
                y = problem_df[attribute]
                if (y > 0).any():
                    ax.set_yscale("log")
                else:
                    print(f"Skipping log scale for {problem_name}, nvar={nvar}, attribute={attribute} because there are no positive values.")
            plt.tight_layout()
            os.makedirs(os.path.join(PLOTS_PATH, f"{problem_name}_({nvar})"), exist_ok=True)
            plt.savefig(os.path.join(PLOTS_PATH, f"{problem_name}_({nvar})", f"mem_vs_{attribute}.pdf"), bbox_inches="tight")
            plt.close()
    
    # time_df = problem_df["stats_elapsed_time"]
    # for attribute in ["nvmops", "neval_obj", "neval_grad"]:
    #     problem_df["temp"] = time_df / problem_df[attribute]
    #     ax = problem_df.plot(x="mem", y="temp", kind="line", marker="o", title = f"{problem_name} ({nvar} variables) – {status} with {solver}")
    #     ax.set_xlabel("mem")
    #     ax.set_ylabel(f"{attribute} ({UNITS.get("stats_elapsed_time", '')} per evaluation)")
    #     ax.legend().remove()
    #     # if attribute in LOG_SCALED_ATTRS:
    #     #     ax.set_yscale("log")
    #     plt.tight_layout()
    #     os.makedirs(os.path.join(PLOTS_PATH, f"{problem_name}_({nvar})",), exist_ok=True)
    #     plt.savefig(os.path.join(PLOTS_PATH, f"{problem_name}_({nvar})", f"mem_vs_time_per_{attribute}_.pdf"), bbox_inches="tight")
    #     plt.close()
        
