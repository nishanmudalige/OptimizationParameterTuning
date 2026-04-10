import os
import pandas as pd
from pathlib import Path
import re
import numpy as np

MAX_TIME = 3600

PATH = "/home/gaoyuc10/projects/AutoJSOSolverSelection.jl/"

path = Path(PATH)
problem = list(path.rglob("problems_instances/logs_*_n*/output.json_*.log"))
solver =  list(path.rglob("solver_instances/logs_solvers_10_cores_new_with_timed/output.json_*.log"))
error_cols = ["error_problem", "error_type"]
alloc_cols = ["alloc_obj", "alloc_grad", "alloc_hprod", "alloc_hess"]
expr_cols = ["is_constant (ExprTree)", "is_linear (ExprTree)"]

problem_cols_reg = ["nvar", 
                "tree_length", 
                "tree_depth", 
                "time_obj", 
                "time_grad"
                ]
solver_cols_reg = ["mem",
              "neval_obj",
              "neval_grad",
              "timed_bytes"]

solver_cols_model = ["stats_elapsed_time", "timed_time"]

join_key = ["problem", "name", "nvar"]


print(problem)
print(solver)

problem_dfs = []
for file in problem:
    df = pd.read_json(file, lines=True)
    problem_dfs.append(df)
problem_df_all = pd.concat(problem_dfs, ignore_index=True)
for col in problem_cols_reg:
    problem_df_all[col] = pd.to_numeric(problem_df_all[col], errors="coerce")
problem_df_all = problem_df_all.copy().replace(r'^\s*$', np.nan, regex=True).replace(['NaN', 'None', 'nan'], np.nan)
problem_df_all = problem_df_all.copy().dropna(subset=problem_cols_reg + join_key)
problem_df_all.to_csv("./datasets/problem_metadata.csv",index=False)

solver_dfs = []
for file in solver:
    df = pd.read_json(file, lines=True)
    df = df.replace(r'^\s*$', np.nan, regex=True).replace(['NaN', 'None', 'nan'], np.nan)
    solver_dfs.append(df)
solver_df_all = pd.concat(solver_dfs, ignore_index=True)
solver_df_all = solver_df_all.copy().dropna(subset=solver_cols_reg + solver_cols_model + join_key)
solver_df_all.to_csv("./datasets/solver_runtimes.csv",index=False)

df = pd.merge(
    solver_df_all,
    problem_df_all,
    on=["problem", "name", "nvar"],
    how="inner",
    suffixes=('_solver', '_problem')
)

df[expr_cols] = df[expr_cols].eq(True)

# This handles: empty strings, whitespace, and literal "NaN"/"None" text
df[error_cols] = df[error_cols].replace(r'^\s*$', np.nan, regex=True).replace(['NaN', 'None', 'nan'], np.nan)
df["total_alloc"] = df[alloc_cols].sum(axis=1) / 1e6

df.to_csv("./datasets/raw.csv", index=False)
# df.loc[df["error_solver"].notna(), "stats_elapsed_time"] = MAX_TIME
df.loc[df["error_solver"] == "Timeout after 3600.0 s", "stats_elapsed_time"] = MAX_TIME #3599.2, 3601.2
df.loc[df["error_solver"].notna(), "total_alloc"] = np.nan
# df["solved"] = True
# df.loc[df[error_cols + ["error_solver"]].notna().any(axis=1), "solved"] = False

final_df = df[df[error_cols].isna().all(axis=1)]
final_df.to_csv("./datasets/final.csv", index=False)
error_df = df[df[error_cols + ["error_solver"]].notna().any(axis=1)]
error_df.to_csv("./datasets/error.csv", index=False)
