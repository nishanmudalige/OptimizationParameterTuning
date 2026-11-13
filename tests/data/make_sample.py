"""Build the small fixture dataset used by the fast tests (run once; the output is committed).

    python tests/data/make_sample.py --data PATH/lbfgs_runs.csv
"""
import argparse
from pathlib import Path

import pandas as pd

COLS = ["name", "nvar", "mem", "tree_length", "tree_depth", "time_obj", "time_grad", "extract_duration_problem",
        "highest_degree (ExprTree)", "count_plus_minus", "count_trigonometric_function",
        "count_exponential_function", "count_op_function", "alloc_obj", "alloc_grad", "total_alloc",
        "neval_obj", "neval_grad", "timed_bytes", "stats_elapsed_time"]

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--n-instances", type=int, default=25)
ap.add_argument("--seed", type=int, default=2026)
a = ap.parse_args()

df = pd.read_csv(a.data, usecols=COLS, low_memory=False)
counts = df.groupby(["name", "nvar"])["mem"].transform("size")
full = df[counts == 100]
keys = full[["name", "nvar"]].drop_duplicates().sample(n=a.n_instances, random_state=a.seed)
sample = full.merge(keys, on=["name", "nvar"]).sort_values(["name", "nvar", "mem"])
out = Path(__file__).with_name("sample_runs.csv")
sample[COLS].to_csv(out, index=False)
print(f"{len(keys)} instances, {len(sample)} rows -> {out} ({out.stat().st_size/1e3:.0f} kB)")
