#!/usr/bin/env python
"""How much do the results depend on the train/test split? (NOT in the paper; see docs/ROBUSTNESS.md.)

    python scripts/robustness_check.py                   # 20 instance-level splits + leave-problems-out, ~10 min
    python scripts/robustness_check.py --repeats 3       # quicker
    python scripts/robustness_check.py --data tests/data/sample_runs.csv --repeats 2 --folds 3    # seconds

Everything is refitted on every split with the hyper-parameters of the committed models (nothing is tuned again)
and evaluated like the paper: performance ratio = run time of the chosen mem / run time of the best mem.

Writes to results/robustness/: instances_repeated.csv, instances_leave_problems_out.csv (one row per test instance
and policy), summary.json and summary.md.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from importlib import metadata

import pandas as pd

from memtune import load_framework, paths
from memtune import robustness as rb
from memtune.data import SPLIT_SEED, load_dataset


def markdown_across(table: pd.DataFrame) -> str:
    lines = ["| Policy | Median | 90th pct. | Mean | Total time / best | mem <= 3 |", "|---|---:|---:|---:|---:|---:|"]
    for policy, row in table.iterrows():
        cell = lambda c, f="{:.2f}": f"{f.format(row[('mean', c)])} ± {f.format(row[('sd', c)])}"  # noqa: E731
        lines.append(f"| {rb.POLICY_LABELS[policy]} | {cell('Q50')} | {cell('Q90')} | {cell('mean')} | "
                     f"{cell('time_ratio')} | {row[('mean', 'small_mem')]:.0%} |")
    return "\n".join(lines)


def markdown_pooled(table: pd.DataFrame) -> str:
    lines = ["| Policy | Median | 90th pct. | Mean (95 % interval) | Total time / best | mem <= 3 |",
             "|---|---:|---:|---:|---:|---:|"]
    for policy, row in table.iterrows():
        lines.append(f"| {rb.POLICY_LABELS[policy]} | {row['Q50']:.2f} | {row['Q90']:.2f} | "
                     f"{row['mean']:.2f} ({row['mean_lo']:.2f} to {row['mean_hi']:.2f}) | {row['time_ratio']:.2f} | "
                     f"{row['small_mem']:.0%} |")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="dataset file or folder (default: data/lbfgs_runs.csv.gz)")
    ap.add_argument("--out", help="output folder (default: results/robustness)")
    ap.add_argument("--repeats", type=int, default=20, help="number of instance-level splits (seeds 0, 1, ...)")
    ap.add_argument("--folds", type=int, default=5, help="folds of the leave-problems-out cross-validation")
    ap.add_argument("--jobs", type=int, default=-1, help="parallel jobs for the random forests (default: all cores)")
    a = ap.parse_args(argv)

    out = paths.results_dir(a.out) if a.out else paths.results_dir() / "robustness"
    out.mkdir(parents=True, exist_ok=True)
    df, _, _ = load_dataset(a.data)
    base = load_framework()
    n_inst, n_prob = len(df[["name", "nvar"]].drop_duplicates()), df["name"].nunique()
    print(f"{n_inst} complete instances of {n_prob} problems")
    start = time.time()
    say = lambda text: print(f"  [{time.time() - start:5.0f} s] {text}", flush=True)  # noqa: E731

    print(f"\n1. The paper's protocol (80 % / 20 % by instance), {a.repeats} other splits")
    repeated = rb.repeated_splits(df, base, range(a.repeats), a.jobs, say)
    per_seed = rb.per_split_table(repeated, "seed")
    across = rb.across_splits(per_seed)
    print("\n" + markdown_across(across))

    print(f"\n2. Leave whole problems out ({a.folds}-fold, grouped by problem name)")
    lopo = rb.leave_problems_out(df, base, a.folds, 0, a.jobs, say)
    pooled = rb.pooled_table(lopo)
    print("\n" + markdown_pooled(pooled))

    print("\n3. The paper's own split (seed 66), refitted with the same recipe")
    paper = rb.per_split_table(rb.repeated_splits(df, base, [SPLIT_SEED], a.jobs), "seed").set_index("policy")
    print(paper[rb.SUMMARY_COLUMNS].reindex(list(rb.POLICIES)).round(3).to_string())

    paired = {name: {p: rb.paired_comparison(rows, p) for p in rb.POLICIES if p != "default_mem5"}
              for name, rows in (("repeated", repeated), ("leave_problems_out", lopo))}
    fixed = {"repeated": sorted(int(v) for v in repeated.groupby("seed")["fixed_train_mem"].first()),
             "leave_problems_out": sorted(int(v) for v in lopo.groupby("fold")["fixed_train_mem"].first())}
    summary = {
        "system": {"platform": platform.platform(), "python": platform.python_version(),
                   "scikit-learn": metadata.version("scikit-learn"), "xgboost": metadata.version("xgboost")},
        "n_instances": n_inst, "n_problems": n_prob, "repeats": a.repeats, "folds": a.folds,
        "repeated_splits": {p: {s: {c: float(across.loc[p, (s, c)]) for c in rb.SUMMARY_COLUMNS}
                                for s in ("mean", "sd")} for p in across.index},
        "leave_problems_out": {p: {c: float(v) for c, v in row.items()} for p, row in pooled.iterrows()},
        "paper_split_refit": {p: {c: float(paper.loc[p, c]) for c in rb.SUMMARY_COLUMNS} for p in paper.index},
        "faster_or_slower_than_default": paired,
        "fixed_mem_chosen_on_training_data": fixed,
    }
    repeated.to_csv(out / "instances_repeated.csv", index=False)
    lopo.to_csv(out / "instances_leave_problems_out.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (out / "summary.md").write_text(
        f"## {a.repeats} instance-level splits (mean ± standard deviation over the splits)\n\n{markdown_across(across)}\n\n"
        f"## Leave whole problems out ({a.folds}-fold; {n_inst} instances pooled)\n\n{markdown_pooled(pooled)}\n",
        encoding="utf-8")
    print(f"\nfixed mem chosen on the training data: {fixed}")
    print(f"faster / slower than the default: {json.dumps(paired)}")
    print(f"\nWrote {out} ({time.time() - start:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
