#!/usr/bin/env python
"""Precompute the interactive demo's data: every instance of the dataset through every stage of the framework.

    python scripts/export_demo_data.py                 # writes site/data/
    python scripts/export_demo_data.py --out DIR

Output:
    index.json               one entry per (problem, size): split, recommendations, observed best
    instances/<id>.json      features, Stage 1 and Stage 2 curves over mem = 1..100, observed values

The static page in site/ reads these files; nothing runs on a server.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from memtune import __version__, load_framework, paths
from memtune.data import GROUP_KEY, N_MEM, TIME_COLUMN, prepare, split_instances
from memtune.models import REQUIRED_VERSIONS
from memtune.predict import MEM_GRID, PROBLEM_FEATURES, explain, recommend_mem

OBSERVED = ["neval_obj", "neval_grad", "timed_bytes", TIME_COLUMN]
DEFAULT_MEM, FIXED_MEM = 5, 87

FEATURE_HELP = {
    "nvar": "number of variables",
    "tree_length": "number of nodes in the objective's expression tree",
    "tree_depth": "depth of the expression tree",
    "highest_degree (ExprTree)": "highest polynomial degree in the expression tree",
    "count_plus_minus": "number of + and - operations",
    "count_trigonometric_function": "number of trigonometric functions",
    "count_exponential_function": "number of exponential functions",
    "count_op_function": "number of other operator functions",
    "time_obj": "time of one objective evaluation (s)",
    "time_grad": "time of one gradient evaluation (s)",
    "extract_duration_problem": "time to extract these features (s)",
    "alloc_obj": "bytes allocated by one objective evaluation",
    "alloc_grad": "bytes allocated by one gradient evaluation",
    "total_alloc": "total bytes allocated while extracting the features",
}


def sig(x, digits=5):
    """Round to significant digits for compact JSON; None for NaN."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return None
    x = float(x)
    return 0.0 if x == 0 else float(f"{x:.{digits}g}")


def slug(name: str, nvar: int) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", f"{name}_{nvar}").strip("-").lower()


def git_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=paths.repo_root(),
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="dataset file or folder (default: data/lbfgs_runs.csv.gz)")
    ap.add_argument("--out", help="output folder (default: site/data)")
    a = ap.parse_args(argv)
    out = Path(a.out) if a.out else paths.repo_root() / "site" / "data"
    (out / "instances").mkdir(parents=True, exist_ok=True)

    cols = sorted(set(GROUP_KEY + PROBLEM_FEATURES + OBSERVED + ["mem", "problem", "status"]))
    raw = pd.read_csv(paths.dataset_path(a.data), usecols=cols, low_memory=False)
    raw = raw.sort_values(GROUP_KEY + ["mem"]).reset_index(drop=True)

    # which instances the paper used for training / testing (complete instances only, seed-66 split)
    train, test = split_instances(prepare(raw[[c for c in raw.columns if c not in ("problem", "status")]]))
    role = {}
    for frame, label in ((train, "train"), (test, "test")):
        for n, v in frame[GROUP_KEY].drop_duplicates().itertuples(index=False):
            role[(n, int(v))] = label

    inst = raw.groupby(GROUP_KEY, sort=True).first().reset_index()  # features are constant per instance
    fw = load_framework()
    curves = explain(inst, fw)
    rec = {key: recommend_mem(inst, fw, stage2=key) for key in ("rf", "gb")}

    index = []
    for i, row in inst.iterrows():
        name, nvar = row["name"], int(row["nvar"])
        sid = slug(name, nvar)
        obs = raw[(raw["name"] == name) & (raw["nvar"] == nvar)].set_index("mem")
        c = curves[curves["problem_index"] == i].set_index("mem")
        times = obs[TIME_COLUMN].reindex(MEM_GRID)
        best_mem = int(times.idxmin()) if times.notna().any() else None
        observed_at = lambda m: sig(times.get(m)) if m in times.index else None  # noqa: E731
        entry = {
            "id": sid, "name": name, "problem": row["problem"], "nvar": nvar,
            "role": role.get((name, nvar), "incomplete"),
            "n_observed": int(obs[TIME_COLUMN].notna().sum()),
            "rec": {k: int(v[i]) for k, v in rec.items()},
            "best_observed_mem": best_mem,
        }
        index.append(entry)
        detail = {
            **entry,
            "features": {f: sig(row[f]) for f in PROBLEM_FEATURES},
            "mem": [int(m) for m in MEM_GRID],
            "stage1": {
                "neval_obj": [sig(v) for v in c["stage1_neval_obj"]],
                "neval_grad": [sig(v) for v in c["stage1_neval_grad"]],
                "bytes": [sig(v) for v in c["stage1_bytes"]],
            },
            "stage2": {"rf": [sig(v) for v in c["stage2_time_rf"]], "gb": [sig(v) for v in c["stage2_time_gb"]]},
            "observed": {
                "neval_obj": [sig(v) for v in obs["neval_obj"].reindex(MEM_GRID)],
                "neval_grad": [sig(v) for v in obs["neval_grad"].reindex(MEM_GRID)],
                "bytes": [sig(v) for v in obs["timed_bytes"].reindex(MEM_GRID)],
                "time": [sig(v) for v in times],
                "status": [None if pd.isna(s) else str(s) for s in obs["status"].reindex(MEM_GRID)],
            },
            "observed_time_at": {"default": observed_at(DEFAULT_MEM), "fixed": observed_at(FIXED_MEM),
                                 "rf": observed_at(entry["rec"]["rf"]), "gb": observed_at(entry["rec"]["gb"]),
                                 "best": observed_at(best_mem) if best_mem else None},
        }
        (out / "instances" / f"{sid}.json").write_text(json.dumps(detail, separators=(",", ":")), encoding="utf-8")

    meta = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "memtune_version": __version__, "git": git_sha(), "libraries": REQUIRED_VERSIONS,
        "n_instances": len(index), "n_problems": int(inst["name"].nunique()), "n_mem": N_MEM,
        "default_mem": DEFAULT_MEM, "fixed_mem": FIXED_MEM, "features": FEATURE_HELP,
        "training_nvar_range": [int(inst["nvar"].min()), int(inst["nvar"].max())],
    }
    (out / "index.json").write_text(json.dumps({"meta": meta, "instances": index}, separators=(",", ":")),
                                    encoding="utf-8")
    size = sum(f.stat().st_size for f in out.rglob("*.json"))
    print(f"{len(index)} instances ({meta['n_problems']} problems) -> {out}  ({size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
