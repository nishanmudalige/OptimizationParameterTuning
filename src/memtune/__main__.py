"""Command line: ``python -m memtune {evaluate,recommend,info}``."""
from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from . import __version__, paths
from .evaluate import evaluate_framework, table4_frame
from .models import REQUIRED_VERSIONS, check_versions


def _cmd_evaluate(a) -> int:
    res = evaluate_framework(a.data, a.models)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print(f"{res['n_instances']} complete instances, {res['n_test_instances']} test instances\n")
    print("Stage 1 (log targets, test set):", json.dumps(res["stage1"]))
    print("Stage 2 (log time, test set):   ", json.dumps(res["stage2"]), "\n")
    print(table4_frame(res).to_string())
    out = paths.results_dir(a.out)
    (out / "evaluation.json").write_text(json.dumps(res, indent=1, sort_keys=True), encoding="utf-8")
    table4_frame(res).to_csv(out / "table4.csv")
    print(f"\nWrote {out / 'evaluation.json'} and {out / 'table4.csv'}")
    return 0


def _cmd_recommend(a) -> int:
    from .predict import recommend_mem

    problems = pd.read_csv(a.features)
    problems["recommended_mem"] = recommend_mem(problems, stage2=a.stage2, models_dir=a.models)
    problems.to_csv(sys.stdout if a.output is None else a.output, index=False)
    return 0


def _cmd_info(a) -> int:
    ds = paths.dataset_path(a.data)
    print(f"memtune {__version__}")
    print("repo root :", paths.repo_root())
    print("data      :", ds, "(found)" if ds.is_file() else "(missing)")
    print("models    :", paths.models_dir(a.models))
    print("results   :", paths.results_dir(a.out, create=False))
    print("required  :", REQUIRED_VERSIONS)
    problems = check_versions(strict=False)
    print("versions  :", "OK" if not problems else problems)
    return 0 if not problems else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="memtune", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn, help_ in (
        ("evaluate", _cmd_evaluate, "recompute the paper's end-to-end results"),
        ("recommend", _cmd_recommend, "recommend mem for problems in a CSV of features"),
        ("info", _cmd_info, "show resolved paths and library versions"),
    ):
        s = sub.add_parser(name, help=help_)
        s.add_argument("--data", help="dataset CSV (or directory containing lbfgs_runs.csv)")
        s.add_argument("--models", help="models directory")
        s.add_argument("--out", help="results directory")
        s.set_defaults(fn=fn)
        if name == "recommend":
            s.add_argument("features", help="CSV with one row per problem (see docs/USAGE.md)")
            s.add_argument("--stage2", choices=["rf", "gb"], default="rf")
            s.add_argument("--output", help="write CSV here instead of stdout")
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
