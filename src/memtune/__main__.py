"""memtune: choose the L-BFGS memory parameter with machine learning.

    python -m memtune reproduce     recompute the paper's results and compare with the reference
    python -m memtune evaluate      recompute the end-to-end results and write them to results/
    python -m memtune recommend     recommend mem for the problems in a CSV file of features
    python -m memtune demo          build and open the interactive demo on this machine
    python -m memtune doctor        check the installation
    python -m memtune info          show resolved paths and library versions
"""
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
    if "one_step_rf" in res["table4"]:
        print("(one_step_rf is not in the paper: a one-step model without the framework's small-mem bias,"
              " see docs/ONE_STEP_MODEL.md)")
    out = paths.results_dir(a.out)
    (out / "evaluation.json").write_text(json.dumps(res, indent=1, sort_keys=True), encoding="utf-8")
    table4_frame(res).to_csv(out / "table4.csv")
    print(f"\nWrote {out / 'evaluation.json'} and {out / 'table4.csv'}")
    return 0


def _cmd_recommend(a) -> int:
    from .predict import recommend_mem

    problems = pd.read_csv(a.features)
    problems["recommended_mem"] = recommend_mem(problems, stage2=a.stage2, models_dir=a.models)
    problems.to_csv(sys.stdout if a.output is None else a.output, index=False, lineterminator="\n")
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


def _cmd_doctor(a) -> int:
    from .doctor import run

    return run(a.data, a.models, quick=a.quick)


def _cmd_demo(a) -> int:
    from .demo import run

    return run(port=a.port, open_browser=not a.no_browser, rebuild=a.rebuild, build_only=a.build_only, data=a.data,
               host=a.host)


def _cmd_reproduce(a) -> int:
    from .reproduce import run

    return run(a.data, a.models, a.out, figures=a.figures)


COMMANDS = (
    ("reproduce", _cmd_reproduce, "recompute the paper's results and compare with the reference"),
    ("evaluate", _cmd_evaluate, "recompute the end-to-end results and write them to results/"),
    ("recommend", _cmd_recommend, "recommend mem for problems in a CSV of features"),
    ("demo", _cmd_demo, "build and open the interactive demo on this machine"),
    ("doctor", _cmd_doctor, "check the installation"),
    ("info", _cmd_info, "show resolved paths and library versions"),
)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="memtune", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"memtune {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn, help_ in COMMANDS:
        s = sub.add_parser(name, help=help_, description=help_)
        s.add_argument("--data", help="dataset CSV (or directory containing lbfgs_runs.csv[.gz])")
        s.add_argument("--models", help="models directory")
        s.add_argument("--out", help="results directory")
        s.set_defaults(fn=fn)
        if name == "recommend":
            s.add_argument("features", help="CSV with one row per problem (see docs/USAGE.md)")
            s.add_argument("--stage2", choices=["rf", "gb", "one_step"], default="rf",
                           help="rf / gb: the paper's framework; one_step: the one-step model (not in the paper)")
            s.add_argument("--output", help="write CSV here instead of stdout")
        elif name == "demo":
            s.add_argument("--port", type=int, default=8000, help="port (default 8000; the next free one is used)")
            s.add_argument("--host", default="127.0.0.1",
                           help="address to listen on (default: this machine only; 0.0.0.0 inside Docker)")
            s.add_argument("--no-browser", action="store_true", help="do not open the browser")
            s.add_argument("--rebuild", action="store_true", help="recompute the demo data even if it is up to date")
            s.add_argument("--build-only", action="store_true", help="compute the demo data and exit")
        elif name == "doctor":
            s.add_argument("--quick", action="store_true", help="skip the computation check (versions and files only)")
        elif name == "reproduce":
            s.add_argument("--figures", action="store_true", help="also write the aggregate figures (PDF)")
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
