"""Recompute the paper's results and compare every value with the stored reference:
``python -m memtune reproduce``.

The reference (``tests/golden/golden_full.json``) was computed by a script that re-implements the original
notebooks without importing this package (``tests/golden/generate_golden.py``). A row is marked ``ok`` when the
recomputed value agrees to a relative tolerance of 1e-6.
"""
from __future__ import annotations

import json
import subprocess
import sys

from . import paths
from .evaluate import evaluate_framework

RTOL = 1e-6

POLICY_LABELS = {
    "baseline1_mem5": "Default mem = 5 (Baseline 1)",
    "baseline2_mem87": "Fixed mem = 87 (Baseline 2)",
    "framework_rf": "Framework",
    "framework_gb": "Alternative, XGBoost Stage 2",
}
METRIC_LABELS = {"Q50": "median ratio", "Q75": "75th pct. ratio", "Q90": "90th pct. ratio", "mean": "mean ratio",
                 "total_time": "total time (s)"}
MODEL_LABELS = {"rf": "random forest", "gb": "XGBoost"}


def comparison_rows(result: dict, reference: dict) -> list:
    """``[(section, quantity, reference value, recomputed value, agrees)]`` for every reported number."""
    rows = []

    def add(section, quantity, ref, got, exact=False):
        ok = (ref == got) if exact else abs(got - ref) <= RTOL * max(1.0, abs(ref))
        rows.append((section, quantity, ref, got, ok))

    add("Data", "complete problem instances", reference["n_instances"], result["n_instances"], exact=True)
    add("Data", "test instances", reference["n_test_instances"], result["n_test_instances"], exact=True)
    add("Data", "test runs", reference["n_test_rows"], result["n_test_rows"], exact=True)
    for stage, title in (("stage1", "Stage 1 accuracy (log targets, test set)"),
                         ("stage2", "Stage 2 accuracy (log time, test set)")):
        for model, label in MODEL_LABELS.items():
            for metric in ("mse", "r2"):
                add(title, f"{label}, {'MSE' if metric == 'mse' else 'R2'}",
                    reference[stage][model][metric], result[stage][model][metric])
    for policy, label in POLICY_LABELS.items():
        for metric, mlabel in METRIC_LABELS.items():
            add("End-to-end performance ratios (43 test instances)", f"{label}, {mlabel}",
                reference["table4"][policy][metric], result["table4"][policy][metric])
    same = all(result["chosen_mem"].get(p) == chosen for p, chosen in reference["chosen_mem"].items())
    rows.append(("End-to-end performance ratios (43 test instances)", "mem chosen for every test instance",
                 "reference", "identical" if same else "DIFFERENT", same))
    return rows


def run(data=None, models=None, out=None, figures: bool = False) -> int:
    root = paths.repo_root()
    golden = root / "tests" / "golden" / "golden_full.json"
    if not golden.is_file():
        print(f"{golden} not found: this command needs a clone of the repository.")
        return 1
    result = evaluate_framework(data, models, one_step=False)
    rows = comparison_rows(result, json.loads(golden.read_text(encoding="utf-8")))

    section = None
    for sec, quantity, ref, got, ok in rows:
        if sec != section:
            section = sec
            print(f"\n{sec}\n  {'quantity':<52}{'reference':>12}{'recomputed':>12}")
        fmt = (lambda v: f"{v:>12}") if isinstance(ref, (int, str)) else (lambda v: f"{v:>12.4f}")
        print(f"  {quantity:<52}{fmt(ref)}{fmt(got)}   {'ok' if ok else 'DIFFERENT'}")

    results = paths.results_dir(out)
    (results / "reproduction.json").write_text(json.dumps(
        [{"section": s, "quantity": q, "reference": r, "recomputed": g, "agrees": bool(ok)}
         for s, q, r, g, ok in rows], indent=1), encoding="utf-8")
    bad = sum(not ok for *_, ok in rows)
    print(f"\n{len(rows) - bad} of {len(rows)} reference values reproduced"
          f" (details: {results / 'reproduction.json'}).")

    if figures:
        print("\nFigures (PDF) ...")
        cmd = [sys.executable, str(root / "scripts" / "plot_global_mem.py"), "--out", str(results)]
        if data:
            cmd += ["--data", str(data)]
        subprocess.run(cmd, check=True)
    if bad:
        print("Some values differ. Check the installation with: python -m memtune doctor")
    return 1 if bad else 0
