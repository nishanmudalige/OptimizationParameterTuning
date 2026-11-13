#!/usr/bin/env python
"""Train the one-step run-time model (an alternative to the paper's framework; NOT in the paper).

    python scripts/train_one_step_model.py              # train with the chosen settings, write models/one_step/
    python scripts/train_one_step_model.py --select     # first rerun the cross-validation that chose them
    python scripts/train_one_step_model.py --out DIR    # write somewhere else (the committed file stays)

The model is trained on the paper's training instances (seed-66 split) only. See memtune/one_step.py
for why it exists.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib

from memtune import paths
from memtune.data import FEATURES, load_dataset
from memtune.evaluate import fixed_mem_score, performance_ratios, summarize_ratios
from memtune.models import MODEL_FILES
from memtune.one_step import PARAMS, cross_validate, fit, within_instance_trend


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="dataset file or folder (default: data/lbfgs_runs.csv.gz)")
    ap.add_argument("--out", help="models folder to write into (default: models/)")
    ap.add_argument("--select", action="store_true", help="rerun the cross-validation over the settings grid")
    a = ap.parse_args(argv)

    _, train, test = load_dataset(a.data)
    if a.select:
        t0 = time.time()
        cv = cross_validate(train)
        print(f"5-fold cross-validation on the {train[['name', 'nvar']].drop_duplicates().shape[0]} training "
              f"instances ({time.time() - t0:.0f} s), best first:")
        print(cv.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        best = cv.iloc[0]
        print(f"\nchosen: min_samples_leaf={int(best['min_samples_leaf'])}, max_features={best['max_features']}"
              f"   (memtune.one_step.PARAMS: {PARAMS})\n")

    model = fit(train)
    target = Path(a.out or paths.models_dir()) / MODEL_FILES["one_step_rf"]
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, target, compress=3)
    print(f"wrote {target} ({target.stat().st_size / 1e6:.1f} MB)")

    pred = model.predict(test[FEATURES].to_numpy(dtype=float))
    rows = {"one_step_rf": pred, "fixed_mem5": fixed_mem_score(test, 5), "fixed_mem87": fixed_mem_score(test, 87)}
    summary = {}
    for name, score in rows.items():
        m = performance_ratios(test, score)
        summary[name] = {**summarize_ratios(m["ratio"]), "total_time": float(m["stats_elapsed_time"].sum())}
    summary["one_step_rf"]["within_instance_corr_mem_time"] = within_instance_trend(test, pred)
    print("test instances (43):")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
