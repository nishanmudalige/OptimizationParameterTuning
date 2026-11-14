"""Check an installation: ``python -m memtune doctor``.

Every check prints one line: ``[ ok ]``, ``[warn]`` (works, but differs from the reference setup) or ``[FAIL]``
(results would be wrong or a command would not run). The exit status is 1 if any check failed.
"""
from __future__ import annotations

import hashlib
import platform
import sys
from importlib import metadata
from pathlib import Path

from . import __version__, paths
from .models import MODEL_FILES, REQUIRED_VERSIONS

OK, WARN, FAIL = "ok", "warn", "FAIL"
SUPPORTED_PYTHON = ((3, 10), (3, 12))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_sums(path: Path) -> dict:
    """``{relative name: sha256}`` from a ``SHA256SUMS`` file."""
    sums = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2:
            sums[parts[1].lstrip("*")] = parts[0]
    return sums


def check_python() -> tuple:
    v = sys.version_info[:2]
    text = f"Python {platform.python_version()} ({sys.executable})"
    if SUPPORTED_PYTHON[0] <= v <= SUPPORTED_PYTHON[1]:
        return OK, text
    return FAIL, text + ": memtune needs Python 3.10, 3.11 or 3.12"


def pinned_requirements() -> dict:
    lock = paths.repo_root() / "requirements-lock.txt"
    pins = {}
    if lock.is_file():
        for line in lock.read_text(encoding="utf-8").splitlines():
            line = line.split("#")[0].strip()
            if "==" in line:
                name, version = line.split("==")
                pins[name.strip()] = version.strip()
    return pins or dict(REQUIRED_VERSIONS)


def check_packages() -> list:
    out = []
    for name, wanted in pinned_requirements().items():
        try:
            have = metadata.version(name)
        except metadata.PackageNotFoundError:
            have = None
        if have == wanted:
            out.append((OK, f"{name} {have}"))
        elif name in REQUIRED_VERSIONS or have is None:
            hint = "the saved models need exactly this version" if name in REQUIRED_VERSIONS else "not installed"
            out.append((FAIL, f"{name}: found {have}, need {wanted} ({hint}); "
                              "run: python -m pip install -r requirements-lock.txt"))
        else:
            out.append((WARN, f"{name}: found {have}, the results were verified with {wanted}"))
    return out


def check_models(directory=None) -> list:
    folder = paths.models_dir(directory)
    sums_file = folder / "SHA256SUMS"
    if not sums_file.is_file():
        return [(FAIL, f"{sums_file} not found")]
    sums = read_sums(sums_file)
    bad, missing = [], []
    for rel in MODEL_FILES.values():
        f = folder / rel
        if not f.is_file():
            missing.append(rel)
        elif sums.get(rel) != sha256(f):
            bad.append(rel)
    if missing or bad:
        return [(FAIL, f"models in {folder}: missing {missing or 'none'}, changed {bad or 'none'}")]
    return [(OK, f"{len(MODEL_FILES)} models match their checksums ({folder})")]


def check_dataset(explicit=None) -> list:
    ds = paths.dataset_path(explicit)
    if not ds.is_file():
        return [(FAIL, f"dataset not found: {ds} (see data/README.md)")]
    sums_file = paths.repo_root() / "data" / "SHA256SUMS"
    wanted = read_sums(sums_file).get(ds.name) if sums_file.is_file() else None
    if wanted is None:
        return [(WARN, f"dataset {ds}: no reference checksum for a file of this name")]
    if sha256(ds) != wanted:
        return [(WARN, f"dataset {ds} differs from the published file; results will differ from the paper")]
    return [(OK, f"dataset matches its checksum ({ds})")]


def check_results() -> list:
    """Recompute the end-to-end table on the small sample and compare with the stored reference."""
    import json

    root = paths.repo_root()
    sample, golden = root / "tests" / "data" / "sample_runs.csv", root / "tests" / "golden" / "golden_sample.json"
    if not (sample.is_file() and golden.is_file()):
        return [(WARN, "sample data or reference results not found; skipped the computation check")]
    from .evaluate import evaluate_framework

    try:
        res = evaluate_framework(sample, one_step=False)
    except Exception as err:  # noqa: BLE001 - report any failure as a failed check
        return [(FAIL, f"could not evaluate the sample data: {type(err).__name__}: {err}")]
    ref = json.loads(golden.read_text(encoding="utf-8"))
    for policy, row in ref["table4"].items():
        for metric, value in row.items():
            got = res["table4"][policy][metric]
            if abs(got - value) > 1e-9 * max(1.0, abs(value)):
                return [(FAIL, f"sample results differ from the reference ({policy} {metric}: {got} vs {value})")]
    if any(res["chosen_mem"].get(k) != v for k, v in ref["chosen_mem"].items()):
        return [(FAIL, "the mem chosen on the sample data differs from the reference")]
    return [(OK, "sample results reproduce the reference exactly")]


def check_demo() -> list:
    from .demo import data_is_current, site_dir

    if not (site_dir() / "index.html").is_file():
        return [(WARN, "site/ not found: the demo needs a clone of the repository")]
    if data_is_current():
        return [(OK, "demo data is built (start it with: python -m memtune demo)")]
    return [(OK, "demo data not built yet (python -m memtune demo builds it in about 10 s)")]


def run(data=None, models=None, quick: bool = False) -> int:
    print(f"memtune {__version__} on {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"repository: {paths.repo_root()}\n")
    checks = [check_python(), *check_packages(), *check_models(models), *check_dataset(data)]
    if not quick and not any(status == FAIL for status, _ in checks):
        checks += check_results()
    checks += check_demo()
    for status, text in checks:
        print(f"[{status:^4}] {text}")
    failed = sum(status == FAIL for status, _ in checks)
    warned = sum(status == WARN for status, _ in checks)
    print()
    if failed:
        print(f"{failed} check(s) failed. See docs/INSTALL.md, section 'Troubleshooting'.")
        return 1
    print("Everything works." + (f" ({warned} warning(s) above.)" if warned else ""))
    return 0
