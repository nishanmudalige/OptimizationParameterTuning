# Tests

```bash
python -m pytest                # everything, about 3 min (the full dataset is data/lbfgs_runs.csv.gz)
python -m pytest -m "not full"  # fast subset, under a minute, uses tests/data/sample_runs.csv
```

## What is checked

| File | Checks |
|---|---|
| `test_regression_sample.py` | the packaged pipeline reproduces the reference results on the sample data: data size, train/test split, Stage 1 and Stage 2 MSE and R², the end-to-end table, and the `mem` chosen for every test instance |
| `test_regression_full.py` | the same on the full dataset, plus the numbers reported in the paper (default `mem = 5`: 90th percentile 2.684, mean 2.262; `mem = 87`: 1.280, 1.093; framework RF: 2.195, 1.637; framework XGBoost: 2.061, 1.459) |
| `test_one_step.py` | the one-step model (not in the paper): its test-set numbers stay as documented and beat the framework and the default; the framework's predicted time rises with `mem` within an instance while the measured and the one-step ones fall; `train_one_step_model.py` rebuilds a model of the same quality (byte-identical on Windows; other systems draw different random numbers) |
| `test_models.py` | the 9 models are byte-identical to the committed ones (SHA-256), load with the pinned libraries and expect the right number of inputs |
| `test_predict.py` | `recommend_mem` gives the same choice as the evaluation pipeline |
| `test_machine_independence.py` | the program works from any working directory, with data in a folder with spaces, with `MEMTUNE_DATA_DIR` or `--data`, gives a clear error for a missing dataset, and no source file contains a machine-specific absolute path |
| `test_build_dataset.py` | the penalty rules for failed runs and the table join of `scripts/build_dataset.py` (synthetic tables) |
| `test_build_dataset_script.py` | runs the script end to end on the example logs in `tests/data/example_logs/` (log reading, join, penalties, `--gzip`, error message) |
| `test_scripts.py` | every script starts (`--help`); `plot_global_mem`, `plot_per_problem --limit 2` and `train_model_comparison --quick` run and write their outputs |
| `test_dataset_file.py` | the shipped `data/lbfgs_runs.csv.gz` has the documented checksums (archive and unzipped content), shape and `mem` range; `python -m gzip -d` restores the CSV; the code finds the `.gz` with or without a plain CSV |
| `test_data.py` | complete-instance filter, split by instance, error message for missing columns |
| `test_demo_site.py` | the interactive demo's data covers all 280 instances, the test problems' recommendations equal the paper's evaluation (and the one-step model's evaluation), `explain` agrees with `recommend_mem`, the page's files and Hugging Face settings exist |
| `test_cli_tools.py` | `memtune doctor` passes on a correct installation and reports a changed model or a missing dataset; `memtune reproduce` confirms all 32 reference values and notices a changed one; `memtune demo` builds its data, recognises damaged data, serves the page, and moves to the next port when one is taken |
| `test_installer.py` | `install.sh`: valid syntax, `--help`, the system report of `--detect`, the plan of `--dry-run` (nothing is created), clear errors for wrong options, and an existing environment is never deleted without asking. Nothing is installed by these tests; the real installation runs in continuous integration on Linux, macOS and Windows |
| `test_robustness.py` | the robustness evaluation (not in the paper): the fixed-`mem` baseline, whole problems held out with every instance tested once, the paper's split protocol with other seeds, the summaries, reproducible refits |
| `test_demo_browser.py` | the demo page in a real browser: it loads without errors, shows a recommendation and all charts, the *Run* button, the model switch and links work, a narrow screen does not scroll sideways, and missing data gives instructions. Needs Playwright (`python -m pip install playwright==1.52.0`, `python -m playwright install chromium`), otherwise skipped; continuous integration always runs it |
| `test_notebooks.py` | notebooks 01 – 06 run top to bottom and leave the models unchanged (needs `pip install -r requirements-notebooks.txt`, otherwise skipped) |

## Where the reference results come from

`tests/golden/generate_golden.py` re-implements the original notebook logic (`final_pipeline.ipynb`,
`multiple_regression_hybrid.ipynb`) **without importing `memtune`**, and writes
`golden_sample.json` and `golden_full.json`. The tests compare the package against these files: exact for
counts and chosen `mem`, relative tolerance 1e-9 for floating-point metrics. If a number ever changes, the test
fails, and the reason must be understood before the reference is updated.

To regenerate (only when the intended behaviour changes):

```bash
python tests/golden/generate_golden.py --data tests/data/sample_runs.csv --models models --out tests/golden/golden_sample.json
python tests/golden/generate_golden.py --data data/lbfgs_runs.csv.gz     --models models --out tests/golden/golden_full.json
```

`tests/data/make_sample.py` rebuilds the sample dataset from the full one (25 random complete instances).
