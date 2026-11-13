# Data

The benchmark ships **compressed** in this folder: `lbfgs_runs.csv.gz` (1.9 MB; the plain CSV is 98 MB, which is
why it is compressed). Everything in `memtune` reads the `.gz` file directly, so **you do not have to unzip it
to use the code**. Unzip it only if you want the plain CSV for another tool (Excel, R, a text editor, a
notebook of your own).

## Unzipping (only if you need the plain CSV)

Run one of these in the repository folder. Each creates `data/lbfgs_runs.csv` and keeps the `.gz` file.

| System | Command |
|---|---|
| Linux / macOS | `gunzip -k data/lbfgs_runs.csv.gz` |
| Windows (PowerShell or cmd) | `python -m gzip -d data\lbfgs_runs.csv.gz` (uses the Python you installed for this project) |
| Any system | `python -m gzip -d data/lbfgs_runs.csv.gz` |
| Windows, graphical | install [7-Zip](https://www.7-zip.org/), right-click the file, *7-Zip → Extract Here* |
| In Python | `pd.read_csv("data/lbfgs_runs.csv.gz")`: pandas decompresses automatically |

Windows Explorer cannot open `.gz` files by itself.

If both `lbfgs_runs.csv` and `lbfgs_runs.csv.gz` exist, `memtune` uses the plain `.csv`. The unzipped file is
git-ignored, so unzipping never dirties the repository.

### Check the files

`data/SHA256SUMS` has the checksum of the archive and of the unzipped CSV (the tests check both):

```bash
cd data && sha256sum -c SHA256SUMS --ignore-missing          # Linux / macOS
```

```powershell
Get-FileHash data\lbfgs_runs.csv.gz -Algorithm SHA256        # Windows: compare with data\SHA256SUMS
```

To keep the data somewhere else (another disk, a cluster folder), set `MEMTUNE_DATA_DIR` or pass `--data`; both
accept the file or its folder ([`../docs/INSTALL.md`](../docs/INSTALL.md)). A small sample with the columns the
pipeline needs is in `tests/data/sample_runs.csv`.

## How the data was produced

1. The L-BFGS runs and the problem features were collected with parallel Julia scripts that run the
   `JSOSolvers.jl` L-BFGS solver for every problem and every `mem` in 1..100 and write JSON-lines logs. Those
   scripts are not part of this repository yet.
2. The logs were merged and filtered with `scripts/build_dataset.py` (originally `json_to_csv_converter.py`),
   which produced the table stored here. `python scripts/build_dataset.py --gzip` writes the same kind of
   file, compressed.

## What it contains

`lbfgs_runs.csv`: one row per L-BFGS run of `JSOSolvers.jl` on one unconstrained test problem of
`OptimizationProblems.jl`, for one value of the memory parameter `mem`.

| | |
|---|---|
| Rows | 21,802 runs (280 problem instances × up to 100 values of `mem`) |
| Problems | 77 problems; instance = (`name`, `nvar`), `nvar` from 99 to 100,000 |
| `mem` | 1 … 100 |
| Complete instances | 215 instances have all 100 runs (21,500 rows); the analysis uses only these |
| Status | 21,487 `first_order` (converged), 315 `time_limit` |
| Time limit | 3,600 s per run |

The file has 97 columns (solver output, problem metadata, expression-tree features). The ones used by
`memtune`:

| Column | Role | Meaning |
|---|---|---|
| `name`, `nvar` | key | problem name and number of variables (an instance) |
| `mem` | parameter | L-BFGS memory, 1..100 |
| `stats_elapsed_time` | Stage 2 target | wall-clock time of the run in seconds |
| `neval_obj`, `neval_grad` | Stage 1 targets | number of objective / gradient evaluations |
| `timed_bytes` | Stage 1 target | bytes allocated during the run |
| `tree_length`, `tree_depth`, `highest_degree (ExprTree)` | feature | size, depth and degree of the objective's expression tree |
| `count_plus_minus`, `count_trigonometric_function`, `count_exponential_function`, `count_op_function` | feature | operator counts in the expression tree |
| `time_obj`, `time_grad` | feature | time of one objective / gradient evaluation (s) |
| `extract_duration_problem` | feature | time spent extracting the features (s) |
| `alloc_obj`, `alloc_grad`, `total_alloc` | feature | bytes allocated by one evaluation, and in total |

Other columns worth knowing: `problem` (problem constructor name), `status`, `iter`, `objective`,
`dual_feas`, `error_solver` / `error_problem` / `error_type` (text of a failure), `timed_time`,
`timed_gctime`, `nvmops`, and the `nlp_*` / `adbackend_*` / `*_available` metadata columns of the
NLPModels objects.

### Failed runs

`scripts/build_dataset.py` assigns a penalty to `stats_elapsed_time` when a run failed: twice the time
limit for an error, the time limit for a recognised time-out message. **In the dataset used in the paper,
all 315 `time_limit` runs carry `stats_elapsed_time = 7200` s (twice the 3,600 s limit)**, so the effective
time limit in the data is 7,200 s. Keep this in mind when you interpret absolute times.

## Intermediate files

`build_dataset.py` also reads/writes `problem_metadata.csv` and `solver_runtimes.csv` (the two tables that
are merged) and `lbfgs_runs_success.csv` (runs without error).
