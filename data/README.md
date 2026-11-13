# Data

The benchmark is **not stored in git** (about 100 MB; `data/*` is ignored except this file). The code looks
for `data/lbfgs_runs.csv`; use `MEMTUNE_DATA_DIR` or `--data` to keep it elsewhere
([`../docs/INSTALL.md`](../docs/INSTALL.md)).

> Status: where the dataset is distributed (GitHub release, Zenodo, …) is still to be decided by the
> authors. Until then, ask the authors for `lbfgs_runs.csv`. A small sample with the same columns that the
> pipeline needs is in `tests/data/sample_runs.csv` (25 instances, 2,500 rows) so you can try everything
> without the full file.

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

## Producing the file yourself

The measurements come from two data-collection steps (problem features, and L-BFGS runs for every `mem`).
Their raw JSON-lines logs are merged by `scripts/build_dataset.py`
([`../docs/USAGE.md`](../docs/USAGE.md), section 3). The distributed data-collection scripts themselves are
not part of this repository yet.

## Intermediate files

`build_dataset.py` also reads/writes `problem_metadata.csv` and `solver_runtimes.csv` (the two tables that
are merged) and `lbfgs_runs_success.csv` (runs without error).
