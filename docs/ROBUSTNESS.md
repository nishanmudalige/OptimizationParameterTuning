# How much do the results depend on the train/test split?

> **This analysis is not in the paper.** The paper's numbers are unchanged; `python -m memtune reproduce` checks
> them.

## Two properties of the paper's split

The paper evaluates on one split of the 215 complete instances: 172 for training and 43 for testing, drawn by
instance with seed 66. An *instance* is a problem at one size (`name`, `nvar`).

1. **43 test instances are few.** A 90th percentile over 43 values is decided by four or five instances, so the
   summary statistics can move noticeably with another split.
2. **Every test instance is a known problem at a new size.** The 43 test instances come from 38 problems, and all
   38 also occur in the training data at another size. The reported numbers therefore describe how well the models
   transfer to a new size of a problem they have seen, not to a new problem.

## What the script does

```bash
python scripts/robustness_check.py                 # about 3 minutes
python scripts/robustness_check.py --repeats 5     # fewer splits
python scripts/robustness_check.py --data tests/data/sample_runs.csv --repeats 2 --folds 3    # a quick trial
```

It evaluates in two ways, both with the paper's performance ratio (run time of the chosen `mem` divided by the run
time of the best `mem`, differences below 0.1 s counted as ties):

| Protocol | What it answers |
|---|---|
| **Repeated splits**: the paper's 80 % / 20 % split by instance, with 20 other seeds | How much do the summary statistics vary with the split? |
| **Leave whole problems out**: 5-fold cross-validation grouped by problem name, so all sizes of a problem are held out together; every instance is tested exactly once | How well do the choices transfer to problems the models have never seen? |

For every split, all models are **fitted again on that split's training data**:

- Stage 1 (random forest and XGBoost) and Stage 2 (random forest and XGBoost) with the hyper-parameters stored in
  the committed models and the procedure of `notebooks/05_hybrid_framework.ipynb`: Stage 2 is trained on 5-fold
  out-of-fold Stage 1 predictions. No hyper-parameter is tuned again.
- The one-step model with its documented settings ([`ONE_STEP_MODEL.md`](ONE_STEP_MODEL.md)).

Five ways of choosing `mem` are compared on every test set:

| Policy | Meaning |
|---|---|
| Default `mem = 5` | the default of `JSOSolvers.jl` |
| Fixed `mem` chosen on the training instances | one value for all problems: the `mem` with the smallest mean performance ratio on the *training* instances of that split. Unlike the paper's `mem = 87`, it uses no test data, so it is a fair fixed baseline |
| Framework | the paper's framework: hybrid Stage 1 and random-forest Stage 2 |
| Alternative, XGBoost Stage 2 | the same Stage 1 with the XGBoost Stage 2 of the journal article; not part of the framework |
| One-step model | not in the paper |

## Output

Everything is written to `results/robustness/`:

| File | Content |
|---|---|
| `summary.md` | the two result tables: mean ± standard deviation over the splits, and the pooled leave-problems-out results with a 95 % bootstrap interval for the mean ratio |
| `summary.json` | the same numbers, the fixed `mem` chosen in every split, the number of test instances on which each policy was faster or slower than the default, and the versions of Python, scikit-learn and XGBoost |
| `instances_repeated.csv`, `instances_leave_problems_out.csv` | one row per test instance and policy: chosen `mem`, its measured time, the best time, the ratio |

Columns of the tables: median, 90th percentile and mean of the performance ratio; *total time / best*, the summed
run time of the chosen runs divided by the summed run time of the best runs; and the share of test instances for
which a policy chose `mem` ≤ 3.

## Reading the results

- **Refitting is not the same as loading the committed models.** With the paper's own split (seed 66), the refitted
  random-forest Stage 1 equals the committed one to 14 digits, but the refitted XGBoost Stage 1 does not reproduce
  the committed file, and Stage 2 depends on it. The script prints the refitted results for seed 66 next to the
  others, so the size of this effect is visible. The published numbers are those of the committed models.
- **The mean ratio is sensitive to single instances.** On a few problems a very small `mem` converges extremely
  slowly (`morebv` with 100 variables: 341 s with `mem = 1`, 0.11 s with the best `mem`), which gives a ratio in
  the thousands. Read the mean together with the median and the 90th percentile.
- **Results differ slightly between systems.** Windows on one side and Linux / macOS on the other draw different
  random numbers for the same seed when a random forest is trained ([`REPRODUCING.md`](REPRODUCING.md)).

The code is in [`src/memtune/robustness.py`](../src/memtune/robustness.py); `tests/test_robustness.py` checks the
splitting rules (no problem on both sides, every instance tested once), the fixed-`mem` baseline and the summaries.
