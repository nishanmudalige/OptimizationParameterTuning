# OptimizationParemeterTuning
A Machine Learning Approach for Hyper-Parameter Tuning of Unconstrained Optimization Solvers

[📄 View the full documentation (PDF)](docs/Documentation/Final_Report/Final_Report_Felix_Gao.pdf)

---
### Things to do

- For March 20: Felix to create Jupyter Notebook document + email Tangi and Nishan with summary
- Tangi: looks at `existing` problems that we skipped in the script.
- Add the new features to feature_cols
- Try to use multivariate regression
- Can we predict stats_elapsed_time based on (neval_obj, neval_grad, timed_bytes)? It would be another regressor :).
- Start writing report with column description
- Finalize the new categorical variable for time
  - Aim for approximately 5 categories
- Finalize the new categorical variable for number of variables
  - Aim for approximately 5 categories

---

## New Ideas

- Repeat analysis with other solvers `trunc`, `tron`, `fomo`
  - See https://github.com/JuliaSmoothOptimizers/JSOSolvers.jl/tree/main/src
  - solvers ending in `ls` are essentially the same as the ones without `ls`
- Create plots of perforance profile
  - Compare `lbfgs` with optimal value of 'mem' and default value of `mem`
  - Compare `lbfgs` with `trunc`
  - Calculate area under the curves
- Tangi and Felix to set up meeting on bash script parallelizing solvers
https://github.com/tmigot/AutoJSOSolverSelection.jl
