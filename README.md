# OptimizationParemeterTuning
A Machine Learning Approach for Hyper-Parameter Tuning of Unconstrained Optimization Solvers

[📄 View the full documentation (PDF)](Documentation/Final_Report/Final_Report_Felix_Gao.pdf)

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

---

## Week 2

- Felix:
  - Examine another solver to repeat analysis
  - Discuss ideas and strategy to complete the work
  - If time permits: read up on interpretting performance profiles
