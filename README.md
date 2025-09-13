# OptimizationParemeterTuning
A Machine Learning Approach for Hyper-Parameter Tuning of Unconstrained Optimization Solvers

---
## Things to do (Week 2):

- Felix:
  - Update Julia script on notebook (`MLParameterSelection.ipynb`) to do the following:
    - Get the sample set of optimization problems from [https://github.com/JuliaSmoothOptimizers/OptimizationProblems.jl](https://github.com/JuliaSmoothOptimizers/OptimizationProblems.jl)
    - Vary the `mem` variable in the `lbfgs` solver over the set of problems and record the following (i.e. write to file as `.csv` or `.txt`)
      - Number of evaluations of objective function
      - Number of evaluations of gradient (first order)
      - Number of variables
      - Optimization problem name/class
      - Solver (right now, should only be `lbfgs`)
      - Runtime
      - Memory used (if possible)
      - Number of steps to solve
      - Number of vector-matrix operations
      - See Tangi's sample script for all fields
    - On HPC, run `lbfgs` on entire subset of problems to generate `.csv`
    - Try plotting results
    - If time permits attempt with another solver
      - **Caution:** Update variables for different solver
      - Priority to `trunk` and `tron`


---

## Things to do (Week 1):

- Felix:
  - Update Julia script on notebook (`MLParameterSelection.ipynb`) to do the following:
    - Get the sample set of optimization problems from [https://github.com/JuliaSmoothOptimizers/OptimizationProblems.jl](https://github.com/JuliaSmoothOptimizers/OptimizationProblems.jl)
    - Vary the `mem` variable in the `lbfgs` solver over the set of problems and record the following (i.e. write to file as `.csv` or `.txt`)
      - Optimization problem name/class
      - Solver (right now, should only be `lbfgs`)
      - Runtime
      - Memory used (if possible)
      - Number of steps to solve
      - Number of vector-matrix operations
    - If this process is taking too long, perform steps above on a subset of the optimization problems
  - Desired outcome for next meeting:
    - Notebook with updated script which performs the tasks above
    - A `.csv` or `.txt` file with a record of the information above
- Nishan:
  - Create GitHub page ✅
  - Add Julia notebook, project summary, update Readme ✅
  - Get HPC working again 
