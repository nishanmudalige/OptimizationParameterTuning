# OptimizationParemeterTuning
A Machine Learning Approach for Hyper-Parameter Tuning of Unconstrained Optimization Solvers

[📄 View the full documentation (PDF)](./documentation.pdf)

---
## Things to do (Week 4):

- Felix:
  - Add following columns into `.csv` file: 
    - Number of iterations ✅
    - Time takes for 1 gradient evaluation ✅
    - Time takes for 1 objective evaluation ✅
  - Develop script for creating plots   
    - Make plot for each solver and each problem ✅  * organized in sub-directories under ./OptimizationParameterTuning/Julia Notebook/plots
    - x-axis is `mem`; vary y-axis with other suitable variables ✅
  - Develop script to vary `nvar` for `NZF1` for scaling `mem`✅ * Made a script to vary 'nvar' to close to 1000 for all scalable problems in OptimizationProblems
  - Document the columns of the data table in a report. ✅

---
## Things to do (Week 3):

Felix's Progress
  - Good news! With a checking mechanism, I no longer need to run the script in one setting. I can always pause and resume next time.
  - Before starting a new run, the script checks if that entry already exists by first loading what's been recorded in the CSV file.
  - It only takes O(1) to perform the check since it's implemented using a hashmap.

- Felix has made very good progress with understanding and implementing the framework. We discussed methods to make implementation more efficient
- Felix:
  - Complete running script and recording results for the `lbfgs` solver ✅
  - Create multiple scripts for `trunk` on subsets of the optimization problems
    - Run each script for a subset of the probelms in a different terminal and let OS manage scheduling.
    - We expece OS to run each script from a termainal instance on a single core, therefore we can get results faster
    - We can combine the results `.csv` files later
  - Repeat for `tron` if time permits
  - Experiment with visualization packages in python if time permints
    - `ggplot2`, `plotly` and `seaborn`  
- Tangi and Nishan:
  - Prepare to discuss methods to visualize results
    
---
## Things to do (Week 2):

- Felix:
  - Update Julia script on notebook (`MLParameterSelection.ipynb`) to do the following:
    - Get the sample set of optimization problems from [https://github.com/JuliaSmoothOptimizers/OptimizationProblems.jl](https://github.com/JuliaSmoothOptimizers/OptimizationProblems.jl)
    - Vary the `mem` variable in the `lbfgs` solver over the set of problems and record the following (i.e. write to file as `.csv` or `.txt`)
      - Number of evaluations of objective function ✅
      - Number of evaluations of gradient (first order) ✅
      - Number of variables ✅
      - Optimization problem name/class ✅
      - Solver (right now, should only be `lbfgs`) ✅
      - Runtime ✅
      - Memory used (if possible) ✅
      - Number of vector-matrix operations ✅
      - See Tangi's sample script for all fields ✅
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
