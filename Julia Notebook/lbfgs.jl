using Pkg
Pkg.activate(".")
Pkg.status()
Pkg.instantiate()
Pkg.update()
Pkg.status()

using BenchmarkTools
using CSV, DataFrames
using Random
using OptimizationProblems
using OptimizationProblems.ADNLPProblems # https://github.com/JuliaSmoothOptimizers/OptimizationProblems.jl
                                         # contains a test set of problems of type ADNLPModel
using NLPModels # see https://github.com/JuliaSmoothOptimizers/NLPModels.jl, it defines an abstract API to access a continuous optimization problem
using ADNLPModels # see https://github.com/JuliaSmoothOptimizers/ADNLPModels.jl
                  # is a concrete implementation of the ty
using SolverParameters # Define the main structure to handle the algorithm's parameters
using JSOSolvers

nlp = OptimizationProblems.ADNLPProblems.arglina(matrix_free = true) # is one of them
meta = OptimizationProblems.meta
problem_names = meta[meta.contype .== :unconstrained .&& .!meta.has_bounds .&& meta.nvar .>= 5, :name]; # LBFGS is an algorithm to solve unconstrained problems
                                                                   # For this example, we select only problems of size up to 10.
problems = [Meta.parse("OptimizationProblems.ADNLPProblems.eval($problem)()") for problem ∈ problem_names]; # https://jso.dev/OptimizationProblems.jl/dev/benchmark/                                                
filename = "./results/lbfgs.csv"
df = DataFrame(
                :status => Symbol[],
                :name => String[],
                :solver => String[],
                :mem => Int[],
                :nvar => Int[],
                :time => Float64[],
                :memory => Float64[],
                :nvmops => Int[],
                :neval_obj => Int[],
                :neval_grad => Int[]
            )


# load from CSV file
completed = Set{Tuple{String, String, Int}}()
if isfile(filename) && filesize(filename) > 0
    df_completed = CSV.read(filename, DataFrame)
    completed = Set(zip(df_completed.name, df_completed.solver, df_completed.mem))
end

i = 0
for pb_expr in problems
    nlp = eval(pb_expr)
    i += 1
    param_set = JSOSolvers.LBFGSParameterSet(nlp)
    r = domain(param_set.mem)
    problem = nlp.meta.name
    solver = "LBFGSSolver"
    for mem in r.lower:r.upper
        key = (problem, solver, mem)
        if key in completed
            @info "Skip $key — already seen"
            continue
        else
            reset!(nlp)
            println("Running $pb_expr with mem=$mem")
            try
                bench = @benchmark JSOSolvers.lbfgs($nlp; mem = $mem)
                stats = JSOSolvers.lbfgs(nlp; mem = mem)
                push!(df, (
                            status = stats.status, 
                            name = problem,
                            solver = solver,
                            mem = mem,
                            nvar = nlp.meta.nvar, 
                            time = minimum(bench).time,
                            memory = minimum(bench).memory, 
                            nvmops = stats.solver_specific[:nprod],
                            neval_obj = nlp.counters.neval_obj,
                            neval_grad = nlp.counters.neval_grad
                        )
                    )
                push!(completed, (problem, solver, mem))
            catch e
                @info "Solver failed on $(nlp.meta.name): $e"
                break
            end
        end
        CSV.write(filename, DataFrame([last(df)]); append=true)
    end
end

