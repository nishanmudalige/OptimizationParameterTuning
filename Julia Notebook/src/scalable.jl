using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()    
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

MAX_TIME = 60.0 * 5

function main()
    OptimizationProblems.NZF1_meta

    meta = OptimizationProblems.meta

    # Dataframe contains all the scalable problems
    scalable_problem_names = meta[(meta.variable_nvar .== true) .& (meta.ncon .== 0), [:name]]

    scalable_problems = (
        eval(Meta.parse("OptimizationProblems.ADNLPProblems.$(row.name)")) for row in eachrow(scalable_problem_names)
    )

    filename = "../results/lbfgs_scalable.csv"
    mkpath(dirname(filename))

    df = DataFrame(
                    :status => Symbol[],
                    :name => String[],
                    :solver => String[],
                    :mem => Int[],
                    :nvar => Int[],
                    :time => Float64[],
                    :memory => Float64[],
                    :num_iter => Int[],
                    :nvmops => Int[],
                    :neval_obj => Int[],
                    :init_eval_obj_time => Float64[],
                    :neval_grad => Int[],
                    :init_eval_grad_time => Float64[],
                    :is_init_run => Bool[]
                )

    i = 0
    for scalable_problem in scalable_problems
        nlp = scalable_problem(; n=1000)
        println(nlp.meta)  # CHANGED: println for readability
        i += 1
        param_set = JSOSolvers.LBFGSParameterSet(nlp)
        r = domain(param_set.mem)
        problem = nlp.meta.name
        solver = "LBFGSSolver"
        is_init_run = true
        for mem in r.lower:r.upper
            for i in 0:is_init_run
                reset!(nlp)
                println("Running $scalable_problem with nvar=$(nlp.meta.nvar) and mem=$mem")
                try
                    init_eval_obj_time = @elapsed obj(nlp, nlp.meta.x0)
                    init_eval_grad_time = @elapsed grad(nlp, nlp.meta.x0)
                    reset!(nlp)
                    bench = @benchmark JSOSolvers.lbfgs($nlp; mem = $mem, max_time=MAX_TIME)
                    reset!(nlp)
                    stats = JSOSolvers.lbfgs(nlp; mem = mem, max_time=MAX_TIME)
                    push!(df, (
                                status = stats.status, 
                                name = problem,
                                solver = solver,
                                mem = mem,
                                nvar = nlp.meta.nvar, 
                                time = minimum(bench).time / 10^9,
                                memory = minimum(bench).memory, 
                                num_iter = stats.iter,
                                nvmops = stats.solver_specific[:nprod],
                                neval_obj = nlp.counters.neval_obj,
                                init_eval_obj_time = init_eval_obj_time,
                                neval_grad = nlp.counters.neval_grad,
                                init_eval_grad_time = init_eval_grad_time,
                                is_init_run = is_init_run
                            )
                        )
                    is_init_run = false
                catch e
                    @info "Solver failed on $(nlp.meta.name): $e"
                    break
                end
                CSV.write(
                    filename,
                    DataFrame([last(df)]);
                    append = isfile(filename) && filesize(filename) > 0
                )
            end
        end
    end
end
main()