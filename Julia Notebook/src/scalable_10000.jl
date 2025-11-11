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
# using SolverBenchmark

MAX_TIME = 60.0 * 10

function main()
    meta = OptimizationProblems.meta

    # Dataframe contains all the scalable problems
    scalable_problem_names = meta[
        (meta.contype .== :unconstrained) .& (.!meta.has_bounds) .& (meta.variable_nvar) .& (meta.ncon .== 0),
        [:nvar, :name],
    ]


    scalable_problem_names = scalable_problem_names[46:79, :]

    scalable_problems = (
        eval(Meta.parse("OptimizationProblems.ADNLPProblems.$(row.name)")) for
        row in eachrow(scalable_problem_names)
    )

    filename = "../results/remaining_scalable_10000.csv"
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
        :init_eval_obj_mem => Float64[],
        :init_eval_obj_alloc => Int[],
        :neval_grad => Int[],
        :init_eval_grad_time => Float64[],
        :init_eval_grad_mem => Float64[],
        :init_eval_grad_alloc => Int[],
        :is_init_run => Bool[],
    )

    i = 0
    for scalable_problem in scalable_problems
        nlp_build_time = @elapsed nlp = scalable_problem(; n = 10000, matrix_free = true)
        i += 1
        param_set = JSOSolvers.LBFGSParameterSet(nlp)
        r = domain(param_set.mem)
        problem = nlp.meta.name
        solver = "LBFGSSolver"
        is_init_run = true
        for mem = r.lower:r.upper
            for i = 0:is_init_run
                reset!(nlp)
                println("Running $scalable_problem with nvar=$(nlp.meta.nvar) and mem=$mem")
                try
                    _, init_eval_obj_time, init_eval_obj_mem, _, init_eval_obj_gcstats =
                        @timed obj(nlp, nlp.meta.x0)
                    _, init_eval_grad_time, init_eval_grad_mem, _, init_eval_grad_gcstats =
                        @timed grad(nlp, nlp.meta.x0)
                    stats, time, memory, _, _ =
                        @timed JSOSolvers.lbfgs(nlp; mem = mem, max_time = MAX_TIME)
                    init_eval_obj_alloc =
                        init_eval_obj_gcstats.poolalloc + init_eval_obj_gcstats.bigalloc
                    init_eval_grad_alloc =
                        init_eval_grad_gcstats.poolalloc + init_eval_grad_gcstats.bigalloc


                    push!(
                        df,
                        (
                            status = stats.status,
                            name = problem,
                            solver = solver,
                            mem = mem,
                            nvar = nlp.meta.nvar,
                            time = time + nlp_build_time,
                            memory = memory / 1e6,
                            num_iter = stats.iter,
                            nvmops = stats.solver_specific[:nprod],
                            neval_obj = nlp.counters.neval_obj,
                            init_eval_obj_time = init_eval_obj_time,
                            init_eval_obj_mem = init_eval_obj_mem / 1e6,
                            init_eval_obj_alloc = init_eval_obj_alloc,
                            neval_grad = nlp.counters.neval_grad,
                            init_eval_grad_time = init_eval_grad_time,
                            init_eval_grad_mem = init_eval_grad_mem / 1e6,
                            init_eval_grad_alloc = init_eval_grad_alloc,
                            is_init_run = is_init_run,
                        ),
                    )
                    is_init_run = false
                catch e
                    @info "Solver failed on $(nlp.meta.name): $e"
                    break
                end
                CSV.write(
                    filename,
                    DataFrame([last(df)]);
                    append = isfile(filename) && filesize(filename) > 0,
                )
            end
        end
    end
end
main()