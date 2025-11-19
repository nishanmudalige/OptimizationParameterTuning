#!/usr/bin/env julia
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using Distributed

# --- spin up workers (tweak N as you like) ---
const NWORKERS = max(1, Sys.CPU_THREADS ÷ 2)
if nprocs() == 1
    addprocs(NWORKERS; exeflags="--project=$(Base.active_project())")
end

# Share the active project path with workers and load deps
const _PROJ = Base.active_project()
@everywhere begin
    using Pkg
    Pkg.activate($_PROJ); Pkg.instantiate()
end

@everywhere begin
    using BenchmarkTools
    using CSV, DataFrames
    using Random
    using OptimizationProblems
    using OptimizationProblems.ADNLPProblems
    using NLPModels
    using ADNLPModels
    using SolverParameters
    using JSOSolvers
    using SolverBenchmark
    using Dates
end

const MAX_TIME = 60.0 * 10  # 10 minutes

# ---------- helpers ----------
# Robust way to fetch a problem constructor by name (no eval)
@everywhere function get_adnlp_constructor(name::AbstractString)
    return getproperty(OptimizationProblems.ADNLPProblems, Symbol(name))
end

# Run ONE job: (problem_name, mem, warmup_flag)
@everywhere function run_one(problem_name::String, mem::Int, do_warmup::Bool)
    # Build problem (time this too, as you did)
    nlp_build_time = @elapsed begin
        nlp = get_adnlp_constructor(problem_name)(; n = 10000, matrix_free = true)
        # optional light warmup to stabilize allocs:
        if do_warmup
            _ = obj(nlp, nlp.meta.x0); _ = grad(nlp, nlp.meta.x0)
        end
        # main run
        reset!(nlp)
        solver = "LBFGSSolver"
        stats, time, memory, _, _ = @timed JSOSolvers.lbfgs(nlp; mem = mem, max_time = MAX_TIME)

        # initial eval stats (measured fresh on the same worker)
        _, init_eval_obj_time, init_eval_obj_mem, _, gc_o = @timed obj(nlp, nlp.meta.x0)
        _, init_eval_grad_time, init_eval_grad_mem, _, gc_g = @timed grad(nlp, nlp.meta.x0)

        init_eval_obj_alloc = gc_o.poolalloc + gc_o.bigalloc
        init_eval_grad_alloc = gc_g.poolalloc + gc_g.bigalloc

        # Return a small NamedTuple (serializable)
        return (
            status = stats.status,
            name = problem_name,
            solver = solver,
            mem = mem,
            nvar = nlp.meta.nvar,
            time = time + nlp_build_time,
            memory = memory / 1e6,
            num_iter = stats.iter,
            nvmops = get(stats.solver_specific, :nprod, missing),
            neval_obj = nlp.counters.neval_obj,
            init_eval_obj_time = init_eval_obj_time,
            init_eval_obj_mem = init_eval_obj_mem / 1e6,
            init_eval_obj_alloc = init_eval_obj_alloc,
            neval_grad = nlp.counters.neval_grad,
            init_eval_grad_time = init_eval_grad_time,
            init_eval_grad_mem = init_eval_grad_mem / 1e6,
            init_eval_grad_alloc = init_eval_grad_alloc,
            is_init_run = do_warmup,
        )
    end
end

# ---------- main ----------
function main()
    meta = OptimizationProblems.meta

    # pick unconstrained, no bounds, scalable, ncon==0
    dfmeta = meta[(meta.contype .== :unconstrained) .& (.!meta.has_bounds) .&
                  (meta.variable_nvar) .& (meta.ncon .== 0), [:nvar, :name]]
    
    dfmeta = dfmeta[52:79, :]

    # Turn into a Vector{String} of problem names
    problems = collect(String.(dfmeta.name))

    # Build the job list: for each problem, try all mem values,
    # and run each with do_warmup = true once (then false)
    # We’ll construct mem domain locally first to avoid extra RPCs.
    # We need an nlp to get mem bounds; do this on master cheaply with n=10.
    function mem_range_for(problem_name)
        nlp = get_adnlp_constructor(problem_name)(; n = 10, matrix_free = true)
        r = JSOSolvers.LBFGSParameterSet(nlp).mem |> domain
        return r.lower:r.upper
    end

    jobs = Vector{Tuple{String,Int,Bool}}()
    for pname in problems
        for m in mem_range_for(pname)
            # run once with warmup=true (collect init stats), then the “real” run
            push!(jobs, (pname, m, true))
            push!(jobs, (pname, m, false))
        end
    end

    # Parallel map: each job returns a NamedTuple
    # You can tune `batch_size` for large job lists.
    results = pmap(jobs) do (pname, m, warm)
        try
            run_one(pname, m, warm)
        catch e
            @info "Solver failed on $pname (mem=$m, warm=$warm): $e"
            # return a minimal row with failure status
            (status = :exception, name = pname, solver = "LBFGSSolver", mem = m,
             nvar = missing, time = missing, memory = missing, num_iter = missing,
             nvmops = missing, neval_obj = missing, init_eval_obj_time = missing,
             init_eval_obj_mem = missing, init_eval_obj_alloc = missing,
             neval_grad = missing, init_eval_grad_time = missing,
             init_eval_grad_mem = missing, init_eval_grad_alloc = missing,
             is_init_run = warm)
        end
    end

    # Build a DataFrame on the master and write once
    df = DataFrame(results)
    filename = joinpath(@__DIR__, "..", "results", "test.csv")
    mkpath(dirname(filename))
    CSV.write(filename, df)  # overwrites; use append=true if you prefer appending
    @info "Wrote $(nrow(df)) rows to $filename"
end

main()