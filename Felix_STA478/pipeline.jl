using Pkg
println("[1/5] Activating Environment...")
Pkg.activate(joinpath(@__DIR__, "..", "..")) 

using AutoJSOSolverSelection
using Dates

function debug_metadata(p_name::String, n_val::Int)
    println("[2/5] Starting metadata search at $(now())...")
    p_sym = Symbol(p_name)
    
    # Strict dictionary setup
    t = Dict(
        :OptimizationProblemsADNLPProblems => (
            Dict(p_sym => (type = Float64, n = n_val)),
        ),
    )
    
    println("[3/5] Calling collect_problems_data_sets (Registry Check)...")
    problems = AutoJSOSolverSelection.collect_problems_data_sets(t)
    
    if isempty(problems)
        println("!!! Result: No problems found for $p_name. Stopping here.")
        return nothing
    end
    println("Found $(length(problems)) problem(s).")

    println("[4/5] Calling collect_problems_data (This is the heavy structural analysis)...")
    # We use a very aggressive 30-second timeout here
    df = AutoJSOSolverSelection.collect_problems_data(
        problems;
        counter = 1,
        timeout = 30, 
        matrix_free = true,
        hessian_free = true,
    )
    
    println("[5/5] Success! Data collected at $(now()).")
    return df
end

# Hardcoded test to bypass ARGS issues
println("--- DEBUG START ---")
test_name = "arglinb"
test_n = 10
df = debug_metadata(test_name, test_n)

if df !== nothing
    println("\n--- RESULTS ---")
    display(df)
end