__precompile__(false)
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()
Pkg.status()
using OptimizationProblems
using ADNLPModels
using Symbolics
using ExpressionTreeForge # A bit outdated so it's not easy to find a compatible version.
using CSV, DataFrames

struct Type_node
    type_node::Symbol
    children::Vector{Type_node}
end

function Base.length(node::Type_node)
    isempty(node.children) ? 1 : sum(child -> length(child), node.children)
end

function depth(node::Type_node)
    isempty(node.children) && return 1
    return 1 + maximum(depth.(node.children))
end

function main()
    meta = OptimizationProblems.meta
    problem_names = meta[(meta.contype .== :unconstrained) .& (.!meta.has_bounds) .& (meta.nvar .>= 5),:name]

    problems = [Meta.parse("OptimizationProblems.ADNLPProblems.$(problem)") 
                    for problem ∈ problem_names];

    filename = "../metadata/length_and_depth.csv"
    mkpath(dirname(filename))

    df = DataFrame(
        :problem => String[],
        :length => Int[],
        :depth => Int[]
    )

    for problem in problems
        try
            nlp = eval(problem)()      # call the zero-arg constructor
            println("Extracting information of $(nlp.meta.name)")
            n = nlp.meta.nvar
            Symbolics.@variables x[1:n]
            fun = nlp.f(x)
            mtk_tree = Symbolics._toexpr(fun)
            expr_tree_Symbolics = transform_to_expr_tree(mtk_tree)

            tree_length = length(expr_tree_Symbolics)
            tree_depth = depth(expr_tree_Symbolics)
            push!(df, (; problem = nlp.meta.name, tree_length, tree_depth))

            CSV.write(
                filename,
                DataFrame([last(df)]);
                append = isfile(filename) && filesize(filename) > 0,
            )
        catch e
            @info "Failed to extract information on $(problem): $e"
        end
    end
end
main()