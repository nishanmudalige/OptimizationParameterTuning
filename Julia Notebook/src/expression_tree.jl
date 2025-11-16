__precompile__(false)
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()
Pkg.status()
using OptimizationProblems
using ADNLPModels
using Symbolics
using CSV, DataFrames
using ExpressionTreeForge

function length(node::ExpressionTreeForge.M_implementation_tree.Type_node{T}) where {T}
    if isempty(node.children)
        return 1
    else
        return sum(length, node.children)
    end
end

function depth(node::ExpressionTreeForge.M_implementation_tree.Type_node{T}) where {T}
    isempty(node.children) && return 1
    return 1 + maximum(depth.(node.children))
end

meta = OptimizationProblems.meta
problem_names = meta[(meta.contype .== :unconstrained) .& (.!meta.has_bounds) .& (meta.nvar .>= 5),:name]

problems = [Meta.parse("OptimizationProblems.ADNLPProblems.$(problem)") 
                    for problem ∈ problem_names];

filename = "./metadata/length_and_depth.csv"
mkpath(dirname(filename))

df = DataFrame(
    :problem => String[],
    :length => Int[],
    :depth => Int[]
)

for pb_expr in problems
    nlp = eval(pb_expr)()     # call the zero-arg constructor
    println("Extracting information of $(nlp.meta.name)")
    try
        n = nlp.meta.nvar
        Symbolics.@variables x[1:n]
        fun = nlp.f(x)
        mtk_tree = Symbolics._toexpr(fun)
        expr_tree_Symbolics = transform_to_expr_tree(mtk_tree)

        tree_length = length(expr_tree_Symbolics)
        tree_depth = depth(expr_tree_Symbolics)
        push!(df, (; problem = nlp.meta.name, length = tree_length, depth = tree_depth))
    catch e
        @info "Failed to extract information on $(nlp.meta.name)): $e"
    end
    CSV.write(filename, df)
end
