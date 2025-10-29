using OptimizationProblems
using ADNLPModels

# Data from OptimizationProblems.jl
# Careful, nvar and ncon are the default, they may vary if variable_nvar or variable_ncon are true.
name_problem = "arglina"
scalable_problem_names = meta[(meta.name .== name_problem), :]

# Get the type
nlp = OptimizationProblems.ADNLPProblems.arglina()
# Return the type of problem in terms of programming
type_of_pb = typeof(nlp)
if type_of_pb <: ADNLPModel
    objective_function = nlp.f # this is a function
    # This gives information on how is computed the gradient.
    gradient_backend = typeof(nlp.adbackend.gradient_backend)
end

nlp.meta # also contains more info on the problem, but mainly useless for unconstrained problems

# How to get the expression tree of the function:
# use Symbolics to write it as an Expr
# then ExpressionTreeForge make a tree out of it.
using Symbolics

n = 100
Symbolics.@variables x[1:n] # must be x
fun = nlp.f(x)
mtk_tree = Symbolics._toexpr(fun)
using ExpressionTreeForge # A bit outdated so it's not easy to find a compatible version.
expr_tree_Symbolics = transform_to_expr_tree(mtk_tree)
