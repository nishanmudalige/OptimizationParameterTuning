using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()    
Pkg.status()

using OptimizationProblems
using ADNLPModels

OptimizationProblems.NZF1_meta

meta = OptimizationProblems.meta

# Dataframe contains all the scalable problems
scalable_problem_names = meta[(meta.variable_nvar .== true) .& (meta.ncon .== 0), [:name]]

scalable_problems = (
    eval(Meta.parse("OptimizationProblems.ADNLPProblems.$(row.name)")) for row in eachrow(scalable_problem_names)
)

for scalable_problem in scalable_problems
    nlp = scalable_problem(; n=1000)
    println(nlp.meta)  # CHANGED: println for readability
end
