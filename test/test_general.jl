using crossover
using JuMP
using LinearAlgebra

include("utils.jl")

path = "/Users/sky/Desktop/computer/sky/solvers/dataset/netlib/feasible"
probs = readdir(path)
probs = sort([f for f in readdir(path) if endswith(f, ".mps")])

# probname = probs[4]  # afiro for julia compiling
probname = probs[3]
println("Problem: $(probname)")

epsilon = 1e-8
lpmethod = 6

presolve = true
# presolve = false

# ols_method = crossover.OLS_QR
# ols_method = crossover.OLS_LSMR
ols_method = crossover.OLS_AUTO
# ols_method = crossover.OLS_PCG

glp = read_general_linear_programming(joinpath(path, probname))
model = copt_solve_general(glp; lpmethod=6, tol=epsilon, presolve=presolve)
solution = get_general_lp_solutions(model)
num_support = calculate_support_number(glp, solution)
println("Original support number: $(num_support)")

# COPT crossover
model_copt_crossover = copt_solve_general(glp; lpmethod=6, crossover_flag=true, tol=epsilon, presolve=presolve)
sol_copt_crossover = get_general_lp_solutions(model_copt_crossover)
num_support_copt_crossover = calculate_support_number(glp, sol_copt_crossover)
println("COPT crossover support number: $(num_support_copt_crossover)")

# crossover.jl

cross = Crossover()
cross.load_general_lp!(glp)
cross.load_general_solution!(solution)
cross.setParam!("max_time", Inf)
cross.setParam!("pdlp_presolve", true)
cross.setParam!("tol_bound", 1e-8)
cross.setParam!("ols_method", ols_method)
cross.setParam!("lp_method", lpmethod)
# cross.setParam!("lp_method", 2)
cross.setParam!("rhs_shift", 10)
cross.setParam!("seed", 0)
# cross.setParam!("gamma", 1.0)
# cross.setParam!("verbose_level", 2)
# cross.setParam!("delta", 1.0)
# cross.setParam!("tol_ols", 1e-12)
# cross.setParam!("max_iter_ols", 1e8)
# cross.setParam!("primal_push_method", crossover.P_OLS)
# cross.setParam!("dual_push_method", crossover.D_OLS)
# cross.setParam!("push_general", false)
# cross.setParam!("primal_push_general", false)
# cross.setParam!("dual_push_general", false)
cross.run!()

num_support_pdlp_crossover = calculate_support_number(glp, cross.gsol)
rank_basic = rank(cross.lp.A[:, cross.sol.basis_status.==crossover.BASIC])
println("PDLP crossover support number: $(num_support_pdlp_crossover)")
println("# row = $(cross.nRows), # col = $(cross.nCols)")
println("Rank of basic columns: $(rank_basic)")

println()
println("Problem: $(probname)")
println("Original support number: $(num_support)")
println("COPT crossover support number: $(num_support_copt_crossover)")
println("PDLP crossover support number: $(num_support_pdlp_crossover)")

nothing