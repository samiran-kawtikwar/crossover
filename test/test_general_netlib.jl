# test script for the netlib dataset

using crossover
using JuMP
using LinearAlgebra
using Dates

include("utils.jl")

N_MAX = 5000
# N_MAX = Inf

@enum TEST_MODE NETLIB_ALL NETLIB_SELF
TEST_MODE_STR = Dict(
    NETLIB_ALL => "netlib_all",
    NETLIB_SELF => "netlib_self",
)

test_mode = NETLIB_ALL
# test_mode = NETLIB_SELF

netlib_path = "/Users/sky/Desktop/computer/sky/solvers/dataset/netlib/feasible"

probs = readdir(netlib_path)
probs = sort([f for f in probs if endswith(f, ".mps")])

save_path = "/Users/sky/Desktop/computer/sky/projects/crossover/MIT-Lu-Lab/crossover/result/"
if !isdir(save_path)
    mkpath(save_path)
end

save_flag = true
# save_flag = false

# settings
time_limit = 300
seed = 0
epsilon = 1e-8
lpmethod = 6
rhs_shift = 10

presolve = true
# presolve = false

# ols_method = crossover.OLS_QR
# ols_method = crossover.OLS_LSMR
ols_method = crossover.OLS_AUTO
# ols_method = crossover.OLS_PCG

save_file = joinpath(save_path, TEST_MODE_STR[test_mode] * "-general" * "-" * string(today()) * ".csv")

# netlib problem set

if test_mode == NETLIB_SELF
    # adlittle and afiro for julia compiling
    selected_problems = [
        3, 4
    ]
elseif test_mode == NETLIB_ALL
    selected_problems = 1:length(probs)
end

for i in selected_problems
    probname = probs[i]
    println("Problem: $(probname)")

    is_lp_loaded = false
    is_lp_solved = false

    record = nothing
    glp = nothing
    num_support = nothing
    num_support_copt_crossover = nothing

    try
        glp = read_general_linear_programming(joinpath(netlib_path, probname))

        @assert glp.nRows < N_MAX or glp.nCols < N_MAX "nRows or nCols is too large"

        is_lp_loaded = true

        model = copt_solve_general(glp; lpmethod=6, tol=epsilon, presolve=presolve)
        solution = get_general_lp_solutions(model)
        is_lp_solved = true

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
        cross.setParam!("max_time", time_limit)
        cross.setParam!("pdlp_presolve", true)
        cross.setParam!("tol_bound", 1e-8)
        cross.setParam!("ols_method", ols_method)
        cross.setParam!("lp_method", lpmethod)
        cross.setParam!("rhs_shift", rhs_shift)
        # cross.setParam!("primal_push_method", crossover.P_OLS)
        # cross.setParam!("dual_push_method", crossover.D_OLS)
        cross.setParam!("seed", seed)
        cross.run!()

        num_support_pdlp_crossover = calculate_support_number(glp, cross.gsol)
        rank_basic = rank(cross.lp.A[:, cross.sol.basis_status.==crossover.BASIC])
        println("PDLP crossover support number: $(num_support_pdlp_crossover)")
        println("# row = $(cross.nRows), # col = $(cross.nCols)")
        println("Rank of basic columns: $(rank_basic)")

        println()
        println("Original support number: $(num_support)")
        println("COPT crossover support number: $(num_support_copt_crossover)")
        println("PDLP crossover support number: $(num_support_pdlp_crossover)")

        record = (
            id=i,
            probname=probname,
            nRows=glp.nRows,
            nCols=glp.nCols,
            orig_support=num_support,
            copt_support=num_support_copt_crossover,
            pdlp_support=num_support_pdlp_crossover,
            pdlp_cross_time=cross.timer.crossover_time,
            pdlp_cross_status=cross.info.crossover_status_str,
            rank_basic=rank_basic,
            is_optimal=cross.sol.is_optimal,
            pres_rel=cross.info.primal_infeasibility_rel,
            dual_rel=cross.info.dual_infeasibility_rel,
            gap_rel=cross.info.duality_gap_rel,
        )

    catch e
        println("Error: $(e)")

        record = (
            id=i,
            probname=probname,
            nRows=is_lp_loaded ? glp.nRows : -1,
            nCols=is_lp_loaded ? glp.nCols : -1,
            orig_support=is_lp_solved ? num_support : -1,
            copt_support=is_lp_solved ? num_support_copt_crossover : -1,
            pdlp_support=-1,
            pdlp_cross_time=-1,
            pdlp_cross_status=-1,
            rank_basic=-1,
            is_optimal=-1,
            pres_rel=-1,
            dual_rel=-1,
            gap_rel=-1,
        )
    end

    println(record)

    if save_flag
        write1record2csv(save_file, record)
    end

end
