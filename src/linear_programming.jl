"""
a standard form of linear programming

min  cost' * x + offset
s.t. A * x = b
     lb <= x <= ub

nRows:  number of constraints
nCols:  number of variables
cost:   cost vector in the MINIMIZATION form
A:      constraint matrix
b:      right-hand side vector
lb:     variable lower bound vector
ub:     variable upper bound vector
sense:  sense of the ORIGINAL objective function
offset: offset of the ORIGINAL objective function
"""
Base.@kwdef mutable struct LinearProgramming
    nRows::Int64 = 0
    nCols::Int64 = 0
    cost::Vector{Float64} = []
    A::SparseArrays.SparseMatrixCSC{Float64,Int64} = SparseMatrixCSC{Float64,Int64}(undef, 0, 0)
    b::Vector{Float64} = []
    lb::Vector{Float64} = []
    ub::Vector{Float64} = []
    sense::SENSE = MINIMIZE
    offset::Float64 = 0.0
end


"""
basic solution for a standard form of linear programming

min  cost' * x + offset
s.t. A * x = b      (y)
     lb <= x <= ub  (s)

nRows:        number of constraints
nCols:        number of variables
col_value:    values of x
col_dual:     values of s
row_value:    values of A * x
row_dual:     values of y
basis_status: basis status vector
dual_status:  dual status vector
is_optimal:   whether the solution is optimal
is_vertex:    whether the solution is a vertex
"""
Base.@kwdef mutable struct Solution
    nRows::Int64 = 0
    nCols::Int64 = 0
    col_value::Vector{Float64} = []
    col_dual::Vector{Float64} = []
    row_value::Vector{Float64} = []
    row_dual::Vector{Float64} = []
    n_basis::Int64 = 0
    n_dual_active::Int64 = 0
    basis_status::Vector{BASIS_STATUS} = []
    dual_status::Vector{DUAL_STATUS} = []
    is_optimal::Bool = false
    is_vertex::Bool = false
end


"""
solving information of a standard form of linear programming

crossover_status:     status of crossover
primal_objective:     primal objective value
dual_objective:       dual objective value
duality_gap:          duality gap
primal_infeasibility: primal infeasibility
dual_infeasibility:   dual infeasibility
"""
Base.@kwdef mutable struct Information
    crossover_status::CROSSOVER_STATUS = UNSTARTED
    crossover_status_str::String = "Crossover unstarted."
    primal_objective::Float64 = +Inf
    dual_objective::Float64 = -Inf
    duality_gap_abs::Float64 = +Inf
    duality_gap_rel::Float64 = +Inf
    primal_infeasibility_abs::Float64 = 1
    primal_infeasibility_rel::Float64 = 1
    dual_infeasibility_abs::Float64 = 1
    dual_infeasibility_rel::Float64 = 1
end


function reformulate_general_lp(glp::GeneralLinearProgramming, full_row_rank::Bool=false)

    if full_row_rank
        return reformulate_full_row_rank_general_lp(glp)
    else
        return reformulate_std_general_lp(glp)
    end

end

function reformulate_std_general_lp(glp::GeneralLinearProgramming)

    idx_eq = (glp.lhs .== glp.rhs)
    idx_ge = (glp.lhs .> -Inf) .& .!idx_eq
    idx_le = (glp.rhs .< +Inf) .& .!idx_eq

    n_eq = sum(idx_eq)
    n_ge = sum(idx_ge)
    n_le = sum(idx_le)

    nRows = n_eq + n_ge + n_le
    nCols = glp.nCols + n_ge + n_le

    cost = [Int(glp.sense) * glp.cost; zeros(n_ge + n_le)]
    A = SparseMatrixCSC{Float64,Int64}(
        [[glp.A[idx_eq, :];
            -glp.A[idx_ge, :];
            glp.A[idx_le, :]] [spzeros(n_eq, n_ge + n_le); sparse(I, n_ge + n_le, n_ge + n_le)]]
    )
    b = [glp.lhs[idx_eq]; -glp.lhs[idx_ge]; glp.rhs[idx_le]]
    lb = [glp.lb; zeros(n_ge + n_le)]
    ub = [glp.ub; fill(Inf, n_ge + n_le)]

    return LinearProgramming(nRows, nCols, cost, A, b, lb, ub, glp.sense, glp.offset)
end

function reformulate_full_row_rank_general_lp(glp::GeneralLinearProgramming)

    nRows = glp.nRows
    nCols = glp.nCols + glp.nRows

    cost = [Int(glp.sense) * glp.cost; zeros(glp.nRows)]
    A = SparseMatrixCSC{Float64,Int64}(
        [glp.A -sparse(I, glp.nRows, glp.nRows)]
    )
    b = zeros(glp.nRows)
    lb = [glp.lb; glp.lhs]
    ub = [glp.ub; glp.rhs]

    return LinearProgramming(nRows, nCols, cost, A, b, lb, ub, glp.sense, glp.offset)
end

function initialize_information!(info::Information)
    info.crossover_status = UNSTARTED
    info.crossover_status_str = "Crossover unstarted."
    info.primal_objective = +Inf
    info.dual_objective = -Inf
    info.duality_gap_abs = +Inf
    info.duality_gap_rel = +Inf
    info.primal_infeasibility_abs = 1.0
    info.primal_infeasibility_rel = 1.0
    info.dual_infeasibility_abs = 1.0
    info.dual_infeasibility_rel = 1.0
end

function cal_optimality!(info::Information, lp::LinearProgramming, sol::Solution; nrm_type::OPTIMALITY_NORM=L_2)
    if nrm_type == L_2
        ord = 2
    elseif nrm_type == L_INF
        ord = Inf
    else
        error("unsupported norm type")
    end

    # primal infeasibility
    sol.row_value = lp.A * sol.col_value
    info.primal_infeasibility_abs = norm(sol.row_value - lp.b, ord)
    info.primal_infeasibility_rel = info.primal_infeasibility_abs / (norm(lp.b, ord) + 1)

    # dual infeasibility
    s = lp.cost - lp.A' * sol.row_dual
    idx_lb = (lp.lb .> -Inf)
    idx_ub = (lp.ub .< +Inf)

    s_pos = max.(s, 0)
    s_neg = -min.(s, 0)

    s_pos[.!idx_lb] .= 0
    s_neg[.!idx_ub] .= 0

    sol.col_dual = s_pos - s_neg
    info.dual_infeasibility_abs = norm(sol.col_dual - s, ord)
    info.dual_infeasibility_rel = info.dual_infeasibility_abs / (norm(lp.cost, ord) + 1)

    # objective gap
    info.primal_objective = Int(lp.sense) * lp.cost' * sol.col_value + lp.offset
    info.dual_objective = Int(lp.sense) * (lp.b' * sol.row_dual + lp.lb[idx_lb]' * s_pos[idx_lb] - lp.ub[idx_ub]' * s_neg[idx_ub]) + lp.offset
    info.duality_gap_abs = abs(info.primal_objective - info.dual_objective)
    info.duality_gap_rel = info.duality_gap_abs / (abs(info.primal_objective) + abs(info.dual_objective) + 1)

end

function check_optimality(info::Information; tol::Float64=1e-6)

    is_optimal = false

    if info.primal_infeasibility_rel < tol && info.dual_infeasibility_rel < tol && info.duality_gap_rel < tol
        is_optimal = true
    end

    return is_optimal

end
