function initialize_iteration!(iter::Iteration, nRows::Int64, nCols::Int64, primal_push_method::PRIMAL_PUSH_METHOD, dual_push_method::DUAL_PUSH_METHOD)
    iter.cost_new = zeros(nCols)
    iter.b_new = zeros(nRows)
    iter.col_value_next = zeros(nCols)
    iter.row_dual_next = zeros(nRows)
    iter.x_v = zeros(nCols)
    iter.y_v = zeros(nRows)
    iter.v_x = zeros(nCols)
    iter.v_y = zeros(nRows)
    iter.nrm_v_x = 0.0
    iter.nrm_v_y = 0.0
    iter.nrm_Av_x = 1.0
    iter.nrm_ATv_y = 1.0
    iter.n_basis_last = 0
    iter.n_dual_active_last = 0
    iter.primal_push_method = primal_push_method
    iter.dual_push_method = dual_push_method
end

function ols_initialize_iteration!(crossover::Crossover)
    crossover.iter.x_v = copy(crossover.sol.col_value)
    crossover.iter.y_v = copy(crossover.sol.row_dual)
    crossover.iter.v_x = zeros(crossover.nCols)
    crossover.iter.v_y = zeros(crossover.nRows)
    crossover.iter.nrm_v_x = 0.0
    crossover.iter.nrm_v_y = 0.0
    crossover.iter.nrm_Av_x = 1.0
    crossover.iter.nrm_ATv_y = 1.0
end

function perturb_cost!(crossover::Crossover; delta::Float64=1.0)
    idx_lb = crossover.lp.lb .> -Inf
    idx_ub = crossover.lp.ub .< +Inf

    crossover.iter.cost_new = rand(crossover.nCols) * delta
    crossover.iter.cost_new[(idx_ub).&(.!idx_lb)] *= -1.0

    crossover.iter.cost_new = crossover.iter.cost_new + crossover.lp.cost / (norm(crossover.lp.cost, Inf) + 1)
end

function perturb_rhs!(crossover::Crossover; delta::Float64=1.0)
    crossover.iter.b_new = rand(crossover.nRows) * delta + crossover.lp.b / (norm(crossover.lp.b, Inf) + 1)
end

function detect_and_fix_nonbasic!(crossover::Crossover; gamma::Float64=1.0, tol::Float64=1.0e-8)
    idx_basic = crossover.sol.basis_status .== BASIC
    idx_fixed = (crossover.lp.lb .== crossover.lp.ub)
    # idx_lb = (crossover.sol.col_value .<= crossover.lp.lb .+ tol) .& .!idx_fixed
    # idx_ub = (crossover.sol.col_value .>= crossover.lp.ub .- tol) .& .!idx_fixed
    idx_rel_lb = (crossover.sol.col_value - crossover.lp.lb) .<= max.(gamma * crossover.sol.col_dual, tol)
    idx_rel_ub = (crossover.lp.ub - crossover.sol.col_value) .<= max.(-gamma * crossover.sol.col_dual, tol)
    idx_lb = ((crossover.sol.col_value .<= crossover.lp.lb .+ tol) .| idx_rel_lb) .& .!idx_fixed
    idx_ub = ((crossover.sol.col_value .>= crossover.lp.ub .- tol) .| idx_rel_ub) .& .!idx_fixed

    idx_fixed_lb = idx_fixed .& (crossover.sol.col_dual .>= 0.0)
    idx_fixed_ub = idx_fixed .& (crossover.sol.col_dual .< 0.0)

    crossover.sol.basis_status[idx_basic.&idx_lb] .= NONBASIC_AT_LOWER_BOUND
    crossover.sol.basis_status[idx_basic.&idx_ub] .= NONBASIC_AT_UPPER_BOUND
    crossover.sol.basis_status[idx_basic.&idx_fixed_lb] .= NONBASIC_AT_LOWER_BOUND
    crossover.sol.basis_status[idx_basic.&idx_fixed_ub] .= NONBASIC_AT_UPPER_BOUND

    crossover.iter.n_basis_last = crossover.sol.n_basis
    crossover.sol.n_basis = sum(crossover.sol.basis_status .== BASIC)
end

function improve_primal_feasibility!(crossover::Crossover)
    if crossover.param.verbose
        println("  - Fixing non-basic variables.")
    end

    idx_lb = crossover.sol.basis_status .== NONBASIC_AT_LOWER_BOUND
    idx_ub = crossover.sol.basis_status .== NONBASIC_AT_UPPER_BOUND
    idx_nonbasic = idx_lb .| idx_ub

    crossover.sol.col_value[idx_lb] = crossover.lp.lb[idx_lb]
    crossover.sol.col_value[idx_ub] = crossover.lp.ub[idx_ub]

    if crossover.param.verbose
        println("  - Modifying b for better primal feas.")
    end
    crossover.iter.b_new = crossover.lp.b - crossover.lp.A[:, idx_nonbasic] * crossover.sol.col_value[idx_nonbasic]
end

function recover_primal_feasibility!(crossover::Crossover)
    idx_basic = crossover.sol.basis_status .== BASIC
    model = copt_load_std_lp(
        crossover.lp.cost[idx_basic],
        crossover.lp.A[:, idx_basic],
        crossover.lp.b - crossover.lp.A[:, .!idx_basic] * crossover.sol.col_value[.!idx_basic],
        crossover.lp.lb[idx_basic],
        crossover.lp.ub[idx_basic],
        crossover.lp.sense,
        0.0
    )

    set_optimizer_attribute(model, "presolve", crossover.param.pdlp_presolve)
    set_optimizer_attribute(model, "lpmethod", crossover.param.lp_method)
    set_optimizer_attribute(model, "crossover", 0)
    set_optimizer_attribute(model, "feastol", crossover.param.tol_feas)
    set_optimizer_attribute(model, "dualtol", crossover.param.tol_pdlp)
    set_optimizer_attribute(model, "barprimaltol", crossover.param.tol_pdlp)
    set_optimizer_attribute(model, "bardualtol", crossover.param.tol_pdlp)
    set_optimizer_attribute(model, "bargaptol", crossover.param.tol_pdlp)
    set_optimizer_attribute(model, "pdlptol", crossover.param.tol_pdlp)

    time_limit = crossover.param.max_time - crossover.timer.get_current_runtime()
    if time_limit > 0.0
        if time_limit < Inf
            set_optimizer_attribute(model, "timelimit", time_limit)
        end
        optimize!(model)

        recover_succ = termination_status(model) == MOI.OPTIMAL
    else
        recover_succ = false
        crossover.info.crossover_status = REACHED_TIME_LIMIT
    end

    if recover_succ
        vars = JuMP.all_variables(model)
        crossover.sol.col_value[idx_basic] = value.(vars)
    else
        if crossover.param.verbose
            println("Primal Feasibility Recover failed.")
        end
    end

    return recover_succ
end

"""
max  b^T y + l^T s^+ - u^T s^-
s.t. A_D^T y = c_D
        c - A^T y in Lambda
"""
function dual_feas_model(crossover::Crossover; tol::Float64=1.0e-8)

    # indices
    idx_dual_active = (crossover.sol.dual_status .== ACTIVE)
    idx_fixed = (crossover.lp.lb .== crossover.lp.ub) .& .!idx_dual_active
    idx_lb = (crossover.sol.col_value .< crossover.lp.lb .+ tol) .& .!idx_dual_active .& .!idx_fixed
    idx_ub = (crossover.sol.col_value .> crossover.lp.ub .- tol) .& .!idx_dual_active .& .!idx_fixed

    # construct matrices
    cost = crossover.lp.b - crossover.lp.A[:, idx_lb] * crossover.lp.lb[idx_lb] - crossover.lp.A[:, idx_ub] * crossover.lp.ub[idx_ub]
    offset = crossover.lp.cost[idx_lb]' * crossover.lp.lb[idx_lb] + crossover.lp.cost[idx_ub]' * crossover.lp.ub[idx_ub]
    # ignore fixed vars for obj

    lhs = copy(crossover.lp.cost)
    rhs = copy(crossover.lp.cost)
    lhs[idx_fixed] .= -Inf
    rhs[idx_fixed] .= +Inf
    lhs[idx_lb] .= -Inf
    rhs[idx_ub] .= +Inf

    lb = fill(-Inf, crossover.nRows)
    ub = fill(+Inf, crossover.nRows)

    model = copt_load_lp(
        cost,
        crossover.lp.A',
        lhs,
        rhs,
        lb,
        ub,
        MAXIMIZE,
        offset
    )

    return model
end

function recover_dual_feasibility!(crossover::Crossover)
    model = dual_feas_model(crossover; tol=crossover.param.tol_bound)

    set_optimizer_attribute(model, "presolve", crossover.param.pdlp_presolve)
    set_optimizer_attribute(model, "lpmethod", crossover.param.lp_method)
    set_optimizer_attribute(model, "crossover", 0)
    set_optimizer_attribute(model, "feastol", crossover.param.tol_feas)
    set_optimizer_attribute(model, "dualtol", crossover.param.tol_pdlp)
    set_optimizer_attribute(model, "barprimaltol", crossover.param.tol_pdlp)
    set_optimizer_attribute(model, "bardualtol", crossover.param.tol_pdlp)
    set_optimizer_attribute(model, "bargaptol", crossover.param.tol_pdlp)
    set_optimizer_attribute(model, "pdlptol", crossover.param.tol_pdlp)

    optimize!(model)

    recover_succ = termination_status(model) == MOI.OPTIMAL
    if recover_succ
        vars = JuMP.all_variables(model)
        crossover.sol.row_dual = value.(vars)
    else
        if crossover.param.verbose
            println("Dual Feasibility Recover failed.")
        end
    end

    return recover_succ
end

function crossover_terminate_criteria(crossover::Crossover)
    is_vertex = (crossover.sol.n_basis == crossover.nRows)
    return is_vertex
end

function primal_push_terminate_criteria(crossover::Crossover)
    is_vertex = (crossover.sol.n_basis <= crossover.nRows && crossover.sol.n_basis >= crossover.iter.n_basis_last) || crossover.sol.n_basis == 0

    return is_vertex
end

function dual_push_terminate_criteria(crossover::Crossover)
    is_vertex = crossover.sol.n_dual_active <= crossover.iter.n_dual_active_last

    return is_vertex
end

"""
    min  c^T x
    s.t. lhs <= A_B^T x_B <= rhs
          lb <=       x_B <= ub
"""
function primal_push_general_model(crossover::Crossover)
    nRows = crossover.nRows
    nCols = crossover.nCols - crossover.nRows

    idx_basic = crossover.sol.basis_status .== BASIC
    col_basic = idx_basic[1:nCols]
    row_basic = idx_basic[nCols+1:end]
    x = crossover.sol.col_value[1:nCols]
    z = crossover.sol.col_value[nCols+1:end]

    if crossover.is_rhs_shifted
        z .-= crossover.param.rhs_shift
    end

    cost = crossover.iter.cost_new[1:nCols][col_basic] + crossover.glp.A[row_basic, col_basic]' * crossover.iter.cost_new[nCols+1:end][row_basic]

    A = crossover.glp.A[:, col_basic]
    lhs = copy(crossover.glp.lhs)
    rhs = copy(crossover.glp.rhs)
    lhs[.!row_basic] = z[.!row_basic]
    rhs[.!row_basic] = z[.!row_basic]
    Ax_nonbasic = crossover.glp.A[:, .!col_basic] * x[.!col_basic]
    lhs = lhs - Ax_nonbasic
    rhs = rhs - Ax_nonbasic

    lb = crossover.glp.lb[col_basic]
    ub = crossover.glp.ub[col_basic]

    offset = crossover.iter.cost_new[nCols+1:end][row_basic]' * Ax_nonbasic[row_basic]

    model = copt_load_lp(
        cost, A, lhs, rhs, lb, ub, MINIMIZE, offset
    )
    set_optimizer_attribute(model, "Logging", 0)

    return model
end

function recover_primal_push_from_general_model(crossover::Crossover, model::GenericModel{Float64}, idx_basic::BitVector)
    vars = JuMP.all_variables(model)
    x_basic = value.(vars)
    nCols = crossover.nCols - crossover.nRows
    col_basic = idx_basic[1:nCols]
    row_basic = idx_basic[nCols+1:end]
    x = crossover.sol.col_value[1:nCols]
    x[col_basic] = x_basic
    Ax = crossover.glp.A * x

    x_and_z_basic = nothing
    if crossover.is_rhs_shifted
        x_and_z_basic = [x_basic; Ax[row_basic] .+ crossover.param.rhs_shift]
    else
        x_and_z_basic = [x_basic; Ax[row_basic]]
    end

    return x_and_z_basic
end

function primal_push_pdlp!(crossover::Crossover; delta::Float64=1.0)
    # PDLP
    idx_basic = crossover.sol.basis_status .== BASIC
    perturb_cost!(crossover; delta=delta)
    model = crossover.is_general_lp && crossover.param.primal_push_general ? primal_push_general_model(crossover) : copt_load_std_lp(
        crossover.iter.cost_new[idx_basic],
        crossover.lp.A[:, idx_basic],
        crossover.iter.b_new,
        crossover.lp.lb[idx_basic],
        crossover.lp.ub[idx_basic],
        MINIMIZE,
        0.0
    )
    set_optimizer_attribute(model, "presolve", crossover.param.pdlp_presolve)
    set_optimizer_attribute(model, "lpmethod", crossover.param.lp_method)
    set_optimizer_attribute(model, "crossover", 0)
    set_optimizer_attribute(model, "feastol", crossover.param.tol_feas)
    set_optimizer_attribute(model, "dualtol", crossover.param.tol_pdlp)
    set_optimizer_attribute(model, "barprimaltol", crossover.param.tol_pdlp)
    set_optimizer_attribute(model, "bardualtol", crossover.param.tol_pdlp)
    set_optimizer_attribute(model, "bargaptol", crossover.param.tol_pdlp)
    set_optimizer_attribute(model, "pdlptol", crossover.param.tol_pdlp)

    time_limit = crossover.param.max_time - crossover.timer.get_current_runtime()
    if time_limit > 0.0
        if time_limit < Inf
            set_optimizer_attribute(model, "timelimit", time_limit)
        end
        optimize!(model)

        primal_push_succ = termination_status(model) == MOI.OPTIMAL
    else
        crossover.info.crossover_status = REACHED_TIME_LIMIT
        primal_push_succ = false
    end

    if primal_push_succ
        if crossover.is_general_lp && crossover.param.primal_push_general
            crossover.sol.col_value[idx_basic] = recover_primal_push_from_general_model(crossover, model, idx_basic)
        else
            vars = JuMP.all_variables(model)
            crossover.sol.col_value[idx_basic] = value.(vars)
        end
    else
        crossover.info.crossover_status = PRIMAL_FAILED
        if crossover.param.verbose
            println("Primal Push failed.")
        end
    end

end

function cal_primal_spiral_ray(crossover::Crossover, cost::Vector{Float64}, A::SparseMatrixCSC{Float64,Int64}, b::Vector{Float64}, eta::Float64, w::Float64, basis::BitVector; tol::Float64=1.0e-8)
    y_v = ols(A[:, basis]', cost[basis], crossover, crossover.sol.row_dual)
    v_x = -eta / w * (cost[basis] - A[:, basis]' * y_v)

    v_x[abs.(v_x).<=tol] .= 0.0

    return v_x, y_v
end

function cal_dual_spiral_ray(crossover::Crossover, cost::Vector{Float64}, A::SparseMatrixCSC{Float64,Int64}, b::Vector{Float64}, eta::Float64, w::Float64, basis::BitVector; tol::Float64=1.0e-8)
    x_v = ols(A[:, basis], b, crossover, crossover.sol.col_value[basis])
    v_y = eta * w * (b - A[:, basis] * x_v)

    v_y[abs.(v_y).<=tol] .= 0.0

    return v_y, x_v
end

function primal_ratio_test(crossover::Crossover)

    idx_neg = crossover.iter.v_x .< -crossover.param.epsilon_zero
    if any(idx_neg)
        theta_arr_neg = (crossover.lp.lb[idx_neg] .- crossover.iter.x_v[idx_neg]) ./ crossover.iter.v_x[idx_neg]
        idx_theta_neg = argmin(theta_arr_neg)
        theta_neg = theta_arr_neg[idx_theta_neg]
    else
        idx_theta_neg = -1
        theta_neg = +Inf
    end

    idx_pos = crossover.iter.v_x .> crossover.param.epsilon_zero
    if any(idx_pos)
        theta_arr_pos = (crossover.lp.ub[idx_pos] .- crossover.iter.x_v[idx_pos]) ./ crossover.iter.v_x[idx_pos]
        idx_theta_pos = argmin(theta_arr_pos)
        theta_pos = theta_arr_pos[idx_theta_pos]
    else
        idx_theta_pos = -1
        theta_pos = +Inf
    end

    if theta_neg < theta_pos
        theta = theta_neg
        idx_theta = idx_theta_neg
    else
        theta = theta_pos
        idx_theta = idx_theta_pos
    end

    return theta, idx_theta
end

function move_x_to_vertex!(crossover::Crossover)
    idx_basic = crossover.sol.basis_status .== BASIC

    crossover.iter.v_x[idx_basic], _ = cal_primal_spiral_ray(crossover, crossover.iter.cost_new, crossover.lp.A, crossover.iter.b_new, 1.0, 1.0, idx_basic; tol=crossover.param.epsilon_zero)

    crossover.iter.nrm_v_x = norm(crossover.iter.v_x, 2)

    if crossover.iter.nrm_v_x > crossover.param.epsilon_zero
        theta, idx_theta = primal_ratio_test(crossover)

        if theta < +Inf
            crossover.sol.col_value[idx_basic] = crossover.sol.col_value[idx_basic] + theta * crossover.iter.v_x[idx_basic]
        else
            crossover.iter.v_x *= -1
            theta, idx_theta = primal_ratio_test(crossover)
            crossover.sol.col_value[idx_basic] = crossover.sol.col_value[idx_basic] + theta * crossover.iter.v_x[idx_basic]
        end
    else
        theta = 0.0
        idx_theta = -1
    end

    return theta, idx_theta
end

function primal_push_ols!(crossover::Crossover; delta::Float64=1.0)
    # OLS
    perturb_cost!(crossover; delta=delta)
    ols_initialize_iteration!(crossover)

    move_x_to_vertex!(crossover)
end

function primal_push!(crossover::Crossover)
    if crossover.param.verbose
        println("Primal push")
    end

    detect_and_fix_nonbasic!(crossover, gamma=crossover.param.gamma, tol=crossover.param.tol_bound)
    improve_primal_feasibility!(crossover)

    if crossover.param.verbose
        println("Primal Push: $(crossover.sol.n_basis) remaining.")
    end

    primal_push_count = 0
    while !primal_push_terminate_criteria(crossover) || primal_push_count <= 1
        if crossover.iter.primal_push_method == P_HYBRID
            primal_push_pdlp!(crossover, delta=crossover.param.delta)
            crossover.iter.primal_push_method = P_OLS
        elseif crossover.iter.primal_push_method == P_PDLP
            primal_push_pdlp!(crossover, delta=crossover.param.delta)
        elseif crossover.iter.primal_push_method == P_OLS
            primal_push_ols!(crossover)
        end

        primal_push_count += 1

        if crossover.param.verbose && crossover.param.verbose_level >= 3
            cal_optimality!(crossover.info, crossover.lp, crossover.sol; nrm_type=crossover.param.nrm_type)
            print_optimality(crossover)
        end

        if norm(crossover.lp.A * crossover.sol.col_value - crossover.lp.b, 2) > crossover.param.tol_recover
            if crossover.param.verbose && crossover.param.verbose_level >= 2
                cal_optimality!(crossover.info, crossover.lp, crossover.sol; nrm_type=crossover.param.nrm_type)
                print_optimality(crossover)
            end

            if crossover.param.verbose
                println("Primal Push - Feasibility Recover.")
            end

            recover_succ = recover_primal_feasibility!(crossover)
            detect_and_fix_nonbasic!(crossover, gamma=crossover.param.gamma, tol=crossover.param.tol_bound)
            if !recover_succ
                break
            end
        else
            detect_and_fix_nonbasic!(crossover, gamma=crossover.param.gamma, tol=crossover.param.tol_bound)
        end

        improve_primal_feasibility!(crossover)

        if crossover.param.verbose
            println("Primal Push: $(crossover.sol.n_basis) remaining.")
        end

        # check time limit
        if crossover.timer.get_current_runtime() > crossover.param.max_time
            crossover.info.crossover_status = REACHED_TIME_LIMIT
            break
        end
    end

    if crossover.param.verbose && crossover.param.verbose_level >= 2
        cal_optimality!(crossover.info, crossover.lp, crossover.sol; nrm_type=crossover.param.nrm_type)
        print_optimality(crossover)
    end

    if crossover.param.verbose && crossover.param.verbose_level >= 3
        s = crossover.lp.cost - crossover.lp.A' * crossover.sol.row_dual
        println("nonbasic_lb_min ", minimum(s[crossover.sol.basis_status.==NONBASIC_AT_LOWER_BOUND]))
        println("nonbasic_ub_max ", maximum(s[crossover.sol.basis_status.==NONBASIC_AT_UPPER_BOUND]))
    end

    if crossover.param.verbose
        print_line()
    end

    # primal push succeeded if crossover_status unchanged
    if crossover.info.crossover_status == UNSTARTED
        crossover.info.crossover_status = PRIMAL_SUCCEEDED
    end
end

function cal_dual_active!(crossover::Crossover; gamma::Float64=1.0, tol::Float64=1.0e-8)
    s = crossover.lp.cost - crossover.lp.A' * crossover.sol.row_dual

    # primal nonbasic update for fixed nonbasic
    idx_nonbasic = crossover.sol.basis_status .!= BASIC
    crossover.sol.basis_status[idx_nonbasic.&(s.>=0.0)] .= NONBASIC_AT_LOWER_BOUND
    crossover.sol.basis_status[idx_nonbasic.&(s.<0.0)] .= NONBASIC_AT_UPPER_BOUND

    # dual active update

    has_lb = (crossover.lp.lb .> -Inf)
    has_ub = (crossover.lp.ub .< +Inf)

    idx_bd = has_lb .& has_ub  # lb <= x <= ub
    idx_lb = has_lb .& .!idx_bd  # lb < x < inf
    idx_ub = has_ub .& .!idx_bd  # -inf < x < ub
    idx_free = .!has_lb .& .!has_ub  # -inf < x < inf
    # idx_fixed = (self.lb .== self.ub)  # lb = x = ub

    idx_basic = crossover.sol.basis_status .== BASIC
    idx_dual_active = crossover.sol.dual_status .== ACTIVE

    dual_active_lb = (s .<= tol) .& idx_lb
    dual_active_ub = (s .>= -tol) .& idx_ub
    dual_active_bd_free = (abs.(s) .<= tol) .& (idx_bd .| idx_free)
    # dual_active_rel_lb = s .<= gamma * max.(crossover.sol.col_value - crossover.lp.lb, tol)
    # dual_active_rel_ub = s .>= gamma * min.(crossover.sol.col_value - crossover.lp.ub, -tol)
    # dual_active_rel_bd_free = abs.(s) .<= gamma * max.(abs.(crossover.sol.col_value), tol)
    # dual_active_lb = ((s .<= tol) .| dual_active_rel_lb) .& idx_lb
    # dual_active_ub = ((s .>= -tol) .| dual_active_rel_ub) .& idx_ub
    # dual_active_bd_free = ((abs.(s) .<= tol) .| dual_active_rel_bd_free) .& (idx_bd .| idx_free)

    crossover.sol.dual_status[dual_active_lb.|dual_active_ub.|dual_active_bd_free.|idx_basic.|idx_dual_active] .= ACTIVE

    crossover.iter.n_dual_active_last = crossover.sol.n_dual_active
    crossover.sol.n_dual_active = sum(crossover.sol.dual_status .== ACTIVE)
end

"""
    min  1^T s^+ + 1^T s^-
    s.t.   A_D^T y = c_D
         c - A^T y in Lambda
"""
function dual_push_model(crossover::Crossover)
    # indices
    idx_dual_active = (crossover.sol.dual_status .== ACTIVE)
    idx_fixed = (crossover.lp.lb .== crossover.lp.ub) .& .!idx_dual_active
    idx_lb = (crossover.sol.basis_status .== NONBASIC_AT_LOWER_BOUND) .& .!idx_dual_active .& .!idx_fixed
    idx_ub = (crossover.sol.basis_status .== NONBASIC_AT_UPPER_BOUND) .& .!idx_dual_active .& .!idx_fixed

    # construct matrices
    weight_lb = ones(sum(idx_lb))
    weight_ub = ones(sum(idx_ub))
    # lp.A = [glp.A -I]
    cost = -crossover.lp.A[:, idx_lb] * weight_lb + crossover.lp.A[:, idx_ub] * weight_ub
    offset = crossover.lp.cost[idx_lb]' * weight_lb - crossover.lp.cost[idx_ub]' * weight_ub
    # ignore fixed vars for obj

    lhs = copy(crossover.lp.cost)
    rhs = copy(crossover.lp.cost)
    lhs[idx_fixed] .= -Inf
    rhs[idx_fixed] .= +Inf
    lhs[idx_lb] .= -Inf
    rhs[idx_ub] .= +Inf

    lb = fill(-Inf, crossover.nRows)
    ub = fill(+Inf, crossover.nRows)

    model = copt_load_lp(
        cost,
        crossover.lp.A',
        lhs,
        rhs,
        lb,
        ub,
        MINIMIZE,
        offset
    )

    return model
end

function dual_push_general_model(crossover)

    # model = dual_push_model(crossover)

    nRows = crossover.nRows
    nCols = crossover.nCols - crossover.nRows

    # indices
    idx_dual_active = (crossover.sol.dual_status .== ACTIVE)
    idx_fixed = (crossover.lp.lb .== crossover.lp.ub) .& .!idx_dual_active
    idx_lb = (crossover.sol.basis_status .== NONBASIC_AT_LOWER_BOUND) .& .!idx_dual_active .& .!idx_fixed
    idx_ub = (crossover.sol.basis_status .== NONBASIC_AT_UPPER_BOUND) .& .!idx_dual_active .& .!idx_fixed

    col_fixed = idx_fixed[1:nCols]
    row_fixed = idx_fixed[nCols+1:end]
    col_lb = idx_lb[1:nCols]
    col_ub = idx_ub[1:nCols]
    row_lhs = idx_lb[nCols+1:end]
    row_rhs = idx_ub[nCols+1:end]

    # construct matrices
    weight_lb = ones(sum(idx_lb))
    weight_ub = ones(sum(idx_ub))
    cost = -crossover.lp.A[:, idx_lb] * weight_lb + crossover.lp.A[:, idx_ub] * weight_ub
    offset = crossover.lp.cost[idx_lb]' * weight_lb - crossover.lp.cost[idx_ub]' * weight_ub
    # ignore fixed vars for obj

    lhs = copy(crossover.glp.cost)
    rhs = copy(crossover.glp.cost)
    lhs[col_fixed] .= -Inf
    rhs[col_fixed] .= +Inf
    lhs[col_lb] .= -Inf
    rhs[col_ub] .= +Inf

    lb = zeros(nRows)
    ub = zeros(nRows)
    lb[row_fixed] .= -Inf
    ub[row_fixed] .= +Inf
    ub[row_lhs] .= +Inf
    lb[row_rhs] .= -Inf

    model = copt_load_lp(
        cost,
        crossover.glp.A',
        lhs,
        rhs,
        lb,
        ub,
        MINIMIZE,
        offset
    )
    set_optimizer_attribute(model, "Logging", 0)
    return model
end

function recover_dual_push_from_general_model(crossover::Crossover, model::GenericModel{Float64})
    vars = JuMP.all_variables(model)
    y = value.(vars)

    return y
end

function dual_push_pdlp!(crossover::Crossover)
    # PDLP
    model = crossover.is_general_lp && crossover.param.dual_push_general ? dual_push_general_model(crossover) : dual_push_model(crossover)

    set_optimizer_attribute(model, "presolve", crossover.param.pdlp_presolve)
    set_optimizer_attribute(model, "lpmethod", crossover.param.lp_method)
    set_optimizer_attribute(model, "crossover", 0)
    set_optimizer_attribute(model, "feastol", crossover.param.tol_feas)
    set_optimizer_attribute(model, "dualtol", crossover.param.tol_pdlp)
    set_optimizer_attribute(model, "barprimaltol", crossover.param.tol_pdlp)
    set_optimizer_attribute(model, "bardualtol", crossover.param.tol_pdlp)
    set_optimizer_attribute(model, "bargaptol", crossover.param.tol_pdlp)
    set_optimizer_attribute(model, "pdlptol", crossover.param.tol_pdlp)

    time_limit = crossover.param.max_time - crossover.timer.get_current_runtime()
    if time_limit > 0.0
        if time_limit < Inf
            set_optimizer_attribute(model, "timelimit", time_limit)
        end
        optimize!(model)

        dual_push_succ = termination_status(model) == MOI.OPTIMAL
    else
        crossover.info.crossover_status = REACHED_TIME_LIMIT
        dual_push_succ = false
    end

    if dual_push_succ
        if crossover.is_general_lp && crossover.param.primal_push_general
            crossover.sol.row_dual = recover_dual_push_from_general_model(crossover, model)
        else
            vars = JuMP.all_variables(model)
            crossover.sol.row_dual = value.(vars)
        end
    else
        crossover.info.crossover_status = DUAL_FAILED
        if crossover.param.verbose
            println("Dual Push failed.")
        end
    end
end

function dual_ratio_test(crossover::Crossover, basis::BitVector)
    ATy_v = crossover.lp.A[:, .!basis]' * crossover.iter.y_v
    ATv_y = crossover.lp.A[:, .!basis]' * crossover.iter.v_y
    s = crossover.lp.cost[.!basis] - ATy_v

    idx_lb = crossover.sol.basis_status[.!basis] .== NONBASIC_AT_LOWER_BOUND
    idx_ub = crossover.sol.basis_status[.!basis] .== NONBASIC_AT_UPPER_BOUND

    idx_neg = (ATv_y .< -crossover.param.epsilon_zero) .& idx_ub
    if any(idx_neg)
        theta_arr_neg = s[idx_neg] ./ ATv_y[idx_neg]
        idx_theta_neg = argmin(theta_arr_neg)
        theta_neg = theta_arr_neg[idx_theta_neg]
    else
        idx_theta_neg = -1
        theta_neg = +Inf
    end

    idx_pos = (ATv_y .> crossover.param.epsilon_zero) .& idx_lb
    if any(idx_pos)
        theta_arr_pos = s[idx_pos] ./ ATv_y[idx_pos]
        idx_theta_pos = argmin(theta_arr_pos)
        theta_pos = theta_arr_pos[idx_theta_pos]
    else
        idx_theta_pos = -1
        theta_pos = +Inf
    end

    if theta_neg < theta_pos
        theta = theta_neg
        idx_theta = idx_theta_neg
    else
        theta = theta_pos
        idx_theta = idx_theta_pos
    end

    if crossover.param.verbose && crossover.param.verbose_level >= 3
        println("  θ_pos = $theta_pos, θ_neg = $theta_neg")
    end

    return theta, idx_theta
end

function move_y_to_vertex!(crossover::Crossover)
    idx_dual_active = crossover.sol.dual_status .== ACTIVE

    crossover.iter.v_y, _ = cal_dual_spiral_ray(crossover, crossover.iter.cost_new, crossover.lp.A, crossover.iter.b_new, 1.0, 1.0, idx_dual_active; tol=crossover.param.epsilon_zero)

    crossover.iter.nrm_v_y = norm(crossover.iter.v_y, 2)

    if crossover.iter.nrm_v_y > crossover.param.epsilon_zero
        theta, idx_theta = dual_ratio_test(crossover, idx_dual_active)

        if theta < +Inf
            crossover.sol.row_dual = crossover.sol.row_dual + theta * crossover.iter.v_y
        else
            crossover.iter.v_y *= -1
            theta, idx_theta = dual_ratio_test(crossover, idx_dual_active)
            crossover.sol.row_dual = crossover.sol.row_dual + theta * crossover.iter.v_y
        end
    else
        theta = 0.0
        idx_theta = -1
    end

    if crossover.param.verbose && crossover.param.verbose_level >= 2
        println("  θ = $theta")
    end

    return theta, idx_theta
end

function dual_push_ols!(crossover::Crossover; delta::Float64=1.0)
    # OLS
    perturb_rhs!(crossover; delta=delta)
    ols_initialize_iteration!(crossover)

    move_y_to_vertex!(crossover)
end

function dual_push!(crossover::Crossover)
    if crossover.param.verbose
        println("Dual push")
    end

    cal_dual_active!(crossover, tol=crossover.param.tol_bound)
    if crossover.param.verbose
        println("Dual Push: # dual active $(crossover.sol.n_dual_active).")
    end

    feas_recover = false
    dual_push_count = 0
    while !dual_push_terminate_criteria(crossover) || feas_recover || dual_push_count <= 1

        if crossover.iter.dual_push_method == D_HYBRID
            dual_push_pdlp!(crossover)
            crossover.iter.dual_push_method = D_OLS
        elseif crossover.iter.dual_push_method == D_PDLP
            dual_push_pdlp!(crossover)
        elseif crossover.iter.dual_push_method == D_OLS
            dual_push_ols!(crossover, delta=crossover.param.delta)
        end

        dual_push_count += 1

        cal_dual_active!(crossover, gamma=crossover.param.gamma, tol=crossover.param.tol_bound)

        if crossover.param.verbose && crossover.param.verbose_level >= 3
            cal_optimality!(crossover.info, crossover.lp, crossover.sol; nrm_type=crossover.param.nrm_type)
            print_optimality(crossover)
        end

        idx_dual_active = crossover.sol.dual_status .== ACTIVE
        if norm(crossover.lp.cost[idx_dual_active] - crossover.lp.A[:, idx_dual_active]' * crossover.sol.row_dual, 2) > crossover.param.tol_recover
            if crossover.param.verbose && crossover.param.verbose_level >= 2
                cal_optimality!(crossover.info, crossover.lp, crossover.sol; nrm_type=crossover.param.nrm_type)
                print_optimality(crossover)
            end

            if crossover.param.verbose
                println("Dual Push - Feasibility Recover.")
            end

            feas_recover = true

            recover_succ = recover_dual_feasibility!(crossover)
            cal_dual_active!(crossover, gamma=crossover.param.gamma, tol=crossover.param.tol_bound)

            if !recover_succ
                break
            end
        else
            feas_recover = false
        end

        if crossover.param.verbose
            println("Dual Push: # dual active $(crossover.sol.n_dual_active).")
        end

        if crossover.param.verbose && crossover.param.verbose_level >= 3
            s = crossover.lp.cost - crossover.lp.A' * crossover.sol.row_dual
            println("nonbasic_lb_min ", minimum(s[crossover.sol.basis_status.==NONBASIC_AT_LOWER_BOUND]))
            println("nonbasic_ub_max ", maximum(s[crossover.sol.basis_status.==NONBASIC_AT_UPPER_BOUND]))
        end

        # check time limit
        if crossover.timer.get_current_runtime() > crossover.param.max_time
            crossover.info.crossover_status = REACHED_TIME_LIMIT
            break
        end
    end

    # dual push succeeded if crossover_status unchanged
    if crossover.info.crossover_status == PRIMAL_SUCCEEDED
        crossover.info.crossover_status = DUAL_SUCCEEDED
    end

    if crossover.param.verbose && crossover.param.verbose_level >= 2
        cal_optimality!(crossover.info, crossover.lp, crossover.sol; nrm_type=crossover.param.nrm_type)
        print_optimality(crossover)
    end
end

function linear_independence_check!(crossover::Crossover)
    if crossover.param.verbose
        println("Linear independence check")
    end

    # record basic indices
    idx_basic = crossover.sol.basis_status .== BASIC

    # original row status
    row_status = fill(false, crossover.nRows)

    # LU factorization of A_basic
    F_basic = lu(crossover.lp.A[:, idx_basic]; check=false)
    is_success_lu_basic = issuccess(F_basic)
    nRows_basic, nCols_basic = size(F_basic.U)
    if is_success_lu_basic && nRows_basic == nCols_basic && nRows_basic == crossover.sol.n_basis
        row_status[F_basic.p[1:nRows_basic]] .= true

        if nRows_basic < crossover.nRows
            # record candidate indices
            idx_dual_active = crossover.sol.dual_status .== ACTIVE
            idx_candidate = idx_dual_active .& .!idx_basic

            # LU factorization of A_candidate
            row_basic = F_basic.p[1:nRows_basic]
            remain_rows = F_basic.p[nRows_basic+1:crossover.nRows]
            # A_candidate = F_basic.Rs[remain_rows] .* crossover.lp.A[remain_rows, idx_candidate] - F_basic.L[nRows_basic+1:crossover.nRows, :] * sparse(F_basic.L[1:nRows_basic, :] \ (F_basic.Rs[row_basic] .* crossover.lp.A[row_basic, idx_candidate]))
            A_candidate = F_basic.Rs[remain_rows] .* crossover.lp.A[remain_rows, idx_candidate] - (F_basic.L[1:nRows_basic, :]' \ F_basic.L[nRows_basic+1:crossover.nRows, :]')' * (F_basic.Rs[row_basic] .* crossover.lp.A[row_basic, idx_candidate])

            F_candidate = lu(A_candidate; check=false)
            is_success_lu_candidate = issuccess(F_candidate)
            nRows_candidate, nCols_candidate = size(F_candidate.U)
            is_lu_lin_ind_succ = false
            if is_success_lu_candidate && nRows_candidate == crossover.nRows - nRows_basic

                # get remaining candidate basic columns
                candidate_col_no = findall(idx_candidate)
                F_candidate_q = F_candidate.q[1:crossover.nRows-nRows_basic]

                # deal with numerical issue, e.g., 1e-19 on the diagonal
                idx_bool_diag_zero = abs.(diag(F_candidate.U)) .<= eps()
                idx_diag_zero = findall(idx_bool_diag_zero)
                n_diag_zero = length(idx_diag_zero)

                is_recover_succ = true

                if n_diag_zero > 0
                    U = F_candidate.U
                    for i in 1:n_diag_zero
                        # eliminate nonzeros
                        row_idx = idx_diag_zero[i]
                        for j = row_idx+1:nRows_candidate
                            if abs(U[row_idx, j]) > eps()
                                if idx_bool_diag_zero[j]
                                    # switch row j and row row_idx
                                    U[j, :], U[row_idx, :] = U[row_idx, :], U[j, :]
                                    idx_bool_diag_zero[j] = false
                                    F_candidate.p[j], F_candidate.p[row_idx] = F_candidate.p[row_idx], F_candidate.p[j]
                                else
                                    # subtract row j from row row_idx
                                    U[row_idx, :] .-= U[row_idx, j] / U[j, j] * U[j, :]
                                end
                            end
                        end
                    end

                    idx_diag_zero = findall(idx_bool_diag_zero)
                    U = U[idx_diag_zero, nRows_candidate+1:nCols_candidate]
                    n_diag_zero = length(idx_diag_zero)
                    for i in 1:n_diag_zero
                        # pivot
                        idx_col_max = argmax(abs.(U[i, :]))
                        col_max = U[i, idx_col_max]
                        if abs(col_max) > eps()
                            F_candidate_q[idx_diag_zero[i]] = F_candidate.q[nRows_candidate+idx_col_max]
                            for j = i+1:n_diag_zero
                                U[j, :] .-= U[j, idx_col_max] / col_max * U[i, :]
                            end
                        else
                            is_recover_succ = false
                            break
                        end
                    end
                end

                if is_recover_succ
                    basic_in_candidate_col_no = candidate_col_no[F_candidate_q]

                    crossover.sol.basis_status[basic_in_candidate_col_no] .= BASIC
                    row_status[remain_rows[F_candidate.p]] .= true

                    crossover.sol.n_basis = sum(crossover.sol.basis_status .== BASIC)
                    is_lu_lin_ind_succ = true
                end
            end
            if !is_lu_lin_ind_succ
                # record LU failure
                # crossover.info.crossover_status = LINEAR_IND_FAILED
                if crossover.param.verbose
                    if !is_success_lu_candidate
                        println("LU factorization of A_candidate failed.")
                    else
                        println("Incorrect LU factorization of A_candidate. Maybe not full row rank.")
                    end
                end

                # try QR factorization, which is more stable
                if crossover.param.verbose
                    println("Try QR factorization.")
                end
                F_qr_candidate = qr(A_candidate)
                nRows_qr_candidate, nCols_qr_candidate = size(F_qr_candidate.R)

                if all(abs.(diag(F_qr_candidate.R)) .> eps())
                    candidate_col_no = findall(idx_candidate)
                    basic_in_candidate_col_no = candidate_col_no[F_qr_candidate.pcol[1:crossover.nRows-nRows_basic]]
                    crossover.sol.basis_status[basic_in_candidate_col_no] .= BASIC
                    row_status[remain_rows[F_qr_candidate.prow]] .= true

                    crossover.sol.n_basis = sum(crossover.sol.basis_status .== BASIC)
                else
                    crossover.info.crossover_status = LINEAR_IND_FAILED
                    if crossover.param.verbose
                        println("QR factorization of A_candidate failed. Maybe not full row rank.")
                    end
                end
            end
        end  # factorize A_candidate
    else  # LU factorization of A_basic failed
        crossover.info.crossover_status = LINEAR_IND_FAILED
        if crossover.param.verbose
            if !is_success_lu_basic
                println("LU factorization of A_basic failed.")
            else
                println("Incorrect LU factorization of A_basic. Maybe not full column rank.")
            end
        end
    end

    # check whether full rank
    rank_basic = sum(row_status)
    if rank_basic == crossover.nRows
        crossover.sol.is_vertex = true
        if crossover.param.verbose
            println("Linear independence check passed.")
        end
    else
        if crossover.param.verbose
            println("Linear independence check failed. Rank of basis is $rank_basic.")
        end
    end

    # linear independence check succeeded if crossover_status unchanged
    if (crossover.info.crossover_status == DUAL_SUCCEEDED || crossover.info.crossover_status == DUAL_SKIPPED) && crossover.sol.is_vertex
        crossover.info.crossover_status = LINEAR_IND_SUCCEEDED
    end
end

function dual_push_and_linear_independence_check!(crossover::Crossover)
    if crossover.sol.n_basis < crossover.nRows
        dual_push!(crossover)
    else
        if crossover.param.verbose
            println("Dual push")
            println("Dual push skipped.")
        end
        crossover.info.crossover_status = DUAL_SKIPPED
    end

    if crossover.param.verbose
        print_line()
    end

    if crossover.info.crossover_status == DUAL_SUCCEEDED || crossover.info.crossover_status == DUAL_SKIPPED
        linear_independence_check!(crossover)

        if crossover_terminate_criteria(crossover)
            crossover.info.crossover_status = SUCCEEDED
        end

        if crossover.param.verbose
            print_line()
        end
    end
end

function crossover_main!(crossover::Crossover)

    # primal push
    primal_push!(crossover)

    if crossover.info.crossover_status == PRIMAL_SUCCEEDED
        # dual push and linear independence check
        dual_push_and_linear_independence_check!(crossover)
    end

end

function cleanup!(crossover::Crossover)
    if crossover.param.verbose
        println("Cleaning up.")
    end

    cleanup_succ = false

    if crossover.sol.n_basis == crossover.nRows
        idx_basic = crossover.sol.basis_status .== BASIC
        crossover.sol.col_value[idx_basic] = crossover.lp.A[:, idx_basic] \ (crossover.lp.b - crossover.lp.A[:, .!idx_basic] * crossover.sol.col_value[.!idx_basic])
        crossover.sol.row_dual = crossover.lp.A[:, idx_basic]' \ crossover.lp.cost[idx_basic]
        cleanup_succ = true
    end

    return cleanup_succ
end