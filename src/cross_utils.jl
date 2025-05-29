function project_to_bounds(x::Vector{Float64}, lb::Vector{Float64}, ub::Vector{Float64})

    return min.(max.(x, lb), ub)

end

function ols(A::AbstractMatrix{Float64}, b::Vector{Float64}, crossover::Crossover, x0::Vector{Float64})
    if crossover.param.ols_method == OLS_AUTO
        if minimum(size(A)) > 5000
            ols_method = OLS_PCG
        elseif minimum(size(A)) >= 1000
            ols_method = OLS_LSMR
        else
            ols_method = OLS_QR
        end
    else
        ols_method = crossover.param.ols_method
    end

    if ols_method == OLS_QR
        # F = SuiteSparse.SPQR.qr(A[:, basis])
        F = qr(A)
        x = F \ b
        # x = A \ b
        return x
    elseif ols_method == OLS_LSMR
        maxiter = max(crossover.param.max_iter_ols, 10 * maximum(size(A)))
        return lsmr(A, b; atol=crossover.param.tol_ols, btol=crossover.param.tol_ols, maxiter=maxiter)
    elseif ols_method == OLS_PCG
        maxiter = max(crossover.param.max_iter_ols, 10 * maximum(size(A)))
        return ols_pcg(A, b, x0, crossover; abs_tol=1e-10, rel_tol=crossover.param.tol_ols, max_iter=maxiter, mu=1e-12)
    end
end

"""
(A'A + μI) x = A'b
"""
function ols_pcg(A::AbstractMatrix, b::Vector{Float64}, x0::Vector{Float64}, crossover::Crossover; abs_tol=1e-10, rel_tol::Float64=1e-8, max_iter::Int=100000, mu::Float64=1e-12)

    nRows, nCols = size(A)

    P = 1 ./ (vec(sqrt.(sum(abs2, A, dims=1))) .+ crossover.param.epsilon_zero)

    ATb = A' * b
    one_plus_nrm_ATb = 1 + norm(ATb)

    x = copy(x0)
    r = ATb - A' * (A * x) - mu * x
    z = P .* r
    p = z
    ρ = r' * z

    for i = 1:max_iter
        # pcg log
        if crossover.param.verbose && crossover.param.verbose_level >= 4
            println("  - PCG iteration: $i, sqrt(ρ): $(sqrt(ρ)), nrm(r): $(norm(r))")
        end

        # check convergence
        if (sqrt(ρ) < rel_tol * one_plus_nrm_ATb || norm(r) < rel_tol * one_plus_nrm_ATb) && norm(r, Inf) < abs_tol
            break
        end

        # pcg
        w = A' * (A * p) + mu * p
        α = ρ / (w' * p)
        x = x + α * p
        r = r - α * w
        z = P .* r
        ρ_new = r' * z
        p = z + (ρ_new / ρ) * p
        ρ = ρ_new
    end

    return x
end

function copt_load_std_lp(cost::Vector{Float64}, A::AbstractMatrix{Float64}, b::Vector{Float64}, lb::Vector{Float64}, ub::Vector{Float64}, sense::SENSE, offset::Float64)
    # initialization
    model = Model(COPT.Optimizer)
    set_optimizer_attribute(model, "Logging", 0)
    # modeling
    nRows, nCols = size(A)
    @variable(model, lb[i] <= x[i=1:nCols] <= ub[i])
    @constraint(model, A * x .== b)
    if sense == MINIMIZE
        @objective(model, Min, Int(sense) * cost' * x + offset)
    else
        @objective(model, Max, Int(sense) * cost' * x + offset)
    end

    return model
end

function copt_load_lp(cost::Vector{Float64}, A::AbstractMatrix{Float64}, lhs::Vector{Float64}, rhs::Vector{Float64}, lb::Vector{Float64}, ub::Vector{Float64}, sense::SENSE, offset::Float64)
    # initialization
    model = Model(COPT.Optimizer)

    # modeling
    nRows, nCols = size(A)
    @variable(model, lb[i] <= x[i=1:nCols] <= ub[i])
    @constraint(model, lhs .<= A * x .<= rhs)
    if sense == MINIMIZE
        @objective(model, Min, Int(sense) * cost' * x + offset)
    else
        @objective(model, Max, Int(sense) * cost' * x + offset)
    end

    return model
end

# gsol to sol
function reformulate_full_row_rank_sol(gsol::GeneralSolution)

    nRows = gsol.nRows
    nCols = gsol.nRows + gsol.nCols
    col_value = [gsol.col_value; gsol.row_value]
    col_dual = [gsol.col_dual; gsol.row_dual]
    row_value = zeros(nRows)
    row_dual = gsol.row_dual
    n_basis = gsol.n_basis
    basis_status = [gsol.col_basis_status; gsol.row_basis_status]
    is_optimal = gsol.is_optimal
    is_vertex = gsol.is_vertex

    return Solution(nRows, nCols, col_value, col_dual, row_value, row_dual, n_basis, 0, basis_status, [], is_optimal, is_vertex)

end

# sol to gsol
function reformulate_full_row_rank_general_sol(sol::Solution)
    nRows = sol.nRows
    nCols = sol.nCols - nRows
    col_value = sol.col_value[1:nCols]
    col_dual = sol.col_dual[1:nCols]
    row_value = sol.col_value[nCols+1:end]
    row_dual = sol.row_dual
    n_basis = sol.n_basis
    col_basis_status = sol.basis_status[1:nCols]
    row_basis_status = sol.basis_status[nCols+1:end]
    is_optimal = sol.is_optimal
    is_vertex = sol.is_vertex

    return GeneralSolution(nRows, nCols, col_value, col_dual, row_value, row_dual, n_basis, col_basis_status, row_basis_status, is_optimal, is_vertex)
end