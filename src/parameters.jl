Base.@kwdef mutable struct Parameter
    max_time::Float64 = 600.0
    tol_bound::Float64 = 1.0e-8
    tol_cross::Float64 = 1.0e-6
    tol_recover::Float64 = 1.0e-1
    tol_pdlp::Float64 = 1.0e-8
    tol_feas::Float64 = 1.0e-8
    epsilon_zero::Float64 = 1.0e-8
    verbose::Bool = true
    verbose_level::Int64 = 1
    seed::Int64 = -1
    primal_push_method::PRIMAL_PUSH_METHOD = P_HYBRID
    dual_push_method::DUAL_PUSH_METHOD = D_HYBRID
    ols_method::OLS_METHOD = OLS_QR
    nrm_type::OPTIMALITY_NORM = L_2
    pdlp_presolve::Bool = true
    tol_ols::Float64 = 1.0e-16
    max_iter_ols::Int64 = 1e8
    lp_method::Int64 = 6  # 6-PDLP or 2-IPM
    primal_push_general::Bool = true
    dual_push_general::Bool = true
    push_general::Bool = true
    delta::Float64 = 1.0
    rhs_shift::Float64 = 0.0
    gamma::Float64 = 1.0
end

function setParam!(param::Parameter, name::String, value)
    if param.verbose
        println("Setting parameter [$name] to [$value].")
    end

    if name == "max_time"
        param.max_time = value
    elseif name == "tol_bound"
        param.tol_bound = value
    elseif name == "tol_cross"
        param.tol_cross = value
    elseif name == "tol_recover"
        param.tol_recover = value
    elseif name == "tol_pdlp"
        param.tol_pdlp = value
    elseif name == "tol_feas"
        param.tol_feas = value
    elseif name == "epsilon_zero"
        param.epsilon_zero = value
    elseif name == "verbose"
        param.verbose = value
    elseif name == "verbose_level"
        param.verbose_level = value
    elseif name == "seed"
        param.seed = value
    elseif name == "primal_push_method"
        param.primal_push_method = value
    elseif name == "dual_push_method"
        param.dual_push_method = value
    elseif name == "ols_method"
        param.ols_method = value
    elseif name == "nrm_type"
        param.nrm_type = value
    elseif name == "pdlp_presolve"
        param.pdlp_presolve = value
    elseif name == "tol_ols"
        param.tol_ols = value
    elseif name == "max_iter_ols"
        param.max_iter_ols = value
    elseif name == "lp_method"
        param.lp_method = value
    elseif name == "primal_push_general"
        param.primal_push_general = value
    elseif name == "dual_push_general"
        param.dual_push_general = value
    elseif name == "push_general"
        param.push_general = value
        param.primal_push_general = value
        param.dual_push_general = value
    elseif name == "delta"
        param.delta = value
    elseif name == "rhs_shift"
        param.rhs_shift = value
    elseif name == "gamma"
        param.gamma = value
    else
        if param.verbose
            println("Unsupported parameter name [$name].")
        end
    end
end