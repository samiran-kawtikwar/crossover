Base.@kwdef mutable struct Iteration
    # problem buffer
    cost_new::Vector{Float64} = []
    b_new::Vector{Float64} = []

    # solution buffer
    col_value_next::Vector{Float64} = []
    row_dual_next::Vector{Float64} = []
    x_v::Vector{Float64} = []
    y_v::Vector{Float64} = []
    v_x::Vector{Float64} = []
    v_y::Vector{Float64} = []
    nrm_v_x::Float64 = 0.0
    nrm_v_y::Float64 = 0.0
    nrm_Av_x::Float64 = 1.0
    nrm_ATv_y::Float64 = 1.0

    # status record
    n_basis_last::Int64 = 0
    n_dual_active_last::Int64 = 0

    # parameters
    primal_push_method::PRIMAL_PUSH_METHOD = P_HYBRID
    dual_push_method::DUAL_PUSH_METHOD = D_HYBRID
end

Base.@kwdef mutable struct Crossover
    nCols::Int64 = 0
    nRows::Int64 = 0
    is_lp_loaded::Bool = false
    is_sol_loaded::Bool = false
    is_general_lp::Bool = false
    lp::LinearProgramming = LinearProgramming()
    sol_origin::Solution = Solution()
    sol::Solution = Solution()
    glp::GeneralLinearProgramming = GeneralLinearProgramming()
    gsol_origin::GeneralSolution = GeneralSolution()
    gsol::GeneralSolution = GeneralSolution()
    param::Parameter = Parameter()
    iter::Iteration = Iteration()
    info::Information = Information()
    timer::CrossoverTimer = CrossoverTimer()

    is_rhs_shifted::Bool = false
end


function load_lp!(crossover::Crossover, lp::LinearProgramming)
    crossover.lp = deepcopy(lp)
    crossover.is_lp_loaded = true
    crossover.is_general_lp = false
    crossover.nRows = lp.nRows
    crossover.nCols = lp.nCols
end

function load_solution!(crossover::Crossover, sol::Solution)
    crossover.sol_origin = deepcopy(sol)
    crossover.is_sol_loaded = true
    crossover.is_general_lp = false
    crossover.nRows = sol.nRows
    crossover.nCols = sol.nCols
end

function load_general_lp!(crossover::Crossover, glp::GeneralLinearProgramming)
    crossover.glp = deepcopy(glp)
    crossover.lp = reformulate_general_lp(glp, true)
    crossover.is_lp_loaded = true
    crossover.is_general_lp = true
    crossover.nRows = glp.nRows
    crossover.nCols = glp.nCols + glp.nRows
end

function load_general_solution!(crossover::Crossover, sol::GeneralSolution)
    crossover.gsol_origin = deepcopy(sol)
    crossover.sol_origin = reformulate_full_row_rank_sol(sol)
    crossover.is_sol_loaded = true
    crossover.is_general_lp = true
    crossover.nRows = sol.nRows
    crossover.nCols = sol.nCols + sol.nRows
end

function shift_rhs!(crossover::Crossover, shift::Float64)

    nRows = crossover.nRows
    nCols = crossover.nCols
    crossover.lp.lb[nCols-nRows+1:end] .+= shift
    crossover.lp.ub[nCols-nRows+1:end] .+= shift

    row_shift = sum(crossover.lp.A[:, nCols-nRows+1:end], dims=2) * shift
    crossover.lp.b .+= row_shift

    crossover.lp.offset -= Int(crossover.lp.sense) * sum(crossover.lp.cost[nCols-nRows+1:end]) * shift

    crossover.sol.col_value[nCols-nRows+1:end] .+= shift
    crossover.sol.row_value .+= row_shift

    crossover.is_rhs_shifted = true

end

function unshift_rhs!(crossover::Crossover)
    if crossover.is_rhs_shifted

        if crossover.param.verbose && crossover.param.verbose_level >= 2
            println("Unshift rhs.")
        end

        shift_rhs!(crossover, -crossover.param.rhs_shift)
    end
end

function initialize!(crossover::Crossover)
    @assert crossover.is_lp_loaded "LP is not loaded"
    @assert crossover.is_sol_loaded "Solution is not loaded"
    @assert crossover.nCols == crossover.sol_origin.nCols && crossover.nCols == crossover.lp.nCols "Inconsistent number of columns"
    @assert crossover.nRows == crossover.sol_origin.nRows && crossover.nRows == crossover.lp.nRows "Inconsistent number of rows"
    @assert crossover.nRows <= crossover.nCols "nRows should be less than or equal to nCols"

    initialize_crossover_solution!(crossover)
    initialize_iteration!(crossover.iter, crossover.nRows, crossover.nCols, crossover.param.primal_push_method, crossover.param.dual_push_method)
    initialize_information!(crossover.info)
end

function initialize_crossover_solution!(crossover::Crossover)

    crossover.sol = deepcopy(crossover.sol_origin)

    crossover.sol.col_dual *= Int(crossover.lp.sense)
    crossover.sol.row_dual *= Int(crossover.lp.sense)

    crossover.sol.basis_status = fill(BASIC, crossover.nCols)
    crossover.sol.dual_status = fill(INACTIVE, crossover.nCols)
    crossover.sol.n_basis = sum(crossover.sol.basis_status .== BASIC)
    crossover.sol.n_dual_active = sum(crossover.sol.dual_status .== ACTIVE)

    if norm(crossover.lp.b) < crossover.param.epsilon_zero && crossover.param.rhs_shift != 0.0

        if crossover.param.verbose && crossover.param.verbose_level >= 2
            println("Shift rhs.")
        end

        shift_rhs!(crossover, crossover.param.rhs_shift)
    end

    if crossover.param.seed >= 0
        Random.seed!(crossover.param.seed)
    end
end

function set_crossover_status_string!(crossover::Crossover)
    if crossover.info.crossover_status == SUCCEEDED
        if crossover.sol.is_optimal
            status_str = "Crossover succeeded."
        else
            status_str = "Crossover succeeded but may be not optimal."
        end
    elseif crossover.info.crossover_status == REACHED_TIME_LIMIT
        status_str = "Crossover reached time limit."
    elseif crossover.info.crossover_status == REACHED_ITERATION_LIMIT
        status_str = "Crossover reached iteration limit."
    elseif crossover.info.crossover_status == UNSTARTED
        status_str = "Crossover unstarted."
    elseif crossover.info.crossover_status == PRIMAL_FAILED
        status_str = "Crossover failed at primal push."
    elseif crossover.info.crossover_status == DUAL_FAILED
        status_str = "Crossover failed at dual push."
    elseif crossover.info.crossover_status == LINEAR_IND_FAILED
        status_str = "Crossover failed at linear independence check."
    else
        status_str = "Crossover failed."
    end

    crossover.info.crossover_status_str = status_str
end

function recover_original_solution!(crossover)
    # change sign for MAXIMIZE
    crossover.sol.col_dual *= Int(crossover.lp.sense)
    crossover.sol.row_dual *= Int(crossover.lp.sense)

    # reformulate for general lp
    if crossover.is_general_lp
        crossover.gsol = reformulate_full_row_rank_general_sol(crossover.sol)
    end

end

function finalize!(crossover::Crossover)
    if crossover.param.verbose
        println("Finalize")
    end

    cal_optimality!(crossover.info, crossover.lp, crossover.sol; nrm_type=crossover.param.nrm_type)
    if check_optimality(crossover.info; tol=crossover.param.tol_cross)
        crossover.sol.is_optimal = true
    elseif crossover.info.crossover_status == SUCCEEDED
        cleanup_succ = cleanup!(crossover)
        if cleanup_succ
            cal_optimality!(crossover.info, crossover.lp, crossover.sol; nrm_type=crossover.param.nrm_type)
            if check_optimality(crossover.info; tol=crossover.param.tol_cross)
                crossover.sol.is_optimal = true
            else
                crossover.sol.is_optimal = false
            end
        else
            crossover.sol.is_optimal = false
        end
    else
        crossover.sol.is_optimal = false
    end

    # recover original solution
    # - to change sign for MAXIMIZE
    # - reformulate for general lp
    unshift_rhs!(crossover)
    recover_original_solution!(crossover)

    if crossover.param.verbose
        print_optimality(crossover)
    end

    # record end time
    toc!(crossover.timer)

    set_crossover_status_string!(crossover)
    if crossover.param.verbose
        println()
        print_time(crossover.timer)
        print_status(crossover)
    end

    if crossover.param.verbose
        print_line()
    end
end

function run!(crossover::Crossover)
    # Record start time
    tic!(crossover.timer)

    # print header
    if crossover.param.verbose
        print_line()
        print_crossover_header()
        print_line()
        print_author()
        print_line()
    end

    # print lp info
    if crossover.param.verbose && crossover.is_lp_loaded
        if crossover.is_general_lp
            println("Reformulated from General Linear Programming")
        else
            println("Standard Linear Programming")
        end
        print_lp_info(crossover.lp)
        print_line()
    end

    # initialize
    initialize!(crossover)

    # crossover
    crossover_main!(crossover)

    # finalize
    finalize!(crossover)
end

function resetParam!(crossover::Crossover)
    if crossover.param.verbose
        println("Resetting parameters.")
    end
    crossover.param = Parameter()
end

Base.getproperty(solver::Crossover, name::Symbol) =
    name == :load_lp! ? (lp) -> load_lp!(solver, lp) :
    name == :load_solution! ? (sol) -> load_solution!(solver, sol) :
    name == :load_general_lp! ? (lp) -> load_general_lp!(solver, lp) :
    name == :load_general_solution! ? (sol) -> load_general_solution!(solver, sol) :
    name == :run! ? () -> run!(solver) :
    name == :setParam! ? (name, value) -> setParam!(solver.param, name, value) :
    name == :resetParam! ? () -> resetParam!(solver) :
    getfield(solver, name)