using JuMP
using COPT
using crossover


function copt_solve(lp::LinearProgramming; lpmethod::Int64=6, presolve::Bool=false, crossover_flag::Bool=false, tol::Float64=1e-6, fease_tol::Float64=1e-8, time_limit::Float64=600.0)

    # initialization
    model = Model(COPT.Optimizer)

    # modeling
    println("adding variables ...")
    @variable(model, lp.lb[i] <= x[i=1:lp.nCols] <= lp.ub[i])
    println("adding constraints ...")
    @constraint(model, lp.A * x .== lp.b)
    println("setting objective ...")
    if lp.sense == crossover.MINIMIZE
        @objective(model, Min, Int(lp.sense) * lp.cost' * x + lp.offset)
    else
        @objective(model, Max, Int(lp.sense) * lp.cost' * x + lp.offset)
    end

    # setting parameters
    set_optimizer_attribute(model, "presolve", presolve)
    set_optimizer_attribute(model, "lpmethod", lpmethod)
    set_optimizer_attribute(model, "crossover", crossover_flag)
    set_optimizer_attribute(model, "dualtol", tol)
    set_optimizer_attribute(model, "barprimaltol", tol)
    set_optimizer_attribute(model, "bardualtol", tol)
    set_optimizer_attribute(model, "bargaptol", tol)
    set_optimizer_attribute(model, "pdlptol", tol)
    set_optimizer_attribute(model, "feastol", fease_tol)
    set_optimizer_attribute(model, "timelimit", time_limit)

    # solve
    optimize!(model)

    return model

    nothing

end


function get_lp_solutions(model::Model)
    lp = lp_matrix_data(model)
    nRows, nCols = size(lp.A)

    vars = JuMP.all_variables(model)
    constrs = JuMP.all_constraints(model, include_variable_in_set_constraints=false)

    col_values = value.(vars)
    col_duals = reduced_cost.(vars)
    row_values = value.(constrs)
    row_duals = dual.(constrs)
    # row_duals = shadow_price.(constrs)  # WARNING: sign of shadow_price is different from dual!

    return Solution(nRows, nCols, col_values, col_duals, row_values, row_duals, 0, 0, [], [], termination_status(model) == MOI.OPTIMAL, false)

end


function copt_solve_general(glp::GeneralLinearProgramming; lpmethod::Int64=6, presolve::Bool=false, crossover_flag::Bool=false, tol::Float64=1e-6, fease_tol::Float64=1e-8, time_limit::Float64=600.0)

    # initialization
    model = Model(COPT.Optimizer)
    # silence model outputs
    set_optimizer_attribute(model, "Logging", 0)
    # modeling
    println("adding variables ...")
    @variable(model, glp.lb[i] <= x[i=1:glp.nCols] <= glp.ub[i])
    println("adding constraints ...")
    @constraint(model, glp.lhs .<= glp.A * x .<= glp.rhs)
    println("setting objective ...")
    if glp.sense == crossover.MINIMIZE
        @objective(model, Min, Int(glp.sense) * glp.cost' * x + glp.offset)
    else
        @objective(model, Max, Int(glp.sense) * glp.cost' * x + glp.offset)
    end

    # setting parameters
    set_optimizer_attribute(model, "presolve", presolve)
    set_optimizer_attribute(model, "lpmethod", lpmethod)
    set_optimizer_attribute(model, "crossover", crossover_flag)
    set_optimizer_attribute(model, "dualtol", tol)
    set_optimizer_attribute(model, "barprimaltol", tol)
    set_optimizer_attribute(model, "bardualtol", tol)
    set_optimizer_attribute(model, "bargaptol", tol)
    set_optimizer_attribute(model, "pdlptol", tol)
    set_optimizer_attribute(model, "feastol", fease_tol)
    set_optimizer_attribute(model, "timelimit", time_limit)

    # solve
    optimize!(model)

    return model

    nothing

end


function get_general_lp_solutions(model::Model)
    glp = lp_matrix_data(model)
    nRows, nCols = size(glp.A)

    vars = JuMP.all_variables(model)
    constrs = JuMP.all_constraints(model, include_variable_in_set_constraints=false)

    col_values = value.(vars)
    col_duals = reduced_cost.(vars)
    row_values = value.(constrs)
    row_duals = dual.(constrs)
    # row_duals = shadow_price.(constrs)  # WARNING: sign of shadow_price is different from dual!

    return GeneralSolution(nRows, nCols, col_values, col_duals, row_values, row_duals, 0, [], [], termination_status(model) == MOI.OPTIMAL, false)

end


function calculate_nonsupport_number(lp::Union{LinearProgramming,GeneralLinearProgramming}, solution::Union{Solution,GeneralSolution}, tol::Float64=1e-8)

    idx_fixed = (lp.lb .== lp.ub)

    # @assert all(lp.lb .- tol .<= solution.col_value .<= lp.ub .+ tol) "col_value must be in bound"

    n_fixed = sum(idx_fixed)

    idx_on_lb = (lp.lb .+ tol .>= solution.col_value) .& .!idx_fixed
    idx_on_ub = (lp.ub .- tol .<= solution.col_value) .& .!idx_fixed

    n_on_lb = sum(idx_on_lb)
    n_on_ub = sum(idx_on_ub)

    support_number = n_fixed + n_on_lb + n_on_ub

    return support_number
end

function calculate_support_number(lp::Union{LinearProgramming,GeneralLinearProgramming}, solution::Union{Solution,GeneralSolution}, tol::Float64=1e-8)
    return lp.nCols - calculate_nonsupport_number(lp, solution, tol)
end

function write1record2csv(filename::String, record::NamedTuple, write_type::String="a")
    @assert split(filename, ".")[end] == "csv" "filename must be csv"

    is_file_flag = isfile(filename)
    open(filename, write_type) do file
        if !is_file_flag
            # write header
            header = join(keys(record), ",")
            println(file, header)
        end

        # write record
        record_str = join([string(v) for v in values(record)], ",")
        println(file, record_str)
    end
end