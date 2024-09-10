"""
a general form of linear programming

min/max  cost' * x + offset
 s.t.    lhs <= A * x <= rhs
         lb <= x <= ub

nRows:  number of constraints
nCols:  number of variables
cost:   cost vector in the ORIGINAL form
A:      constraint matrix
lhs:    left-hand side vector
rhs:    right-hand side vector
lb:     variable lower bound vector
ub:     variable upper bound vector
sense:  sense of the ORIGINAL objective function
offset: offset of the ORIGINAL objective function
"""
Base.@kwdef mutable struct GeneralLinearProgramming
    nRows::Int64 = 0
    nCols::Int64 = 0
    cost::Vector{Float64} = []
    A::SparseArrays.SparseMatrixCSC{Float64,Int64} = SparseMatrixCSC{Float64,Int64}(undef, 0, 0)
    lhs::Vector{Float64} = []
    rhs::Vector{Float64} = []
    lb::Vector{Float64} = []
    ub::Vector{Float64} = []
    sense::SENSE = MINIMIZE
    offset::Float64 = 0.0
end


"""
basic solution for a general form of linear programming

min  cost' * x + offset
s.t. lhs <= A * x <= rhs (y)
     lb <= x <= ub       (s)

nRows:            number of constraints
nCols:            number of variables
col_value:        values of x
col_dual:         values of s
row_value:        values of A * x
row_dual:         values of y
col_basis_status: column basis status vector
row_basis_status: row basis status vector
is_optimal:       whether the solution is optimal
is_vertex:        whether the solution is a vertex
"""
Base.@kwdef mutable struct GeneralSolution
    nRows::Int64 = 0
    nCols::Int64 = 0
    col_value::Vector{Float64} = []
    col_dual::Vector{Float64} = []
    row_value::Vector{Float64} = []
    row_dual::Vector{Float64} = []
    n_basis::Int64 = 0
    col_basis_status::Vector{BASIS_STATUS} = []
    row_basis_status::Vector{BASIS_STATUS} = []
    is_optimal::Bool = false
    is_vertex::Bool = false
end


function read_general_linear_programming(file::String)
    model = read_from_file(file)
    relax_integrality(model)
    lp = lp_matrix_data(model)

    nRows, nCols = size(lp.A)
    cost = lp.c
    A = lp.A
    lhs = lp.b_lower
    rhs = lp.b_upper
    lb = lp.x_lower
    ub = lp.x_upper
    sense = lp.sense == MIN_SENSE ? MINIMIZE :
            lp.sense == MAX_SENSE ? MAXIMIZE :
            lp.sense == FEASIBILITY_SENSE ? FEASIBILITY : MINIMIZE
    offset = lp.c_offset

    return GeneralLinearProgramming(nRows, nCols, cost, A, lhs, rhs, lb, ub, sense, offset)

end