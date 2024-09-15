# crossover.jl inspired by PDHG

This repository contains experimental code of a randomized crossover approach for the linear programming (LP) in the general form

$$
\begin{aligned}
\min\ & c^\top x \\
\text{s.t.}\ & lhs \leq Ax \leq rhs \\
& lb \leq x \leq ub.
\end{aligned}
$$

## Setup

A one-time step is required to set up the necessary packages on the local machine:
```shell
julia --project -e 'import Pkg; Pkg.instantiate()'
```

We use cuPDLP-C (`LpMethod=6`) in [COPT](https://shanshu.ai/copt) to obtain the initial optimal solution for crossover, and to solve auxiliary LPs in our crossover. Please apply for a free academic license via the hyperlink for reproducing the experiments.

## Test

Test scripts are located in `test/`.
 - `test_general.jl` for a single LP instance.
 - `test_general_netlib.jl` for testing all the NETLIB LP instances.

Remember to change the `path`/`netlib_path` (and `save_path`) in the test scripts to load LPs in `.mps` format (and to save the results).

## Parameters

| name | default | description |
| --- | --- | --- |
| `max_time` | 600 | Time limit for crossover. |
| `tol_bound` | 1e-8 | Tolerance to bound for non-basis identification. |
| `tol_cross` | 1e-6 | Relative tolerance for optimality after crossover. |
| `tol_recover` | 1e-1 | Tolerance to recover feasibility during crossover. |
| `tol_pdlp` | 1e-8 | Optimality tolerance of auxiliary LPs. |
| `tol_feas` | 1e-8 | Feasibility tolerance of auxiliary LPs. |
| `epsilon_zero` | 1e-8 | Tolerance to zero. |
| `verbose` | true | Whether to log. |
| `verbose_level` | 1 | Level to log. |
| `seed` | -1 | Random seed. -1 means not set. |
| `primal_push_method` | P_HYBRID | Primal push method. |
| `dual_push_method` | D_HYBRID | Dual push method. |
| `ols_method` | OLS_QR | OLS method. |
| `nrm_type` | L_2 | Norm type used to calculate the optimality. |
| `pdlp_presolve` | true | Whether to presolve in auxiliary LPs. |
| `tol_ols` | 1e-16 | Tolerance in iterative OLS solver. |
| `lp_method` | 6 | LP method for auxiliary LPs. 6 is PDLP; 2 is IPM. |
| `max_iter_ols` | 1e8 | Iteration limit in iterative OLS solver. |
| `primal_push_general` | true | Whether to use primal auxiliary LPs in the general form. |
| `dual_push_general` | true | Whether to use dual auxiliary LPs in the general form. |
| `push_general` | true | Whether to use auxiliary LPs in the general form. |
| `delta` | 1.0 | Weight of perturbation. |
| `rhs_shift` | 0.0 | Right-hand-side shift. |
| `gamma` | 1.0 | Ratio of primal variable to dual slack for non-basis identification. |
