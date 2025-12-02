import pyGinkgo as pg
import numpy as np

def main():
    # Loading sparse matrix on reference device
    fn = "bcsstk01.mtx"
    A = pg.read ( path = fn , dtype = "double" , format = "Csr", device = "cpu" )

    n_rows = A.shape[0]

    # RHS b and initial guess x
    b = pg.as_tensor ( device = "cpu", dim = ( n_rows, 1 ), dtype = "double", fill = 1.0 )
    x = pg.as_tensor ( device = "cpu", dim = ( n_rows, 1 ), dtype = "double", fill = 0.0 )

    # Solver params for GC + Jacobi preconditioners
    solver_params = {
        "type": "solver::Cg",
        "preconditioner": {
            "type": "preconditioner::Jacobi"
        },
        "criteria": [
            {"type": "Iteration", "max_iters": 1000},
            {"type": "ResidualNorm", "reduction_factor": 1e-6},
        ],
    }

    # Solving Ax=b with config_solve
    logger, x_sol = pg.config_solve(A, b, x, solver_args=solver_params)


    # Printing logger info
    print("Solver logger:", logger)

    # Printing first 8 elems of solution vector x
    print("\nSolution x (first 8):")
    for i in range(min(8, x_sol.shape[0])):
        print(x_sol.at(i))

if __name__ == "__main__":
    main()
