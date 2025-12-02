import pyGinkgo as pg
import numpy as np
import time

def main():
    # Loading sparse matrix on reference device
    fn = "bcsstk01.mtx"
    A = pg.read ( path = fn , dtype = "double" , format = "Csr", device = "cpu" )

    n_rows = A.shape[0]

    # RHS b and initial guess x
    b = pg.as_tensor ( device = "cpu", dim = ( n_rows, 1 ), dtype = "double", fill = 1.0 )
    x = pg.as_tensor ( device = "cpu", dim = ( n_rows, 1 ), dtype = "double", fill = 0.0 )

    # Solver params for GC + Jacobi preconditioners, iterations + residual criteria
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

    # Timing the Solve
    start_time = time.perf_counter()

    # Solving Ax=b with config_solve
    logger, x_sol = pg.config_solve(A, b, x, solver_args=solver_params)

    end_time = time.perf_counter()
    wall_time = end_time - start_time

    # Calculate residual r = A x_sol - b
    r = pg.as_tensor(device="cpu", dim=(n_rows, 1), dtype="double", fill=0.0)
    A.apply(x_sol, r) # r = A x_sol
    # Need to convert r and b to NumPy arrays to allow subtraction and computing of the norm
    r_np = np.array([r.at(i) for i in range(n_rows)], dtype=float)
    b_np = np.array([b.at(i) for i in range(n_rows)], dtype=float)
    res_vec = r_np - b_np
    residual_norm = np.linalg.norm(res_vec, ord=2)


    # Printing Performance data
    print("\n ---Performance Data---")
    print(f"Wall-clock solve time: {wall_time:.6e} seconds")
    print(f"Final residual 2-norm: {residual_norm:.6e}")

    print("\n--- Raw Ginkgo logger ---")
    print(logger) # Not exposed; can't inspect it


    # Printing first 8 elems of solution vector x
    print("\nSolution x (first 8):")
    for i in range(min(8, x_sol.shape[0])):
        print(x_sol.at(i))

if __name__ == "__main__":
    main()
