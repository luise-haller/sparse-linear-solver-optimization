import pyGinkgo as pg
import numpy as np
import time
import csv
from pathlib import Path

RESULT_FILE = "results03.csv"

def run_single_solve(tolerance, run_id):
    # Loading sparse matrix on reference device
    fn = "bcsstk03.mtx"
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
            {"type": "ResidualNorm", "reduction_factor": tolerance},
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
    r_np = np.array([r.at(i) for i in range(n_rows)], dtype=float)
    b_np = np.array([b.at(i) for i in range(n_rows)], dtype=float)
    res_vec = r_np - b_np
    residual_norm = np.linalg.norm(res_vec, ord=2)

    b_norm = np.linalg.norm(b_np, ord=2)
    relative_residual = residual_norm / b_norm


    # Console feedback on performance data
    print("\n ---Performance Data---")
    print(f"\nRun {run_id}, tol={tolerance:g}")
    print(f"Wall-clock solve time: {wall_time:.6e} seconds")
    print(f"Final residual 2-norm: {residual_norm:.6e}")
    print(f"Relative residual 2-norm: {relative_residual:.6e}")

    return wall_time, residual_norm, relative_residual

def append_result_row(path, row, write_header=False):
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(
                ["run_id", "tolerance", "wall_time_s", 
                "residual_norm", "relative_residual"]
            )
        writer.writerow(row)

def main():
    
    # Experiment variables for testing
    tolerances = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]
    runs_per_tol = 10

    results_path = Path(RESULT_FILE)
    write_header = not results_path.exists()

    run_id = 0
    for tol in tolerances:
        for _ in range(runs_per_tol):
            run_id += 1
            wall_time, res_norm, rel_res = run_single_solve(tol, run_id)
            row = [run_id, tol, wall_time, res_norm, rel_res]
            append_result_row(results_path, row, write_header=write_header)
            write_header = False

    print(f"\nSaved results to {results_path.resolve()}")
    

    

if __name__ == "__main__":
    main()
