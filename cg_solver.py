import pyGinkgo as pg
import numpy as np
import time
import csv
from pathlib import Path

RESULT_FILES = [
    "results/results01.csv",
    "results/results02.csv", 
    "results/results03.csv",
    "results/results04.csv",
    "results/results05.csv",
    "results/results06.csv",
    "results/results08.csv",
    "results/results11.csv",
    "results/results14.csv",
    "results/results15.csv",
    "results/results16.csv",
    "results/results18.csv",
]

MATRIX_FILES = [
    "matrices/bcsstk01.mtx",
    "matrices/bcsstk02.mtx",
    "matrices/bcsstk03.mtx",
    "matrices/bcsstk04.mtx",
    "matrices/bcsstk05.mtx",
    "matrices/bcsstk06.mtx",
    "matrices/bcsstk08.mtx",
    "matrices/bcsstk11.mtx",
    "matrices/bcsstk14.mtx",
    "matrices/bcsstk15.mtx",
    "matrices/bcsstk16.mtx",
    "matrices/bcsstk18.mtx",
]

"""Run single CG solve with Jacobi preconditioner and given tolerance."""
def run_solver(A, b, x, tolerance, run_id):
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


    # Calculate relative residual norm
    n_rows = A.shape[0]
    r = pg.as_tensor(device="cpu", dim=(n_rows, 1), dtype="double", fill=0.0)
    A.apply(x_sol, r) # r = A x_sol
    r_np = np.array([r.at(i) for i in range(n_rows)], dtype=float)
    b_np = np.array([b.at(i) for i in range(n_rows)], dtype=float)
    residual_norm = np.linalg.norm(r_np - b_np, ord=2)
    b_norm = np.linalg.norm(b_np, ord=2)
    relative_residual = residual_norm / b_norm


    # Console feedback on performance data
    print("\n ---Performance Data---")
    print(f"\nRun {run_id}, tol={tolerance:g}")
    print(f"Wall-clock solve time: {wall_time:.6e} seconds")
    print(f"Final residual 2-norm: {residual_norm:.6e}")
    print(f"Relative residual 2-norm: {relative_residual:.6e}")

    return wall_time, residual_norm, relative_residual

"""Run full tolerance experiment for one matrix and save to CSV."""
def run_matrix_experiment(matrix_path, result_path, matrix_id):
    print(f"\n=== Processing {matrix_path} (Matrix {matrix_id}) ===")
    
    # Load matrix
    A = pg.read(path=matrix_path, dtype="double", format="Csr", device="cpu")
    n_rows = A.shape[0]
    print(f"Matrix dimensions: {A.shape}")

    # RHS b and initial guess x
    b = pg.as_tensor(device="cpu", dim=(n_rows, 1), dtype="double", fill=1.0)
    x = pg.as_tensor(device="cpu", dim=(n_rows, 1), dtype="double", fill=0.0)

    tolerances = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]
    runs_per_tol = 10

    results_path = Path(result_path)
    write_header = not results_path.exists()

    run_id = 0
    for tol in tolerances:
        for _ in range(runs_per_tol):
            run_id += 1
            wall_time, res_norm, rel_res = run_solver(A, b, x, tol, run_id)
            row = [run_id, tol, wall_time, res_norm, rel_res]

            append_result_row(results_path, row, write_header=write_header)
            write_header = False

    print(f"Saved {matrix_id} results to {results_path.resolve()}")

"""Append a result row to CSV file."""
def append_result_row(path, row, write_header=False):
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        
        if write_header:
            writer.writerow(
                ["run_id", "tolerance", "wall_time_s", 
                "residual_norm", "relative_residual"]
            )
        writer.writerow(row)

"""Run experiments on all bcsstk matrices."""
def main():
    for i, (matrix_file, result_file) in enumerate(zip(MATRIX_FILES, RESULT_FILES), 1):
        run_matrix_experiment(matrix_file, result_file, i)

if __name__ == "__main__":
    main()
