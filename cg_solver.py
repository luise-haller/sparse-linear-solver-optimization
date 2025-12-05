import pyGinkgo as pg
import numpy as np
import time
import csv
from pathlib import Path
import os

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

FIXED_TOL = 1e-6

PRECONDITIONERS = [
    "preconditioner::Jacobi",
    "preconditioner::Ic",
    "preconditioner::Ilu"
]

"""Run single CG solve with Jacobi preconditioner and given tolerance."""
def run_solver(A, b, x, prec_type, run_id):
    solver_params = {
        "type": "solver::Cg",
        "preconditioner": {
            "type": prec_type
        },
        "criteria": [
            {"type": "Iteration", "max_iters": 1000},
            {"type": "ResidualNorm", "reduction_factor": FIXED_TOL},
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

    print(
        f"Run {run_id}, prec={prec_type}, tol={FIXED_TOL:g}: "
        f"{wall_time:.6e}s, rel_res={relative_residual:.6e}"
    )
    return wall_time, residual_norm, relative_residual

"""Run full tolerance experiment for one matrix and save to CSV."""
def run_matrix_experiment(matrix_filename, result_filename, matrix_id):
    matrix_path = Path(matrix_filename)
    print(f"\n---Processing {matrix_path} (Matrix {matrix_id})---")
    
    if not matrix_path.exists():
        print(f"ERROR: Matrix file not found: {matrix_path.absolute()}")
        return

    A = pg.read(path=str(matrix_path), dtype="double", format="Csr", device="cpu")
    n_rows = A.shape[0]
    print(f"Matrix loaded successfully: {A.shape}")

    b = pg.as_tensor(device="cpu", dim=(n_rows, 1), dtype="double", fill=1.0)
    x = pg.as_tensor(device="cpu", dim=(n_rows, 1), dtype="double", fill=0.0)

    results_path = Path(result_filename)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not results_path.exists()

    run_id = 0
    runs_per_prec = 10

    for prec in PRECONDITIONERS:
        print(f"\n-- Preconditioner: {prec} --")
        for _ in range(runs_per_prec):
            run_id += 1
            wall_time, res_norm, rel_res = run_solver(A, b, x, prec, run_id)
            row = [run_id, prec, FIXED_TOL, wall_time, res_norm, rel_res]
            append_result_row(results_path, row, write_header=write_header)
            write_header = False

    print(f"Saved results to {results_path.absolute()}")

"""Append a result row to CSV file."""
def append_result_row(path, row, write_header=False):
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(
                [
                    "run_id",
                    "preconditioner",
                    "tolerance",
                    "wall_time_s",
                    "residual_norm",
                    "relative_residual",
                ]
            )
        writer.writerow(row)

def clear_results_files():
    for path in RESULT_FILES:
        p = Path(path)
        if p.exists():
            p.unlink()


"""Run experiments on all bcsstk matrices."""
def main():
    clear_results_files()
    
    for i, (matrix_file, result_file) in enumerate(zip(MATRIX_FILES, RESULT_FILES), 1):
        run_matrix_experiment(matrix_file, result_file, i)

if __name__ == "__main__":
    main()
