# run_one_instance.py
import sys
from pathlib import Path
from gurobi_functions import (
    read_dataset,
    window_tightening,
    solve_gurobi
)

def main():
    if len(sys.argv) != 3:
        print("Usage: python run_one_instance.py <instance_path> <ub_file>")
        sys.exit(1)

    instance_path = Path(sys.argv[1])
    sol_file = Path(sys.argv[2])
    time_limit = 297

    # -------- Pipeline --------

    # Read dataset
    n, durations, ready_dates, due_dates, deadlines, successors = read_dataset(instance_path)

    # Window tightening
    #new_ready_dates, new_deadlines = window_tightening(n, ready_dates, durations, deadlines, successors)

    # Solve
    solve_gurobi(n, durations, ready_dates, deadlines, due_dates, successors, sol_file, time_limit)

if __name__ == "__main__":
    main()
