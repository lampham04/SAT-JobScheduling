# run_one_instance.py
import sys
from pathlib import Path
from Incremental_SAT_functions import (
    read_dataset,
    window_tightening
)
from cplex_Lmax import solve_Lmax_cplex

def main():
    if len(sys.argv) != 3:
        print("Usage: python run_one_instance.py <instance_path> <ub_file>")
        sys.exit(1)

    instance_path = Path(sys.argv[1])
    sol_file = Path(sys.argv[2])
    time_limit = 30

    # -------- Pipeline --------

    # Read dataset
    n, weights, durations, ready_dates, due_dates, deadlines, successors = \
    read_dataset(instance_path)

    # Window tightening
    new_ready_dates, new_deadlines = window_tightening(
    n, ready_dates, durations, deadlines, successors
    )

    solve_Lmax_cplex(n, durations, new_ready_dates, new_deadlines, due_dates, successors, sol_file, time_limit)

if __name__ == "__main__":
    main()
