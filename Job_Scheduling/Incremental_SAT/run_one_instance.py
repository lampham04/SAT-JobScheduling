# run_one_instance.py
import sys
from pathlib import Path
from Incremental_SAT_functions import (
    read_dataset,
    window_tightening,
    solve_SAT,
    compute_UB_Lmax,
    incremental_SAT_Lmax
)


def main():
    if len(sys.argv) != 3:
        print("Usage: python run_one_instance.py <instance_path> <ub_file>")
        sys.exit(1)

    instance_path = Path(sys.argv[1])
    sol_file = Path(sys.argv[2])

    # -------- Pipeline --------

    # Read dataset
    n, weights, durations, ready_dates, due_dates, deadlines, successors = \
        read_dataset(instance_path)

    # Window tightening
    new_ready_dates, new_deadlines = window_tightening(
        n, ready_dates, durations, deadlines, successors
    )

    # Initial SAT solve
    cnf, next_var, schedule, valid, S, is_sat = solve_SAT(
        n, durations, new_ready_dates, new_deadlines, successors
    )

    if not is_sat:
        sol_file.write_text("UNSAT")
        return

    # Initial UB
    UB = compute_UB_Lmax(schedule, durations, due_dates)

    # Incremental SAT
    incremental_SAT_Lmax(durations, due_dates, S, cnf, UB, sol_file, new_ready_dates, new_deadlines, successors)


if __name__ == "__main__":
    main()
