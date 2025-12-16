# run_one_instance.py

import sys
from pathlib import Path

from Incremental_SAT_functions import (
    read_dataset,
    window_tightening,
    solve_SAT,
    compute_UB,
    incremental_SAT
)


def main():
    if len(sys.argv) != 3:
        print("Usage: python run_one_instance.py <instance_path> <ub_file>")
        sys.exit(1)

    instance_path = Path(sys.argv[1])
    ub_file = Path(sys.argv[2])

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
        ub_file.write_text("UNSAT")
        return

    # Initial UB
    UB = compute_UB(schedule, durations, weights, due_dates)

    # Save initial UB
    ub_file.write_text(str(UB))

    # Incremental SAT (UB được cập nhật & ghi file bên trong)
    incremental_SAT(
        weights,
        durations,
        due_dates,
        S,
        cnf,
        UB,
        valid,
        next_var,
        ub_file
    )


if __name__ == "__main__":
    main()
