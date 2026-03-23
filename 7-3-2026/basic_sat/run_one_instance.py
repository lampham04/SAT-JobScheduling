# run_one_instance.py
import sys
from pathlib import Path
from functions import (
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
    n, durations, ready_dates, due_dates, deadlines, successors = read_dataset(instance_path)

    # Window tightening
    new_ready_dates, new_deadlines = window_tightening(
        n, ready_dates, durations, deadlines, successors
    )

    # Initial SAT solve
    cnf, schedule, valid_starts, S, is_sat = solve_SAT(
        n, durations, new_ready_dates, new_deadlines, successors
    )

    if not is_sat:
        sol_file.write_text("UNSAT")
        return

    # Initial UB
    UB = compute_UB_Lmax(schedule, durations, due_dates)

    with open(sol_file, "w") as f:
        f.write(f"Lmax = {str(UB)} \n")
        f.write("Schedule: \n")
        for i, start in sorted(schedule.items(), key=lambda x: x[1]):
            f.write(f"Job {i}: start = {start}, end = {start + durations[i]} \n")

    # Incremental SAT
    incremental_SAT_Lmax(durations, due_dates, S, cnf, UB, sol_file, valid_starts)


if __name__ == "__main__":
    main()
