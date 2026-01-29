from pathlib import Path
from Incremental_SAT_functions import (
    read_dataset,
    window_tightening,
    solve_SAT,
    compute_UB_Lmax,
    incremental_SAT_Lmax
)

BASE_DIR = Path(__file__).resolve().parent

instance_path = BASE_DIR / "datasets" / "S" / "30_10_005_125_25_3.GSP"
ub_file = BASE_DIR / "datasets" / "UB" / "30_10_005_125_25_3.GSP.ub"

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

# Initial UB
UB = compute_UB_Lmax(schedule, durations, due_dates)

# Save initial UB
ub_file.write_text(str(UB))

# Incremental SAT (UB được cập nhật & ghi file bên trong)
incremental_SAT_Lmax(
    durations,
    weights,
    due_dates,
    S,
    cnf,
    UB,
    ub_file,
    ready_dates,
    deadlines,
    successors
)