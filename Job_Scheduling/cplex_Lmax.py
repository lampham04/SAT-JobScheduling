from docplex.mp.model import Model

from Incremental_SAT_functions import (read_dataset, window_tightening)

def solve_Lmax_cplex(n, durations, ready_dates, deadlines, due_dates, successors, sol_file, time_limit):
    """
    Single machine scheduling with r_i, deadline_i, precedence
    Objective: minimize Lmax
    """

    mdl = Model("SingleMachine_Lmax")
    jobs = list(range(1, n + 1))

    # -----------------------
    # Variables
    # -----------------------
    S = {i: mdl.continuous_var(lb=0, name=f"S_{i}") for i in jobs}
    Lmax = mdl.continuous_var(lb=0, name="Lmax")

    # ordering variables for machine capacity
    x = {}
    for i in jobs:
        for j in jobs:
            if i < j:
                x[(i, j)] = mdl.binary_var(name=f"x_{i}_{j}")

    # Big-M
    M = max(deadlines.values()) + max(durations.values())

    # -----------------------
    # Constraints
    # -----------------------

    # Release dates + deadlines
    for i in jobs:
        mdl.add_constraint(S[i] >= ready_dates[i], ctname=f"release_{i}")
        mdl.add_constraint(S[i] + durations[i] <= deadlines[i], ctname=f"deadline_{i}")

    # Precedence
    for i in jobs:
        for j in successors.get(i, []):
            mdl.add_constraint(
                S[j] >= S[i] + durations[i],
                ctname=f"prec_{i}_{j}"
            )

    # Single machine capacity
    for i in jobs:
        for j in jobs:
            if i < j:
                mdl.add_constraint(
                    S[i] + durations[i] <= S[j] + M * (1 - x[(i, j)]),
                    ctname=f"cap1_{i}_{j}"
                )
                mdl.add_constraint(
                    S[j] + durations[j] <= S[i] + M * x[(i, j)],
                    ctname=f"cap2_{i}_{j}"
                )

    # Lmax definition
    for i in jobs:
        mdl.add_constraint(
            Lmax >= S[i] + durations[i] - due_dates[i],
            ctname=f"Lmax_{i}"
        )

    mdl.parameters.timelimit = time_limit
    # -----------------------
    # Objective
    # -----------------------
    mdl.minimize(Lmax)

    # -----------------------
    # Solve
    # -----------------------
    sol = mdl.solve()

    if not sol:
        print("❌ No feasible schedule")

    schedule = {i: sol[S[i]] for i in jobs}

    with open(sol_file, "w") as f:
        f.write(f"Lmax = {str(sol[Lmax])} \n")
        f.write("Schedule: \n")
        for i, start in sorted(schedule.items(), key=lambda x: x[1]):
            #print(f"Job {i}: start = {start}, end = {start + durations[i]}")
            f.write(f"Job {i}: start = {start}, end = {start + durations[i]} \n")


def main():
    instance_path = r"C:\Users\LamPham\Desktop\Lab\Job_Scheduling\data\datasets\30_05_025_125_00_4.GSP"
    sol_file = r"C:\Users\LamPham\Desktop\Lab\Job_Scheduling\data\solutions_cplex\30_05_025_125_00_4.GSP.txt"

    # -------- Pipeline --------

    # Read dataset
    n, weights, durations, ready_dates, due_dates, deadlines, successors = \
    read_dataset(instance_path)

    # Window tightening
    new_ready_dates, new_deadlines = window_tightening(
    n, ready_dates, durations, deadlines, successors
    )

    solve_Lmax_cplex(n, durations, new_ready_dates, new_deadlines, due_dates, successors, sol_file, 10)

if __name__ == "__main__":
    main()



