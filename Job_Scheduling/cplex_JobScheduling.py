from docplex.mp.model import Model
from Incremental_SAT_functions import (
    read_dataset,
    window_tightening,
)


def solve_with_cplex(jobs, durations, ready_dates, due_dates,
                     deadlines, weights, successors):
    """
    valid[i] = list of feasible start times for job i
    """

    mdl = Model("1_machine_TWT_compact")

    # -----------------------------
    # Valid start times
    # -----------------------------
    valid = {}
    for i in jobs:
        r, p, dl = ready_dates[i], durations[i], deadlines[i]
        last_start = dl - p
        if last_start < r:
            return None, None
        valid[i] = list(range(r, last_start + 1))

    # -----------------------------
    # Start variables S[i,t]
    # -----------------------------
    S = {
        (i, t): mdl.binary_var(name=f"S_{i}_{t}")
        for i in jobs
        for t in valid[i]
    }

    # Each job starts exactly once
    for i in jobs:
        mdl.add_constraint(
            mdl.sum(S[i, t] for t in valid[i]) == 1,
            ctname=f"start_once_{i}"
        )

    # -----------------------------
    # One-machine constraints (NO A variables)
    # -----------------------------
    all_times = range(
        min(ready_dates.values()),
        max(deadlines.values())
    )

    for tau in all_times:
        mdl.add_constraint(
            mdl.sum(
                S[i, t]
                for i in jobs
                for t in valid[i]
                if t <= tau < t + durations[i]
            ) <= 1,
            ctname=f"machine_{tau}"
        )

    # -----------------------------
    # Completion times
    # -----------------------------
    C = {
        i: mdl.sum((t + durations[i]) * S[i, t] for t in valid[i])
        for i in jobs
    }

    # -----------------------------
    # Precedence constraints
    # -----------------------------
    for i in successors:
        for j in successors[i]:
            mdl.add_constraint(
                C[i] <= mdl.sum(t * S[j, t] for t in valid[j]),
                ctname=f"prec_{i}_{j}"
            )

    # -----------------------------
    # Tardiness variables
    # -----------------------------
    T = {
        i: mdl.continuous_var(lb=0, name=f"T_{i}")
        for i in jobs
    }

    for i in jobs:
        mdl.add_constraint(
            T[i] >= C[i] - due_dates[i],
            ctname=f"tard_{i}"
        )

    # -----------------------------
    # Objective
    # -----------------------------
    mdl.minimize(
        mdl.sum(weights[i] * T[i] for i in jobs)
    )

    # -----------------------------
    # Solve
    # -----------------------------
    sol = mdl.solve(log_output=True)
    if sol is None:
        return None, None

    schedule = {
        i: t for i in jobs for t in valid[i]
        if sol.get_value(S[i, t]) > 0.5
    }

    UB = sum(
        weights[i] * max(0, schedule[i] + durations[i] - due_dates[i])
        for i in jobs
    )

    for i, start in sorted(schedule.items(), key=lambda x: x[1]):
        print(f"Job {i}: start = {start}, end = {start + durations[i]}")

    return UB, schedule

def validate_schedule(
    schedule,
    durations,
    ready_dates,
    deadlines,
    successors
):
    """
    Validate a schedule for hard constraints.

    schedule   : dict {job: start_time}
    jobs       : list of jobs
    durations  : dict {job: p_i}
    ready_dates: dict {job: r_i}
    deadlines  : dict {job: δ_i}
    successors : dict {i: [j1, j2, ...]}

    Returns:
        (is_valid: bool, violations: list of strings)
    """

    jobs = list(range(1, len(schedule) + 1))
    violations = []

    # -------------------------
    # 0) All jobs scheduled?
    # -------------------------
    for i in jobs:
        if i not in schedule:
            violations.append(f"Job {i} has no start time")

    if violations:
        return False, violations

    # -------------------------
    # 1) Release date + deadline
    # -------------------------
    for i in jobs:
        start = schedule[i]
        end = start + durations[i]

        if start < ready_dates[i]:
            violations.append(
                f"Release violation: job {i}, start={start}, r_i={ready_dates[i]}"
            )

        if end > deadlines[i]:
            violations.append(
                f"Deadline violation: job {i}, end={end}, δ_i={deadlines[i]}"
            )

    # -------------------------
    # 2) One-machine constraint
    # -------------------------
    intervals = []
    for i in jobs:
        intervals.append((schedule[i], schedule[i] + durations[i], i))

    intervals.sort()  # sort by start time

    for k in range(len(intervals) - 1):
        s1, e1, i1 = intervals[k]
        s2, e2, i2 = intervals[k + 1]

        if e1 > s2:
            violations.append(
                f"Machine overlap: job {i1} [{s1},{e1}) overlaps job {i2} [{s2},{e2})"
            )

    # -------------------------
    # 3) Precedence constraints
    # -------------------------
    for i in successors:
        for j in successors[i]:
            if schedule[i] + durations[i] > schedule[j]:
                violations.append(
                    f"Precedence violated: {i} ≺ {j}, "
                    f"Ci={schedule[i] + durations[i]}, Sj={schedule[j]}"
                )

    # -------------------------
    # Final result
    # -------------------------
    return len(violations) == 0, violations


def main():
    instance_path = r"C:\Users\LamPham\Desktop\Lab\Job_Scheduling\datasets\S\30_10_005_125_25_2.GSP"
    jobs = list(range(1, 31))

    # Read dataset
    n, weights, durations, ready_dates, due_dates, deadlines, successors = \
        read_dataset(instance_path)

    # Window tightening
    new_ready_dates, new_deadlines = window_tightening(
        n, ready_dates, durations, deadlines, successors
    )

    #print(new_deadlines)

    UB, schedule = solve_with_cplex(jobs, durations, new_ready_dates, due_dates, new_deadlines, weights, successors)
    print(UB)

    num_violations, violations = validate_schedule(schedule, durations, ready_dates, deadlines, successors)
    print("Violations: ", violations)

if __name__ == "__main__":
    main()