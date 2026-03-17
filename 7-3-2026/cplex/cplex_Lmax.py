from docplex.mp.model import Model
import numpy as np
from collections import deque
import pandas as pd
from docplex.cp.model import CpoModel

### Read in dataset
def read_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    idx = 0

    # n
    assert lines[idx].startswith("n")
    idx += 1
    n = int(lines[idx])
    idx += 1

    # helper to read a list of n integers
    def read_values(label):
        nonlocal idx
        assert label in lines[idx]
        idx += 1
        values = list(map(int, lines[idx].split()))
        idx += 1
        return {i + 1: values[i] for i in range(n)}

    # read main data
    weights = read_values("weight")
    durations = read_values("duration")
    due_dates = read_values("due date")
    ready_dates = read_values("ready date")
    deadlines = read_values("deadline")

    # precedence relations
    assert "precedence relations" in lines[idx]
    idx += 1

    successors = {i: [] for i in range(1, n + 1)}

    for job in range(1, n + 1):
        parts = list(map(int, lines[idx].split()))
        idx += 1

        num_succ = parts[0]
        succs = parts[1:]

        # bỏ successor > n (ví dụ job 31)
        successors[job] = [s for s in succs if s <= n]

    return n, durations, ready_dates, due_dates, deadlines, successors


# Tighten time window using precedence constraints
def window_tightening(n, ready_dates, durations, deadlines, successors):
    predecessors = {i: [] for i in range(1, n+1)}
    for i in range(1, n+1):
        for s in successors[i]:
            predecessors[s].append(i)

    indeg = {i: len(predecessors[i]) for i in range(1, n+1)}
    q = deque([i for i in range(1, n+1) if indeg[i] == 0])
    topo = []

    while q:
        u = q.popleft()
        topo.append(u)
        for v in successors[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)

    if len(topo) != n:
        raise ValueError("Precedence graph has a cycle!")

    new_ready_dates = {i: ready_dates[i] for i in range(1, n+1)}
    for i in topo:
        if predecessors[i]:
            new_ready_dates[i] = max(ready_dates[i], max(new_ready_dates[j] + durations[j] for j in predecessors[i]))

    new_deadlines = {i: deadlines[i] for i in range(1, n+1)}

    for i in reversed(topo):
        if successors[i]:
            new_deadlines[i] = min(new_deadlines[i], min(new_deadlines[s] - durations[s] for s in successors[i]))

    return new_ready_dates, new_deadlines

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
        return violations

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
    return violations

def solve_MP(n, durations, ready_dates, deadlines, due_dates, successors, sol_file, time_limit):
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
        print("No feasible schedule")
        return

    schedule = {i: int(np.round(sol[S[i]])) for i in jobs}

    
    with open(sol_file, "w") as f:
        f.write(f"Lmax = {int(np.round(sol[Lmax]))} \n")
        f.write("Schedule: \n")
        for i, start in sorted(schedule.items(), key=lambda x: x[1]):
            f.write(f"Job {i}: start = {start}, end = {start + durations[i]} \n")



def solve_CP(n, durations, ready_dates, deadlines, due_dates, successors, sol_file, time_limit):
    jobs = list(range(1, n + 1))
    
    model = CpoModel()
    
    # ------------------------
    # 1) Decision variables — interval var per job
    # ------------------------
    tasks = {}
    for i in jobs:
        tasks[i] = model.interval_var(
            size=durations[i],
            start=(ready_dates[i], deadlines[i] - durations[i]),  # start in [ready, last_valid_start]
            end=(ready_dates[i] + durations[i], deadlines[i]),     # end in [earliest_end, deadline]
            name=f"task_{i}"
        )
    
    # ------------------------
    # 2) No overlap — single machine
    # ------------------------
    model.add(model.no_overlap(tasks.values()))

    # ------------------------
    # 3) Precedence constraints
    # ------------------------
    for i in jobs:
        for j in successors.get(i, []):
            model.add(model.end_before_start(tasks[i], tasks[j]))

    # ------------------------
    # 4) Minimize Lmax
    # ------------------------
    lmax = model.integer_var(min=0, name="Lmax")
    for i in jobs:
        tardiness = model.max([0, model.end_of(tasks[i]) - due_dates[i]])
        model.add(lmax >= tardiness)
    
    model.minimize(lmax)

    # ------------------------
    # 5) Solve
    # ------------------------
    print("\n=== SOLVING CP ===")
    solution = model.solve(TimeLimit=time_limit, LogVerbosity="Quiet")

    if solution:
        schedule = {}
        for i in jobs:
            start = solution.get_var_solution(tasks[i]).get_start()
            schedule[i] = start

        lmax_value = solution.get_var_solution(lmax).get_value()
        print(f"Lmax = {lmax_value}")

        violations = validate_schedule(schedule, durations, ready_dates, deadlines, successors)
        if (len(violations) > 0):
            print(violations)
            return
        
        with open(sol_file, "w") as f:
            f.write(f"Lmax = {lmax_value} \n")
            f.write("Schedule: \n")
            for i, start in sorted(schedule.items(), key=lambda x: x[1]):
                f.write(f"Job {i}: start = {start}, end = {start + durations[i]} \n")
    else:
        print("No solution found")
        return

def main():
    instance_path = r"C:\Users\LamPham\Desktop\Lab\\7-3-2026\datasets\\40-S\\40_05_005_125_25_1.GSP"
    sol_file = r"C:\Users\LamPham\Desktop\Lab\\7-3-2026\solutions_cplex\\40-S\\40_05_005_125_25_1.GSP.txt"

    # -------- Pipeline --------

    # Read dataset
    n, durations, ready_dates, due_dates, deadlines, successors = read_dataset(instance_path)

    # Window tightening
    new_ready_dates, new_deadlines = window_tightening(n, ready_dates, durations, deadlines, successors)

    # Solve
    solve_CP(n, durations, new_ready_dates, new_deadlines, due_dates, successors, sol_file, 300)

if __name__ == "__main__":
    main()



