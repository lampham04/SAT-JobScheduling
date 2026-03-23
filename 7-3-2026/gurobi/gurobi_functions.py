from collections import deque
from gurobipy import Model, GRB
import numpy as np

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

def solve_gurobi(n, durations, ready_dates, deadlines, due_dates, successors, sol_file, time_limit):
    jobs = list(range(1, n + 1))

    model = Model("job_scheduling")
    model.setParam("OutputFlag", 0)
    model.setParam("TimeLimit", time_limit)

    # ------------------------
    # 1) Decision variables
    # ------------------------
    start = {}
    for i in jobs:
        start[i] = model.addVar(
            lb=ready_dates[i],
            ub=deadlines[i] - durations[i],
            vtype=GRB.CONTINUOUS,  # match CPLEX MP continuous
            name=f"start_{i}"
        )

    lmax = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="Lmax")

    # ------------------------
    # 2) No overlap
    # ------------------------
    for i in jobs:
        for j in jobs:
            if i < j:
                z = model.addVar(vtype=GRB.BINARY, name=f"z_{i}_{j}")
                M_ij = max(
                    deadlines[i],
                    deadlines[j]
                )
                model.addConstr(start[i] + durations[i] <= start[j] + M_ij * (1 - z))
                model.addConstr(start[j] + durations[j] <= start[i] + M_ij * z)

    # ------------------------
    # 3) Precedence
    # ------------------------
    for i in jobs:
        for j in successors.get(i, []):
            model.addConstr(start[j] >= start[i] + durations[i])

    # ------------------------
    # 4) Release dates and deadlines
    # ------------------------
    for i in jobs:
        model.addConstr(start[i] >= ready_dates[i])
        model.addConstr(start[i] + durations[i] <= deadlines[i])

    # ------------------------
    # 5) Lmax
    # ------------------------
    for i in jobs:
        model.addConstr(lmax >= start[i] + durations[i] - due_dates[i])

    model.setObjective(lmax, GRB.MINIMIZE)

    # ------------------------
    # 6) Solve
    # ------------------------
    print("\n=== SOLVING GUROBI ===")
    model.optimize()

    if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        schedule = {i: int(np.round(start[i].X)) for i in jobs}
        lmax_val = int(np.round(lmax.X))
        print(f"Lmax = {lmax_val}")

        with open(sol_file, "w") as f:
            f.write(f"Lmax = {lmax_val} \n")
            f.write("Schedule: \n")
            for i, s in sorted(schedule.items(), key=lambda x: x[1]):
                f.write(f"Job {i}: start = {s}, end = {s + durations[i]} \n")

        return schedule, lmax_val
    else:
        print("No solution found")
        return None, None

def main():
    instance_path = r"C:\Users\LamPham\Desktop\Lab\\7-3-2026\datasets\\10-S\\10_05_005_100_25_1.GSP"
    sol_file = r"C:\Users\LamPham\Desktop\Lab\\7-3-2026\solutions_gurobi\\10-S\\10_05_005_100_25_1.GSP.txt"

    # -------- Pipeline --------

    # Read dataset
    n, durations, ready_dates, due_dates, deadlines, successors = read_dataset(instance_path)

    # Window tightening
    #new_ready_dates, new_deadlines = window_tightening(n, ready_dates, durations, deadlines, successors)

    # Solve
    solve_gurobi(n, durations, ready_dates, deadlines, due_dates, successors, sol_file, 300)

if __name__ == "__main__":
    main()



