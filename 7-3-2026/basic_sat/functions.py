from pysat.formula import CNF
from pysat.solvers import Solver 
from pysat.card import CardEnc, EncType
from collections import deque

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

        succs = parts[1:]

        # bỏ successor > n (job 31)
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


# Solve for SAT
def solve_SAT(n, durations, ready_dates, deadlines, successors, verbose=False):
    jobs = list(range(1, n + 1))
    horizon = max(deadlines.values())

    cnf = CNF()
    var_counter = 1

    # ------------------------
    # 1) S variables
    # ------------------------
    S = {}
    valid_starts = {}

    for i in jobs:
        last_start = deadlines[i] - durations[i]
        if last_start < ready_dates[i]:
            print(f"Job {i} infeasible")
            return None, None, None, None, False

        valid_starts[i] = list(range(ready_dates[i], last_start + 1))
        for t in valid_starts[i]:
            S[(i, t)] = var_counter
            var_counter += 1

    # ------------------------
    # 2) Exactly one start per job
    # ------------------------
    for i in jobs:
        lits = [S[(i, t)] for t in valid_starts[i]]
        enc = CardEnc.equals(lits=lits, bound=1, encoding=EncType.seqcounter, top_id=var_counter-1)
        cnf.extend(enc.clauses)
        var_counter = enc.nv + 1


    # ------------------------
    # 3) No overlap — pairwise
    # ------------------------
    for i in jobs:
        for j in jobs:
            if i >= j:
                continue
            for t_i in valid_starts[i]:
                for t_j in valid_starts[j]:
                    # check if i and j overlap
                    if not (t_i + durations[i] <= t_j or t_j + durations[j] <= t_i):
                        cnf.append([-S[(i, t_i)], -S[(j, t_j)]])

    # ------------------------
    # 4) Precedence
    # ------------------------
    for i in jobs:
        for j in successors.get(i, []):
            for t_i in valid_starts[i]:
                for t_j in valid_starts[j]:
                    if t_j < t_i + durations[i]:
                        cnf.append([-S[(i, t_i)], -S[(j, t_j)]])


    # ------------------------
    # 7) Solve
    # ------------------------
    print("\n=== SOLVING (HARD CONSTRAINTS ONLY) ===")
    solver = Solver(name="g421", bootstrap_with=cnf)
    is_sat = solver.solve()

    if not is_sat:
        if (verbose):
            print("UNSAT — no feasible schedule.")
        return None, None, None, None, is_sat
    
    model = set(solver.get_model())

    # Extract schedule
    schedule = {}
    for (i,t), vid in S.items():
        if vid in model:
            schedule[i] = t

    if (verbose):
        print("Feasible schedule found")
    solver.delete()

    return cnf, schedule, valid_starts, S, is_sat

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

def compute_UB_Lmax(schedule, durations, due_dates):
    Lmax = 0
    for i in schedule:
        L = max(0, schedule[i] + durations[i] - due_dates[i])
        if L > Lmax:
            Lmax = L

    #print("Lmax UB: ", Lmax)

    return Lmax

def incremental_SAT_Lmax(durations, due_dates, S, cnf, UB, sol_file, valid_starts, verbose=False):
    solver = Solver(name='g421', bootstrap_with=cnf)
    iteration_count = 0
    var_to_S = {v: (i, t) for (i, t), v in S.items()}

    print("\n=== SOLVING INCREMENTAL SAT ===")

    while True:
        if UB <= 0:
            break
        iteration_count += 1
        if (verbose):
            print("\n==============================")
            print("Trying with Lmax UB =", UB)
            print("Iteration:", iteration_count)

        # add not S[i,t] that can violate Lmax UB for each iteration
        for (i, t), var in S.items():
            if t + durations[i] - due_dates[i] >= UB:
                solver.add_clause([-var]) 

        if solver.solve():
            if (verbose):
                print("SAT")
            model = solver.get_model()
            best_schedule = {}
            Lmax = 0

            for j in range(0, len(S)):
                if model[j] > 0:
                    i, t = var_to_S[model[j]]
                    Late = max(0, t + durations[i] - due_dates[i])
                    if Late > Lmax:
                        Lmax = Late
                    best_schedule[i] = t

            if (verbose):
                print("New Lmax UB:", Lmax)
                if (UB == Lmax):
                    print("Warning: UB not decreasing, possible bug.")

            UB = Lmax
            with open(sol_file, "w") as f:
                f.write(f"Lmax = {str(UB)} \n")
                f.write("Schedule: \n")
                for i, start in sorted(best_schedule.items(), key=lambda x: x[1]):
                    f.write(f"Job {i}: start = {start}, end = {start + durations[i]} \n")
        else:
            if (verbose):
                print("UNSAT")
            break

    solver.delete()
    print("Incremental SAT finished.")
    print("Best Lmax UB found:", UB)




def main():
    instance_path = r"C:\Users\LamPham\Desktop\Lab\\7-3-2026\datasets\\10-S\\10_05_005_100_25_1.GSP"
    sol_file = r"C:\Users\LamPham\Desktop\Lab\\7-3-2026\solutions_basic_sat\\10-S\\10_05_005_100_25_1.GSP.txt"

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
    incremental_SAT_Lmax(durations, due_dates, S, cnf, UB, sol_file, valid_starts, verbose=True)

if __name__ == "__main__":
    main()