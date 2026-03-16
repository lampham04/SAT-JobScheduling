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
    jobs = list(range(1, n+1))
    horizon = max(deadlines.values())

    cnf = CNF()
    var_counter = 1    # next free var id

    # Dictionaries mapping (i,t) -> var_id
    S = {}
    A = {}
    L = {}
    valid_starts = {}

    # ------------------------
    # 1) CREATE S and A variables
    # ------------------------
    for i in jobs:
        last_start = deadlines[i] - durations[i]
        if last_start < ready_dates[i]:
            if (verbose):
                print(f"Job {i} impossible: last_start < ready ({last_start} < {ready_dates[i]})")
            return None, None, None, None, None, False

        valid_starts[i] = list(range(ready_dates[i], last_start + 1))

        for t in valid_starts[i]:
            # S variables
            S[(i,t)] = var_counter
            var_counter += 1

    for i in jobs:
        # A variables
        for t in range(ready_dates[i], deadlines[i]):
            A[(i,t)] = var_counter
            var_counter += 1

    if (verbose):
        print(f"Created S variables: {len(S)}, A variables: {len(A)}. Next var id = {var_counter}")

    # ------------------------
    # 2) Activation S -> A
    # ------------------------
    s_to_a_clauses = 0
    for i in jobs:
        p_i = durations[i]
        for t0 in valid_starts[i]:
            s_lit = S[(i,t0)]
            for t in range(t0, t0 + p_i):
                if (i,t) in A:
                    # clause: ¬S[i,t0] ∨ A[i,t]
                    cnf.append([-s_lit, A[(i,t)]])
                    s_to_a_clauses += 1

    if (verbose):
        print("S->A clauses:", s_to_a_clauses)

    # ------------------------
    # 3) Capacity: at most one job active at each time t 
    # ------------------------
    cap_clauses = 0
    for t in range(horizon):
        active_vars = [A[(i,t)] for i in jobs if (i,t) in A]
        if len(active_vars) > 1:
            enc = CardEnc.atmost(lits=active_vars, bound=1, encoding=EncType.seqcounter, top_id=var_counter-1)
            cnf.extend(enc.clauses)
            cap_clauses += len(enc.clauses)
            var_counter = enc.nv + 1

    if(verbose):
        print(f"Capacity clauses: {cap_clauses}. Next var id = {var_counter}")

    # ------------------------
    # 5) Build L variable then link it to S. This also enforces exactly one start time for each job
    # ------------------------
    l_count = 0
    l_clauses = 0

    for j in jobs:
        times = valid_starts[j]
        if not times:
            continue
        t_min = times[0]
        t_max = times[-1]

        # create L variables for t_min..t_max
        for t in range(t_min, t_max + 1):
            L[(j,t)] = var_counter
            var_counter += 1
            l_count += 1

        cnf.append([L[(j, t_max)]]) # L_i, t_max = 1
        cnf.append([-L[(j, t_min)], S[(j, t_min)]]) # L_i, 1 -> S_i, 1
        l_clauses += 2

        for t in range(t_min, t_max + 1):
            cnf.append([L[(j, t)], -S[(j ,t)]]) # S_i, t -> L_i, t
            l_clauses += 1

        for t in range(t_min + 1, t_max + 1):
            cnf.append([-S[(j, t)], -L[(j, t - 1)]])
            cnf.append([L[(j, t)], -L[(j, t - 1)]])
            cnf.append([-L[(j, t)], L[(j, t - 1)], S[(j, t)]]) # Si, t <-> Li, t ^ -Li, t-1
            l_clauses += 3

    if (verbose):
        print("L variables: ", l_count)
        print("L clauses: ", l_clauses)

    # ------------------------
    # 6) Precedence constraints: For i -> j, forbid L[j,t_i + p_i - 1] when S[i,t_i] = 1
    #    only when i finish in j's valid_starts range
    # ------------------------
    prec_clauses = 0
    for i in jobs:
        for j in successors.get(i, []):
            if not valid_starts[i] or not valid_starts[j]:
                # infeasible handled earlier
                continue
            t_min_j = valid_starts[j][0]
            t_max_j = valid_starts[j][-1]
            p_i = durations[i]

            for t_i in valid_starts[i]:
                finish = t_i + p_i - 1
                # Only add clause if finish lies within j's valid_starts range
                if finish < t_min_j or finish > t_max_j:
                    continue
                cnf.append([-S[(i,t_i)], -L[(j, finish)]])
                prec_clauses += 1

    if (verbose):
        print("Precedence clauses:", prec_clauses)

        print("Total clauses: ", prec_clauses + l_clauses + cap_clauses + s_to_a_clauses )
        print("Total variables: ", var_counter - 1)

    # ------------------------
    # 7) Solve
    # ------------------------
    print("\n=== SOLVING (HARD CONSTRAINTS ONLY) ===")
    solver = Solver(name="g421", bootstrap_with=cnf)
    is_sat = solver.solve()

    if not is_sat:
        if (verbose):
            print("UNSAT — no feasible schedule.")
        return None, None, None, None, None, is_sat
    
    model = set(solver.get_model())

    # Extract schedule
    schedule = {}
    for (i,t), vid in S.items():
        if vid in model:
            schedule[i] = t

    if (verbose):
        print("Feasible schedule found")
    solver.delete()

    return cnf, schedule, valid_starts, S, L, is_sat

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

def incremental_SAT_Lmax(durations, due_dates, S, L, cnf, UB, sol_file, valid_starts, verbose=False):
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

        for j in range(1, len(durations) + 1):
            if(due_dates[j] + UB - durations[j] - 1 < valid_starts[j][-1]):
                solver.add_clause([L[(j, due_dates[j] + UB - durations[j] - 1)]])

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


