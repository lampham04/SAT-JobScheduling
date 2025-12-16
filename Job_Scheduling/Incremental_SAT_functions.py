from pysat.formula import CNF
from pysat.solvers import Solver   
from pysat.card import CardEnc, EncType
from pypblib import pblib
from collections import deque
import pandas as pd

# READ filename
def read_filenames(path):
    df = pd.read_excel(path, engine="xlrd")

    df["OPT VALUE"] = pd.to_numeric(df["OPT VALUE"], errors="coerce")

    df_filtered = df[(df["OPT VALUE"] <= 1000) & (df["PT"] == "S")]
    df_filtered = df_filtered.reset_index(drop=True)

    filenames = df_filtered["filename"].tolist()

    return df_filtered, filenames


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

    return n, weights, durations, ready_dates, due_dates, deadlines, successors


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
def solve_SAT(n, durations, ready_dates, deadlines, successors):
    """
    SAT encoding using manual var_counter and CardEnc with top_id.
    Returns (sat_bool, schedule_dict_or_None).
    """

    jobs = list(range(1, n+1))
    horizon = max(deadlines.values())   # we will consider time points 0..horizon-1 for activity

    cnf = CNF()
    var_counter = 1    # next free var id

    # Dictionaries mapping (i,t) -> var_id
    S = {}
    A = {}
    SC = {}
    valid = {}

    # ------------------------
    # 1) CREATE S and A variables (manual numbering)
    # ------------------------
    for i in jobs:
        r_i = ready_dates[i]
        p_i = durations[i]
        dl_i = deadlines[i]

        last_start = dl_i - p_i
        if last_start < r_i:
            print(f"Job {i} impossible: last_start < ready ({last_start} < {r_i})")
            return False, None

        valid[i] = list(range(r_i, last_start + 1))

        for t in valid[i]:
            S[(i,t)] = var_counter
            var_counter += 1

    for i in jobs:
        r_i = ready_dates[i]
        p_i = durations[i]
        dl_i = deadlines[i]

        last_start = dl_i - p_i
        if last_start < r_i:
            print(f"Job {i} impossible: last_start < ready ({last_start} < {r_i})")
            return False, None

        # A defined for t in [r_i, dl_i-1]
        for t in range(r_i, dl_i):
            A[(i,t)] = var_counter
            var_counter += 1

    print(f"Created S variables: {len(S)}, A variables: {len(A)}. Next var id = {var_counter}")

    # ------------------------
    # 2) Exactly-one start per job using CardEnc.equals with top_id
    # ------------------------
    total_start_once_clauses = 0
    for i in jobs:
        lits = [S[(i,t)] for t in valid[i]]
        enc = CardEnc.equals(lits=lits, bound=1, encoding=EncType.seqcounter, top_id=var_counter-1)
        cnf.extend(enc.clauses)
        total_start_once_clauses += len(enc.clauses)
        # update var_counter to next free id
        var_counter = enc.nv + 1
    print("Start-once clauses (total): ", total_start_once_clauses, ". Next var id = ", var_counter)

    # ------------------------
    # 3) Activation S -> A
    # ------------------------
    s_to_a_clauses = 0
    for i in jobs:
        p_i = durations[i]
        for t0 in valid[i]:
            s_lit = S[(i,t0)]
            for t in range(t0, t0 + p_i):
                if (i,t) in A:
                    # clause: ¬S[i,t0] ∨ A[i,t]
                    cnf.append([-s_lit, A[(i,t)]])
                    s_to_a_clauses += 1
    print("S->A clauses: ", s_to_a_clauses)

    # ------------------------
    # 4) Capacity: at most one active at each time t using CardEnc.atmost with top_id
    # ------------------------
    cap_clauses = 0
    for t in range(horizon):
        active_vars = [A[(i,t)] for i in jobs if (i,t) in A]
        if len(active_vars) > 1:
            enc = CardEnc.atmost(lits=active_vars, bound=1, encoding=EncType.seqcounter, top_id=var_counter-1)
            cnf.extend(enc.clauses)
            cap_clauses += len(enc.clauses)
            var_counter = enc.nv + 1
    print("Capacity clauses (total): ", cap_clauses, ". Next var id = ", var_counter)

    # ------------------------
    # 5) Build SC prefix variables and clauses (SC[j,t] = ∃u ≤ t: S[j,u] = 1)
    #    We'll define SC[j,t] on the interval [min(valid[j]), max(valid[j])] for each j.
    #    Clauses:
    #      (A) S[j,u] -> SC[j,u]
    #      (B) SC[j,t] -> SC[j,t+1]
    #      (C) SC[j,t] -> OR_{u ≤ t} S[j,u]  (reverse implication to prevent SC=true without S)
    #    We do NOT force SC[last] = 1 (exact-one on S ensures existence).
    # ------------------------
    sc_count = 0
    s_to_sc_count = 0
    sc_chain_count = 0
    sc_reverse_count = 0

    for j in jobs:
        times = valid[j]
        if not times:
            continue
        t_min = times[0]
        t_maxj = times[-1]

        # create SC variables for t_min..t_maxj
        for t in range(t_min, t_maxj + 1):
            SC[(j,t)] = var_counter
            var_counter += 1
            sc_count += 1

        # (A) S -> SC at same t
        for t in times:
            cnf.append([-S[(j,t)], SC[(j,t)]])
            s_to_sc_count += 1

        # (B) SC[t] -> SC[t+1]
        for t in range(t_min, t_maxj):
            cnf.append([-SC[(j,t)], SC[(j,t+1)]])
            sc_chain_count += 1

        # (C) SC[t] -> OR_{u ≤ t} S[j,u]
        # Build the reverse clauses to ensure SC can't be true unless some S[u] true
        for t in range(t_min, t_maxj + 1):
            ors = [S[(j,u)] for u in times if u <= t]
            if not ors:
                # no starts <= t (shouldn't happen), skip
                continue
            clause = [-SC[(j,t)]] + ors
            cnf.append(clause)
            sc_reverse_count += 1

    print(f"SC vars created: {sc_count}, S->SC: {s_to_sc_count}, SC chain: {sc_chain_count}, SC reverse: {sc_reverse_count}")
    print("next var id after SC creation: ", var_counter)
    # ------------------------
    # 6) Precedence constraints: For i -> j, forbid SC[j,finish] when S[i,t_i] = 1
    #    Clause: ¬S[i,t_i] ∨ ¬SC[j, finish]  only when finish in SC domain
    # ------------------------
    prec_clauses = 0
    for i in jobs:
        for j in successors.get(i, []):
            if not valid[i] or not valid[j]:
                # infeasible handled earlier
                continue
            t_min_j = valid[j][0]
            t_max_j = valid[j][-1]
            p_i = durations[i]

            for t_i in valid[i]:
                finish = t_i + p_i - 1
                # Only add clause if finish lies within SC domain for j
                if finish < t_min_j or finish > t_max_j:
                    continue
                cnf.append([-S[(i,t_i)], -SC[(j, finish)]])
                prec_clauses += 1
    print("Precedence clauses: ", prec_clauses)


    # ------------------------
    # 7) Solve
    # ------------------------
    print("\n=== SOLVING (HARD CONSTRAINTS ONLY) ===")
    solver = Solver(name="g421", bootstrap_with=cnf)
    is_sat = solver.solve()

    if not is_sat:
        print("UNSAT — no feasible schedule.")
        return None, None, None, None, None. is_sat
    
    model = set(solver.get_model())

    # Extract schedule
    schedule = {}
    for (i,t), vid in S.items():
        if vid in model:
            schedule[i] = t

    # # print schedule ordered by start
    # order = sorted(schedule.items(), key=lambda x: x[1])
    print("\nFeasible schedule found:")
    solver.delete()
    # print("{:<8} {:<10} {:<10}".format("Job", "Start", "End"))
    # print("-"*32)
    # for i, st in order:
    #     en = st + durations[i]
    #     print("{:<8} {:<10} {:<10}".format(i, st, en))
    # print("\nJob order:", [i for i,_ in order])

    return cnf, var_counter, schedule, valid, S, is_sat


def compute_UB(schedule, durations, weights, due_dates):
    UB = 0
    for i in schedule:
        tard = max(0, schedule[i] + durations[i] - due_dates[i])
        UB += (tard * weights[i])

    print("UB: ", UB)

    return UB


def incremental_SAT(weights, durations, due_dates, S, cnf, UB, valid, next_var_id, ub_file):
    config = pblib.PBConfig()
    pb2 = pblib.Pb2cnf(config)
    formula = []

    tardiness_lits = []
    tardiness_weights = []

    for i, times in valid.items():
        w_i = weights.get(i, 1)
        p_i = durations[i]
        d_i = due_dates[i]

        for t in times:
            T_i_t = max(0, t + p_i - d_i)
            if T_i_t > 0:
                tardiness_lits.append(S[(i,t)])
                tardiness_weights.append(w_i * T_i_t)

    x_vars = []
    for _ in range(UB):
        x_vars.append(next_var_id)
        next_var_id += 1

    for i in range(len(x_vars)-1):
        cnf.append([-x_vars[i], x_vars[i+1]])

    lits = tardiness_lits + x_vars
    new_weights = tardiness_weights + [1] * len(x_vars)

    print("Encoding UB constraints")
    max_var = pb2.encode_leq(new_weights, lits, UB, formula, next_var_id)
    print("UB constraints encoded")
    cnf.extend(formula)
    next_var_id = max_var

    solver = Solver(name='g421', bootstrap_with=cnf)
    MAX_ITERATION = 100
    iteration_count = 0
    var_to_S = {v: (i, t) for (i, t), v in S.items()}


    while True:
        if iteration_count == MAX_ITERATION or UB <= 0:
            break
        iteration_count += 1
        print("\n==============================")
        print("Trying with UB =", UB)
        print("Iteration:", iteration_count)

        solver.add_clause([x_vars[UB-1]])

        if solver.solve():
            print("SAT")
            model = solver.get_model()
            total_weight = 0

            for j in range(0, len(S)):
                if model[j] > 0:
                    i, t = var_to_S[model[j]]
                    tardy = max(0, t + durations[i] - due_dates[i])
                    total_weight += tardy * weights[i]
            print("New UB:", total_weight)
            UB = total_weight
            with open(ub_file, "w") as f:
                f.write(str(UB))
        else:
            print("UNSAT")
            break

    solver.delete()
    print("Incremental SAT finished.")
    print("Best UB found:", UB)

