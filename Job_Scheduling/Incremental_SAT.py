from pysat.formula import CNF
from pysat.solvers import Glucose3   # or Glucose42 if you prefer
from pysat.card import CardEnc, EncType
from pypblib import pblib


def solve_with_topid(n, durations, ready_dates, deadlines, successors):
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

        # A defined for t in [r_i, dl_i-1]
        for t in range(r_i, dl_i):
            A[(i,t)] = var_counter
            var_counter += 1

    print(f"Created S variables: {len(S)}, A variables: {len(A)}. next var id = {var_counter}")

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
    print("Start-once clauses (total):", total_start_once_clauses, "next var id =", var_counter)

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
    print("S->A clauses:", s_to_a_clauses)

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
    print("Capacity clauses (total):", cap_clauses, "next var id =", var_counter)

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
    print("next var id after SC creation:", var_counter)
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
    print("Precedence clauses:", prec_clauses)


    # ------------------------
    # 7) Solve
    # ------------------------
    print("\n=== SOLVING (HARD CONSTRAINTS ONLY) ===")
    solver = Glucose3()
    solver.append_formula(cnf)
    sat = solver.solve()

    if not sat:
        print("UNSAT — no feasible schedule.")
        return None

    model = set(solver.get_model())

    # Extract schedule
    schedule = {}
    for (i,t), vid in S.items():
        if vid in model:
            schedule[i] = t

    # detect missing starts
    missing = [i for i in jobs if i not in schedule]
    if missing:
        print("WARNING: some jobs missing start times:", missing)

    # print schedule ordered by start
    order = sorted(schedule.items(), key=lambda x: x[1])
    print("\nFeasible schedule found:")
    print("{:<8} {:<10} {:<10}".format("Job", "Start", "End"))
    print("-"*32)
    for i, st in order:
        en = st + durations[i]
        print("{:<8} {:<10} {:<10}".format(i, st, en))
    print("\nJob order:", [i for i,_ in order])

    return cnf, var_counter, schedule, valid, S


def compute_UB(schedule, durations, weights, due_dates):
    UB = 0
    for i in schedule:
        tard = max(0, schedule[i] + durations[i] - due_dates[i])
        UB += (tard * weights[i])

    print("UB: ", UB)

    return UB


def incremental_SAT(weights, durations, due_dates, S, cnf, UB, valid, next_var_id):
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
    weights = tardiness_weights + [1] * len(x_vars)

    print("Encoding 1")
    max_var = pb2.encode_leq(weights, lits, UB, formula, next_var_id)
    print("encoding 2")
    for clause in formula:
        cnf.extend(clause)
    next_var_id = max_var

    print(next_var_id)

n = 30

weights = {
    1: 2,   2: 5,   3: 3,   4: 23,  5: 17,
    6: 8,   7: 12,  8: 1,   9: 4,  10: 5,
    11: 7,  12: 2,  13: 4,  14: 7,  15: 20,
    16: 5,  17: 22, 18: 7,  19: 4,  20: 1,
    21: 14, 22: 0,  23: 26, 24: 16, 25: 6,
    26: 15, 27: 8,  28: 4,  29: 8,  30: 4
}

durations = {
    1: 18,  2: 71,  3: 1,   4: 4,   5: 30,
    6: 97,  7: 73,  8: 83,  9: 34, 10: 22,
    11: 33, 12: 30, 13: 35, 14: 63, 15: 25,
    16: 84, 17: 87, 18: 83, 19: 46, 20: 63,
    21: 76, 22: 60, 23: 8,  24: 85, 25: 85,
    26: 26, 27: 83, 28: 86, 29: 36, 30: 82
}


ready_dates = {
    1: 4,     2: 28,    3: 85,    4: 69,    5: 373,
    6: 397,   7: 413,   8: 425,   9: 410,   10: 478,
    11: 696,  12: 768,  13: 812,  14: 902,  15: 903,
    16: 906,  17: 1012, 18: 1113, 19: 1163, 20: 1079,
    21: 1335, 22: 1294, 23: 1330, 24: 1497, 25: 1432,
    26: 1587, 27: 1338, 28: 1341, 29: 929,  30: 982
}


due_dates = {
    1: 83,    2: 619,   3: 649,   4: 724,   5: 1117,
    6: 565,   7: 1237,  8: 593,   9: 1001, 10: 677,
    11: 1370, 12: 1167, 13: 1348, 14: 1175, 15: 1000,
    16: 1212, 17: 1238, 18: 1990, 19: 1908, 20: 1323,
    21: 1915, 22: 2143, 23: 2037, 24: 2135, 25: 1981,
    26: 1870, 27: 2029, 28: 1641, 29: 1437, 30: 1084
}

deadlines = {
    1: 2201,  2: 1270,  3: 1213,  4: 2356,  5: 2392,
    6: 636,   7: 1773,  8: 1362,  9: 3116, 10: 2786,
    11: 2037, 12: 2393, 13: 3582, 14: 1627, 15: 2062,
    16: 3511, 17: 3599, 18: 4220, 19: 2048, 20: 2703,
    21: 2856, 22: 3939, 23: 3812, 24: 3881, 25: 3363,
    26: 3804, 27: 2312, 28: 3626, 29: 3426, 30: 2384
}


successors = {
    1: [2],
    2: [8, 7, 3],
    3: [11, 6],
    4: [8, 6],
    5: [6],
    6: [19, 12, 9],
    7: [19, 12, 9],
    8: [19, 9],
    9: [23, 17, 15, 14, 13],
    10: [22, 19, 17, 14, 13],
    11: [23, 19, 17, 13],
    12: [23, 16, 13],
    13: [21, 20, 18],
    14: [21, 20, 18],
    15: [22, 20, 18],
    16: [22, 21, 18],
    17: [21, 18],
    18: [28, 25, 24],
    19: [24, 21],
    20: [25, 24],
    21: [27, 25],
    22: [30, 24],
    23: [30, 24],
    24: [27, 26],
    25: [30, 26],
    26: [29],
    27: [29],
    28: [29],
    29: [],
    30: []
}

predecessors = {i: [] for i in range(1, n+1)}
for i in range(1, n+1):
    for s in successors[i]:
        predecessors[s].append(i)

from collections import deque

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


cnf, next_var, schedule, valid, S = solve_with_topid(n, durations, new_ready_dates, new_deadlines, successors)
UB = compute_UB(schedule, durations, weights, due_dates)
incremental_SAT(weights, durations, due_dates, S, cnf, UB, valid, next_var)