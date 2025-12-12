from pypblib import pblib
from pysat.solvers import Solver


def main():

    demo_incremental_sat()


def demo_incremental_sat():
    n = 20
    MAX_ITERATION = 100
    config = pblib.PBConfig()
    pb2 = pblib.Pb2cnf(config)
    formula = []

    literals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    weights_hard = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    weights = [1, 2, 3, 4, 1, 2, 3, 1, 2, 3, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
    # weights = weights_hard
    first_free_var = n+1


    # Hard constraint: sum weights_hard * literals >= 6
    max_vars = pb2.encode_geq(weights, literals, 6, formula, first_free_var)



    # Start optimizing

    solver = Solver(name='g421')

    for clause in formula:
        solver.add_clause(clause)

    total_weight = 0

    print("Solving first problem:")
    if solver.solve():
        print("SAT")
        print(solver.get_model())

        for i in range(n):
            if solver.get_model()[i] > 0:
                total_weight += weights[i]
                # print(f"{i} weight:", weights[i])
        print("Total weight:", total_weight)

    else:
        print("UNSAT")


    print("Starting incremental SAT")

    UB = total_weight
    last_UB = -1


    # Try to find solutions with total_weight <= UB
    print("max_vars:", max_vars)
    first_free_var = max_vars
    x_vars = []
    for i in range(UB):

        first_free_var += 1
        print(f"X[{i}]:", first_free_var)
        x_vars.append(first_free_var)
        # x_vars.append(max_vars + i + 1)


    print("x_vars:", x_vars)

    # Create clause for x_vars: x[i] -> x[i+1]

    for i in range(UB - 1):
        solver.add_clause([-x_vars[i], x_vars[i + 1]])

    iteration_count = 0

    lits = literals.copy()
    lits.extend(x_vars)

    weight_for_lits = weights.copy()
    weight_for_lits.extend([1] * UB)

    formula = []
    max_vars = pb2.encode_leq(weight_for_lits, lits, UB, formula, first_free_var)
    for clause in formula:
        solver.add_clause(clause)

    first_free_var = max_vars

    #
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
            print(solver.get_model())

            total_weight = 0
            for i in range(n):
                if solver.get_model()[i] > 0:
                    total_weight += weights[i]
                    # print(f"{literals[i]} weight:", weights[i])
            print("Total weight:", total_weight)
            # last_UB = total_weight
            UB = total_weight
        else:
            print("UNSAT")
            break

    print("Incremental SAT finished.")
    print("Best UB found:", UB)


if __name__ == "__main__":
    main()
