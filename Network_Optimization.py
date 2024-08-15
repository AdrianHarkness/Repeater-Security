#!/usr/bin/env python3

import cvxpy as cp
import numpy as np

#Adversarial Error Maximization
def w_tilde(Qx, delta_prime, p, m):
    # Ensure that p and m have the correct length
    assert len(p) == len(m), "Length of p must be equal to length of m"
    assert np.isclose(np.sum(m), 1), "Sum of m must be 1"
    assert np.all(p >= 0), "All elements of p must be greater than or equal to 0"
    assert np.all(p < 1/2), "All elements of p must be less than 1/2"
    assert np.all(m >= 0), "All elements of m must be greater than or equal to 0"
    assert np.all(m <= 1), "All elements of m must be less than or equal to 1"
    K = len(p)

    # Define the optimization variables
    w = cp.Variable()
    x = cp.Variable(K)

    # Define the constraints
    constraints = [
        x >= 0,
        x <= 1,
        w == cp.sum(cp.multiply(m,x)),

        #v1
        # Qx - cp.sum(m * p) - w + 2 * cp.sum(cp.multiply(m, cp.multiply(x, p))) <= delta_prime,
        # Qx - cp.sum(m * p) - w + 2 * cp.sum(cp.multiply(m, cp.multiply(x, p))) >= -delta_prime

        #v2
        # Qx - w - cp.sum(cp.multiply(m, cp.multiply((1-x), p))) <= delta_prime,
        # Qx - w - cp.sum(cp.multiply(m, cp.multiply((1-x), p))) >= -delta_prime

        #v3
        Qx + cp.sum(cp.multiply(m, (2*cp.multiply(x,p) - x - p)))  <= delta_prime,
        Qx + cp.sum(cp.multiply(m, (2*cp.multiply(x,p) - x - p))) >= -delta_prime

    ]

    # Define the objective function
    objective = cp.Maximize(w)

    # Formulate the problem
    prob = cp.Problem(objective, constraints)

    # Solve the problem
    prob.solve()

    #Print the results
    # if prob.status == cp.OPTIMAL or prob.status == cp.OPTIMAL_INACCURATE:
    #     print("Optimal Value: w = ", w.value)
    #     #print("Optimal Solution: x = ", x.value)
    # else:
    #     print("Something went wrong!")
    #     print(prob.status)

    if K == 1:
        assert np.isclose(w.value, (Qx - np.sum(p) + delta_prime) / (1 - 2 * np.sum(p)), atol=1e-2), \
            f"w = {w.value} is not equal to (Qx - np.sum(p) + delta_prime) / (1 - 2 * np.sum(p)) = {(Qx - np.sum(p) + delta_prime) / (1 - 2 * np.sum(p))}"
    #If number of signals too small, delta_prime gets too large and w will be greater than Qx!!
    # elif K > 1:
    #     assert w.value <= Qx, f"w = {w.value} is greater than Qx = {Qx}"

    return w.value