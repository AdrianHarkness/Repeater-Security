#!/usr/bin/env python3

"""
Example usage: ./Network_Optimization.py 0.08 0.005 0.01 0.02 0.03 0.04 0.05 0.06 0.2 0.2 0.2 0.2 0.1 0.1

This corresponds to Qx = .08, delta = .005, p =  [0.01 0.02 0.03 0.04 0.05 0.06], and m = [0.2 0.2 0.2 0.2 0.1 0.1]
"""

import cvxpy as cp
import numpy as np
import sys

def main(Qx, delta, p, m):
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
    omega = cp.Variable(K)

    # Define the constraints
    constraints = [
        omega >= 0,
        omega <= 1,
        w == cp.sum(omega),
        Qx - cp.sum(m * p) - w + 2 * cp.sum(cp.multiply(m, cp.multiply(omega, p))) <= delta,
        Qx - cp.sum(m * p) - w + 2 * cp.sum(cp.multiply(m, cp.multiply(omega, p))) >= -delta
    ]

    # Define the objective function
    objective = cp.Maximize(w)

    # Formulate the problem
    prob = cp.Problem(objective, constraints)

    # Solve the problem
    prob.solve()

    # Print the results
    if prob.status == cp.OPTIMAL or prob.status == cp.OPTIMAL_INACCURATE:
        print("Optimal Value: w = ", w.value)
        print("Optimal Solution: Omega = ", omega.value)
    else:
        print("Something went wrong!")
        print(prob.status)

    if K == 1:
        assert np.isclose(w.value, (Qx - np.sum(p) + delta) / (1 - 2 * np.sum(p)))
    elif K > 1:
        assert w.value <= Qx

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: ./Network_Optimization.py Qx delta p m")
        print("Example: ./Network_Optimization.py 0.08 0.005 0.01 0.02 0.03 0.04 0.05 0.2 0.2 0.2 0.2 0.2")
        sys.exit(1)

    Qx = float(sys.argv[1])
    delta = float(sys.argv[2])

    # Calculate the number of elements in p and m
    num_elements = (len(sys.argv) - 3) // 2

    # Parse p and m as lists of floats from the command line arguments
    p = np.array([float(x) for x in sys.argv[3:3 + num_elements]])
    m = np.array([float(x) for x in sys.argv[3 + num_elements:]])

    main(Qx, delta, p, m)