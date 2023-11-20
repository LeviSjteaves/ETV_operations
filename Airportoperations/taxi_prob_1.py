# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 15:27:26 2023

@author: levi1
"""

from scipy.optimize import linprog

# Define coefficients for the linear part of the objective function
c_linear = [-1, -2, -3]  # Negate for minimization

# Define the constant term
constant_term = 10  # Replace with your actual constant term

# Set up the inequality constraints
A = [[-1, 1, 0]]  # Example inequality constraints
b = [1]  # Example right-hand side value

# Set up the bounds for the decision variables
bounds = [(0, None), (0, None), (0, None)]  # Lower bounds only, adjust as needed

# Solve the linear programming problem
result = linprog(c_linear, A_ub=A, b_ub=b, bounds=bounds)

# Subtract the constant term from the optimal objective value
optimal_objective = result.fun - constant_term

# Display results
print('Optimal solution for decision variables:', result.x)
print('Optimal objective value (excluding constant term):', result.fun)
print('Optimal value for the constant term:', constant_term)
print('Optimal objective value (including constant term):', optimal_objective)