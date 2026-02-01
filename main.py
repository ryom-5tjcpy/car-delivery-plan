# main.py

import openjij as oj
from openjij import BinaryQuadraticModel
# import pandas as pd # pandas is not used in the new problem, so it can be removed

def solve_combinatorial_problem(n: int, k: int):
    """
    Solves a combinatorial optimization problem with the constraint sum(x_i) = k
    for n binary variables x_i using quantum annealing with OpenJij.

    Parameters:
    n (int): The number of binary variables x_i.
    k (int): The target sum for the binary variables.

    Returns:
    openjij.response.Response: The sampleset containing the results from the sampler.
    """

    # H = (sum(x_i) - k)^2 = sum(x_i)^2 - 2k*sum(x_i) + k^2
    # Since x_i are binary, x_i^2 = x_i
    # sum(x_i)^2 = sum(x_i) + 2 * sum_{i<j} x_i*x_j
    # So, H = sum(x_i) + 2 * sum_{i<j} x_i*x_j - 2k*sum(x_i) + k^2
    # H = (1 - 2k)*sum(x_i) + 2 * sum_{i<j} x_i*x_j + k^2

    linear_terms = {i: (1 - 2 * k) for i in range(n)}
    quadratic_terms = {(i, j): 2 for i in range(n) for j in range(i + 1, n)}

    bqm = BinaryQuadraticModel(linear_terms, quadratic_terms, 0.0, 'BINARY')

    # Set up the OpenJij sampler (default is SA, simulated annealing)
    sampler = oj.SASampler()

    # Run the problem on the sampler
    sampleset = sampler.sample(bqm, num_reads=100) # Increased num_reads for potentially better results

    return sampleset

if __name__ == "__main__":
    # Example usage:
    n_vars = 5
    target_k = 2
    print(f"Solving for n={n_vars}, k={target_k}")
    sampleset = solve_combinatorial_problem(n_vars, target_k)

    # Print the results
    print("--------------------------------------------------------------------------------")
    print("Sampleset Results:")  

    # Verify the constraint for the best solution
    best_sample = sampleset.first.sample
    sum_x = sum(best_sample[i] for i in range(n_vars))
    print(f"Best sample: {best_sample}")
    print(f"Sum of x_i in best sample: {sum_x}")
    print(f"Target k: {target_k}")
    if sum_x == target_k:
        print("Constraint sum(x_i) = k is satisfied!")
    else:
        print("Constraint sum(x_i) = k is NOT satisfied for the best sample.")

    print("\nTrying another example:")
    n_vars_2 = 10
    target_k_2 = 3
    print(f"Solving for n={n_vars_2}, k={target_k_2}")
    sampleset_2 = solve_combinatorial_problem(n_vars_2, target_k_2)
    print("--------------------------------------------------------------------------------")
    print("Sampleset Results:")
    best_sample_2 = sampleset_2.first.sample
    sum_x_2 = sum(best_sample_2[i] for i in range(n_vars_2))
    print(f"Best sample: {best_sample_2}")
    print(f"Sum of x_i in best sample: {sum_x_2}")
    print(f"Target k: {target_k_2}")
    if sum_x_2 == target_k_2:
        print("Constraint sum(x_i) = k is satisfied!")
    else:
        print("Constraint sum(x_i) = k is NOT satisfied for the best sample.")
