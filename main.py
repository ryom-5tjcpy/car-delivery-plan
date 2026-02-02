# main.py

import numpy as np
import openjij as oj
from openjij import BinaryQuadraticModel
import pandas as pd

def create_evaluate1(n: int, delivery: pd.Series):
    e = {(i, j): 0 for i in range(n) for j in range(i + 1, n)}
    s = delivery.str.split('→')

    for i in range(n):
        for j in range(i + 1, n):
            a = s[i]
            b = s[j]

            m1 = len(a)
            m2 = len(b)
            if m1 == 1 and m2 == 1:
                e[i, j] = 0 if a[0] == b[0] else 2
            elif m1 == 1 and m2 == 2:
                if a[0] == b[1]:
                    e[i, j] = 1
                elif a[0] == b[0]:
                    e[i, j] = 1
                else:
                    e[i, j] = 2
            elif m1 == 2 and m2 == 2:
                if a[0] == b[0]:
                    e[i, j] = 0 if a[1] == b[1] else 1
                else:
                    e[i, j] = 1 if a[1] == b[1] else 2

    return e



def solve_combinatorial_problem(n: int, k: int, e):
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
    for i in quadratic_terms:
        quadratic_terms[i] = quadratic_terms[i] + e[i]

    bqm = BinaryQuadraticModel(linear_terms, quadratic_terms, 0.0, 'BINARY')

    # Set up the OpenJij sampler (default is SA, simulated annealing)
    sampler = oj.SASampler()

    # Run the problem on the sampler
    sampleset = sampler.sample(bqm, num_reads=100) # Increased num_reads for potentially better results

    return sampleset

if __name__ == "__main__":
    input_data = pd.read_excel('data/sample/山陽IC_INPUT.xlsx', header=1, usecols='B:J')
    delivery_vehicles = pd.read_csv('data/sanyo/delivery-vehicle.csv')

    input_data = input_data.drop(87).reset_index(drop=True)
    delivery = input_data['回送先']

    n = len(input_data)
    k = delivery_vehicles['n'][0]

    e = create_evaluate1(n, delivery)

    samplest = solve_combinatorial_problem(n, k, e)
    print(samplest)
    print(sum(samplest.first.sample[i] for i in range(n)))

    key = np.zeros(n, dtype=bool)
    for i in range(n):
        key[i] = samplest.first.sample[i] == 1
    
    print(input_data[key])
