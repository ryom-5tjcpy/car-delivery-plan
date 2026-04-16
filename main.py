# main.py

import numpy as np
import openjij as oj
from openjij import BinaryQuadraticModel
import pandas as pd

def get_source_and_destination(delivery):
    n = len(delivery)
    if n == 2:
        return delivery[0], delivery[1]
    
    return None, delivery[0]

def create_weights(n: int, deliveries: pd.Series, src: str):
    quad = {(i, j): 0 for i in range(n) for j in range(i + 1, n)}
    diag = {i: 0 for i in range(n)}

    for i in range(n):
        s_i = deliveries[i]
        src_i, des_i = get_source_and_destination(s_i)

        if src_i == None:
            if des_i != src:
                diag[i] = 2
            else:
                diag[i] = 10
        elif src_i != src:
            diag[i] = 10

        for j in range(i + 1, n):
            s_j = deliveries[j]
            src_j, des_j = get_source_and_destination(s_j)

            if src_i == src_j:
                quad[i, j] = 0 if des_i == des_j else 5 if src_i == None else 1
            else:
                quad[i, j] = 10


    return diag, quad


def solve_combinatorial_problem(n: int, k: int, linear: dict, quadratic: dict):
    """
    Solves a combinatorial optimization problem with the constraint sum(x_i) = k
    for n binary variables x_i using quantum annealing with OpenJij.

    Parameters:
    n (int): The number of binary variables x_i.
    k (int): The target sum for the binary variables.

    Returns:
    openjij.response.Response: The sampleset containing the results from the sampler.
    """
    """
    追加したパラメータ：
    eval: 評価関数の係数を格納した配列。
          eval[0]とeval[1]があり、eval[0]は線形項の係数、eval[1]は交互作用の係数である。
    """

    # H = (sum(x_i) - k)^2 = sum(x_i)^2 - 2k*sum(x_i) + k^2
    # Since x_i are binary, x_i^2 = x_i
    # sum(x_i)^2 = sum(x_i) + 2 * sum_{i<j} x_i*x_j
    # So, H = sum(x_i) + 2 * sum_{i<j} x_i*x_j - 2k*sum(x_i) + k^2
    # H = (1 - 2k)*sum(x_i) + 2 * sum_{i<j} x_i*x_j + k^2

    linear_terms = {i: (linear[i] + 1 - 2 * k) + linear[i] for i in range(n)}
    quadratic_terms = {(i, j): quadratic[i, j] + 2 for i in range(n) for j in range(i + 1, n)}

    bqm = BinaryQuadraticModel(linear_terms, quadratic_terms, 0.0, 'BINARY')

    # Set up the OpenJij sampler (default is SA, simulated annealing)
    sampler = oj.SASampler()

    # Run the problem on the sampler
    sampleset = sampler.sample(bqm, num_reads=1000) # Increased num_reads for potentially better results

    return sampleset

def conv_delivery(delivery: pd.Series):
    """
    パラメータの表記ゆれを変形し，
    回収先と配送先に分ける．
    
    :param delivery: 回収地点と配送先が記入されたデータ
    :type delivery: pd.Series
    """
    # 表記ゆれを変形する
    delivery = delivery.str.replace("（法人）"  , "")
    delivery = delivery.str.replace("B", "店")

    # " → "で出発地点と配送先に分ける
    return delivery.str.split(' → ')

if __name__ == "__main__":
    input_data = pd.read_excel('data/sample/山陽IC_INPUT.xlsx', header=1, usecols='B:J')
    delivery_vehicles = pd.read_csv('data/sanyo/delivery-vehicle.csv')

    input_data = input_data.drop(87).reset_index(drop=True)
    deliveries = input_data['回送先']

    n = len(input_data)
    k = delivery_vehicles['n'][0]
    conversed_deliveries = conv_delivery(deliveries)
    diag, quad = create_weights(n, conversed_deliveries, "野田店")

    sampleset = solve_combinatorial_problem(n, k, diag, quad)
    #print(sampleset)
    print(sum(sampleset.first.sample[i] for i in range(n)))
    print(sampleset.first.energy)

    key = np.zeros(n, dtype=bool)
    for i in range(n):
        key[i] = sampleset.first.sample[i] == 1
    
    print(deliveries[key])