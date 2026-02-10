# main.py

import numpy as np
import openjij as oj
from openjij import BinaryQuadraticModel
import pandas as pd

def create_evaluate1(n: int, delivery: pd.Series, sp: str):
    """
    量子アニーリングの評価関数の係数を作成する関数。
    
    :param n: deliveryパラメータの個数
    :type n: int
    :param delivery: 出発地点と配送先を記入した配列
    :type delivery: pd.Series
    :param sp: 出発地点
    :type sp: str

    Returns:
    評価関数の係数
    """
    # 係数の初期化
    quad = {(i, j): 0 for i in range(n) for j in range(i + 1, n)}
    diag = {i: 1 for i in range(n)}

    for i in range(n):
        s_i = delivery[i]
        m_i = len(s_i)

        # 線形項の係数を調整する
        # 出発地点がパラメータspと異なる場合はペナルティが大きくなる。
        # 出発地点が不明なものは少しだけペナルティが大きいが、出発地点が異なる場合と比べると小さい。
        if m_i == 2 and s_i[0] != sp:
            diag[i] = 10
        elif m_i == 1:
            diag[i] = 2

        # 交互作用項の係数を調整する
        for j in range(i + 1, n):
            s_j = delivery[j]

            m_j = len(s_j)
            # s_i, s_jともに配送先しか記載がない場合
            if m_i == 1 and m_j == 1:
                quad[i, j] = 0 if s_i[0] == s_j[0] else 5
            # s_iが配送先しか記載がない場合
            elif m_i == 1 and m_j == 2:
                if s_i[0] == s_j[1]:
                    quad[i, j] = 0
                elif s_i[0] == s_j[0]:
                    quad[i, j] = 10
                else:
                    quad[i, j] = 20
            # s_jが配送先しか記載がない場合
            elif m_i == 2 and m_j == 1:
                if s_i[0] == s_j[0]:
                    quad[i, j] = 10
                elif s_i[1] == s_j[0]:
                    quad[i, j] = 0
                else:
                    quad[i, j] = 20
            # s_i, s_jともに出発地点が記載されている場合
            elif m_i == 2 and m_j == 2:
                if s_i[0] == s_j[0]:
                    quad[i, j] = 0 if s_i[1] == s_j[1] else 2
                else:
                    quad[i, j] = 10

    return diag, quad

def solve_combinatorial_problem(n: int, k: int, eval):
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

    linear_terms = {i: (eval[0][i] + 1 - 2 * k) + eval[0][i] for i in range(n)}
    quadratic_terms = {(i, j): eval[1][i, j] + 2 for i in range(n) for j in range(i + 1, n)}

    bqm = BinaryQuadraticModel(linear_terms, quadratic_terms, 0.0, 'BINARY')

    # Set up the OpenJij sampler (default is SA, simulated annealing)
    sampler = oj.SASampler()

    # Run the problem on the sampler
    sampleset = sampler.sample(bqm, num_reads=10000) # Increased num_reads for potentially better results

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
    delivery = input_data['回送先']

    n = len(input_data)
    k = delivery_vehicles['n'][0]
    conversed_delivery = conv_delivery(delivery)

    eval = create_evaluate1(n, conversed_delivery, "野田店")

    sampleset = solve_combinatorial_problem(n, k, eval)
    #print(sampleset)
    print(sum(sampleset.first.sample[i] for i in range(n)))
    print(sampleset.first.energy)

    key = np.zeros(n, dtype=bool)
    for i in range(n):
        key[i] = sampleset.first.sample[i] == 1
    
    print(delivery[key])