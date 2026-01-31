# main.py

import openjij as oj
from openjij import BinaryQuadraticModel

# Define the problem (simple Ising model)
# H = -J_12 * s_1 * s_2 - h_1 * s_1 - h_2 * s_2
# J_12 = -1, h_1 = 0, h_2 = 0

bqm = BinaryQuadraticModel({0: 0.0, 1: 0.0}, {(0, 1): -1.0}, 0.0, 'SPIN')

# Set up the OpenJij sampler (default is SA, simulated annealing)
sampler = oj.SASampler()

# Run the problem on the sampler
sampleset = sampler.sample(bqm, num_reads=10)

# Print the results
print(sampleset.first.sample)
print(sampleset.first.energy)
