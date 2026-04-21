import numpy as np
import pandas as pd
import openjij as oj
from openjij import BinaryQuadraticModel

input_data = pd.read_excel('data/sample/山陽IC_INPUT.xlsx', header=1, usecols='B:J')

origins_and_destinations = input_data['回送先']
origins_and_destinations = origins_and_destinations.dropna()
origins_and_destinations = origins_and_destinations.str.replace("（法人）", "")
origins_and_destinations = origins_and_destinations.str.replace("（岡山B）", "")

origins_and_destinations = origins_and_destinations.str.split(" → ")
df = pd.DataFrame(columns=['origin', 'destination'], dtype = 'str')
for i, r in enumerate(origins_and_destinations):
    N_DATA = len(r)
    if N_DATA >= 2:
        df.loc[i] = [r[0], r[1]]
    else:
        df.loc[i] = [np.nan, r[0]]

df = df.dropna()

coords = {
    "野田店": np.array([34.654796952803004, 133.90363607639333]),
    "高野店": np.array([35.07383804491031, 134.0504428999516]),
    "津山店": np.array([35.05447437087948, 133.97621559056554]),
    "水島店": np.array([34.52257077684557, 133.766949565224]),
    "吉備路店": np.array([34.671719620256056, 133.77132609451738]),
    "倉敷中央店": np.array([34.58715902699975, 133.77061928562654]),
    "倉敷中島店": np.array([34.58135157279875, 133.73489229636525]),
    "中庄店": np.array([34.63701941823053, 133.81998295403938]),
    "平島店": np.array([34.705126772918085, 134.05694605404224]),
    "十日市店": np.array([34.705126772918085, 134.05694605404224]),
    "高屋店": np.array([34.6740798816033, 133.97155189275097]),
    "玉野紅陽台店": np.array([34.54297235593657, 133.89015340112158]),
    "中古車C": np.array([34.72258633711081, 133.9839577391999]),
    "赤磐C": np.array([34.748790746336795, 134.0236317423281]),
    "津山B": np.array([35.05537760000001, 133.99844677116405]),
    "真庭B": np.array([35.00940701246312, 133.73504602883594]),
    "PDI": np.array([34.737335733341055, 134.01831081349152]),
    "営業所": np.array([34.622486758516715, 133.80092669685916])
}

def eauclid_norm(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

load_capacity = 5
lam_load_cap = 10000
N_DATA = len(df)

linear_terms = {i: lam_load_cap * (1 - 2 * load_capacity) for i in range(N_DATA)}
quadratic_terms = {(i, j): 2 * lam_load_cap for i in range(N_DATA) for j in range(i + 1, N_DATA)}

for i in range(N_DATA):
    origin_i = coords.get(df['origin'].iloc[i], np.zeros(2))
    destination_i = coords.get(df['destination'].iloc[i], np.zeros(2))
    for j in range(i + 1, N_DATA):
        origin_j = coords.get(df['origin'].iloc[j], np.zeros(2))
        destination_j = coords.get(df['destination'].iloc[j], np.zeros(2))
        
        # Example QUBO term (replace with actual logic)
        quadratic_terms[(i, j)] += eauclid_norm(destination_i - origin_i, destination_j - origin_j)

bqm = BinaryQuadraticModel(linear=linear_terms, quadratic=quadratic_terms, offset=0.0, vartype='BINARY')

sampler = oj.SASampler()

sampleset = sampler.sample(bqm, num_reads=1000)

print(sum(sampleset.first.sample[i] for i in range(N_DATA)))
print(sampleset.first.energy)

key = np.zeros(N_DATA, dtype=bool)
for i in range(N_DATA):
    key[i] = sampleset.first.sample[i] == 1

print(df[key])