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
    "中古車C": np.array([34.748790746336795, 134.0236317423281]),
    "赤磐C": np.array([34.72391911569691, 133.98226146378104]),
    "津山B": np.array([35.05537760000001, 133.99844677116405]),
    "真庭B": np.array([35.00940701246312, 133.73504602883594]),
    "PDI": np.array([34.737335733341055, 134.01831081349152]),
    "営業所": np.array([34.622486758516715, 133.80092669685916]),
    "○○水島店": np.array([34.52978972589754, 133.75017584976348]),
    "○○泉田": np.array([34.63200770824816, 133.92227706152727]),
    "○○岡山": np.array([34.628346669104545, 133.87781675432046]),
    "○○東岡山": np.array([34.67051745172791, 133.9734545314257]),
    "○○オート": np.array([34.927103787508734, 133.49877569916796]),
    "○○オートセンター": np.array([34.55142595564274, 133.87368200370835]),
    "○○会社": np.array([34.47946213728613, 133.8019328807972]),
    "○○倉敷": np.array([34.586288134222016, 133.7966207167915]),
    "○○岡山青江店": np.array([34.52978972589754, 133.75017584976348]),
    "○○車体": np.array([34.6362446037451, 133.9263989083228]),
    "○○車輛": np.array([34.68094485197269, 133.99488344034324]),
    "△△車輛": np.array([34.71033749824907, 133.86233713956912]),
    "○○岡山営業所": np.array([34.6199524505259, 133.87903214522797]),
    "○○岡山新見店": np.array([34.994260979999055, 133.44116203495304]),
    "○○岡山津山店": np.array([35.05477433641165, 134.00309984286147]),
    "倉敷店": np.array([34.586852950295494, 133.78487538709288]),
    "備前店": np.array([34.74636215944565, 134.20202166391874]),
    "○○倉敷店": np.array([34.586288134222016, 133.7966207167915]),
    "○○特販部": np.array([34.63200770824816, 133.92227706152727])
}

def eauclid_norm(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

load_capacity = 5
lam_load_cap = 10000
N_DATA = len(df)

def get_equality_constraint(n: int, k: int, lam: float):
    linear_terms = {i: lam * (1 - 2 * k) for i in range(n)}
    quadratic_terms = {(i, j): 2 * lam for i in range(n) for j in range(i + 1, n)}
    return linear_terms, quadratic_terms

def get_movement_distance_constraint(n: int):
    quadratic_terms = {}
    for i in range(n):
        o_i = df['origin'].iloc[i]
        origin_i = coords[o_i] if pd.notna(o_i) else coords['PDI']
        destination_i = coords[df["destination"].iloc[i]]

        for j in range(i + 1, n):
            o_j = df['origin'].iloc[j]
            origin_j = coords[o_j] if pd.notna(o_j) else coords['PDI']
            destination_j = coords[df["destination"].iloc[j]]

            quadratic_terms[(i, j)] = eauclid_norm(destination_i - origin_i, destination_j - origin_j)

    return quadratic_terms

linear_terms = {}
quadratic_terms = {}

linear_equ, quadratic_equ = get_equality_constraint(N_DATA, load_capacity, lam_load_cap)
qua = get_movement_distance_constraint(N_DATA)

for k in range(2):
    for i in range(N_DATA):
        linear_terms[k * N_DATA + i] = linear_equ[i]

        for j in range(i + 1, N_DATA):
            quadratic_terms[k * N_DATA + i, k * N_DATA + j] = qua[i, j] + quadratic_equ[i, j]

bqm = BinaryQuadraticModel(linear=linear_terms, quadratic=quadratic_terms, offset=0.0, vartype='BINARY')

sampler = oj.SASampler()

sampleset = sampler.sample(bqm, num_reads=1000)

print(sum(sampleset.first.sample[i] for i in range(N_DATA)))

key = np.zeros(N_DATA, dtype=bool)
for i in range(N_DATA):
    key[i] = sampleset.first.sample[i] == 1

print(df[key])

print()

key = np.zeros(N_DATA, dtype=bool)
for i in range(N_DATA):
    if sampleset.first.sample[N_DATA + i] == 1:
        key[i] = True

print(df[key])