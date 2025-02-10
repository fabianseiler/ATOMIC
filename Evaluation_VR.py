import json
import pathlib
import pickle
import numpy as np
from src.util import comb8

"""
put deviation results in ./deviation_results/algorithm/dev_R{r}_V{v}
"""

r_on, r_off, v_on, v_off = "10k", "1000k", "-10m", "700m"

def get_combination_from_result_index(index: int, dev_r: int, dev_v: int) -> str:
    if dev_r == 0 and dev_v == 0:
        return f'r_{r_on}_{r_off}_v_{v_on}_{v_off}'
    elif dev_r == 0 and dev_v != 0:
        v_on_c, v_off_c = comb8(dev_v, v_on, v_off)
        return f'r_{r_on}_{r_off}_v_{v_on_c[index]}_{v_off_c[index]}'
    elif dev_r != 0 and dev_v == 0:
        r_on_c, r_off_c = comb8(dev_r, r_on, r_off)
        return f'r_{r_on_c[index]}_{r_off_c[index]}_v_{v_on}_{v_off}'
    else:
        r_on_c, r_off_c = comb8(dev_r, r_on, r_off)
        v_on_c, v_off_c = comb8(dev_v, v_on, v_off)
        return f'r_{r_on_c[index%8]}_{r_off_c[index%8]}_v_{v_on_c[index//8]}_{v_off_c[index//8]}'

dir = pathlib.Path(__file__).parent / 'deviation_results'

# algorithms = ['exact_rohani', 'exact_karimi', 'exact_seiler', 'exact_teimoory']
algorithms = ['exact_rohani', 'exact_karimi', 'exact_seiler']

configs = []
for algo in algorithms:
    with open(f'./configs/Serial_{algo}.json', 'r') as f:
        configs.append(json.load(f))

# print(configs)


dev_r = [0, 10, 20, 30, 40, 50]
dev_v = [0, 1, 2, 3, 4, 5, 6]

passmatrices = []

for _, _ in enumerate(algorithms):
    passmatrices.append(np.zeros((len(dev_r), len(dev_v))))


for algo_index, algo in enumerate(algorithms):
    memristor_names = configs[algo_index]['memristors']
    output_names = configs[algo_index]['outputs']
    output_indices = [memristor_names.index(name) for name in output_names]
    
    output_values = [value for value in configs[algo_index]['output_states'].values()]

    print(f'\n -------- Algorithm: {algo} -------- \n')

    for r in dev_r:
        for v in dev_v:
            file = dir / algo / f'dev_R{r}_V{v}'

            passing = True
            with open(file, 'rb') as f:
                data = pickle.load(f)

            print(f'Algorithm: {algo}, Deviation R: {r}, Deviation V: {v}')

            for input_index, results in enumerate(data):
                name = bin(input_index)[2:].zfill(len(configs[algo_index]["inputs"]))
                # print(f'Input {name}')
                for res_index, res in enumerate(results): # either 1, 8 or 64 results
                    for i, output_index in enumerate(output_indices):
                        if abs(res[output_index] - output_values[i][input_index]) > 0.33:
                            print(f'Input {name} failed at {get_combination_from_result_index(res_index, r, v)}')
                            print(f'Output {output_names[i]}: {res[output_index]} Expected: {output_values[i][input_index]}')
                            passing = False
                            break
                    if not passing:
                        break
                if not passing:
                    break
            print('Passed' if passing else 'Failed')
            passmatrices[algo_index][dev_r.index(r), dev_v.index(v)] = 1 if passing else 0
            print('------------------------')

print('Pass Matrices')
print("  V --> \n\
R\n\
|\n\
v")


for i, algo in enumerate(algorithms):
    print(f'Pass Matrix for {algo}')
    print(passmatrices[i])
                