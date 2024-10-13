import numpy as np
import matplotlib.pyplot as plt
import os

from read_file import read_data
from algorithms import *
from utils import *

def main():
    n_exp = 100
    sz_sample = 50
    degs = list(range(0, 91, 10))

    np.random.seed(42)

    mean_errors_R = []
    mean_errors_T = []

    for d in degs:

        P, q_as, q_bs = read_data(os.path.join('./data', f'xx_{d//10}0_10_5_0_10000_45_384_288_100.data'))
        R, t = P[:, :-1], P[:, -1].reshape(-1, 1)
        samples = generate_samples(q_as, q_bs, sz_sample, n_exp)
        
        errs_R, errs_T = [], []
        for q_a, q_b in samples:
            q_a, q_b, T_a, T_b = hartley_normalization(q_a, q_b)
            E = eight_point_algorithm(q_a, q_b)
            #E = recover_actual_essential_matrix(E, T_a, T_b)
            u, d_, vT = decompose_essential_matrix(E)
            R_b, t_b = find_rotation_and_translation2(q_a, q_b, u, d_, vT)

            e_R, e_T = calculate_error(R, t, R_b, t_b)
            errs_R.append(e_R)
            errs_T.append(e_T)

        mean_err_R, mean_err_T = np.mean(errs_R), np.mean(errs_T)
        print(f'Degree {d}:')
        print(f'        Mean rotation error: {mean_err_R}')
        print(f'     Mean translation error: {mean_err_T}')
        print()

        mean_errors_R.append(mean_err_R)
        mean_errors_T.append(mean_err_T)
    
    plt.plot(degs, mean_errors_R, '-o')
    plt.plot(degs, mean_errors_T, '-o')
    plt.show()

main()